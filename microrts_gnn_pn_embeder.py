import sys,os,time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from collections import deque
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse

from utils import layer_init, calculate_gae, remake_mask, MaskedCategorical

lr = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
max_clip_range = 4
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
seed = 1
num_envs = 32
num_steps = 512
num_closest = 8

cuda = True
if cuda and torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')
pae_length = 256


map_name = 'TwoBasesWorkers12x12'
if map_name == 'TwoBasesWorkers12x12':
    h=12
    w=12
    map_path = 'maps/12x12/TwoBasesWorkers12x12.xml'
elif map_name == 'basesWorkers16x16':
    h = 16
    w = 16
    map_path = 'maps/16x16/basesWorkers16x16.xml'
elif map_name == 'basesWorkers8x8':
    h = 8
    w = 8
    map_path = 'maps/8x8/basesWorkers8x8.xml'

action_space = [h*w, 6, 4, 4, 4, 4, 7, 49]
observation_space = [h,w,27]

def process_states(states, selected_units, map_size):
    states = torch.tensor(states)
    nodes_features = []
    for state,selected_unit in zip(states,selected_units):
        if selected_unit == -1:
            nodes_features.append(torch.zeros((num_closest,29)))
        else:
            selected_unit_x = selected_unit//map_size
            selected_unit_y = selected_unit%map_size
            nodes_feature = []
            for y in range(map_size):
                for x in range(map_size):
                    if state[y][x][13] != 1:
                        d = abs(selected_unit_x-x)+abs(selected_unit_y-y)
                        if d >0:
                            nodes_feature.append((d,torch.cat((torch.tensor([x/map_size,y/map_size]),state[y][x]))))
                        elif d == 0:
                            nodes_feature.append((0,torch.cat((torch.tensor([x/map_size,y/map_size]),state[y][x]))))
            #sort by distance and get the first 8
            if len(nodes_feature) > num_closest:
                nodes_feature.sort(key=lambda x:x[0])
                nodes_feature = nodes_feature[:num_closest]
            else:
                for _ in range(num_closest-len(nodes_feature)):
                    nodes_feature.append((-1,torch.zeros(29)))
            nodes_features.append(torch.stack([x[1] for x in nodes_feature]))
    return torch.stack(nodes_features)

                

#main network
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.embedding = layer_init(nn.Linear(29, 8), std=0.01)

        self.encoder = nn.LSTM(8, 128, batch_first=True)
        self.decoder = nn.LSTM(128, 128, batch_first=True)

        self.policy_type = layer_init(nn.Linear(128, 6), std=0.01)
        self.policy_move = layer_init(nn.Linear(128, 4), std=0.01)
        self.policy_harvest = layer_init(nn.Linear(128, 4), std=0.01)
        self.policy_return = layer_init(nn.Linear(128, 4), std=0.01)
        self.policy_produce = layer_init(nn.Linear(128, 4), std=0.01)
        self.policy_produce_type = layer_init(nn.Linear(128, 7), std=0.01)
        self.policy_attack = layer_init(nn.Linear(128, 49), std=0.01)
        
        self.value = layer_init(nn.Linear(128, 1), std=1)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = self.embedding(x)
        # Encoder
        encoder_outputs, hidden = self.encoder(x)

        # Decoder
        decoder_state = hidden
        decoder_input = torch.zeros((batch_size, 1, 128)).to(x.device)
        decoder_outputs = []

        for t in range(seq_len):
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        hidden_z = decoder_outputs[-1][:,-1,:]

        type_distris = MaskedCategorical(self.policy_type(hidden_z))
        move_distris = MaskedCategorical(self.policy_move(hidden_z))
        harvest_distris = MaskedCategorical(self.policy_harvest(hidden_z))
        return_distris = MaskedCategorical(self.policy_return(hidden_z))
        produce_distris = MaskedCategorical(self.policy_produce(hidden_z))
        produce_type_distris = MaskedCategorical(self.policy_produce_type(hidden_z))
        attack_distris = MaskedCategorical(self.policy_attack(hidden_z))

        value = self.value(hidden_z)

        return [type_distris,move_distris,harvest_distris,return_distris,produce_distris,produce_type_distris,attack_distris],value

class Agent:
    def __init__(self,net:ActorCritic) -> None:
        self.net = net
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.pae_length = pae_length
        self.action_space = action_space
        self.out_comes = deque( maxlen= 1000)
        self.env = MicroRTSVecEnv(
                num_envs=self.num_envs,
                max_steps=5000,
                ai2s=[microrts_ai.coacAI for _ in range(self.num_envs)],
                map_path=map_path,
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
        self.obs = self.env.reset()
        self.exps_list = [[] for _ in range(self.num_envs)]
    
    @torch.no_grad()
    def get_sample_actions(self,processed_states,units):
        distris,_ = self.net(processed_states.to(device))
        
        for i in range(len(units)): 
            if units[i] == -1:
                units[i] = 0

        action_components = [torch.Tensor(units)]

        action_mask_list = np.array(self.env.vec_client.getUnitActionMasks(np.array(units))).reshape(len(units), -1)
        
        action_mask_list = remake_mask(action_mask_list)

        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_space[1:], dim=1)
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris,action_masks)]
            
        actions = torch.stack(action_components)
        masks = torch.Tensor(action_mask_list)
        log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris,actions[1:])])
        
        return actions.T.cpu().numpy(), masks.cpu().numpy(),log_probs.T.cpu().numpy()
    
    def sample_env(self, check=False): 
        if check:
           step_record_dict = dict()
           rewards = []
           log_probs = [] 
        while len(self.exps_list[0]) < self.num_steps:
            self.env.render()
            unit_masks = np.array(self.env.vec_client.getUnitLocationMasks()).reshape(self.num_envs, -1)
            #randomly sample units from unit_mask where the value is 1 return the index of the unit
            unit_list = []
            for unit_mask in unit_masks:
                if np.sum(unit_mask) == 0:
                    unit_list.append(-1)
                else:
                    unit_list.append(np.random.choice(np.where(unit_mask == 1)[0]))
            processed_states = process_states(self.obs, unit_list, w)

            action,mask,log_prob=self.get_sample_actions(processed_states,unit_list)
            action = action.astype(np.int32)
            next_obs, rs, done_n, infos = self.env.step(action)
        
            if check:
                rewards.append(np.mean(rs))
                log_probs.append(np.mean(log_prob))
            
            for i in range(self.num_envs):
                if done_n[i]:
                    done = True
                else:
                    done = False
                self.exps_list[i].append([processed_states[i],action[i],rs[i],mask[i],done,log_prob[i]])
                if check:
                    if done_n[i]:
                        if infos[i]['raw_rewards'][0] > 0:
                            self.out_comes.append(1.0)
                        else:
                            self.out_comes.append(0.0)
                
            self.obs=next_obs

        train_exps = self.exps_list
        self.exps_list = [ exps[self.pae_length:self.num_steps] for exps in self.exps_list ]

        if check:
            mean_win_rates = np.mean(self.out_comes) if len(self.out_comes)>0 else 0.0
            print(mean_win_rates)

            step_record_dict['sum_rewards'] = np.sum(rewards)
            step_record_dict['mean_rewards'] = np.mean(rewards)
            step_record_dict['mean_log_probs'] = np.mean(log_probs)
            step_record_dict['mean_win_rates'] = mean_win_rates
            return train_exps, step_record_dict
        
        return train_exps

class Calculator:
    def __init__(self,net:ActorCritic) -> None:
        self.net = net
        self.train_version = 0
        self.pae_length = pae_length
        
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda', 0)
        else:
            self.device = torch.device('cpu')
        
        self.calculate_net = ActorCritic()
        self.calculate_net.to(self.device)
    
        self.share_optim = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        
        
        self.processed_states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None

    def begin_batch_train(self, samples_list: list):    
        s_processed_states = [np.array([np.array(s[0]) for s in samples]) for samples in samples_list]
        s_actions = [np.array([s[1] for s in samples]) for samples in samples_list]
        s_masks = [np.array([s[3] for s in samples]) for samples in samples_list]
        s_log_probs = [np.array([s[5] for s in samples]) for samples in samples_list]
        
        s_rewards = [np.array([s[2] for s in samples]) for samples in samples_list]
        s_dones = [np.array([s[4] for s in samples]) for samples in samples_list]
        
        self.processed_states = [torch.Tensor(states).to(self.device) for states in s_processed_states]
        self.actions = [torch.Tensor(actions).to(self.device) for actions in s_actions]
        self.old_log_probs = [torch.Tensor(log_probs).to(self.device) for log_probs in s_log_probs]
        self.marks = [torch.Tensor(marks).to(self.device) for marks in s_masks]
        self.rewards = s_rewards
        self.dones = s_dones
        
        self.processed_states_list = torch.cat([states[0:self.pae_length] for states in self.processed_states])
        self.actions_list = torch.cat([actions[0:self.pae_length] for actions in self.actions])
        self.old_log_probs_list = torch.cat([old_log_probs[0:self.pae_length] for old_log_probs in self.old_log_probs])
        self.marks_list = torch.cat([marks[0:self.pae_length] for marks in self.marks])

    def calculate_samples_gae(self):
        np_advantages = []
        np_returns = []
        
        for states,rewards,dones in zip(self.processed_states,self.rewards,self.dones):
            with torch.no_grad():
                _,values = self.calculate_net(states)
                            
            advantages,returns = calculate_gae(values.cpu().numpy().reshape(-1),rewards,dones,gamma,gae_lambda)
            np_advantages.extend(advantages[0:self.pae_length])
            np_returns.extend(returns[0:self.pae_length])
            
        np_advantages = np.array(np_advantages)
        np_returns = np.array(np_returns)
        
        return np_advantages, np_returns
        
    def end_batch_train(self):
        self.processed_states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None

    def get_pg_loss(self,ratio,advantage):      
        clip_coef = clip_range
        max_clip_coef = max_clip_range
        positive = torch.where(ratio >= 1.0 + clip_coef, 0 * advantage,advantage)
        negtive = torch.where(ratio <= 1.0 - clip_coef,0 * advantage,torch.where(ratio >= max_clip_coef, 0 * advantage,advantage))
        return torch.where(advantage>=0,positive,negtive)*ratio
        
    def get_prob_entropy_value(self,states, actions, masks):
        distris,values = self.calculate_net(states)
        action_masks = torch.split(masks, action_space[1:], dim=1)
        distris = [dist.update_masks(mask,device=self.device) for dist,mask in zip(distris,action_masks)]
        log_probs = torch.stack([dist.log_prob(action) for dist,action in zip(distris,actions[1:])])
        entropys = torch.stack([dist.entropy() for dist in distris])
        return log_probs.T, entropys.T, values

    def generate_grads(self):
        grad_norm = max_grad_norm
        
        self.calculate_net.load_state_dict(self.net.state_dict())
        np_advantages,np_returns = self.calculate_samples_gae()
        
        np_advantages = (np_advantages - np_advantages.mean()) / np_advantages.std()
                                                    
        advantage_list = torch.Tensor(np_advantages.reshape(-1,1)).to(self.device)    
        returns_list = torch.Tensor(np_returns.reshape(-1,1)).to(self.device)
        

        mini_batch_number = 1
        mini_batch_size = advantage_list.shape[0]

        for i in range(mini_batch_number):
            start_index = i*mini_batch_size
            end_index = (i+1)* mini_batch_size
            
            mini_states = self.processed_states_list[start_index:end_index]
            mini_actions = self.actions_list[start_index:end_index]
            mini_masks = self.marks_list[start_index:end_index]
            mini_old_log_probs = self.old_log_probs_list[start_index:end_index]
            
            self.calculate_net.load_state_dict(self.net.state_dict())
                
            mini_new_log_probs,mini_entropys,mini_new_values = self.get_prob_entropy_value(mini_states,mini_actions.T,mini_masks)
                        
            mini_advantage = advantage_list[start_index:end_index]
            mini_returns = returns_list[start_index:end_index]
            
            ratio1 = torch.exp(mini_new_log_probs-mini_old_log_probs)
            pg_loss = self.get_pg_loss(ratio1,mini_advantage)

            # Policy loss
            pg_loss = -torch.mean(pg_loss)
            
            entropy_loss = -torch.mean(mini_entropys)
            
            v_loss = F.mse_loss(mini_new_values, mini_returns)

            loss = pg_loss + ent_coef * entropy_loss + v_loss*vf_coef

            self.calculate_net.zero_grad()

            loss.backward()
            
            grads = [
                param.grad.data.cpu().numpy()
                if param.grad is not None else None
                for param in self.calculate_net.parameters()
            ]
                
            # Updating network parameters
            for param, grad in zip(self.net.parameters(), grads):
                param.grad = torch.FloatTensor(grad)
                
            if grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),grad_norm)
            self.share_optim.step()

if __name__ == "__main__":
    evaluate = False

    for i in range(10):
        commemt = "gnn_np_"+map_name+"_v"+str(i)
        writer = SummaryWriter(comment=commemt)
        net = ActorCritic().to(device)
        if evaluate:
            net.load_state_dict(torch.load("saved_model/gnn/ppo_model_3000.pkl"))
        parameters = sum([np.prod(p.shape) for p in net.parameters()])
        print("parameters size is:",parameters)

        agent = Agent(net)
        if not evaluate:
            calculator = Calculator(net)

        MAX_VERSION = 5000
        REPEAT_TIMES = 10

        for version in range(MAX_VERSION):
            strat_time = time.time()
            samples_list,infos = agent.sample_env(check=True)

            for (key,value) in infos.items():
                writer.add_scalar(key,value,version)

            print("version:",version,"reward:",infos["mean_rewards"])
            if not evaluate:
                samples = []

                for s in samples_list:
                    samples.append(s)
            
                calculator.begin_batch_train(samples)
                for _ in range(REPEAT_TIMES):
                    calculator.generate_grads()
                calculator.end_batch_train()
            
                if (version+1) % 500 == 0:
                    torch.save(net.state_dict(), "model/ppo_model_"+commemt+"_"+str(version)+".pkl")
                    torch.save(net, "model/ppo_model_"+commemt+"_"+str(version)+".pth")
            print("time:",time.time()-strat_time)
        writer.close()            


