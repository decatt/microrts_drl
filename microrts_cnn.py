import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import torch
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from collections import deque
import numpy as np
import torch.nn.functional as F
from nets import CNNNet as ActorCritic
from utils import calculate_gae, remake_mask
import torch.nn as nn
import wandb
import random
import argparse

lr = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
max_clip_range = 4
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
cuda = True
device = 'cuda'
pae_length = 256
num_envs = 32
num_steps = 512

ai_dict = {
    "coacAI": microrts_ai.coacAI,
    "rojo": microrts_ai.rojo,
    "mayari": microrts_ai.mayari,
    "randomAI": microrts_ai.randomAI,
    "passiveAI": microrts_ai.passiveAI,
    "workerRushAI": microrts_ai.workerRushAI,
    "lightRushAI": microrts_ai.lightRushAI,
}

parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--map_name', type=str, default='TwoBasesBarracks16x16')
parser.add_argument('--op_ai', type=str, default='coacAI')
args = parser.parse_args()

op_ai = ai_dict[args.op_ai]

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
                ai2s=[op_ai for _ in range(self.num_envs)],
                map_path=map_path,
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
        self.obs = self.env.reset()
        self.exps_list = [[] for _ in range(self.num_envs)]
    
    @torch.no_grad()
    def get_sample_actions(self,states, unit_masks):
        states = torch.Tensor(states)
        distris = self.net.get_distris(states)
        
        unit_masks = torch.Tensor(unit_masks)
        distris[0].update_masks(unit_masks)
        
        units = distris[0].sample()
        action_components = [units]

        action_mask_list = np.array(self.env.vec_client.getUnitActionMasks(units.cpu().numpy())).reshape(len(units), -1)
        action_mask_list = remake_mask(action_mask_list)
        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_space[1:], dim=1) 
        
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris[1:],action_masks)]
            
        actions = torch.stack(action_components)
        masks = torch.cat((unit_masks, torch.Tensor(action_mask_list)), 1)
        log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris,actions)])
        
        return actions.T.cpu().numpy(), masks.cpu().numpy(),log_probs.T.cpu().numpy()
    
    def sample_env(self, check=False):  
        if check:
           step_record_dict = dict()
           rewards = []
           log_probs = [] 
        while len(self.exps_list[0]) < self.num_steps:
            unit_mask = np.array(self.env.vec_client.getUnitLocationMasks()).reshape(self.num_envs, -1)
            action,mask,log_prob=self.get_sample_actions(self.obs, unit_mask)
            next_obs, rs, done_n, infos = self.env.step(action)
        
            if check:
                rewards.append(np.mean(rs))
                log_probs.append(np.mean(log_prob))
            
            for i in range(self.num_envs):
                if done_n[i]:
                    done = True
                else:
                    done = False
                self.exps_list[i].append([self.obs[i],action[i],rs[i],mask[i],done,log_prob[i]])
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
        
        self.calculate_net = ActorCritic(cnn_output_dim=cnn_output_dim,pos_output=action_space[0])
        self.calculate_net.to(self.device)
    
        self.share_optim = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        
        
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None

    def begin_batch_train(self, samples_list: list):    
        s_states = [np.array([s[0] for s in samples]) for samples in samples_list]
        s_actions = [np.array([s[1] for s in samples]) for samples in samples_list]
        s_masks = [np.array([s[3] for s in samples]) for samples in samples_list]
        s_log_probs = [np.array([s[5] for s in samples]) for samples in samples_list]
        
        s_rewards = [np.array([s[2] for s in samples]) for samples in samples_list]
        s_dones = [np.array([s[4] for s in samples]) for samples in samples_list]
        
        self.states = [torch.Tensor(states).to(self.device) for states in s_states]
        self.actions = [torch.Tensor(actions).to(self.device) for actions in s_actions]
        self.old_log_probs = [torch.Tensor(log_probs).to(self.device) for log_probs in s_log_probs]
        self.marks = [torch.Tensor(marks).to(self.device) for marks in s_masks]
        self.rewards = s_rewards
        self.dones = s_dones
        
        self.states_list = torch.cat([states[0:self.pae_length] for states in self.states])
        self.actions_list = torch.cat([actions[0:self.pae_length] for actions in self.actions])
        self.old_log_probs_list = torch.cat([old_log_probs[0:self.pae_length] for old_log_probs in self.old_log_probs])
        self.marks_list = torch.cat([marks[0:self.pae_length] for marks in self.marks])

    def calculate_samples_gae(self):
        np_advantages = []
        np_returns = []
        
        for states,rewards,dones in zip(self.states,self.rewards,self.dones):
            with torch.no_grad():
                values = self.calculate_net.get_value(states)
                            
            advantages,returns = calculate_gae(values.cpu().numpy().reshape(-1),rewards,dones,gamma,gae_lambda)
            np_advantages.extend(advantages[0:self.pae_length])
            np_returns.extend(returns[0:self.pae_length])
            
        np_advantages = np.array(np_advantages)
        np_returns = np.array(np_returns)
        
        return np_advantages, np_returns
        
    def end_batch_train(self):
        self.states_list = None
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
        distris = self.calculate_net.get_distris(states)
        values = self.calculate_net.get_value(states)
        action_masks = torch.split(masks, action_space, dim=1)
        distris = [dist.update_masks(mask,device=self.device) for dist,mask in zip(distris,action_masks)]
        log_probs = torch.stack([dist.log_prob(action) for dist,action in zip(distris,actions)])
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
            
            mini_states = self.states_list[start_index:end_index]
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
    map_name = args.map_name
    if map_name == "basesWorkers16x16noResources":
        map_path = "maps/16x16/basesWorkers16x16NoResources.xml"
        h=16
        w=16
        cnn_output_dim = 32*6*6
    elif map_name == "basesWorkers12x12":
        map_path = "maps/12x12/basesWorkers12x12.xml"
        h=12
        w=12
        cnn_output_dim = 32*4*4
    elif map_name == "TwoBasesBarracks16x16":
        map_path = "maps/16x16/TwoBasesBarracks16x16.xml"
        h=16
        w=16
        cnn_output_dim = 32*6*6
    action_space = [w*h, 6, 4, 4, 4, 4, 7, 49]
    observation_space = [w,h,27]

    comment = "ppo_"+map_name

    seed = random.randint(0,100000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    net = ActorCritic(cnn_output_dim=cnn_output_dim,pos_output=action_space[0])
    net.load_state_dict(torch.load("saved_model\ppo_agent\ppo_basesWorkers16x16noResources.pt"))
    parameters = sum([np.prod(p.shape) for p in net.parameters()])
    print("parameters size is:",parameters)

    agent = Agent(net)
    calculator = Calculator(net)

    MAX_VERSION = 5000
    REPEAT_TIMES = 10

    wandb.init(
        # set the wandb project where this run will be logged
        project='microrts_ppo',
        name = comment+str(seed),
        group = comment,

        # track hyperparameters and run metadata
        config={
        "base_ai":"none",
        "op_ai": args.op_ai,
        "map": map_name,
        "epochs": MAX_VERSION,
        "samples_per_epochs":num_envs*pae_length,
        "start_win_rate":0,
        "temperature_coefficient":-2,
        }
    )
    for _ in range(20):
        agent.sample_env(check=True)
        print("checking...")

    for version in range(MAX_VERSION):
        samples_list,infos = agent.sample_env(check=True)

        infos["global_steps"] = version*num_envs*pae_length
        wandb.log(infos)

        print("version:",version,"reward:",infos["mean_rewards"])
        win_rate = infos["mean_win_rates"]

        samples = []

        for s in samples_list:
            samples.append(s)
        
        calculator.begin_batch_train(samples)
        for _ in range(REPEAT_TIMES):
            calculator.generate_grads()
        calculator.end_batch_train()

    torch.save(net.state_dict(), "saved_model/ppo_agent/"+comment+str(seed)+".pt")
        
    wandb.finish()

