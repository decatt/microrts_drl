import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai

from utils import layer_init, one_hot, MaskedCategorical

kernel_size = 7
num_closest = 16

map_path = 'maps/16x16/basesWorkers16x16.xml'
action_space = [6, 4, 4, 4, 4, 7, 49]

def pad_tensor(tensor, padding):
    # Pads a 4D tensor along the height and width dimensions
    batch_size, h, w, _ = tensor.shape
    tensor_padded = torch.zeros((batch_size, h + 2*padding, w + 2*padding, 27))
    tensor_padded[:, padding:-padding, padding:-padding, :] = tensor
    return tensor_padded

def extract_centered_regions(tensor, points):
    # Assumption: kernel_size is odd
    padding = kernel_size // 2
    tensor_padded = pad_tensor(tensor, padding)

    batch_size, h, w, _ = tensor.shape
    output_tensor = torch.zeros((batch_size, kernel_size, kernel_size, 27))

    for batch_index in range(batch_size):
        pos = points[batch_index]
        if pos >=0:
            x = pos // w + padding
            y = pos % w + padding
            output_tensor[batch_index] = tensor_padded[batch_index, x-padding:x+padding+1, y-padding:y+padding+1, :]
        else:
            output_tensor[batch_index] = torch.zeros((kernel_size, kernel_size, 27))

    return output_tensor

def process_states(states, selected_units):
    states = torch.tensor(states)
    _, h, w, _ = states.shape
    pv_state = extract_centered_regions(states, selected_units)
    nodes_features = []
    for state,selected_unit in zip(states,selected_units):
        if selected_unit == -1:
            nodes_features.append(torch.zeros((num_closest,29)))
        else:
            selected_unit_x = selected_unit//w
            selected_unit_y = selected_unit%w
            nodes_feature = []
            for y in range(w):
                for x in range(h):
                    if state[y][x][13] != 1:
                        d = abs(selected_unit_x-x)+abs(selected_unit_y-y)
                        if d >0:
                            nodes_feature.append((d,torch.cat((torch.tensor([x/w,y/h]),state[y][x]))))
                        elif d == 0:
                            nodes_feature.append((0,torch.cat((torch.tensor([x/w,y/h]),state[y][x]))))
            #sort by distance and get the first 8
            if len(nodes_feature) > num_closest:
                nodes_feature.sort(key=lambda x:x[0])
                nodes_feature = nodes_feature[:num_closest]
            else:
                for _ in range(num_closest-len(nodes_feature)):
                    nodes_feature.append((-1,torch.zeros(29)))
            nodes_features.append(torch.stack([x[1] for x in nodes_feature]))
    return pv_state, torch.stack(nodes_features)

#main network
class MixNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32*2*2, 128)),
            nn.ReLU(),
        )

        self.encoder = nn.GRU(29, 128, batch_first=True)
        self.decoder = nn.GRU(128, 128, batch_first=True)

        self.dis = layer_init(nn.Linear(256, 78), std=0.01)
        
        self.value = layer_init(nn.Linear(256, 1), std=1)
        
    def forward(self,cnn_states,linears_states):
        cnn_states = cnn_states.permute((0, 3, 1, 2))
        z_cnn = self.encoder_cnn(cnn_states)

        batch_size = linears_states.size(0)
        seq_len = linears_states.size(1)

        # Encoder
        encoder_outputs, hidden = self.encoder(linears_states)

        # Decoder
        decoder_state = hidden
        decoder_input = torch.zeros((batch_size, 1, 128)).to(linears_states.device)
        decoder_outputs = []

        for t in range(seq_len):
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        z_pn = decoder_outputs[-1][:,-1,:]

        policy_network = torch.cat((z_cnn,z_pn),dim=1)

        distris = self.dis(policy_network)
        
        value = self.value(policy_network)

        return distris, value

class MixStateAgent:
    def __init__(self,num_envs:int, model_path:str=None) -> None:
        self.net = MixNet()
        self.num_envs = num_envs
        if model_path is not None:
            self.net.load_state_dict(torch.load(model_path))
    
    @torch.no_grad()
    def get_actions(self, states:np.ndarray, envs:MicroRTSVecEnv)->np.ndarray:
        unit_masks = np.array(envs.vec_client.getUnitLocationMasks()).reshape(self.num_envs, -1)
            
        selected_units = []
        for unit_mask in unit_masks:
            if np.sum(unit_mask) == 0:
                selected_units.append(-1)
            else:
                selected_units.append(np.random.choice(np.where(unit_mask == 1)[0]))

        pv_state, nodes_features = process_states(states,selected_units)
        
        distris,_ = self.net(pv_state, nodes_features)
        distris = torch.split(distris, action_space, dim=1)
        distris = [MaskedCategorical(dist) for dist in distris]
        
        for i in range(len(selected_units)): 
            if selected_units[i] == -1:
                selected_units[i] = 0

        action_components = [torch.Tensor(selected_units)]

        action_mask_list = np.array(envs.vec_client.getUnitActionMasks(np.array(selected_units))).reshape(len(selected_units), -1)
        
        action_masks = torch.split(torch.Tensor(action_mask_list), action_space, dim=1)
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris,action_masks)]
            
        actions = torch.stack(action_components, dim=1).cpu().numpy()
        return actions.astype(np.int32)
    
    @torch.no_grad()
    def get_action_distribution(self, states:np.ndarray, envs:MicroRTSVecEnv)->torch.Tensor:
        unit_masks = np.array(envs.vec_client.getUnitLocationMasks()).reshape(self.num_envs, -1)
            
        selected_units = []
        for unit_mask in unit_masks:
            if np.sum(unit_mask) == 0:
                selected_units.append(-1)
            else:
                selected_units.append(np.random.choice(np.where(unit_mask == 1)[0]))

        pv_state, nodes_features = process_states(states,selected_units)
        
        distris,_ = self.net(pv_state, nodes_features)
        return distris
    
if __name__ == "__main__":
    num_envs = 4
    path = 'model\ppo_model_mix_TwoBasesWorkers12x12_v0_0.pkl'
    agent = MixStateAgent(num_envs,path)
    envs = MicroRTSVecEnv(
                num_envs=num_envs,
                max_steps=5000,
                ai2s=[microrts_ai.coacAI for _ in range(num_envs)],
                map_path=map_path,
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
    states = envs.reset()
    for i in range(10000):
        envs.render()
        actions = agent.get_actions(states,envs)
        states,_,_,_ = envs.step(actions)