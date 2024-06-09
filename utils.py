import numba
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

@numba.njit()
def calculate_gae(values,rewards,dones,gamma,gae_lambda):
    if len(values.shape) != 1:
        return None,None
    
    length = values.shape[0]
    dones[length-1] = True
    advantages = np.zeros(length, dtype=np.float32)
    returns = np.zeros(length, dtype=np.float32)
    
    last_gae = 0.0
    for index in range(length-1,-1,-1):
        if dones[index]:
            delta = rewards[index] - values[index]
            last_gae = delta
        else:
            delta = rewards[index] + gamma * values[index+1] - values[index]
            last_gae = delta + gamma * gae_lambda * last_gae
            
        advantages[index] = last_gae
        returns[index] = last_gae + values[index]
                                         
    return advantages, returns

class MaskedCategorical:
    def __init__(self, probs):
        self.origin_probs = probs
        self.probs = F.softmax(probs,dim=-1)
        self.dist = Categorical(self.probs)
            
    def update_masks(self,masks,device = 'cpu'):
        if masks is None:
            return self
        probs = torch.lerp(self.origin_probs, torch.tensor(-1e+8).to(device), 1.0 - masks)
        self.probs = F.softmax(probs,dim=-1)
        self.dist = Categorical(self.probs)
        return self
    
    def sample(self):
        actions = self.dist.sample()
        return actions
    
    def one_hot_sample(self):
        actions = self.dist.sample()
        actions = one_hot(actions, self.probs.shape[-1])
        return actions
    
    def log_prob(self,actions):
        return self.dist.log_prob(actions)
    
    def entropy(self):
        return self.dist.entropy()
    
    def argmax(self):
        return torch.argmax(self.probs,dim=-1)
    
    def argmin(self):
        return torch.argmin(self.probs,dim=-1)
    

def data_argument(exp_lists):
    new_exp_lists = []
    for exp_list in exp_lists:
        new_exp_list1 = []
        new_exp_list2 = []
        new_exp_list3 = []
        for item in exp_list:
            state,action,reward,mask,done,log_prob = item
            # rotate state, action and mask
            state1 = torch.rot90(state,1,[0,1])
            action1 = rotate_action(action,1)
            mask1 = rotate_data(mask,1)
            new_exp_list1.append([state1,action1,reward,mask1,done,log_prob])

            state2 = torch.rot90(state,2,[0,1])
            action2 = rotate_action(action,2)
            mask2 = rotate_data(mask,2)
            new_exp_list2.append([state2,action2,reward,mask2,done,log_prob])

            state3 = torch.rot90(state,3,[0,1])
            action3 = rotate_action(action,3)
            mask3 = rotate_data(mask,3)
            new_exp_list3.append([state3,action3,reward,mask3,done,log_prob])
        new_exp_lists.append(exp_list)
        new_exp_lists.append(new_exp_list1)
        new_exp_lists.append(new_exp_list2)
        new_exp_lists.append(new_exp_list3)
    return new_exp_lists

# 0321
def rotate_action(action, rotate_time):
    new_action = action.copy()
    new_action[2] = (action[2]-rotate_time)%4
    new_action[3] = (action[3]-rotate_time)%4
    new_action[4] = (action[4]-rotate_time)%4
    new_action[5] = (action[5]-rotate_time)%4

    atk = np.array(range(49)).reshape(7,7)
    atk_pos = action[7]
    atk_pos_x = atk_pos//7
    atk_pos_y = atk_pos%7

    new_action[7] = np.rot90(atk,rotate_time)[atk_pos_x,atk_pos_y]
    return new_action

def rotate_data(data, rotate_time):
    new_data = data.copy()
    new_data[6:10] =  np.roll(data[6:10],-rotate_time)
    new_data[10:14] = np.roll(data[10:14],-rotate_time)
    new_data[14:18] = np.roll(data[14:18],-rotate_time)
    new_data[18:22] = np.roll(data[18:22],-rotate_time)


    new_data[29:78] = np.rot90(data[29:78].reshape(7,7), rotate_time).reshape(49)
    return new_data


def remake_mask(masks):
    for i in range(len(masks)):
        masks[i][26] = 0
        masks[i][27] = 0
        if masks[i][3] == 1:
            masks[i][1] = 0
            masks[i][2] = 0
            masks[i][4] = 0
        if masks[i][5] == 1:
            masks[i][1] = 0
            masks[i][2] = 0
            masks[i][3] = 0
            masks[i][4] = 0

    return masks

def one_hot(indexes, depth):
    return np.eye(depth)[indexes]

if __name__ == "__main__":
    indexes = torch.tensor([1,2,3,4,5,6,7,8,9])
    depth = 10
    res = one_hot(indexes, depth)
    print(res)