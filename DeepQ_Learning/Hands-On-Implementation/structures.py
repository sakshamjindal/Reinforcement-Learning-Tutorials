import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

Experience = collections.namedtuple(
    'Experience', field_names = ['state', 'action', 'reward',
                                 'done', 'new_state'])

# Code for Experience Buffer

class ExperienceBuffer:
    
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen = capacity)
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        
        indices = np.random.choice(len(self.buffer), batch_size,
                                  replace = False)
        # unzipping and seprating out the iterables
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), \
                np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=np.uint8), \
                np.array(next_states)
 

class Agent:
    
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
        
    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        
        
    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device = "cpu"):
        
        """
        The main method of the agent is to perform a step in
        the environment and store its result int the buffer.
        We take an action and perform a random action, we take the
        random action; otherwise, we use the past model to obtain 
        the Q-values for all possible actions and choose the best
        """
        done_reward = None
        
        # Exploitation vs explorations
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy = False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v* = net(state_v)
            _, act_v = torch.max(q_vals_v, dim = 1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        
        exp = Experience(self.state, action, reward, 
                        is_done, new_state)
        
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
            
        return done_reward
    
def calc_loss(batch, net, tgt_net, device="cpu"):
    
    states, actions, rewards, dones, next_states = batch
    
    states_v = torch.tensor(np.array(
        states, copy = False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy = False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    
    # pass observations to the first model and 
    # extract the specific Q-values for the taken 
    # actions usig the gather () tensor operation
    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    next_state_values = tgt_net(next_states_v).max(1)[0]
    
    # to make discounted reward of the last step in the 
    # episode, then our value of the action doesn't have
    # discounted rewarsd = 0
    next_state_values[done_mask] = 0.0
    
    # nullify the gradients from it's computational graph
    # to prevent gradietns from flowing into the NN
    # to calculate the Q approximations for the next states
    # without this the backpropgatin of the loss will start 
    # to affect both the prediction for the current state 
    # an the next state
    next_state_values = next_state_values.detach()
    
    expected_state_action_values = next_state_values*GAMMA + \
                                    rewards_v
    
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)


