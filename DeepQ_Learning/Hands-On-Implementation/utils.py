import torch
import numpy as np
import torch.nn as nn

GAMMA = 0.99 # Gamma for Bellman approximations
BATCH_SIZE = 32 # Batch size sampled from the replay buffer
REPLAY_SIZE = 10000 # Maximum capacity of the buffer
REPLAY_START_SIZE = 10000 # Count of frames we wait for before starting training to populate the replay buffer
LEARNING_RATE = 0.0001 # learning rate used in adam optimiser
SYNC_TARGET_FRAMES = 1000 # Model sync frequency


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
    
    
    # nullify the gradients from it's computational graph
    # to prevent gradietns from flowing into the NN
    # to calculate the Q approximations for the next states
    # without this the backpropgatin of the loss will start 
    # to affect both the prediction for the current state 
    # an the next state
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        # to make discounted reward of the last step in the 
        # episode, then our value of the action doesn't have
        # discounted rewarsd = 0
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    
    expected_state_action_values = next_state_values*GAMMA + \
                                    rewards_v
    
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)

