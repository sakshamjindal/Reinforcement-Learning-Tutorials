import time
import numpy as np
import collections
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from tensorboardX import SummaryWriter


import wrappers
import dqn_model
from structures import Experience, ExperienceBuffer, Agent
from utils import calc_loss


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0

GAMMA = 0.99 # Gamma for Bellman approximations
BATCH_SIZE = 32 # Batch size sampled from the replay buffer
REPLAY_SIZE = 10000 # Maximum capacity of the buffer
REPLAY_START_SIZE = 10000 # Count of frames we wait for before starting training to populate the replay buffer
LEARNING_RATE = 0.0001 # learning rate used in adam optimiser
SYNC_TARGET_FRAMES = 1000 # Model sync frequency

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda:0"

    env = wrappers.make_env(args.env)

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape,
                            env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    # training loop to create an optimizer, a buffer for full episdeo rewards, a counters
    # of frames and several variables to tack our speed
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1

        # count the iterations and decrease the epsilon
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                    frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), m_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            
            # everytime our mean reward for the last 100 episodes 
            # reaches a maximum, report this and save the model 
            # parameters
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env +
                        "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward

            # if our mean reward exceeds the specified boundary,
            # we stop training
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        # Here we check if our buffer is large enough for training
        # In the beginning we should wait for for enough data
        # to start the training.
        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # training loop : zero gradients, sample data batches
        # for experience replay buffer, calculate loss and perform
        # optimisation step to minimise the loss
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)

        try:
            loss_t = calc_loss(batch, net, tgt_net, device=device)
        except:
            import ipdb
            ipdb.set_trace()
        loss_t.backward()
        optimizer.step()

    writer.close()
            