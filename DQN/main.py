"""
Assignment 3: Deep Q-Network (DQN) for Maze Navigation
Developed for the Principles of Autonomy and Decision Making course (SoSe 25).
This script implements a DQN agent to solve a continuous maze environment.
"""

import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Custom imports
from env import ContinuousMazeEnv
from DQN_model import Qnet
from utils import ReplayBuffer, train


def get_args():
    parser = argparse.ArgumentParser(description='DQN Training and Testing')
    parser.add_argument('--train', action='store_true', help='Train the DQN agent')
    parser.add_argument('--test', action='store_true', help='Test the DQN agent')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    return parser.parse_args()


def train_dqn(render_mode=False):
    # Hyperparameters
    learning_rate = 0.005
    gamma = 0.98
    buffer_limit = 50_000
    batch_size = 32
    num_episodes = 10_000
    max_steps = 500
    
    # Environment Setup
    env = ContinuousMazeEnv(render_mode="human" if render_mode else None)
    dim_actions = 4
    dim_states = 2

    q_net = Qnet(dim_actions=dim_actions, dim_states=dim_states)
    q_target = Qnet(dim_actions=dim_actions, dim_states=dim_states)
    q_target.load_state_dict(q_net.state_dict())

    memory = ReplayBuffer(buffer_limit=buffer_limit)
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    rewards = []
    epsilon_decay = []
    success_counter = 0

    print("Starting Training...")

    for n_epi in range(num_episodes):
        # Epsilon decay: Linear annealing
        epsilon = max(0.1, 1.0 - (n_epi * 0.4) * 0.001)

        s, _ = env.reset()
        done = False
        episode_reward = 0.0

        for _ in range(max_steps):
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, _, _ = env.step(a)
            
            if render_mode:
                env.render()

            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            episode_reward += r

            if done:
                break

        if memory.size() > 2000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % 20 == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(f"Episode: {n_epi}, Reward: {episode_reward:.2f}, Buffer: {memory.size()}, Epsilon: {epsilon:.2f}")

        rewards.append(episode_reward)
        epsilon_decay.append(epsilon)

        # Success check
        if r >= 10.0:
            success_counter += 1
            print(f"[EP {n_epi}] Goal reached! Success count = {success_counter}")
        else:
            success_counter = 0

        # Stop condition
        if success_counter >= 100 and epsilon <= 0.1:
            print(f"\nTraining complete after {n_epi} episodes!")
            torch.save(q_net.state_dict(), "dqn_final.pth")
            np.save("rewards_final.npy", np.array(rewards))
            np.save("epsilon_final.npy", np.array(epsilon_decay))
            
            plot_results(rewards, epsilon_decay)
            break
            
    env.close()


def plot_results(rewards, epsilon_decay):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    axs[0].plot(rewards, label='Reward per Episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Rewards')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epsilon_decay, label='Epsilon Decay', color='orange')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Epsilon')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("combined_training_plot.png")
    plt.show()


def test_dqn(model_path="dqn_final.pth"):
    print(f"Testing DQN from {model_path}...")
    env = ContinuousMazeEnv(render_mode="human")
    dim_actions = 4
    dim_states = 2

    dqn = Qnet(dim_actions=dim_actions, dim_states=dim_states)
    try:
        dqn.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Train first.")
        return

    for ep in range(5):
        s, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = dqn(torch.from_numpy(s).float()).argmax().item()
            s_prime, reward, done, _, _ = env.step(action)
            env.render()
            time.sleep(0.05)
            s = s_prime
            episode_reward += reward

        print(f"Test Episode {ep+1} Reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    args = get_args()
    
    if args.train:
        train_dqn(render_mode=args.render)
    elif args.test:
        test_dqn()
    else:
        print("Please specify --train or --test")
