"""
Assignment 2: Q-Learning Agent for Trash Collection
Developed for the Principles of Autonomy and Decision Making course (SoSe 25).
This script trains and evaluates a Q-learning agent in the custom trash collector environment.
"""

# Imports:
# --------
from custom_env import create_env , run_agent
from Q_learning import train_q_learning, visualize_q_table, plot_policy_split_qtables
import numpy as np
import argparse


# Argument parsing:
parser = argparse.ArgumentParser(description="Run the Q-learning agent for the trash collection environment.")
parser.add_argument('--train', action='store_true', help='Train the Q-learning agent')
parser.add_argument('--visualize', action='store_true', help='Visualize the results')
parser.add_argument('--run_agent', action='store_true', help='Run the trained agent')
args = parser.parse_args()
# User definitions:
# -----------------
train = args.train  # If True, the agent will be trained
visualize_results = args.visualize  # If True, the results will be visualized
run_agent_ = args.run_agent # If True, the agent will run after training

"""
NOTE: Sometimes a fixed initializtion might push the agent to a local minimum.
In this case, it is better to use a random initialization.  
"""
random_initialization = True  # If True, the Q-table will be initialized randomly

learning_rate = 0.1 # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 3000  # Number of episodes



#trash and bin coordinates:
trash_positions = [(1,1),(3,3)]
bin_positions = [(4,0),(4,4)]


# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env(bin_positions=bin_positions,
                     random_initialization=random_initialization)

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)
    
if run_agent_:
    # Run the trained agent:
    # ----------------------
    env = create_env(bin_positions=bin_positions,
                     random_initialization=random_initialization)
    run_agent(env,q_table1_path="q_table1.npy",
                  q_table2_path="q_table2.npy"
                )

if visualize_results:
    # Visualize the Q-table:
    # ----------------------

    '''
   visualize_q_table(trash_positions=trash_positions,
                      bin_positions=bin_positions,
                      q_values_path="q_table1.npy",
                      title="Q-table for Phase 1")
   visualize_q_table(trash_positions=trash_positions,
                      bin_positions=bin_positions,
                      q_values_path="q_table2.npy",
                      title="Q-table for Phase 2")
   '''
    q_table1 = np.load("q_table1.npy")
    q_table2 = np.load("q_table2.npy")
    plot_policy_split_qtables(q_table1,q_table2)
    