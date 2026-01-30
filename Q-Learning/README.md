# Q-Learning Agent

This project implements a Q-Learning agent to solve the Robot Trash Collector environment.

## Overview

The goal is to train an autonomous agent that learns the optimal policy for collecting trash and depositing it in the correct bins using the classical Q-Learning algorithm.

## Training Details

- **Algorithm**: Q-Learning (Off-policy RL)
- **State Space**: Discretized grid positions.
- **Action Space**: 6 discrete actions (Up, Down, Left, Right, Pick, Drop).
- **Hyperparameters**:
  - Learning Rate ($\alpha$): 0.1
  - Discount Factor ($\gamma$): 0.99
  - Epsilon ($\epsilon$): 1.0 (with decay)

## How to Run

### Training the Agent

To train the agent from scratch:

```bash
python main.py --train
```

### Visualizing Results

To visualize the learned Q-table and policy:

```bash
python main.py --visualize
```

### Running the Trained Agent

To see the trained agent in action:

```bash
python main.py --run_agent
```

## Results

The agent successfully learns to navigate the grid, pick up both types of trash, and deliver them to their respective bins while minimizing steps.
