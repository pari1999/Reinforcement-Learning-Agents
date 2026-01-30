# Deep Q-Network (DQN)

Implementation of a Deep Q-Network (DQN) to solve a continuous maze environment.

## Overview

In this project, we transition from tabular Q-learning to Deep Reinforcement Learning. Using a Neural Network as a function approximator (DQN), the agent learns to navigate a continuous state space to reach a goal.

## Key Components

- **DQN Model**: A multi-layer perceptron built with PyTorch.
- **Experience Replay**: A buffer to store and sample past transitions, breaking temporal correlations and improving stability.
- **Target Network**: A separate network used to provide stable targets during training, updated periodically.
- **Continuous Environment**: A 2D maze with continuous state observations.

## How to Run

### Training

To start the training process:

```bash
python main.py --train
```

### Testing

To test the performance of a trained model:

```bash
python main.py --test
```

### Rendering

To enable visual rendering during training or testing:

```bash
python main.py --render
```

## Training Progress

The repository includes `combined_training_plot.png`, which shows the rewards and epsilon decay throughout the training process. The model weights are saved as `.pth` files.
