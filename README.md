# Reinforcement Learning Agents 🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A portfolio of Reinforcement Learning (RL) projects exploring autonomous decision-making in simulated environments. This repository contains implementations of Tabular Q-Learning and Deep Q-Networks (DQN) applied to customer robotic tasks.



## 📂 Projects

### 1. Robot Trash Collector (Q-Learning)
**Location:** [`Q-Learning/`](./Q-Learning)

An autonomous agent trained to clean a grid-world environment.
- **Goal:** Navigate a 5x5 grid, collect plastic and organic trash, and deposit them in the correct bins.
- **Algorithm:** Tabular Q-Learning with Epsilon-Greedy exploration.
- **Features:**
  - Custom Gymnasium environment with PyGame rendering.
  - State space includes robot position, trash status, and held items.
  - Multi-stage reward system (pickup, delivery, step penalties).

### 2. Continuous Maze Navigation (DQN)
**Location:** [`DQN/`](./DQN)

A Deep Reinforcement Learning agent solving a continuous state-space maze.
- **Goal:** Navigate a continuous 2D plane to reach a target zone.
- **Algorithm:** Deep Q-Network (DQN) with Experience Replay and Target Networks.
- **Stack:** PyTorch, Neural Networks (MLP).
- **Features:**
  - Continuous coordinate input.
  - Epsilon decay for exploration-exploitation balance.
  - Training visualizations (Reward vs. Episode).

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Reinforcement-Learning-Agents.git
   cd Reinforcement-Learning-Agents
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Trash Collector (Q-Learning)
To train the agent and visualize the result:
```bash
cd Q-Learning
# Train the agent
python main.py --train --visualize

# Run the pre-trained agent
python main.py --run_agent
```

### Running the Maze Solver (DQN)
To train the deep learning model:
```bash
cd DQN
# Train with rendering enabled
python main.py --train --render

# Test the trained model
python main.py --test
```

## 📊 Results

| Project | Algorithm | Converged Episode (approx) | Success Rate |
|---------|-----------|----------------------------|--------------|
| Trash Collector | Q-Learning | ~2500 | >95% |
| Maze Solver | DQN | ~800 | Stable |

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
