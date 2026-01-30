# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table1_save_path="q_table1.npy",
                     q_table2_save_path="q_table2.npy"):

    # Initialize the Q-table:
    # -----------------------
    q_table1 = np.zeros((env.grid_size, env.grid_size,2,2, env.action_space.n))
    q_table2 = np.zeros((env.grid_size, env.grid_size,2,2, env.action_space.n))


    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(no_episodes):
        state, _ = env.reset()

        state = tuple(state)

        state = (state[0], state[1], int(env.carrying_trash_1), int(env.carrying_trash_2))
        total_reward = 0
        MAX_STEPS_PER_EPISODE = 500
        steps = 0
        phase = 1

        print(f"\n[EPISODE {episode+1} START] Trash1: {env.trash1_pos}, Trash2: {env.trash2_pos}, Carrying1: {env.carrying_trash_1}, Carrying2: {env.carrying_trash_2}")



        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        while True:

            #choosing the q table b\ased on the phase
            current_q_table = q_table1 if phase == 1 else q_table2
           

            #! Step 3: Define your Exploration vs. Exploitation
            #! -------
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(current_q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            env.render()
            print(f"Phase: {phase}, Reward: {reward}, Action: {action}")

            next_state = (next_state[0], next_state[1], int(env.carrying_trash_1), int(env.carrying_trash_2))

            total_reward += reward

            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------
            current_q_table[state][action] = current_q_table[state][action] + alpha * \
                (reward + gamma *
                 np.max(current_q_table[next_state]) - current_q_table[state][action])

            state = next_state
            
            steps +=1

            # Phase switch logic
            if phase == 1 and np.array_equal(env.trash1_pos, [-2, -2]):
                phase = 2
            if steps >= MAX_STEPS_PER_EPISODE:
                print(f"Episode {episode + 1}: Reached max steps, stopping episode.")
                done = True

            #! Step 5: Stop the episode if the agent reaches Goal or Hell-states
            #! -------
            if done:
                break

        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")

    #! Step 8: Save the trained Q-table
    #! -------
    np.save(q_table1_save_path, q_table1)
    np.save(q_table2_save_path, q_table2)

    print("Saved the Q-table.")


# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(trash_positions,
                      bin_positions,
                      q_values_path,
                      actions=["Up", "Down", "Right", "Left","Pickup+down","Drop+right"],
                      title=None
                      ):

    # Load the Q-table:
    # -----------------
        try:
            q_table = np.load(q_values_path)

            # Force a new figure for each Q-table
            fig, axes = plt.subplots(1, 6, figsize=(20, 5))
            fig.suptitle(title if title else q_values_path, fontsize=18)


            for i, action in enumerate(actions):
                ax = axes[i]
                heatmap_data = q_table[:, :, 0, 0, i].copy()


                # Mask goals and trash for clarity
                #mask = np.zeros_like(heatmap_data, dtype=bool)
               # mask[bin_positions[0]] = True
               # mask[bin_positions[1]] = True
               # mask[trash_positions[0]] = True
               # mask[trash_positions[1]] = True

                sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                            ax=ax, cbar=False, annot_kws={"size": 9})

                ax.text(bin_positions[0][1] + 0.5, bin_positions[0][0] + 0.5, 'P', color='green',
                        ha='center', va='center', weight='bold', fontsize=14)
                ax.text(bin_positions[1][1] + 0.5, bin_positions[1][0] + 0.5, 'B', color='green',
                        ha='center', va='center', weight='bold', fontsize=14)
                ax.text(trash_positions[0][1] + 0.5, trash_positions[0][0] + 0.5, 'P', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)
                ax.text(trash_positions[1][1] + 0.5, trash_positions[1][0] + 0.5, 'B', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

                ax.set_title(f'Action: {action}')

            plt.tight_layout()
            plt.show()

        except FileNotFoundError:
            print(f"Q-table not found at path: {q_values_path}")


def plot_policy_split_qtables(q_table1, q_table2, title_prefix="Policy"):
    """
    Plots the best action for:
    - (0,0) and (1,0) from q_table1
    - (0,1) from q_table2
    """

    grid_size = q_table1.shape[0]
    action_symbols = ['↑', '↓', '←', '→', 'P', 'D']

    # Define action priority: ↓ > → > ↑ > ← > P > D
    action_priority = [1, 2, 0, 3, 4, 5]

    carry_states = [
        (q_table1, 0, 0, f"{title_prefix} (C1=0, C2=0)"),
        (q_table1, 1, 0, f"{title_prefix} (C1=1, C2=0)"),
        (q_table2, 0, 0, f"{title_prefix} (C1=0, C2=0) [from q_table2]"),
        (q_table2, 0, 1, f"{title_prefix} (C1=0, C2=1)")
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for idx, (q_table, c1, c2, title) in enumerate(carry_states):
        policy = np.full((grid_size, grid_size), '', dtype=object)

        for i in range(grid_size):
            for j in range(grid_size):
                state_action_values = q_table[i, j, c1, c2]

                # Use priority-based argmax
                best_action = max(action_priority, key=lambda a: state_action_values[a])
                policy[i, j] = action_symbols[best_action]

        ax = axes[idx]
        ax.set_xticks(np.arange(grid_size))
        ax.set_yticks(np.arange(grid_size))
        ax.set_xticklabels(np.arange(grid_size))
        ax.set_yticklabels(np.arange(grid_size))
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_title(title, fontsize=14)
        ax.invert_yaxis()
        ax.grid(True)

        for i in range(grid_size):
            for j in range(grid_size):
                ax.text(j, i, policy[i, j], ha='center', va='center', fontsize=18)

    plt.tight_layout()
    plt.show()

