import numpy as np
import random
import matplotlib.pyplot as plt

# Environment setup
class PricingEnvironment:
    def __init__(self, demand_levels, price_levels):
        self.demand_levels = demand_levels
        self.price_levels = price_levels
        self.n_states = len(demand_levels)
        self.n_actions = len(price_levels)
        self.state = self.reset()

    def reset(self):
        self.state = random.choice(range(self.n_states))
        return self.state

    def step(self, action):
        price = self.price_levels[action]
        demand = self.demand_levels[self.state]

        # Define demand-price relationship for 10 levels
        if demand == 1:
            sales = demand * np.exp(-0.01 * price)
        elif demand == 2:
            sales = demand * np.exp(-0.008 * price)
        elif demand == 3:
            sales = demand * np.exp(-0.006 * price)
        elif demand == 4:
            sales = demand * np.exp(-0.004 * price)
        elif demand == 5:
            sales = demand * np.exp(-0.002 * price)
        elif demand == 6:
            sales = demand * np.exp(-0.0015 * price)
        elif demand == 7:
            sales = demand * np.exp(-0.001 * price)
        elif demand == 8:
            sales = demand * np.exp(-0.0005 * price)
        elif demand == 9:
            sales = demand * np.exp(-0.0002 * price)
        elif demand == 10:
            sales = demand * np.exp(-0.0001 * price)
        else:
            raise ValueError("Demand level out of range.")

        revenue = price * sales
        self.state = random.choice(range(self.n_states))  # New demand level
        return self.state, revenue

# Parameters
demand_levels = np.arange(1, 11, 1)
price_levels = np.arange(150, 600, 100)

n_actions = len(price_levels)
n_states = len(demand_levels)

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000
max_steps_per_episode = 100  # Maximum steps per episode to ensure progress

# Initialize Q-table
q_table = np.zeros((n_states, n_actions))

# Environment
env = PricingEnvironment(demand_levels, price_levels)

# Training
rewards = []
average_rewards = []
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    for step in range(max_steps_per_episode):
        # Choose action
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # Take action
        next_state, reward = env.step(action)
        total_reward += reward

        # Update Q-value
        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])

        state = next_state

    rewards.append(total_reward)
    average_rewards.append(np.mean(rewards[-100:]))  # Rolling average of last 100 rewards

    # Debug print to trace learning
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Average Reward: {average_rewards[-1]:.2f}")

# Evaluate policy
optimal_policy = np.argmax(q_table, axis=1)
print("\nOptimal policy (best price for each demand level):")
for state, action in enumerate(optimal_policy):
    print(f"Demand level {demand_levels[state]}: Set price to {price_levels[action]}")

# Visualization of Q-table
plt.figure(figsize=(18, 5))
#Darker colors (depending on the colormap) typically represent higher Q-values, indicating that those actions are more valuable in the corresponding states.
plt.subplot(1, 3, 1)
plt.imshow(q_table, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xlabel('Price level index')
plt.ylabel('Demand level index')
plt.title('Q-table Heatmap')

# Policy visualization
plt.subplot(1, 3, 2)
plt.bar(range(n_states), [price_levels[action] for action in optimal_policy], tick_label=demand_levels)
plt.xlabel('Demand level')
plt.ylabel('Optimal price level')
plt.title('Optimal Policy')

# Reward evolution plot
plt.subplot(1, 3, 3)
plt.plot(rewards, label='Total Reward')
plt.plot(average_rewards, label='Average Reward (last 100 episodes)', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Evolution Over Episodes')
plt.legend()

plt.tight_layout()
plt.show()

# Additional analysis: Q-value changes over time
q_values_over_time = np.zeros((episodes, n_states, n_actions))

for episode in range(episodes):
    state = env.reset()
    for step in range(max_steps_per_episode):
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        next_state, reward = env.step(action)
        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])
        q_values_over_time[episode, state, action] = q_table[state, action]

        state = next_state

# Plot Q-value changes
plt.figure(figsize=(12, 8))
for state in range(n_states):
    for action in range(n_actions):
        plt.plot(q_values_over_time[:, state, action], label=f'State {demand_levels[state]}, Action {price_levels[action]}')

plt.xlabel('Episode')
plt.ylabel('Q-value')
plt.title('Q-value Changes Over Time')
plt.legend()
plt.show()
