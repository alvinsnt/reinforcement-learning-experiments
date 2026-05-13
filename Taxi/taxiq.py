# Modified from https://github.com/johnnycode8/gym_solutions/blob/main/taxi_q.py

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# VARIABLE ASSIGNMENTS
LEARNING_RATE_A = 0.9
DISCOUNT_FACTOR_G = 0.9
EPSILON = 0.0
EPSILON_DECAY_RATE = 0.000
MIN_EPSILON = 0.0
MOVING_AVG_WINDOW = 100

MODEL_FILE = 'taxi.pkl'
PLOT_FILE = 'taxi-q.png'

def run(episodes, is_training=True, render=False):

    # Start timer
    start_time = time.time()

    env = gym.make('Taxi-v4', render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open(MODEL_FILE, 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = LEARNING_RATE_A
    discount_factor_g = DISCOUNT_FACTOR_G
    epsilon = EPSILON
    epsilon_decay_rate = EPSILON_DECAY_RATE
    min_epsilon = MIN_EPSILON

    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)

    for i in range(episodes):

        state = env.reset()[0]
        terminated = False
        truncated = False

        rewards = 0
        steps = 0

        while(not terminated and not truncated):

            # Epsilon-greedy action selection
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            rewards += reward
            steps += 1

            # Q-Learning Update Rule
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )
            
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)

        rewards_per_episode[i] = rewards
        steps_per_episode[i] = steps

    env.close()

    # End timer
    end_time = time.time()
    training_time = end_time - start_time

    # FINAL STATISTIC

    last_100_avg_reward = np.mean(rewards_per_episode[-100:])
    last_100_avg_steps = np.mean(steps_per_episode[-100:])

    print("\n===== Training Statistics =====")
    print(f"Final Average Reward (Last 100 Episodes): {last_100_avg_reward:.2f}")
    print(f"Average Steps (Last 100 Episodes): {last_100_avg_steps:.2f}")
    print(f"Training Time: {training_time:.2f} seconds")

    # PLOT
    
    moving_avg_rewards = np.convolve(
        rewards_per_episode,
        np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW,
        mode='valid'
    )

    plt.figure(figsize=(15,5))

    # Plot 1: Reward per episode
    plt.subplot(1,3,1)
    plt.plot(rewards_per_episode)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Plot 2: Moving average reward
    plt.subplot(1,3,2)
    plt.plot(moving_avg_rewards)
    plt.title(f'Moving Average Reward ({MOVING_AVG_WINDOW})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

    # Plot 3: Steps per episode
    plt.subplot(1,3,3)
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()
    plt.savefig(PLOT_FILE)

    if is_training:
        f = open(MODEL_FILE,"wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(5000)

    # Uncomment line below to render the Taxi
    # run(3, is_training=False, render=True)