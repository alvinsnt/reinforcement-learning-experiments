import gymnasium as gym
import numpy as np
import pandas as pd
import time

# CONFIGURATION VARIABLES

EPISODES = 5000
TRIALS = 1
INIT_SEED = 100

ALPHAS = [0.0]
GAMMAS = [0.0]
EPSILONS = [0.0]

# Fixed parameters for each experiment group
FIXED_GAMMA_ALPHA_EPSILON = 0.9
FIXED_EPSILON_ALPHA_GAMMA = 0.2
FIXED_ALPHA_GAMMA_EPSILON = 0.3


OUTPUT_CSV = "taxi_hyperparam_optimization.csv"

# SINGLE EXPERIMENT

def run_experiment(
    episodes,
    alpha,
    gamma,
    epsilon,
    seed
):

    start_time = time.time()
    env = gym.make('Taxi-v4')

    # Set seeds
    np.random.seed(seed)

    q = np.zeros((env.observation_space.n, env.action_space.n))
    rng = np.random.default_rng(seed)
    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):

        # Seed environment reset
        state = env.reset(seed=seed + episode)[0]

        terminated = False
        truncated = False

        total_reward = 0

        while not terminated and not truncated:

            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning update
            q[state, action] = q[state, action] + alpha * (
                reward + gamma * np.max(q[new_state, :]) - q[state, action]
            )

            state = new_state
            total_reward += reward

        rewards_per_episode[episode] = total_reward

    env.close()

    training_time = time.time() - start_time
    last_100_avg_reward = np.mean(rewards_per_episode[-100:])

    return last_100_avg_reward, training_time


# REPEATED EXPERIMENTS

def run_multiple_trials(
    trials,
    episodes,
    alpha,
    gamma,
    epsilon
):

    rewards = []
    training_times = []

    for trial in range(trials):

        # Different seed per trial
        seed = INIT_SEED + trial

        print(
            f"Trial {trial+1}/{trials} | "
            f"Seed={seed} | "
            f"alpha={alpha}, gamma={gamma}, epsilon={epsilon}"
        )

        avg_reward, training_time = run_experiment(
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            seed=seed
        )

        rewards.append(avg_reward)
        training_times.append(training_time)

    return {
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "mean_avg_reward_last_100": np.mean(rewards),
        "std_avg_reward_last_100": np.std(rewards),
        "mean_training_time_sec": np.mean(training_times)
    }


# MAIN

if __name__ == '__main__':

    results = []

    # -----------------------------------------------------
    # 1. ALPHA vs EPSILON (gamma fixed)
    # -----------------------------------------------------

    for alpha in ALPHAS:
        for epsilon in EPSILONS:

            result = run_multiple_trials(
                trials=TRIALS,
                episodes=EPISODES,
                alpha=alpha,
                gamma=FIXED_GAMMA_ALPHA_EPSILON,
                epsilon=epsilon
            )

            result["experiment_type"] = "alpha_vs_epsilon"
            results.append(result)

    # -----------------------------------------------------
    # 2. ALPHA vs GAMMA (epsilon fixed)
    # -----------------------------------------------------

    for alpha in ALPHAS:
        for gamma in GAMMAS:

            result = run_multiple_trials(
                trials=TRIALS,
                episodes=EPISODES,
                alpha=alpha,
                gamma=gamma,
                epsilon=FIXED_EPSILON_ALPHA_GAMMA
            )

            result["experiment_type"] = "alpha_vs_gamma"
            results.append(result)

    # -----------------------------------------------------
    # 3. GAMMA vs EPSILON (alpha fixed)
    # -----------------------------------------------------

    for gamma in GAMMAS:
        for epsilon in EPSILONS:

            result = run_multiple_trials(
                trials=TRIALS,
                episodes=EPISODES,
                alpha=FIXED_ALPHA_GAMMA_EPSILON,
                gamma=gamma,
                epsilon=epsilon
            )

            result["experiment_type"] = "gamma_vs_epsilon"
            results.append(result)

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nResults saved to {OUTPUT_CSV}")