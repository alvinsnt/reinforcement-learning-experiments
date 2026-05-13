import gymnasium as gym
import numpy as np
import pickle
import imageio

# Run taxiq.py or taxi-sarsa.py first to get .pkl file

MODEL_FILE = "taxi.pkl"
GIF_FILE = "Taxi/Results/taxi.gif"

# Load trained Q-table
with open(MODEL_FILE, "rb") as f:
    q = pickle.load(f)

# Create environment with rgb_array rendering
env = gym.make("Taxi-v4", render_mode="rgb_array")

state = env.reset()[0]
terminated = False
truncated = False

frames = []

while not terminated and not truncated:

    # Capture frame
    frame = env.render()
    frames.append(frame)

    # Select best action from Q-table
    action = np.argmax(q[state, :])

    # Step environment
    state, reward, terminated, truncated, _ = env.step(action)

env.close()

# Save GIF
imageio.mimsave(GIF_FILE, frames, fps=5)

print(f"GIF saved as {GIF_FILE}")