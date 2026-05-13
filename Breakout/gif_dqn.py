"""
Save trained DQN Breakout agent gameplay as GIF
Loads weights from dqn_breakout.pt
"""

import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import imageio
import ale_py

from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
SEED = 123

ENV_ID = "ALE/Breakout-v5"

FRAME_STACK = 4
FRAME_SIZE  = 84

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "results2m/dqn_breakout2m.pt" # change to model you want to use

GIF_PATH = "results2m/breakout_dqn2m_seed123.gif" # change gif save path as you wish

MAX_FRAMES = 10000     # stop recording after this many frames

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────
def make_env():

    # rgb_array needed for recording frames
    env = gym.make(ENV_ID, render_mode="rgb_array")

    env = AtariPreprocessing(
        env,
        screen_size=FRAME_SIZE,
        grayscale_obs=True,
        frame_skip=1,
        noop_max=30,
        grayscale_newaxis=False,
        scale_obs=False,
        terminal_on_life_loss=False
    )

    env = FrameStackObservation(env, stack_size=FRAME_STACK)

    return env

# ──────────────────────────────────────────────
# Q-Network
# ──────────────────────────────────────────────
class QNetwork(nn.Module):

    def __init__(self, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        dummy = torch.zeros(1, FRAME_STACK, FRAME_SIZE, FRAME_SIZE)
        conv_out = int(np.prod(self.conv(dummy).shape[1:]))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):

        x = x.float() / 255.0

        return self.fc(self.conv(x))

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():

    env = make_env()

    n_actions = env.action_space.n

    q_net = QNetwork(n_actions).to(DEVICE)

    # Load model
    q_net.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    q_net.eval()

    obs, _ = env.reset(seed=SEED)

    frames = []

    total_reward = 0

    for frame_idx in range(MAX_FRAMES):

        # Save rendered frame
        frame = env.render()
        frames.append(frame)

        with torch.no_grad():

            state = torch.tensor(
                np.array(obs),
                dtype=torch.uint8,
                device=DEVICE
            ).unsqueeze(0)

            action = int(
                q_net(state).argmax(dim=1).item()
            )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        done = terminated

        if done:
            break

    env.close()

    print(f"Episode reward: {total_reward}")
    print(f"Saving GIF to: {GIF_PATH}")

    # Save GIF
    imageio.mimsave(
        GIF_PATH,
        frames,
        fps=30
    )

    print("Done!")

# ──────────────────────────────────────────────

if __name__ == "__main__":
    main()