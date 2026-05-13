"""
DQN Agent for ALE/Breakout-v5
==============================
Implements a full Deep Q-Network (DQN) pipeline with:
  - CNN Q-network + target network
  - Experience replay buffer
  - ε-greedy exploration
  - Atari preprocessing (grayscale, resize, frame stack)
  - Reward clipping
  - Periodic evaluation
  - Training / evaluation reward plots
  - Summary report
"""

import os
import time
import random
import platform
import collections
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordEpisodeStatistics,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 0.  Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1.  Hyperparameters
ENV_ID            = "ALE/Breakout-v5"
TOTAL_STEPS       = 2_000_000      # total environment steps
REPLAY_CAPACITY   = 100_000        # replay buffer size
BATCH_SIZE        = 32
GAMMA             = 0.99           # discount factor
LR                = 1e-4           # Adam learning rate
TARGET_UPDATE     = 1_000          # Update target network every N step
TRAIN_START       = 10_000         # steps before first gradient update
TRAIN_FREQ        = 4              # update every N env steps
EPS_START         = 1.0            # ε start
EPS_END           = 0.05           # ε end
EPS_DECAY_STEPS   = 500_000        # ε decay steps
EVAL_FREQ         = 50_000         # evaluate every N steps
EVAL_EPISODES     = 5            # episodes per evaluation
FRAME_STACK       = 4             # frames stacked as state
FRAME_SIZE        = 84            # height & width after resize
CLIP_REWARD       = True          # clip rewards to {-1, 0, +1}
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2.  Environment factory
def make_env(seed=None, eval_mode=False):
    """
    Creates and wraps an ALE/Breakout-v5 environment with standard
    Atari preprocessing:
      - Grayscale conversion
      - Frame resizing to FRAME_SIZE × FRAME_SIZE
      - Frame skipping (skip=4)
      - Max-pooling over last 2 raw frames (flicker removal)
      - Frame stacking (FRAME_STACK consecutive frames)
    """
    env = gym.make(ENV_ID, render_mode=None)
    env = AtariPreprocessing(
        env,
        screen_size=FRAME_SIZE,
        grayscale_obs=True,
        frame_skip=1,
        noop_max=30,
        grayscale_newaxis=False,
        scale_obs=False,           # keep uint8; we normalise in the network
    )
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


# 3.  Q-Network (Nature DQN architecture)
class QNetwork(nn.Module):
    """
    CNN-based Q-network following the Nature DQN paper (Mnih et al., 2015).
    Input : (batch, 4, 84, 84)  uint8 → float32 /255 internally
    Output: (batch, n_actions)
    """
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute flattened size dynamically
        dummy = torch.zeros(1, FRAME_STACK, FRAME_SIZE, FRAME_SIZE)
        conv_out = int(np.prod(self.conv(dummy).shape[1:]))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0          # normalise to [0, 1]
        return self.fc(self.conv(x))


# 4.  Replay Buffer
Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)

class ReplayBuffer:
    """
    Circular replay buffer storing (s, a, r, s', done) tuples.
    States are stored as uint8 to save memory.
    """
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store as uint8 numpy arrays to minimise RAM
        self.buffer.append(Transition(
            np.array(state, dtype=np.uint8),
            action,
            reward,
            np.array(next_state, dtype=np.uint8),
            done,
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states      = torch.tensor(np.stack([t.state      for t in batch]), dtype=torch.uint8, device=DEVICE)
        actions     = torch.tensor([t.action              for t in batch], dtype=torch.long,  device=DEVICE)
        rewards     = torch.tensor([t.reward              for t in batch], dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.uint8, device=DEVICE)
        dones       = torch.tensor([t.done                for t in batch], dtype=torch.float32, device=DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# 5.  DQN Agent
class DQNAgent:
    def __init__(self, n_actions: int):
        self.n_actions  = n_actions
        self.q_net      = QNetwork(n_actions).to(DEVICE)
        self.target_net = QNetwork(n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=LR)
        self.replay     = ReplayBuffer(REPLAY_CAPACITY)
        self.step_count = 0
        self.loss_sum   = 0.0
        self.loss_steps = 0

    # ε schedule
    def epsilon(self) -> float:
        frac = min(self.step_count / EPS_DECAY_STEPS, 1.0)
        return EPS_START + frac * (EPS_END - EPS_START)

    def select_action(self, state) -> int:
        if random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(np.array(state), dtype=torch.uint8, device=DEVICE).unsqueeze(0)
            return int(self.q_net(s).argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        if CLIP_REWARD:
            reward = float(np.sign(reward))
        self.replay.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (double DQN style: action chosen by online net)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q       = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets      = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.loss_sum   += loss.item()
        self.loss_steps += 1

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# 6.  Evaluation
def evaluate(agent: DQNAgent, n_episodes: int = EVAL_EPISODES) -> float:
    """Run n_episodes with ε=0.05 (greedy) and return mean episode return."""
    eval_env = make_env(seed=SEED + 1000)
    returns  = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                s  = torch.tensor(np.array(obs), dtype=torch.uint8, device=DEVICE).unsqueeze(0)
                a  = int(agent.q_net(s).argmax(dim=1).item())
            obs, r, term, trunc, _ = eval_env.step(a)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    eval_env.close()
    return float(np.mean(returns))


# 7.  Training loop
def train():
    print(f"Device : {DEVICE}")
    print(f"Env    : {ENV_ID}")
    print(f"Steps  : {TOTAL_STEPS:,}\n")

    env   = make_env(seed=SEED)
    n_act = env.action_space.n
    agent = DQNAgent(n_actions=n_act)

    # ── logging ──
    train_rewards  = []   # raw episode rewards during training
    train_steps_ep = []   # env step at episode end
    eval_rewards   = []   # mean eval reward
    eval_steps     = []   # env step at evaluation

    # ── training state ──
    obs, _   = env.reset(seed=SEED)
    ep_ret   = 0.0
    t_start  = time.time()
    step     = 0

    while step < TOTAL_STEPS:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done   = terminated or truncated
        ep_ret += reward

        agent.store(obs, action, reward, next_obs, done)
        obs    = next_obs
        step  += 1
        agent.step_count = step

        # ── gradient update ──
        if step >= TRAIN_START and step % TRAIN_FREQ == 0:
            agent.update()

        # ── sync target network ──
        if step % TARGET_UPDATE == 0:
            agent.sync_target()

        # ── episode end ──
        if done:
            train_rewards.append(ep_ret)
            train_steps_ep.append(step)
            ep_ret  = 0.0
            obs, _  = env.reset()

        # ── periodic evaluation ──
        if step % EVAL_FREQ == 0:
            mean_ret = evaluate(agent)
            eval_rewards.append(mean_ret)
            eval_steps.append(step)
            elapsed  = time.time() - t_start
            loss_avg = agent.loss_sum / max(agent.loss_steps, 1)
            print(
                f"Step {step:>7,}/{TOTAL_STEPS:,} | "
                f"ε={agent.epsilon():.3f} | "
                f"EvalRet={mean_ret:6.1f} | "
                f"Loss={loss_avg:.4f} | "
                f"Elapsed={str(timedelta(seconds=int(elapsed)))}"
            )
            agent.loss_sum   = 0.0
            agent.loss_steps = 0

    env.close()
    total_time = time.time() - t_start
    return agent, train_rewards, train_steps_ep, eval_rewards, eval_steps, total_time


# 8.  Plotting
def moving_average(x, window=50):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_results(train_rewards, train_steps, eval_rewards, eval_steps,
                 total_time, final_eval):
    fig = plt.figure(figsize=(16, 10), facecolor="#0f1117")
    fig.suptitle(
        f"DQN on {ENV_ID}",
        fontsize=18, fontweight="bold", color="white", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_colors = {"axes.facecolor": "#1a1d27",
                 "grid.color": "#2e3248",
                 "text.color": "white",
                 "axes.labelcolor": "#a0a8c8",
                 "xtick.color": "#6b7394",
                 "ytick.color": "#6b7394",
                 "axes.edgecolor": "#2e3248"}

    def style_ax(ax, title):
        for k, v in ax_colors.items():
            if k == "axes.facecolor":
                ax.set_facecolor(v)
            elif k == "axes.labelcolor":
                ax.xaxis.label.set_color(v); ax.yaxis.label.set_color(v)
            elif k == "axes.edgecolor":
                for spine in ax.spines.values(): spine.set_edgecolor(v)
        ax.tick_params(colors="#6b7394")
        ax.set_title(title, color="#c8d0f0", fontsize=11, pad=8)
        ax.grid(True, color="#2e3248", linewidth=0.6, alpha=0.8)

    # ── (a) Raw training reward ──
    ax0 = fig.add_subplot(gs[0, 0])
    style_ax(ax0, "Training Episode Reward")
    ax0.plot(train_steps, train_rewards, color="#4d8ef5", linewidth=0.6, alpha=0.5, label="Episode")
    ax0.set_xlabel("Environment Steps"); ax0.set_ylabel("Return")
    ax0.legend(facecolor="#1a1d27", edgecolor="#2e3248", labelcolor="white", fontsize=9)

    # ── (b) Moving average ──
    ax1 = fig.add_subplot(gs[0, 1])
    style_ax(ax1, "Moving-Average Training Reward (window=50)")
    window = 50
    if len(train_rewards) >= window:
        ma = moving_average(train_rewards, window)
        ma_steps = train_steps[window - 1:]
        ax1.plot(ma_steps, ma, color="#f5a04d", linewidth=1.5, label=f"MA-{window}")
    else:
        ax1.plot(train_steps, train_rewards, color="#f5a04d", linewidth=1.5)
    ax1.set_xlabel("Environment Steps"); ax1.set_ylabel("Return")
    ax1.legend(facecolor="#1a1d27", edgecolor="#2e3248", labelcolor="white", fontsize=9)

    # ── (c) Evaluation reward ──
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, f"Evaluation Reward (every {EVAL_FREQ:,} steps, {EVAL_EPISODES} eps)")
    ax2.plot(eval_steps, eval_rewards, color="#5df5b0", linewidth=2, marker="o",
             markersize=4, label="Mean eval return")
    ax2.set_xlabel("Environment Steps"); ax2.set_ylabel("Return")
    ax2.legend(facecolor="#1a1d27", edgecolor="#2e3248", labelcolor="white", fontsize=9)

    # ── (d) Summary card ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#1a1d27")
    for spine in ax3.spines.values(): spine.set_edgecolor("#2e3248")
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title("Run Summary", color="#c8d0f0", fontsize=11, pad=8)

    hw = f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    lines = [
        ("Training steps",        f"{TOTAL_STEPS:,}"),
        ("Final eval reward",     f"{final_eval:.2f}"),
        ("Training time",         str(timedelta(seconds=int(total_time)))),
        ("Hardware",              hw),
        ("Device",                str(DEVICE).upper()),
        ("Replay buffer",         f"{REPLAY_CAPACITY:,}"),
        ("Batch size",            str(BATCH_SIZE)),
        ("Frame stack",           str(FRAME_STACK)),
        ("Frame size",            f"{FRAME_SIZE}×{FRAME_SIZE}"),
        ("Reward clipping",       str(CLIP_REWARD)),
        ("ε start → end",         f"{EPS_START} → {EPS_END}"),
        ("ε decay steps",         f"{EPS_DECAY_STEPS:,}"),
        ("Target update freq",    f"every {TARGET_UPDATE:,} steps"),
        ("Learning rate",         str(LR)),
        ("Discount (γ)",          str(GAMMA)),
    ]

    for i, (label, value) in enumerate(lines):
        y = 0.97 - i * 0.065
        ax3.text(0.02, y, label, transform=ax3.transAxes,
                 color="#a0a8c8", fontsize=8.5, va="top")
        ax3.text(0.98, y, value, transform=ax3.transAxes,
                 color="#ffffff", fontsize=8.5, va="top", ha="right", fontweight="bold")

    out = "dqn_breakout_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nPlot saved → {out}")
    plt.show()


# 9.  Report
def print_report(eval_rewards, eval_steps, total_time):
    final_eval = eval_rewards[-1] if eval_rewards else float("nan")
    hw = (
        f"GPU – {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB)"
        if torch.cuda.is_available()
        else f"CPU – {platform.processor() or platform.machine()}"
    )
    print("\n" + "═" * 54)
    print("  DQN / ALE Breakout-v5 — Run Report")
    print("═" * 54)
    print(f"  Training steps          : {TOTAL_STEPS:>12,}")
    print(f"  Final avg eval reward   : {final_eval:>12.2f}")
    print(f"  Training time           : {str(timedelta(seconds=int(total_time))):>12}")
    print(f"  Hardware                : {hw}")
    print(f"  Device (PyTorch)        : {str(DEVICE).upper():>12}")
    print(f"  Peak eval reward        : {max(eval_rewards, default=0):>12.2f}")
    print("═" * 54 + "\n")
    return final_eval


# 10.  Entry point
if __name__ == "__main__":
    agent, train_rewards, train_steps, eval_rewards, eval_steps, total_time = train()

    final_eval = print_report(eval_rewards, eval_steps, total_time)
    plot_results(train_rewards, train_steps, eval_rewards, eval_steps, total_time, final_eval)

    # Optionally save the trained model
    save_path = "dqn_breakout.pt"
    torch.save(agent.q_net.state_dict(), save_path)
    print(f"Model weights saved → {save_path}")