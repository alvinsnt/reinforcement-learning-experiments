# Reinforcement-Learning-Experiments

A collection of reinforcement learning experiments ranging from tabular Q-learning in Gymnasium's Taxi-v4 environment to Deep Q-Networks (DQN) for Atari Breakout using PyTorch and Gymnasium.

---

## Demonstrations

### Taxi-v4 Q-Learning

<p align="center">
  <img src="Taxi/Results/taxi.gif" alt="Taxi-v4 Q-Learning Demo" width="600">
</p>

The agent learns to efficiently pick up and drop off passengers using tabular Q-learning.

---

### Atari Breakout DQN

<p align="center">
  <img src="Breakout/Results2m/breakout_dqn2m_seed202.gif" alt="Breakout DQN Demo" width="600">
</p>

A Deep Q-Network (DQN) agent trained for 2 million steps to play Atari Breakout ALE/Breakout-v5.

---

# Setup

## 1. Python Environment

This project was tested with **Python 3.12.12**.

If you already manage Python environments with another tool, you may skip this section. The examples below use `pyenv`.

Install and configure pyenv by following the instructions in the official repository:

- https://github.com/pyenv/pyenv

Install Python 3.12.12:

```bash
pyenv install 3.12.12
```

Verify the active Python version:

```bash
$ pyenv version
3.12.12
```

If another version is active, you can switch versions with one of the following commands:

```bash
pyenv shell 3.12.12
```

```bash
pyenv local 3.12.12
```

```bash
pyenv global 3.12.12
```

For project-specific environments, using `pyenv local 3.12.12` inside the repository directory is recommended.

---

## 2. Create a Virtual Environment

Create a virtual environment:

```bash
python -m venv <path/to/venv>
```

Activate the virtual environment:

```bash
source <path/to/venv>/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

You may modify the configuration variables at the top of each program before running.


```bash
python Taxi/taxiq.py
```
```bash
python Taxi/taxi-sarsa.py
```
```bash
python Taxi/taxi-hyperparam.py
```
```bash
python Breakout/dqn_breakout.py
```

---

## Tested System Configuration

The experiments were tested on the following system:

| Component | Specification |
|---|---|
| OS | Arch Linux x86_64 |
| Host | OMEN by HP Obelisk Desktop 875-1xxx |
| Kernel | Linux 6.19.14-arch1-1 |
| CPU | Intel Core i9-9900K |
| RAM | 48 GiB |
| GPU | NVIDIA GeForce RTX 2080 Ti |
| VRAM | 11 GiB |
| CUDA | 13.2 |

---