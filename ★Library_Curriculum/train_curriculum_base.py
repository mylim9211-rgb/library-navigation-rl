# train_curriculum_base.py
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from env_curriculum import LibraryShelfEnvAG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 baseline 모델은 따로 저장
MODEL_PATH_BASE = "library_curriculum_base.pt"


# ----------------------------------------------------
# 1. Standard DQN (지금 쓰는 robust 모델과 동일 구조)
# ----------------------------------------------------
class StandardDQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------
# 2. Replay Buffer
# ----------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=device),
            torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(ns), dtype=torch.float32, device=device),
            torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------
# 3. 학습 메인 로직 (Baseline: 항상 S에서 시작)
# ----------------------------------------------------
def main():
    EPISODES = 2000
    BATCH_SIZE = 64
    LR = 0.0005
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = 200

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    env = LibraryShelfEnvAG()
    state_dim = env.state_dim
    n_actions = env.n_actions

    policy_net = StandardDQN(state_dim, n_actions).to(device)
    target_net = StandardDQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer()

    print(f"🚀 [Baseline] Curriculum Grid 학습 시작 (Device: {device})")
    print("👉 항상 S에서 시작 (random_start=False)")
    print("👉 Double DQN + SmoothL1Loss + grad clipping")

    rewards_history = []

    for ep in range(EPISODES):
        # ✅ Baseline: 항상 S에서 시작 (random_start=False)
        state = env.reset(random_start=False)
        total_reward = 0.0
        done = False

        while not done:
            # ε-greedy
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    q = policy_net(torch.tensor(state, dtype=torch.float32, device=device))
                    action = int(q.argmax().item())

            next_state, reward, done, info = env.step(action)

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 학습
            if len(memory) > 1000:
                s_b, a_b, r_b, ns_b, d_b = memory.sample(BATCH_SIZE)

                with torch.no_grad():
                    next_actions = policy_net(ns_b).argmax(dim=1, keepdim=True)
                    next_q = target_net(ns_b).gather(1, next_actions)
                    target = r_b + GAMMA * next_q * (1 - d_b)

                current_q = policy_net(s_b).gather(1, a_b)
                loss = nn.SmoothL1Loss()(current_q, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # epsilon decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # target 네트워크 동기화
        if ep % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards_history[-100:])
            print(
                f"[Baseline] Ep {ep + 1:4d} | Avg Score: {avg_r:6.2f} | "
                f"Eps: {epsilon:.2f}"
            )

    # 모델 저장
    torch.save(policy_net.state_dict(), MODEL_PATH_BASE)
    print(f"\n✅ Baseline 학습 종료! 모델 저장: {MODEL_PATH_BASE}")


if __name__ == "__main__":
    main()
