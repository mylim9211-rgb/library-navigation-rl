import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# 💡 환경은 어려운 'Advanced'를 사용
from env_advanced import LibraryShelfEnvAG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파일명: Baseline(기준점) 모델
MODEL_PATH = "library_advanced_baseline.pt"


# ----------------------------------------------------
# 1. Standard DQN (Step 2에서 썼던 그 녀석)
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
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=device),
            torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(ns), dtype=torch.float32, device=device),
            torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buf)


# ----------------------------------------------------
# 3. 학습 메인
# ----------------------------------------------------
def main():
    EPISODES = 4000
    BATCH_SIZE = 64
    LR = 0.0005
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = 500

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.999

    env = LibraryShelfEnvAG()
    state_dim = env.state_dim

    # 🚨 수정됨: env_advanced.py에 n_actions가 없으면 그냥 4로 설정
    # (아까 env 수정하셨으면 env.n_actions 쓰셔도 됩니다)
    try:
        n_actions = env.n_actions
    except AttributeError:
        n_actions = 4

    # 모델: Standard DQN
    policy_net = StandardDQN(state_dim, n_actions).to(device)
    target_net = StandardDQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer()

    print(f"🚀 [비교실험: Baseline] Advanced 환경에서 Standard DQN 학습 시작...")

    rewards_history = []

    for ep in range(EPISODES):
        # Curriculum: 0.0 -> 0.8 (Step 2와 동일 조건 비교)
        frac = ep / EPISODES
        random_prob = 0.0 + (0.8 - 0.0) * frac

        use_random_start = (random.random() < random_prob)
        state = env.reset(random_start=use_random_start)

        total_reward = 0
        done = False

        while not done:
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

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if ep % TARGET_UPDATE_FREQ == 0:
            # 🚨 여기가 수정된 부분입니다! (policy -> policy_net)
            target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards_history[-100:])
            print(f"Ep {ep + 1:4d} | Avg Score: {avg_r:6.2f} | Eps: {epsilon:.2f} | RandProb: {random_prob:.2f}")

    torch.save(policy_net.state_dict(), MODEL_PATH)
    print(f"\n✅ Baseline 학습 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()