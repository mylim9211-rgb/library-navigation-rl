import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# 💡 환경 파일명을 env_curriculum으로 수정하여 임포트
from env_curriculum import LibraryShelfEnvAG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "library_simple_robust.pt"


# ----------------------------------------------------
# 1. Standard DQN (Step 2 전용 모델)
# - Random Start 상황에서 위치와 타겟의 상관관계를 충분히 학습할 수 있는 128 노드 구성
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
# 3. 학습 메인 로직
# ----------------------------------------------------
def main():
    # --- 하이퍼파라미터 설정 ---
    EPISODES = 2000  # 일반화 성능(Random Start) 확보를 위해 2000회 수행
    BATCH_SIZE = 64
    LR = 0.0005  # 적절한 학습률
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = 200  # 타겟 네트워크 업데이트 주기

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    env = LibraryShelfEnvAG()
    state_dim = env.state_dim
    n_actions = env.n_actions

    # 모델 생성 (Standard DQN)
    policy_net = StandardDQN(state_dim, n_actions).to(device)
    target_net = StandardDQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer()

    print(f"🚀 [Step 2: Simple Robust] 학습 시작 (Device: {device})")
    print("👉 환경 파일: env_curriculum.py 사용")
    print("👉 전략: Random Start 50% 확률 적용 + Double DQN 로직")

    rewards_history = []

    for ep in range(EPISODES):
        # 💡 일반화 테스트: 50% 확률로 랜덤 시작
        use_random_start = (random.random() < 0.5)

        state = env.reset(random_start=use_random_start)
        total_reward = 0
        done = False

        while not done:
            # Action 선택
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    q = policy_net(torch.tensor(state, dtype=torch.float32, device=device))
                    action = int(q.argmax().item())

            # Step
            next_state, reward, done, info = env.step(action)

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 학습 수행 (데이터 축적 후)
            if len(memory) > 1000:
                s_b, a_b, r_b, ns_b, d_b = memory.sample(BATCH_SIZE)

                # [Double DQN 적용]
                with torch.no_grad():
                    # 차기 행동 선택: Policy Net
                    next_actions = policy_net(ns_b).argmax(dim=1, keepdim=True)
                    # 가치 평가: Target Net
                    next_q = target_net(ns_b).gather(1, next_actions)
                    target = r_b + GAMMA * next_q * (1 - d_b)

                current_q = policy_net(s_b).gather(1, a_b)

                # 안정적인 수렴을 위한 SmoothL1Loss 사용
                loss = nn.SmoothL1Loss()(current_q, target)

                optimizer.zero_grad()
                loss.backward()
                # 그래디언트 클리핑으로 안정성 강화
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # Epsilon Decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # 타겟 네트워크 동기화
        if ep % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(total_reward)

        # 로그 출력 (100 에피소드 단위)
        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards_history[-100:])
            print(
                f"Ep {ep + 1:4d} | Avg Score: {avg_r:6.2f} | Eps: {epsilon:.2f} | Mode: {'Random' if use_random_start else 'Fixed'}")

    # 최종 모델 저장
    torch.save(policy_net.state_dict(), MODEL_PATH)
    print(f"\n✅ Step 2 학습 종료! 모델 저장됨: {MODEL_PATH}")


if __name__ == "__main__":
    main()