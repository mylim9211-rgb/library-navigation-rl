# train_random_start_curriculum.py
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from library_env_random_start import LibraryShelfEnv, DQN, DuelingDQN

# --------------------------------------------------
# 공통 설정
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

EPISODES = 4000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 5e-4
MIN_REPLAY_SIZE = 1000

USE_DOUBLE = True        # Double DQN 사용
USE_DUELING = True      # 필요하면 True로 바꿔도 됨

# 🔥 커리큘럼: 랜덤 시작 비율이 서서히 증가 (0.0 → 0.8)
RANDOM_START_MIN = 0.0
RANDOM_START_MAX = 0.8

MODEL_PATH = "library_shelf_random_start_curriculum.pt"


# --------------------------------------------------
# Replay Buffer
# --------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)

        s = np.array(s, dtype=np.float32)
        ns = np.array(ns, dtype=np.float32)

        return (
            torch.tensor(s, dtype=torch.float32, device=device),
            torch.tensor(a, dtype=torch.long, device=device),
            torch.tensor(r, dtype=torch.float32, device=device),
            torch.tensor(ns, dtype=torch.float32, device=device),
            torch.tensor(d, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buf)


# --------------------------------------------------
# 평가: 랜덤 시작 + 랜덤 타겟 기준 성능
# --------------------------------------------------
def eval_policy(env, policy, n_episodes=50):
    policy.eval()

    successes = 0
    steps_list = []

    for _ in range(n_episodes):
        target_idx = random.randint(0, len(env.target_keys) - 1)
        s = env.reset(target_idx=target_idx, random_start=True)

        done = False
        steps = 0
        while not done and steps < env.max_steps:
            state_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = policy(state_t)[0]
            a = int(q.argmax().item())
            s, r, done, info = env.step(a)
            steps += 1

        if info.get("reached_goal", False):
            successes += 1
            steps_list.append(steps)

    success_rate = successes / n_episodes if n_episodes > 0 else 0.0
    avg_steps = float(np.mean(steps_list)) if steps_list else None

    print("\n📊 [평가] 랜덤 시작 + 랜덤 타겟")
    print(f"   성공률: {successes}/{n_episodes} ({success_rate*100:.1f}%)")
    if avg_steps is not None:
        print(f"   성공 에피소드 평균 스텝: {avg_steps:.1f}")
    else:
        print("   성공한 에피소드가 없어 평균 스텝 없음")

    return success_rate, avg_steps


# --------------------------------------------------
# 학습 함수
# --------------------------------------------------
def train_agent():
    env = LibraryShelfEnv()
    state_dim = len(env.reset())
    n_actions = 4

    # 네트워크 선택
    if USE_DUELING:
        policy = DuelingDQN(state_dim, n_actions).to(device)
        target = DuelingDQN(state_dim, n_actions).to(device)
    else:
        policy = DQN(state_dim, n_actions).to(device)
        target = DQN(state_dim, n_actions).to(device)

    target.load_state_dict(policy.state_dict())
    opt = optim.Adam(policy.parameters(), lr=LR)
    buf = ReplayBuffer()

    epsilon = 1.0
    eps_min = 0.05
    eps_decay = 0.999

    step_count = 0
    rewards_log = []
    success_log = []

    for ep in range(EPISODES):
        # 🔥 커리큘럼: 에피소드가 진행될수록 랜덤 시작 비율을 올림
        frac = ep / EPISODES
        random_start_prob = RANDOM_START_MIN + (RANDOM_START_MAX - RANDOM_START_MIN) * frac
        random_start_prob = max(0.0, min(1.0, random_start_prob))

        use_random_start = (random.random() < random_start_prob)
        s = env.reset(random_start=use_random_start)

        done = False
        ep_reward = 0.0
        success = False

        while not done:
            # ε-greedy
            if random.random() < epsilon:
                a = random.randint(0, n_actions - 1)
            else:
                state_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q = policy(state_t)[0]
                a = int(q.argmax().item())

            ns, r, done, info = env.step(a)
            buf.push(s, a, r, ns, float(done))
            s = ns
            ep_reward += r
            if info.get("reached_goal", False):
                success = True

            # 일정 이상 쌓이면 학습
            if len(buf) >= max(BATCH_SIZE, MIN_REPLAY_SIZE):
                bs, ba, br, bns, bd = buf.sample(BATCH_SIZE)

                q_values = policy(bs).gather(1, ba.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    if USE_DOUBLE:
                        # Double DQN
                        next_q_online = policy(bns)
                        next_actions = next_q_online.argmax(dim=1)
                        next_q_target = target(bns).gather(
                            1, next_actions.unsqueeze(1)
                        ).squeeze(1)
                        target_q = br + GAMMA * next_q_target * (1 - bd)
                    else:
                        # Vanilla / Dueling
                        next_q = target(bns).max(1)[0]
                        target_q = br + GAMMA * next_q * (1 - bd)

                loss = nn.MSELoss()(q_values, target_q)
                opt.zero_grad()
                loss.backward()
                opt.step()

            step_count += 1
            if step_count % 500 == 0:
                target.load_state_dict(policy.state_dict())

        epsilon = max(eps_min, epsilon * eps_decay)
        rewards_log.append(ep_reward)
        success_log.append(1 if success else 0)

        # 로그
        if (ep + 1) % 200 == 0:
            avg_r = float(np.mean(rewards_log[-200:]))
            succ_rate = float(np.mean(success_log[-200:]) * 100)
            tag = "Double" if USE_DOUBLE else "Vanilla"
            if USE_DUELING:
                tag += "+Dueling"
            print(
                f"[{tag} | ep={ep+1}/{EPISODES}] "
                f"AvgR(최근200)={avg_r:.3f} | "
                f"Succ={succ_rate:.1f}% | "
                f"eps={epsilon:.2f} | "
                f"rand_start_prob={random_start_prob:.2f}"
            )

    # 모델 저장
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "use_double": USE_DOUBLE,
            "use_dueling": USE_DUELING,
            "random_start_min": RANDOM_START_MIN,
            "random_start_max": RANDOM_START_MAX,
            "episodes": EPISODES,
        },
        MODEL_PATH,
    )
    print(f"\n💾 모델 저장 완료: {MODEL_PATH}")

    # 최종 평가 (랜덤 시작 + 랜덤 타겟)
    eval_policy(env, policy, n_episodes=50)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    print("🤖 랜덤 시작점 DQN (커리큘럼) 학습 시작...")
    train_agent()
    print("🎉 학습 종료")
