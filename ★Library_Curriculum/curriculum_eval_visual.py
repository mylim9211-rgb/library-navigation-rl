# curriculum_eval_visual.py
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from env_curriculum import LibraryShelfEnvAG  # 너가 쓰는 Curriculum Grid 환경

# 🔹 한글 폰트 설정 (Windows 기준)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Robust / Baseline 모델 경로만 바꿔가면서 쓰면 됨
MODEL_PATH = "library_simple_robust.pt"      # Robust
# MODEL_PATH = "library_curriculum_base.pt"  # Baseline 평가하고 싶으면 이걸로 교체


# --------------------------------------------------
# 1. 네트워크 구조 (Robust 학습 때 썼던 StandardDQN과 동일)
# --------------------------------------------------
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


# --------------------------------------------------
# 2. 모델 로드
# --------------------------------------------------
def load_policy(model_path=MODEL_PATH):
    env = LibraryShelfEnvAG()
    state_dim = env.state_dim
    n_actions = env.n_actions

    policy = StandardDQN(state_dim, n_actions).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    print(f"📦 모델 로드 완료: {model_path}")
    return env, policy


# --------------------------------------------------
# 3. 에피소드 실행 함수 (greedy 정책)
# --------------------------------------------------
def run_episode(env, policy, random_start=False, target_idx=None):
    s = env.reset(random_start=random_start, target_idx=target_idx)

    done = False
    steps = 0
    reached = False

    while not done and steps < env.max_steps:
        state_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = policy(state_t)[0]
        a = int(q.argmax().item())

        s, r, done, info = env.step(a)
        steps += 1

        if info.get("reached_goal", False):
            reached = True

    return reached, steps


# --------------------------------------------------
# 4. S-start vs Random-start 성능 평가
# --------------------------------------------------
def eval_start_condition(env, policy, n_episodes=200):
    results = {}

    for mode_name, random_flag in [("S-start", False), ("Random-start", True)]:
        success = 0
        steps_success = []

        for _ in range(n_episodes):
            reached, steps = run_episode(env, policy, random_start=random_flag)
            if reached:
                success += 1
                steps_success.append(steps)

        success_rate = success / n_episodes * 100.0
        avg_steps = np.mean(steps_success) if steps_success else None
        results[mode_name] = (success_rate, avg_steps)

        if avg_steps is not None:
            print(f"[{mode_name}] 성공률: {success_rate:.1f}% | "
                  f"성공 시 평균 스텝: {avg_steps:.1f}")
        else:
            print(f"[{mode_name}] 성공률: {success_rate:.1f}% | 성공 에피소드 없음")

    return results


# --------------------------------------------------
# 5. 타겟별 성공률 (랜덤 시작 기준)
# --------------------------------------------------
def eval_targetwise(env, policy, n_episodes_per_target=50):
    target_keys = env.target_keys  # ['A','B','C','D','E','F'] 같은 구조라고 가정
    success_dict = {}

    for idx, key in enumerate(target_keys):
        success = 0
        for _ in range(n_episodes_per_target):
            reached, steps = run_episode(
                env,
                policy,
                random_start=True,
                target_idx=idx
            )
            if reached:
                success += 1
        rate = success / n_episodes_per_target * 100.0
        success_dict[key] = rate
        print(f"타겟 {key}: 성공률 {rate:.1f}%")

    return success_dict


# --------------------------------------------------
# 6. 바 차트 시각화 (S vs Random, 타겟별 성공률)
# --------------------------------------------------
def plot_summary(start_results, target_success):
    # (1) S-start vs Random-start
    labels = list(start_results.keys())
    rates = [start_results[k][0] for k in labels]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, rates)
    for i, v in enumerate(rates):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")
    plt.ylim(0, 100)
    plt.ylabel("Success Rate (%)")
    plt.title("Curriculum Grid – S 시작 vs Random 시작 성공률")
    plt.tight_layout()
    plt.show()

    # (2) 타겟별 성공률
    keys = list(target_success.keys())
    vals = [target_success[k] for k in keys]

    plt.figure(figsize=(6, 4))
    plt.bar(keys, vals)
    for i, v in enumerate(vals):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")
    plt.ylim(0, 100)
    plt.ylabel("Success Rate (%)")
    plt.title("랜덤 시작 시 타겟별 성공률")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 7. 메인
# --------------------------------------------------
if __name__ == "__main__":
    env, policy = load_policy()

    print("\n=== [1] S-start vs Random-start 평가 ===")
    start_results = eval_start_condition(env, policy, n_episodes=200)

    print("\n=== [2] 타겟별 성공률 평가 (랜덤 시작) ===")
    target_success = eval_targetwise(env, policy, n_episodes_per_target=50)

    print("\n=== [3] 요약 그래프 출력 ===")
    plot_summary(start_results, target_success)

    print("\n✅ Curriculum Grid Evaluation & Visualization 완료")
