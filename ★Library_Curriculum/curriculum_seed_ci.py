# curriculum_seed_ci.py
import random
import numpy as np
import torch
import torch.nn as nn

from env_curriculum import LibraryShelfEnvAG  # 이미 쓰고 있던 환경

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 여기만 바꿔가면서 사용하면 됨
#  - Robust 모델: "library_simple_robust.pt"
#  - Baseline 모델: "library_curriculum_base.pt"
MODEL_PATH = "library_curriculum_base.pt"


# --------------------------------------------------
# 1. 네트워크 정의 (훈련 때 썼던 StandardDQN이랑 동일하게)
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
    ckpt = torch.load(model_path, map_location=device)
    policy.load_state_dict(ckpt)
    policy.eval()

    print(f"📦 체크포인트 로드: {model_path}")
    return env, policy


# --------------------------------------------------
# 3. 한 번의 평가 (S-start / Random-start 선택)
# --------------------------------------------------
def eval_once(env, policy, n_episodes=100, random_start=False):
    successes = 0
    steps_success = []
    stuck_count = 0

    for _ in range(n_episodes):
        s = env.reset(random_start=random_start)
        done = False
        steps = 0
        prev_pos = None
        stuck_steps = 0

        while not done and steps < env.max_steps:
            state_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = policy(state_t)[0]
            a = int(q.argmax().item())

            s, r, done, info = env.step(a)
            steps += 1

            # 간단한 stuck 감지 (같은 자리 반복 등)
            if prev_pos == env.agent_pos if hasattr(env, "agent_pos") else None:
                stuck_steps += 1
            else:
                stuck_steps = 0
            prev_pos = getattr(env, "agent_pos", None)

        if info.get("reached_goal", False):
            successes += 1
            steps_success.append(steps)
        if not info.get("reached_goal", False):
            stuck_count += 1

    success_rate = successes / n_episodes * 100.0
    avg_steps = float(np.mean(steps_success)) if steps_success else None
    stuck_rate = stuck_count / n_episodes * 100.0

    return success_rate, avg_steps, stuck_rate


# --------------------------------------------------
# 4. Seed 변화 실험 + Random-start 신뢰구간
# --------------------------------------------------
def main():
    print("device:", device)
    env, policy = load_policy()

    # ---------- Part 1: Seed 변화 실험 ----------
    seeds = [1, 42, 2025]
    print("\n==============================================")
    print("🔹 [Part 1] Curriculum Grid – Seed 변화 실험 요약 (Random-start 기준)")
    print("==============================================")
    print("Seed | Random-start 성공률(%)")

    seed_results = []
    for sd in seeds:
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sd)

        sr, avg_steps, stuck = eval_once(env, policy, n_episodes=100, random_start=True)
        seed_results.append(sr)
        print(f"{sd:4d} | {sr:6.1f}")

    # ---------- Part 2: Random-start 신뢰구간 ----------
    print("\n==============================================")
    print("🔹 [Part 2] Curriculum Grid – Random-start 신뢰구간 분석")
    print("==============================================")

    eval_success_rates = []
    N_EVAL = 5  # 5번 반복 평가

    for i in range(N_EVAL):
        sd = 1000 + i  # 평가용 seed
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sd)

        sr, avg_steps, stuck = eval_once(env, policy, n_episodes=100, random_start=True)
        eval_success_rates.append(sr)
        print(f"Eval {i+1}: 성공률 = {sr:.1f}%")

    mean_sr = float(np.mean(eval_success_rates))
    std_sr = float(np.std(eval_success_rates, ddof=0))
    ci = 1.96 * std_sr  # Simple Grid에서 쓴 방식 그대로 사용

    print("\n📊 Random-start 성능 요약 (Curriculum Grid)")
    print(f" - 평균 성공률        : {mean_sr:.2f}%")
    print(f" - 표준편차 (std)     : {std_sr:.2f}")
    print(f" - 95% 신뢰구간 (CI)  : {mean_sr:.2f} ± {ci:.2f} (%)")

    print("\n✅ Curriculum Grid Seed & CI 실험 완료")


if __name__ == "__main__":
    main()
