# simple_seed_ci.py
import random
import numpy as np
import torch

from library_env_random_start import (
    LibraryShelfEnv,
    DQN,
    DuelingDQN,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 🔹 여기만 네가 사용하는 Dueling 모델 파일로 맞춰줘!
MODEL_PATH = "library_shelf_random_start_curriculum.pt"

# seed 실험에 사용할 시드 값들
SEEDS = [1, 42, 2025]

# 신뢰구간 계산을 위해 랜덤 출발 평가를 몇 번 반복할지
N_EVALS_FOR_CI = 5
EPISODES_PER_EVAL = 100


# --------------------------------------------------
# 1. 모델 로드
# --------------------------------------------------
def load_policy(model_path=MODEL_PATH):
    env = LibraryShelfEnv()
    state_dim = len(env.reset())
    n_actions = 4

    ckpt = torch.load(model_path, map_location=device)
    use_dueling = ckpt.get("use_dueling", False)

    print(f"\n📦 체크포인트 로드: {model_path}")
    print(f"   - dueling : {use_dueling}")

    if use_dueling:
        policy = DuelingDQN(state_dim, n_actions).to(device)
    else:
        policy = DQN(state_dim, n_actions).to(device)

    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return env, policy


# --------------------------------------------------
# 2. 평가 함수 (S 시작 / 랜덤 시작)
# --------------------------------------------------
def evaluate(env, policy, random_start=False, episodes=100):
    successes = 0
    steps_list = []

    for _ in range(episodes):
        # 타겟 A/B/C 중 하나 랜덤 선택
        target_idx = random.randint(0, len(env.target_keys) - 1)

        s = env.reset(target_idx=target_idx, random_start=random_start)
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

    success_rate = successes / episodes * 100.0
    avg_steps = float(np.mean(steps_list)) if steps_list else None
    return success_rate, avg_steps


# --------------------------------------------------
# 3. Seed 설정 함수
# --------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# 4. 메인: Seed별 성능 + 신뢰구간
# --------------------------------------------------
if __name__ == "__main__":
    env, policy = load_policy()

    print("\n==============================================")
    print("🔹 [Part 1] Seed 변경에 따른 성능 비교")
    print("==============================================")
    print("Seed | S-start 성공률(%) | Random-start 성공률(%)")

    seed_results_random = []

    for sd in SEEDS:
        set_seed(sd)
        sr_s, _ = evaluate(env, policy, random_start=False, episodes=EPISODES_PER_EVAL)

        set_seed(sd)
        sr_r, _ = evaluate(env, policy, random_start=True, episodes=EPISODES_PER_EVAL)

        seed_results_random.append(sr_r)
        print(f"{sd:4d} | {sr_s:7.1f}           | {sr_r:7.1f}")

    print("\n==============================================")
    print("🔹 [Part 2] Random-start 반복 평가 기반 신뢰구간")
    print("==============================================")

    # 하나의 기준 seed를 사용해서 N_EVALS_FOR_CI번 반복 평가
    base_seed = 42
    random_start_rates = []

    for i in range(N_EVALS_FOR_CI):
        set_seed(base_seed + i)  # 살짝씩 다른 시드
        sr, _ = evaluate(env, policy, random_start=True, episodes=EPISODES_PER_EVAL)
        random_start_rates.append(sr)
        print(f"Eval {i+1}: 성공률 = {sr:.1f}%")

    rates = np.array(random_start_rates)
    mean = rates.mean()
    std = rates.std(ddof=1) if len(rates) > 1 else 0.0
    ci_95 = 1.96 * std  # 대략적인 95% CI

    print("\n📊 랜덤 출발 성능 요약")
    print(f" - 평균 성공률        : {mean:.2f}%")
    print(f" - 표준편차 (std)     : {std:.2f}")
    print(f" - 95% 신뢰구간 (CI)  : {mean:.2f} ± {ci_95:.2f} (%)")

    print("\n✅ Seed 변화 & 신뢰구간 실험 완료")
