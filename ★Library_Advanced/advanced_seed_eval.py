# advanced_seed_eval.py
import math
import random
import numpy as np
import torch
import torch.nn as nn

from env_advanced import LibraryShelfEnvAG, DuelingDQN, device

# 🔧 학습해둔 최종 모델 경로 (Double + Dueling DQN)
MODEL_PATH = "library_AG_double_dueling.pt"

# 🔧 실험에 사용할 seed 목록 (원하는 대로 바꿔도 됨)
SEEDS = [1, 42, 2025]
N_EPISODES = 200  # seed당 평가 에피소드 수


# ----------------------------------------------------
# 1. 모델 로드
# ----------------------------------------------------
def load_policy(env):
    state_dim = env.state_dim
    n_actions = 4

    policy = DuelingDQN(state_dim, n_actions).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    if "state_dict" in ckpt:
        policy.load_state_dict(ckpt["state_dict"])
    else:
        # state_dict만 저장했을 때
        policy.load_state_dict(ckpt)
    policy.eval()

    print(f"📦 Final Model 로드 완료: {MODEL_PATH}")
    return policy


# ----------------------------------------------------
# 2. 평가 함수 (S-start / Random-start 공용)
# ----------------------------------------------------
def eval_mode(env, policy, num_episodes=200, random_start=True):
    """
    random_start=True  : 매 에피소드 랜덤 위치 + 랜덤 타겟
    random_start=False : 항상 S에서 시작 + 랜덤 타겟
    """
    target_keys = env.target_keys

    total_success = 0
    total_steps_success = 0
    total_episodes = num_episodes

    for ep in range(num_episodes):
        ti = random.randint(0, len(target_keys) - 1)
        state = env.reset(target_idx=ti, random_start=random_start)

        done = False
        steps = 0
        info = {}

        while not done and steps < env.max_steps:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = int(policy(state_t).argmax().item())

            state, reward, done, info = env.step(a)
            steps += 1

        reached = info.get("reached_goal", False)

        if reached:
            total_success += 1
            total_steps_success += steps

    success_rate = total_success / total_episodes * 100.0
    avg_steps_success = (
        total_steps_success / total_success if total_success > 0 else 0.0
    )

    return success_rate, avg_steps_success, total_success, total_episodes


# ----------------------------------------------------
# 3. 이항분포 기반 95% 신뢰구간 계산
#    (전체 에피소드 수 = seed * N_EPISODES 기준)
# ----------------------------------------------------
def binom_ci(success, total, alpha=0.05):
    p_hat = success / total
    se = math.sqrt(p_hat * (1 - p_hat) / total)
    z = 1.96  # 95% CI
    low = p_hat - z * se
    high = p_hat + z * se
    return p_hat * 100, low * 100, high * 100


# ----------------------------------------------------
# 4. 메인: seed별 평가 + CI 계산
# ----------------------------------------------------
def main():
    all_s_rates = []
    all_r_rates = []

    total_s_success = 0
    total_s_episodes = 0
    total_r_success = 0
    total_r_episodes = 0

    print("==================================")
    print("🎲 Advanced Grid – seed별 평가 결과")
    print("==================================")

    for sd in SEEDS:
        # seed 설정
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sd)

        env = LibraryShelfEnvAG()
        policy = load_policy(env)

        # S-start
        s_rate, s_avg_steps, s_succ, s_total = eval_mode(
            env, policy, num_episodes=N_EPISODES, random_start=False
        )
        # Random-start
        r_rate, r_avg_steps, r_succ, r_total = eval_mode(
            env, policy, num_episodes=N_EPISODES, random_start=True
        )

        all_s_rates.append(s_rate)
        all_r_rates.append(r_rate)

        total_s_success += s_succ
        total_s_episodes += s_total
        total_r_success += r_succ
        total_r_episodes += r_total

        print(f"\n[Seed {sd}]")
        print(f"  S-start   성공률: {s_rate:5.1f}% | 평균 스텝(성공 시): {s_avg_steps:5.1f}")
        print(f"  Random    성공률: {r_rate:5.1f}% | 평균 스텝(성공 시): {r_avg_steps:5.1f}")

    # --- 전체 에피소드 기준 CI ---
    s_mean, s_low, s_high = binom_ci(total_s_success, total_s_episodes)
    r_mean, r_low, r_high = binom_ci(total_r_success, total_r_episodes)

    print("\n==================================")
    print("📊 전체 에피소드 기준 95% 신뢰구간")
    print("==================================")
    print(
        f"S-start   : {s_mean:5.1f}%  (95% CI: {s_low:5.1f}% ~ {s_high:5.1f}%) "
        f"[총 {total_s_success}/{total_s_episodes} 성공]"
    )
    print(
        f"Random    : {r_mean:5.1f}%  (95% CI: {r_low:5.1f}% ~ {r_high:5.1f}%) "
        f"[총 {total_r_success}/{total_r_episodes} 성공]"
    )

    # --- 참고용: seed 평균 기준 CI (optional) ---
    s_mean_seed = float(np.mean(all_s_rates))
    r_mean_seed = float(np.mean(all_r_rates))
    print("\n(참고) seed별 성공률 평균")
    print(f"  S-start  seed 평균: {s_mean_seed:5.2f}%")
    print(f"  Random   seed 평균: {r_mean_seed:5.2f}%")


if __name__ == "__main__":
    main()
