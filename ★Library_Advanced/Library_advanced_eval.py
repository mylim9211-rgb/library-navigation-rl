# advanced_eval_summary.py
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from env_advanced import LibraryShelfEnvAG, DuelingDQN, device

MODEL_PATH = "library_AG_double_dueling.pt"

# 슬라이드용 seed (대표값용) + 신뢰구간 계산용 seed 목록
REPRESENTATIVE_SEED = 42
SEEDS_FOR_CI = [1, 42, 2025]
N_EPISODES = 200  # seed당 평가 에피소드 수


# --------------------------------------------------
# 1. 모델 로드
# --------------------------------------------------
def load_policy(env):
    state_dim = env.state_dim
    n_actions = 4

    policy = DuelingDQN(state_dim, n_actions).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    # state_dict만 저장했는지, dict로 저장했는지 둘 다 대응
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        policy.load_state_dict(ckpt["state_dict"])
    else:
        policy.load_state_dict(ckpt)

    policy.eval()
    return policy


# --------------------------------------------------
# 2. 에피소드 실행 & 평가 함수
# --------------------------------------------------
def run_episode(env, policy, random_start=False, target_idx=None):
    if target_idx is None:
        # 타겟 랜덤 선택
        target_idx = random.randint(0, len(env.target_keys) - 1)

    s = env.reset(target_idx=target_idx, random_start=random_start)

    done = False
    steps = 0
    info = {}

    while not done and steps < env.max_steps:
        state_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = policy(state_t)[0]
        a = int(q.argmax().item())

        s, r, done, info = env.step(a)
        steps += 1

    reached = info.get("reached_goal", False)
    return reached, steps, target_idx


def eval_over_episodes(env, policy, n_episodes=200, random_start=False):
    """
    S-start / Random-start 공용 평가 함수
    - 전체 성공률, 성공 시 평균 스텝
    - 타겟별 성공률 (슬라이드에서 필요하면 쓰면 됨)
    """
    target_keys = env.target_keys

    total_success = 0
    total_success_steps = 0

    per_target = defaultdict(lambda: {"total": 0, "success": 0})

    for _ in range(n_episodes):
        reached, steps, ti = run_episode(env, policy, random_start=random_start)
        key = target_keys[ti]

        per_target[key]["total"] += 1
        if reached:
            per_target[key]["success"] += 1
            total_success += 1
            total_success_steps += steps

    success_rate = total_success / n_episodes * 100.0
    avg_steps_success = (
        total_success_steps / total_success if total_success > 0 else 0.0
    )

    target_success = {}
    for k in target_keys:
        tot = per_target[k]["total"]
        suc = per_target[k]["success"]
        rate = suc / tot * 100.0 if tot > 0 else 0.0
        target_success[k] = rate

    return success_rate, avg_steps_success, target_success


# --------------------------------------------------
# 3. seed별 Random-start 성공률로부터 신뢰구간 계산
# --------------------------------------------------
def eval_random_start_over_seeds():
    random_rates = []

    for sd in SEEDS_FOR_CI:
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sd)

        env = LibraryShelfEnvAG()
        policy = load_policy(env)

        r_rate, r_steps, _ = eval_over_episodes(
            env, policy, n_episodes=N_EPISODES, random_start=True
        )
        random_rates.append(r_rate)
        print(f"[CI용 Seed {sd}] Random-start 성공률: {r_rate:.1f}%")

    random_rates = np.array(random_rates)
    mean = float(random_rates.mean())
    std = float(random_rates.std(ddof=1))  # sample std
    ci_half = 1.96 * std  # N이 작아서 t 대신 그냥 1.96 사용 (Curriculum과 동일 방식)

    return mean, std, ci_half, random_rates


# --------------------------------------------------
# 4. 메인: 대표 seed에 대한 슬라이드 숫자 + CI 계산
# --------------------------------------------------
def main():
    # 1) 대표 seed 기준 S-start / Random-start 성능 (슬라이드 상단 박스용)
    print("======================================")
    print(f"🎯 Representative Seed = {REPRESENTATIVE_SEED}")
    print("======================================")

    random.seed(REPRESENTATIVE_SEED)
    np.random.seed(REPRESENTATIVE_SEED)
    torch.manual_seed(REPRESENTATIVE_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(REPRESENTATIVE_SEED)

    env = LibraryShelfEnvAG()
    policy = load_policy(env)

    s_rate, s_steps, _ = eval_over_episodes(
        env, policy, n_episodes=N_EPISODES, random_start=False
    )
    r_rate, r_steps, target_success = eval_over_episodes(
        env, policy, n_episodes=N_EPISODES, random_start=True
    )

    print("\n[Success Rate]")
    print(f"  • S-start   : 성공률 {s_rate:.1f}%")
    print(f"  • Random-start : 성공률 {r_rate:.1f}%")

    print("\n[Average Steps]")
    print(f"  • S-start   : {s_steps:.1f} 평균 스텝")
    print(f"  • Random-start : {r_steps:.1f} 평균 스텝")

    print("\n[랜덤 시작 시 타겟별 성공률]")
    for k in env.target_keys:
        print(f"  - {k} : {target_success[k]:.1f}%")

    # 2) seed별 Random-start 성공률 기반 신뢰구간
    print("\n======================================")
    print("📊 Random-start seed 변화 실험 (CI 계산용)")
    print("======================================")

    mean, std, ci_half, rates = eval_random_start_over_seeds()

    print("\n[Random-start 신뢰구간(Seed 기반)]")
    print(f"  • 평균   : {mean:.2f}%")
    print(f"  • 표준 편차 : {std:.2f}")
    print(f"  • 95% CI : {mean:.2f} ± {ci_half:.2f} %")
    print(f"    (seed별 값: {', '.join(f'{r:.1f}' for r in rates)})")


if __name__ == "__main__":
    main()
