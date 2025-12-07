# simple_eval_visual.py
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from library_env_random_start import (
    LibraryShelfEnv,
    DQN,
    DuelingDQN,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 🔹 Simple Grid용 Dueling 모델 경로 (너 지금 쓰는 curriculum 파라미터)
MODEL_PATH = "library_shelf_random_start_curriculum.pt"

SEEDS = [1, 42, 2025]
EPISODES_PER_EVAL = 100
N_EVALS_FOR_CI = 5


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
# 2. 평가 함수 (성공률 + 평균 스텝)
# --------------------------------------------------
def evaluate(env, policy, random_start=False, episodes=100):
    successes = 0
    steps_list = []

    for _ in range(episodes):
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# 3. 메인
# --------------------------------------------------
if __name__ == "__main__":
    env, policy = load_policy()

    # ------------------------------
    # (1) S-start vs Random-start 비교
    # ------------------------------
    set_seed(42)
    sr_s, steps_s = evaluate(env, policy, random_start=False, episodes=EPISODES_PER_EVAL)

    set_seed(42)
    sr_r, steps_r = evaluate(env, policy, random_start=True, episodes=EPISODES_PER_EVAL)

    print("\n=== Simple Grid – S-start vs Random-start ===")
    print(f"S-start   : 성공률 = {sr_s:.1f}%, 평균 스텝 = {steps_s}")
    print(f"Random    : 성공률 = {sr_r:.1f}%, 평균 스텝 = {steps_r}")

    # ------------------------------
    # (2) Seed별 Random-start 성능
    # ------------------------------
    seed_rates = []
    print("\n=== Seed별 Random-start 성공률 ===")
    print("Seed | Random-start 성공률(%)")
    for sd in SEEDS:
        set_seed(sd)
        sr, _ = evaluate(env, policy, random_start=True, episodes=EPISODES_PER_EVAL)
        seed_rates.append(sr)
        print(f"{sd:4d} | {sr:7.1f}")

    # ------------------------------
    # (3) Random-start 반복 평가 + 신뢰구간
    # ------------------------------
    random_start_rates = []
    print("\n=== Random-start 반복 평가 (CI 계산용) ===")
    for i in range(N_EVALS_FOR_CI):
        set_seed(1000 + i)
        sr, _ = evaluate(env, policy, random_start=True, episodes=EPISODES_PER_EVAL)
        random_start_rates.append(sr)
        print(f"Eval {i+1}: 성공률 = {sr:.1f}%")

    rates = np.array(random_start_rates)
    mean = rates.mean()
    std = rates.std(ddof=1) if len(rates) > 1 else 0.0
    ci_95 = 1.96 * std

    print("\n📊 Random-start 성능 요약")
    print(f" - 평균 성공률        : {mean:.2f}%")
    print(f" - 표준편차 (std)     : {std:.2f}")
    print(f" - 95% 신뢰구간 (CI)  : {mean:.2f} ± {ci_95:.2f} (%)")

    # ------------------------------
    # (4) 시각화 – S vs Random 성공률 막대그래프
    # ------------------------------
    labels = ["S-start", "Random-start"]
    values = [sr_s, sr_r]

    plt.figure()
    plt.title("Simple Grid – S vs Random 성공률")
    plt.bar(labels, values)
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # (5) 시각화 – Random-start 성공률 Error bar
    # ------------------------------
    plt.figure()
    plt.title("Simple Grid – Random-start 성공률 (평균 ± 95% CI)")
    x = [0]
    plt.errorbar(x, [mean], yerr=[ci_95], fmt='o')
    plt.xlim(-1, 1)
    plt.ylabel("Success Rate (%)")
    plt.xticks([])
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.show()

    print("\n✅ Simple Grid Evaluation Metrics & 시각화 완료")
