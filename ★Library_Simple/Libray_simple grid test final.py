# test_compare_simple.py
import random
import numpy as np
import torch

from library_env_random_start import LibraryShelfEnv, DQN, DuelingDQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 🔥 테스트할 모델 파일명 입력
# --------------------------
MODEL_PATH = "library_shelf_random_start_curriculum.pt"
# MODEL_PATH = "library_shelf_random_start_dueling.pt"


# --------------------------------------------------
# 1. 모델 로드
# --------------------------------------------------
def load_policy(model_path):
    env = LibraryShelfEnv()
    state_dim = len(env.reset())
    n_actions = 4

    ckpt = torch.load(model_path, map_location=device)
    use_dueling = ckpt.get("use_dueling", False)

    print(f"\n📦 모델 로드: {model_path}")
    print(f"   - dueling : {use_dueling}")

    if use_dueling:
        policy = DuelingDQN(state_dim, n_actions).to(device)
    else:
        policy = DQN(state_dim, n_actions).to(device)

    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return env, policy


# --------------------------------------------------
# 2. 테스트 함수 (S 시작 / 랜덤 시작)
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

    success_rate = successes / episodes * 100
    avg_steps = np.mean(steps_list) if steps_list else None
    return success_rate, avg_steps


# --------------------------------------------------
# 3. 메인 실행
# --------------------------------------------------
if __name__ == "__main__":
    env, policy = load_policy(MODEL_PATH)

    print("\n==================================================")
    print("🧪 Step 1: S에서 시작 테스트 (100판)")
    sr_s, steps_s = evaluate(env, policy, random_start=False)
    print(f"   ✔ 성공률: {sr_s:.1f}% | 평균 스텝: {steps_s}")

    print("--------------------------------------------------")
    print("🧪 Step 2: 랜덤 위치에서 시작 테스트 (100판)")
    sr_r, steps_r = evaluate(env, policy, random_start=True)
    print(f"   ✔ 성공률: {sr_r:.1f}% | 평균 스텝: {steps_r}")

    print("==================================================")
    print("📝 최종 비교 결과")
    print(f" 1) S 고정 출발   : {sr_s:.1f}% (avg steps={steps_s})")
    print(f" 2) 랜덤 출발     : {sr_r:.1f}% (avg steps={steps_r})")
    print("==================================================")
