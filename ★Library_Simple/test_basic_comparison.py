# test_basic_comparison.py
import os
import random
import numpy as np
import torch
import torch.nn as nn

# 💡 Step 1 환경 (Basic)
from library_env_random_start import LibraryShelfEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 💡 학습된 모델 파일
MODEL_PATH = "library_shelf_random_start_curriculum.pt"


# -------------------------------------------------------------
# 1. 모델 구조 (train 코드와 동일해야 함)
# -------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# 2. 모델 로드
# -------------------------------------------------------------
def load_policy(env):
    if not os.path.exists(MODEL_PATH):
        print(f"🚨 모델 파일({MODEL_PATH})이 없습니다. 먼저 학습을 돌리세요!")
        return None

    ckpt = torch.load(MODEL_PATH, map_location=device)

    # 환경에서 정보 가져오기
    dummy_s = env.reset()
    state_dim = len(dummy_s)
    n_actions = 4

    policy = DQN(state_dim, n_actions).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    print(f"📦 Step 1 모델 로드 완료: {MODEL_PATH}")
    return policy


# -------------------------------------------------------------
# 3. 평가 함수
# -------------------------------------------------------------
def run_test(env, policy, mode_name, random_start, episodes=100):
    print(f"\n🧪 [{mode_name}] 테스트 진행 중 ({episodes}판)...")
    success_count = 0
    total_steps = 0

    for ep in range(episodes):
        # 타겟 랜덤 설정
        target_idx = random.randint(0, len(env.target_keys) - 1)

        # 시작 위치 설정 (핵심!)
        state = env.reset(target_idx=target_idx, random_start=random_start)

        done = False
        steps = 0

        while not done and steps < env.max_steps:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = policy(s_t)[0]
            action = int(q.argmax().item())

            state, reward, done, info = env.step(action)
            steps += 1

            if info.get("reached_goal", False):
                success_count += 1
                total_steps += steps

    success_rate = success_count / episodes * 100
    avg_steps = total_steps / success_count if success_count > 0 else 0

    print(f"   결과: 성공률 {success_rate:.1f}% | 평균 스텝 {avg_steps:.1f}")
    return success_rate, avg_steps


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    env = LibraryShelfEnv()
    policy = load_policy(env)

    if policy:
        print("=" * 50)
        print("📊 Step 1 (Basic) : Vanilla Double DQN 성능 검증")
        print("=" * 50)

        # 1. 고정 위치(S) 출발 테스트
        acc_S, step_S = run_test(env, policy, "S 출발 (Fixed)", random_start=False)

        # 2. 랜덤 위치 출발 테스트
        acc_R, step_R = run_test(env, policy, "랜덤 출발 (Random)", random_start=True)

        print("\n" + "=" * 50)
        print(f"📝 최종 요약")
        print(f"1. S 출발   : {acc_S:.1f}% (난이도 하)")
        print(f"2. 랜덤 출발: {acc_R:.1f}% (난이도 중)")
        print("=" * 50)

        if acc_R > 90:
            print("✅ 결론: Basic 환경은 기본 모델로도 충분히 정복 가능하다!")
        else:
            print("⚠️ 결론: 아직 학습이 더 필요하다.")