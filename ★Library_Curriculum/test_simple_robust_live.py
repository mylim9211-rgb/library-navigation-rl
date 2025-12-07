# test_simple_robust_live.py
# ==========================================
# 🚨 흰 화면 방지 코드
# ==========================================
import matplotlib

try:
    matplotlib.use("TkAgg")
except:
    pass
# ==========================================

import os
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# 💡 환경 파일명 수정 (env_curriculum)
from env_curriculum import LibraryShelfEnvAG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 💡 새로 학습한 모델 파일명
MODEL_PATH = "library_simple_robust.pt"


# ----------------------------------------------------
# 1. 모델 구조 정의 (train_simple_robust.py와 동일해야 함!)
#    (DuelingDQN_Large 대신 StandardDQN 사용)
# ----------------------------------------------------
class StandardDQN(torch.nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------
# 2. 모델 불러오기 (로직 단순화)
# ----------------------------------------------------
def load_policy(env):
    if not os.path.exists(MODEL_PATH):
        print(f"🚨 모델 파일({MODEL_PATH})이 없습니다. train_simple_robust.py를 먼저 실행하세요!")
        return None

    # 모델 껍데기 생성
    state_dim = env.state_dim
    n_actions = env.n_actions
    policy = StandardDQN(state_dim, n_actions).to(device)

    # 가중치 로드
    # (이번 코드는 state_dict를 직접 저장했으므로 바로 로드)
    try:
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        policy.eval()
        print(f"📦 Robust 모델 로드 완료: {MODEL_PATH}")
    except Exception as e:
        print(f"🚨 모델 로드 실패: {e}")
        return None

    return policy


# ----------------------------------------------------
# 3. 시각화 함수 (기존과 동일)
# ----------------------------------------------------
def render_step(env, traj, ax):
    visual = env.base_map.copy()

    # 타겟 서가 강조 (4)
    key = env.current_target_key
    for tx, ty in env.targets[key]:
        visual[ty, tx] = 4

    cmap = colors.ListedColormap([
        "#e0e0e0",  # 0 바닥
        "#000000",  # 1 벽/장애물
        "#8B4513",  # 2 일반 서가
        "#d17f00",  # 3 벤치
        "#4B0082",  # 4 타겟 서가
    ])

    ax.clear()
    ax.imshow(visual, cmap=cmap, origin="upper", vmin=0, vmax=4)

    if len(traj) > 0:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]

        # 경로선
        ax.plot(xs, ys, "-", linewidth=2, color="cyan", label="Path")
        # 시작점
        ax.scatter(xs[0], ys[0], c="green", s=80, label="Start", zorder=5)
        # 현재 로봇
        ax.scatter(xs[-1], ys[-1], c="blue", s=80, label="Robot", zorder=6)

    ax.set_title(f"Target: {env.current_target_key} | Steps: {env.steps}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")


# ----------------------------------------------------
# 4. 테스트 루프
# ----------------------------------------------------
def test_live(num_episodes=5, random_start=True, sleep_time=0.1):
    env = LibraryShelfEnvAG()
    policy = load_policy(env)
    if policy is None:
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.show(block=False)

    success_count = 0
    start_str = "랜덤위치" if random_start else "고정위치(S)"
    print(f"\n🎬 [Simple Robust] 라이브 테스트 시작! ({start_str})")

    for ep in range(num_episodes):
        # 타겟 랜덤 선택
        target_idx = random.randint(0, len(env.target_keys) - 1)

        # Robust 모델은 50:50으로 학습했으므로 둘 다 잘해야 함
        state = env.reset(target_idx=target_idx, random_start=random_start)

        traj = [env.pos]
        done = False
        info = {}
        steps = 0

        print(f"\n▶ Episode {ep + 1}/{num_episodes} | Target: {env.current_target_key} | Start: {env.pos}")

        while not done and steps < env.max_steps:
            render_step(env, traj, ax)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(sleep_time)

            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = policy(s_t)[0]
            action = int(q.argmax().item())

            state, reward, done, info = env.step(action)
            steps += 1

            if not traj or traj[-1] != env.pos:
                traj.append(env.pos)

        # 마지막 장면
        render_step(env, traj, ax)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.5)

        if info.get("reached_goal", False):
            success_count += 1
            print(f"   ✔ 성공! (steps={steps})")
        else:
            print(f"   ✖ 실패.. (steps={steps})")

    plt.ioff()
    plt.show()

    print("\n=====================================")
    print(f"🎯 최종 결과 ({start_str}): {num_episodes}판 중 {success_count}판 성공")
    print(f"🔥 성공률: {success_count / num_episodes * 100:.1f}%")
    print("=====================================\n")


if __name__ == "__main__":
    # 1. 랜덤 시작 테스트 (이게 잘 되어야 Robust 모델임!)
    test_live(num_episodes=5, random_start=True, sleep_time=0.05)

    # 2. 고정 시작 테스트 (이건 당연히 잘해야 함)
    # test_live(num_episodes=3, random_start=False, sleep_time=0.05)