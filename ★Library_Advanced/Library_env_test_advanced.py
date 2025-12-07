# test_advanced.py
# ==========================================
# 🚨 흰 화면 방지 코드 (맨 위에 있어야 함)
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

from env_advanced import LibraryShelfEnvAG, DQN, DuelingDQN, device

# 학습 후 생성된 모델 파일 경로 (Double + Dueling 버전 추천)
MODEL_PATH = "library_AG_double_dueling.pt"


def load_policy(env):
    if not os.path.exists(MODEL_PATH):
        print(f"🚨 모델 파일({MODEL_PATH})이 없습니다. 먼저 train_advanced.py를 실행하세요!")
        return None

    ckpt = torch.load(MODEL_PATH, map_location=device)
    use_dueling = ckpt.get("use_dueling", False)
    state_dim = env.state_dim
    n_actions = 4

    if use_dueling:
        policy = DuelingDQN(state_dim, n_actions).to(device)
    else:
        policy = DQN(state_dim, n_actions).to(device)

    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    print(f"📦 모델 로드 완료: {MODEL_PATH} (use_dueling={use_dueling})")
    return policy


def render_step(env, traj, ax):
    """
    env 상태와 현재까지의 traj를 이용해,
    로봇이 서가를 향해 '길을 찾아가는' 모습을 한 프레임 그려줌.
    """
    visual = env.base_map.copy()

    # 타겟 서가 강조 (값 4)
    key = env.current_target_key
    for tx, ty in env.targets[key]:
        visual[ty, tx] = 4

    # 색 설정: 0 바닥, 1 벽, 2 서가, 3 벤치, 4 타겟서가
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

        # 지나온 경로 (얇은 선)
        ax.plot(xs, ys, "-", linewidth=2, color="cyan", label="Path")

        # 시작 지점 (녹색)
        ax.scatter(xs[0], ys[0], c="green", s=80, label="Start", zorder=5)

        # 현재 위치 (파란색 점)
        ax.scatter(xs[-1], ys[-1], c="blue", s=80, label="Robot", zorder=6)

    ax.set_title(f"Target: {env.current_target_key} | Steps: {env.steps}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")


def test(num_episodes=5, random_start=True, sleep_time=0.05):
    env = LibraryShelfEnvAG()
    policy = load_policy(env)
    if policy is None:
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.show(block=False)

    success_count = 0

    start_mode = "랜덤 시작" if random_start else "S에서 시작"
    print(f"\n🎬 테스트 시작! (에피소드 {num_episodes}판, {start_mode})")

    for ep in range(num_episodes):
        # 타겟도 랜덤으로 하나 뽑아서 (A~G)
        target_idx = random.randint(0, len(env.target_keys) - 1)
        state = env.reset(target_idx=target_idx, random_start=random_start)

        traj = [env.pos]
        done = False
        info = {}
        steps = 0

        print(f"\n▶ Episode {ep+1}/{num_episodes} | Target: {env.current_target_key}")

        while not done and steps < env.max_steps:
            # 시각화: 로봇이 한 칸씩 움직이는 모습
            render_step(env, traj, ax)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(sleep_time)

            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = policy(state_t)[0]
            action = int(q.argmax().item())

            state, reward, done, info = env.step(action)
            steps += 1

            if not traj or traj[-1] != env.pos:
                traj.append(env.pos)

        # 마지막 프레임 한 번 더 그리기
        render_step(env, traj, ax)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.2)

        if info.get("reached_goal", False):
            success_count += 1
            print(f"   ✔ 성공! (steps={steps})")
        else:
            print(f"   ✖ 실패.. (steps={steps})")

        # 한 에피소드 끝날 때마다 잠깐 멈춤
        time.sleep(0.5)

    plt.ioff()
    plt.show()

    print("\n=====================================")
    print(f"🎯 총 {num_episodes}판 중 {success_count}판 성공")
    print(f"🔥 성공률: {success_count/num_episodes*100:.1f}%")
    print("=====================================\n")


if __name__ == "__main__":
    # 기본: 랜덤 시작 환경에서 5판 정도 길찾기 시연
    test(num_episodes=5, random_start=True, sleep_time=0.05)

    # S에서만 시작하는 버전도 보고 싶으면:
    # test(num_episodes=5, random_start=False, sleep_time=0.05)
