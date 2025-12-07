# test_simple_live.py
# ==========================================
# Simple 환경에서 학습된 에이전트가
# 서가 사이를 이동해 타겟을 찾는 모습을
# 실시간으로 보여주는 라이브 데모 코드
# ==========================================

import matplotlib

# 🚨 흰 화면 방지 (TkAgg 안 되면 그냥 넘어감)
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import os
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# 한글 깨짐 방지 (윈도우 기준)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 💡 simple 환경 & 네트워크
from library_env_random_start import LibraryShelfEnv, DQN, DuelingDQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 💡 simple 환경에서 학습한 모델 파일
MODEL_PATH = "library_shelf_random_start_curriculum.pt"


# ----------------------------------------------------
# 1. 모델 로드 (DQN / DuelingDQN 자동 선택)
# ----------------------------------------------------
def load_policy(env):
    if not os.path.exists(MODEL_PATH):
        print(f"🚨 모델 파일({MODEL_PATH})이 없습니다. train_random_start.py를 먼저 실행하세요!")
        return None

    # 상태 차원 / 행동 수
    state_dim = len(env.reset())
    n_actions = 4  # 상/하/좌/우 고정

    ckpt = torch.load(MODEL_PATH, map_location=device)

    # {"state_dict": ..., "use_dueling": bool, ...} 또는 state_dict 자체
    if isinstance(ckpt, dict):
        use_dueling = ckpt.get("use_dueling", False)
        state_dict = ckpt.get("state_dict", ckpt)
    else:
        use_dueling = False
        state_dict = ckpt

    if use_dueling:
        policy = DuelingDQN(state_dim, n_actions).to(device)
    else:
        policy = DQN(state_dim, n_actions).to(device)

    try:
        policy.load_state_dict(state_dict)
        policy.eval()
        print(f"📦 Simple 모델 로드 완료: {MODEL_PATH}")
        print(f"   - use_dueling : {use_dueling}")
    except Exception as e:
        print(f"🚨 state_dict 로드 실패: {e}")
        return None

    return policy


# ----------------------------------------------------
# 2. 한 스텝씩 그리기 (타겟 = 보라색)
# ----------------------------------------------------
def render_step(env, traj, ax):
    # 기본 맵 복사
    visual = env.base_map.copy()

    # 🔥 현재 타겟(A/B/C) 서가를 보라색(4)으로 강조
    if hasattr(env, "target_keys") and hasattr(env, "targets"):
        key = env.target_keys[env.current_target_idx]
        for tx, ty in env.targets[key]:
            visual[ty, tx] = 4

    cmap = colors.ListedColormap(
        [
            "#e0e0e0",  # 0 바닥
            "#000000",  # 1 벽/장애물
            "#8B4513",  # 2 일반 서가
            "#d17f00",  # 3 벤치/기타
            "#4B0082",  # 4 타겟 서가 (보라색)
        ]
    )

    ax.clear()
    ax.imshow(visual, cmap=cmap, origin="upper", vmin=0, vmax=4)

    # 이동 경로 / 시작점 / 로봇 위치
    if traj:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]

        ax.plot(xs, ys, "-", linewidth=2, color="cyan", label="Path")
        ax.scatter(xs[0], ys[0], c="green", s=80, label="Start", zorder=5)
        ax.scatter(xs[-1], ys[-1], c="blue", s=80, label="Robot", zorder=6)

    steps = getattr(env, "steps", len(traj))
    cur_key = env.target_keys[env.current_target_idx]
    title = f"Simple Env | Target: {cur_key} | Steps: {steps}"
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")


# ----------------------------------------------------
# 3. 라이브 테스트 루프
# ----------------------------------------------------
def test_live(num_episodes=5, random_start=True, sleep_time=0.1):
    env = LibraryShelfEnv()
    policy = load_policy(env)
    if policy is None:
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.show(block=False)

    success_count = 0
    start_str = "랜덤 시작" if random_start else "S에서 시작"
    print(f"\n🎬 [Simple Env] 라이브 테스트 시작! ({start_str})")

    for ep in range(num_episodes):
        # 타겟 인덱스 랜덤 선택 (A/B/C)
        target_idx = random.randint(0, len(env.target_keys) - 1)

        # reset 시그니처에 맞게 호출
        state = env.reset(target_idx=target_idx, random_start=random_start)

        traj = [env.pos]
        done = False
        info = {}
        steps = 0

        cur_key = env.target_keys[env.current_target_idx]
        print(
            f"\n▶ Episode {ep + 1}/{num_episodes} | "
            f"Target: {cur_key} | Start: {env.pos}"
        )

        max_steps = env.max_steps

        while not done and steps < max_steps:
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

            if traj[-1] != env.pos:
                traj.append(env.pos)

            if info.get("reached_goal", False):
                break

        # 마지막 프레임 렌더
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
    # 랜덤 시작 기준 데모
    test_live(num_episodes=5, random_start=True, sleep_time=0.05)

    # S에서 시작 데모를 보고 싶으면 아래 주석 해제
    # test_live(num_episodes=3, random_start=False, sleep_time=0.05)
