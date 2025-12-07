import random
import torch
import matplotlib.pyplot as plt
from matplotlib import colors

from library_env_random_start import (
    LibraryShelfEnv,
    DQN,
    DuelingDQN,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 🔹 train_random_start.py에서 저장한 모델 이름 그대로 맞춰줘
MODEL_PATH = "library_shelf_random_start_curriculum.pt"


# --------------------------------------------------
# 1. 모델 로드
# --------------------------------------------------
def load_policy(model_path=MODEL_PATH):
    env = LibraryShelfEnv()
    state_dim = len(env.reset())
    n_actions = 4

    ckpt = torch.load(model_path, map_location=device)
    use_dueling = ckpt.get("use_dueling", False)
    random_start_prob = ckpt.get("random_start_prob", None)

    print(f"✅ 체크포인트 로드: {model_path}")
    print(f"   - dueling            : {use_dueling}")
    if random_start_prob is not None:
        print(f"   - random_start_prob  : {random_start_prob}")

    if use_dueling:
        policy = DuelingDQN(state_dim, n_actions).to(device)
    else:
        policy = DQN(state_dim, n_actions).to(device)

    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return env, policy


# --------------------------------------------------
# 2. 경로 시각화 함수
# --------------------------------------------------
def visualize_episode_path(env, traj, title="Trajectory"):
    fig, ax = plt.subplots(figsize=(10, 4))
    visual = env.base_map.copy()

    # 현재 타겟 서가를 보라색으로 강조
    key = env.target_keys[env.current_target_idx]
    for tx, ty in env.targets[key]:
        visual[ty, tx] = 4

    cmap = colors.ListedColormap([
        "#e0e0e0",  # 0: 바닥
        "#000000",  # 1: 벽
        "#8B4513",  # 2: 서가
        "#d17f00",  # 3: 벤치
        "#4B0082",  # 4: 타겟 서가
    ])

    ax.imshow(visual, cmap=cmap, origin="upper", vmin=0, vmax=4)

    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    ax.plot(xs, ys, marker="o", linewidth=2, markersize=4)
    ax.scatter(xs[0], ys[0], c="green", s=80, label="Start")
    ax.scatter(xs[-1], ys[-1], c="blue", s=80, label="End")

    ax.set_title(title)
    ax.axis("off")
    ax.legend()
    plt.show()


# --------------------------------------------------
# 3. 인터랙티브 테스트
# --------------------------------------------------
def run_interactive_test(env, policy):
    while True:
        print("\n=== 테스트 모드 선택 ===")
        print(" 1) S에서 시작해서 서가로 가기")
        print(" 2) 랜덤 위치에서 시작해서 서가로 가기")
        print(" Q) 종료")
        mode = input("선택: ").strip().upper()

        if mode in ["Q", "QUIT", "EXIT"]:
            break
        if mode not in ["1", "2"]:
            print("잘못된 입력입니다.")
            continue

        # 타겟 서가 선택
        choices_str = ",".join(env.target_keys)
        shelf = input(f"타겟 서가 선택 ({choices_str}, R=랜덤): ").strip().upper()

        if shelf == "R":
            target_idx = random.randint(0, len(env.target_keys) - 1)
        elif shelf in env.target_keys:
            target_idx = env.target_keys.index(shelf)
        else:
            print("잘못된 서가 입력입니다.")
            continue

        random_start = (mode == "2")
        s = env.reset(target_idx=target_idx, random_start=random_start)

        done = False
        steps = 0
        traj = [env.pos]

        while not done and steps < env.max_steps:
            state_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = policy(state_t)[0]
            a = int(q.argmax().item())

            s, r, done, info = env.step(a)
            steps += 1
            traj.append(env.pos)

        shelf_name = env.target_keys[target_idx]
        print(
            f"\n🎯 타겟 서가: {shelf_name} | "
            f"시작: {'랜덤' if random_start else 'S'} | "
            f"도착여부: {info.get('reached_goal', False)} | "
            f"steps={steps}"
        )

        title = f"Start={'Random' if random_start else 'S'}, Target={shelf_name}"
        visualize_episode_path(env, traj, title=title)


# --------------------------------------------------
# 메인
# --------------------------------------------------
if __name__ == "__main__":
    env, policy = load_policy()
    run_interactive_test(env, policy)
