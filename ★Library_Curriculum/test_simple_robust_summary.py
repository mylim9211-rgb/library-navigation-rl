import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import colors

# 💡 환경 파일명 확인 (env_curriculum)
from env_curriculum import LibraryShelfEnvAG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 💡 새로 학습한 모델 파일
MODEL_PATH = "library_simple_robust.pt"


# -------------------------------------------------------------
# 1. 모델 구조 정의 (train_simple_robust.py와 동일해야 함)
# -------------------------------------------------------------
class StandardDQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# 2. 모델 로드
# -------------------------------------------------------------
def load_policy(env):
    if not os.path.exists(MODEL_PATH):
        print(f"🚨 모델 파일({MODEL_PATH})이 없습니다. train_simple_robust.py를 먼저 실행하세요!")
        return None

    state_dim = env.state_dim
    n_actions = env.n_actions

    policy = StandardDQN(state_dim, n_actions).to(device)

    try:
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        policy.eval()
        print(f"📦 Robust 모델 로드 완료: {MODEL_PATH}")
    except Exception as e:
        print(f"🚨 모델 로드 에러: {e}")
        return None

    return policy


# -------------------------------------------------------------
# 3. Trajectory 시각화 (PNG 저장용)
# -------------------------------------------------------------
def visualize_episode_path(env, traj, title, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 3))
    visual = env.base_map.copy()

    # 타겟 강조
    key = env.current_target_key
    for tx, ty in env.targets[key]:
        visual[ty, tx] = 4

    cmap = colors.ListedColormap([
        "#e0e0e0",  # 바닥
        "#000000",  # 벽
        "#8B4513",  # 서가
        "#d17f00",  # 벤치
        "#4B0082",  # 타겟
    ])

    ax.imshow(visual, cmap=cmap, origin="upper", vmin=0, vmax=4)

    if traj:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, "-o", color="cyan", markersize=3, linewidth=1.5)
        ax.scatter(xs[0], ys[0], c="green", s=60, label="Start", zorder=5)
        ax.scatter(xs[-1], ys[-1], c="blue", s=60, label="End", zorder=6)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# -------------------------------------------------------------
# 4. 대량 성능 평가 (통계 산출)
# -------------------------------------------------------------
def evaluate_policy(env, policy, num_episodes=200, random_start=True):
    mode = "랜덤 시작" if random_start else "S에서 시작"
    print(f"\n==============================================")
    print(f"🎬 [Robust 통계 평가 | {mode}] ({num_episodes}회)")
    print(f"==============================================")

    target_keys = env.target_keys
    stats = {k: {"total": 0, "success": 0} for k in target_keys}

    total_success = 0
    total_steps = 0
    success_steps = 0

    stuck_ep = 0

    for _ in range(num_episodes):
        ti = random.randint(0, len(target_keys) - 1)
        key = target_keys[ti]

        state = env.reset(target_idx=ti, random_start=random_start)

        steps = 0
        reached = False
        stuck = False
        traj = [env.pos]

        while steps < env.max_steps:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = int(policy(s_t)[0].argmax())

            state, reward, done, info = env.step(a)
            steps += 1
            traj.append(env.pos)

            if info.get("reached_goal", False):
                reached = True
                break

            # env_curriculum.py에서 done=True를 풀었으므로,
            # 여기서는 통계용으로만 Stuck 여부를 체크 (4번 왕복)
            if len(traj) > 4:
                p1, p2, p3, p4 = traj[-4:]
                if p1 == p3 and p2 == p4:
                    stuck = True
                    # 통계에서는 Stuck을 실패로 칠지 말지 결정 가능
                    # 여기선 그냥 "Stuck 발생" 카운트만 하고 계속 진행

        # 결과 집계
        if reached:
            total_success += 1
            success_steps += steps
            stats[key]["success"] += 1

        if stuck:
            stuck_ep += 1

        total_steps += steps
        stats[key]["total"] += 1

    # 지표 계산
    succ_rate = total_success / num_episodes * 100.0
    stuck_rate = stuck_ep / num_episodes * 100.0
    avg_succ_steps = success_steps / total_success if total_success > 0 else 0.0

    print(f"🎯 성공률: {total_success}/{num_episodes} ({succ_rate:.1f}%)")
    print(f"🔁 Stuck 발생률: {stuck_rate:.1f}%")
    print(f"📏 성공 시 평균 스텝: {avg_succ_steps:.1f}")

    return {
        "succ_rate": succ_rate,
        "stuck_rate": stuck_rate,
        "per_target": stats
    }


# -------------------------------------------------------------
# 5. 그래프 그리기
# -------------------------------------------------------------
def make_graphs(res_S, res_R, env, save_dir="robust_results"):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 폰트
    plt.rcParams["axes.unicode_minus"] = False

    # 1) 성공률 비교
    plt.figure(figsize=(6, 4))
    x = ["S 시작 (Fixed)", "랜덤 시작 (Random)"]
    y = [res_S["succ_rate"], res_R["succ_rate"]]
    plt.bar(x, y, color=['skyblue', 'salmon'])
    plt.ylim(0, 110)
    plt.title("Simple Robust 모델 - 일반화 성능 비교")
    plt.ylabel("Success Rate (%)")

    # 값 표시
    for i, v in enumerate(y):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.savefig(os.path.join(save_dir, "success_rate.png"), dpi=150)
    plt.close()

    # 2) 타겟별 성공률 (랜덤 기준)
    targets = env.target_keys
    rates = []
    for k in targets:
        info = res_R["per_target"][k]
        rate = info["success"] / info["total"] * 100.0 if info["total"] > 0 else 0.0
        rates.append(rate)

    plt.figure(figsize=(7, 4))
    plt.bar(targets, rates, color='mediumpurple')
    plt.ylim(0, 110)
    plt.title("랜덤 시작 시 타겟별 성공률")
    plt.ylabel("Success Rate (%)")
    plt.savefig(os.path.join(save_dir, "target_breakdown.png"), dpi=150)
    plt.close()


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    # 시드 고정 (재현성)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = LibraryShelfEnvAG()
    policy = load_policy(env)
    if policy is None:
        return

    # 1. 고정 위치(S) 평가 - 200회
    res_S = evaluate_policy(env, policy, num_episodes=200, random_start=False)

    # 2. 랜덤 위치 평가 - 200회
    res_R = evaluate_policy(env, policy, num_episodes=200, random_start=True)

    # 3. 그래프 저장
    make_graphs(res_S, res_R, env, save_dir="../★★Library_curriculum/robust_results")

    # 4. 샘플 경로 저장 (랜덤 1개)
    #    (보고서에 "이런 식으로 찾았다" 보여주기용)
    state = env.reset(random_start=True)
    traj = [env.pos]
    done = False
    while not done and len(traj) < 100:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(policy(s_t)[0].argmax())
        state, _, done, _ = env.step(a)
        traj.append(env.pos)

    visualize_episode_path(env, traj, "Sample Path (Random Start)",
                           "../★★Library_curriculum/robust_results/sample_path.png")

    print("\n✅ 모든 통계 및 그래프 저장 완료: robust_results 폴더 확인!")


if __name__ == "__main__":
    main()