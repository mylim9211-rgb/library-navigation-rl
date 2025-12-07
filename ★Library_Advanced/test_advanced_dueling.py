import os
import random
import numpy as np
import torch
import torch.nn as nn

# 💡 환경: 최종 보스 'Advanced'
from env_advanced import LibraryShelfEnvAG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 💡 모델: Double Dueling DQN (최종 학습 파일)
MODEL_PATH = "library_AG_double_dueling.pt"


# -------------------------------------------------------------
# 1. Dueling DQN 구조 (train_advanced.py와 동일)
# -------------------------------------------------------------
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


# -------------------------------------------------------------
# 2. 모델 로드
# -------------------------------------------------------------
def load_policy(env):
    if not os.path.exists(MODEL_PATH):
        print(f"🚨 모델 파일({MODEL_PATH})이 없습니다! train_advanced.py를 먼저 실행하세요.")
        return None

    ckpt = torch.load(MODEL_PATH, map_location=device)

    # 체크포인트 구조 확인
    state_dim = env.state_dim
    # env에 n_actions가 없으면 4로 처리
    n_actions = getattr(env, 'n_actions', 4)

    policy = DuelingDQN(state_dim, n_actions).to(device)

    try:
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            policy.load_state_dict(ckpt['state_dict'])
        else:
            policy.load_state_dict(ckpt)

        policy.eval()
        print(f"📦 Final Model 로드 완료: {MODEL_PATH}")
    except Exception as e:
        print(f"🚨 모델 로드 실패: {e}")
        return None

    return policy


# -------------------------------------------------------------
# 3. 상세 평가 함수 (S 시작 / 랜덤 시작 분리)
# -------------------------------------------------------------
def eval_mode(env, policy, num_episodes=200, random_start=True):
    """
    random_start=True  : 매 에피소드 랜덤 위치 + 랜덤 타겟
    random_start=False : 항상 S에서 시작 + 랜덤 타겟
    """
    mode_name = "랜덤 시작" if random_start else "S에서 시작"
    print(f"\n==================================")
    print(f"🎬 [Final Model | {mode_name}] 성능 평가 ({num_episodes}회)")
    print(f"==================================")

    target_keys = env.target_keys

    # 타겟별 통계 저장소
    per_target = {
        k: {
            "total": 0, "success": 0, "steps": 0, "success_steps": 0, "fail_steps": 0,
            "wall_hit_episodes": 0, "stuck_episodes": 0, "timeout_episodes": 0,
        }
        for k in target_keys
    }

    # 전체 통계 변수
    total_success = 0
    total_steps = 0
    total_success_steps = 0
    total_fail_steps = 0

    wall_hit_total = 0
    stuck_total = 0
    timeout_total = 0

    for ep in range(num_episodes):
        # 타겟 랜덤
        ti = random.randint(0, len(target_keys) - 1)
        target_key = target_keys[ti]

        # 시작 위치 설정
        state = env.reset(target_idx=ti, random_start=random_start)

        done = False
        info = {}
        steps = 0

        # 에피소드 내 상태 추적
        hit_wall_ep = False
        stuck_ep = False
        traj = [env.pos]  # 경로 기록 (Stuck 판정용)

        while not done and steps < env.max_steps:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = int(policy(state_t).argmax().item())

            state, reward, done, info = env.step(a)
            steps += 1
            traj.append(env.pos)

            # env_advanced.py의 실제 리턴값과 매칭 (없을 경우를 대비해 수동 체크)
            # 만약 env에서 info에 hit_wall을 안 주면, 좌표 변화 없음으로 체크 가능
            if info.get("hit_wall", False) or (len(traj) > 1 and traj[-1] == traj[-2]):
                hit_wall_ep = True  # 제자리 걸음이면 벽 충돌로 간주

        # 종료 후 분석
        reached = info.get("reached_goal", False)

        # Stuck 판정 (4번 왕복)
        if len(traj) > 4:
            p1, p2, p3, p4 = traj[-4:]
            if p1 == p3 and p2 == p4 and not reached:
                stuck_ep = True

        # Timeout 판정
        is_timeout = False
        if (not reached) and (steps >= env.max_steps):
            is_timeout = True

        # ----- 통계 집계 -----
        total_steps += steps
        if reached:
            total_success += 1
            total_success_steps += steps
        else:
            total_fail_steps += steps

        if hit_wall_ep: wall_hit_total += 1
        if stuck_ep: stuck_total += 1
        if is_timeout: timeout_total += 1

        # 타겟별 집계
        tg = per_target[target_key]
        tg["total"] += 1
        tg["steps"] += steps
        if reached:
            tg["success"] += 1
            tg["success_steps"] += steps
        else:
            tg["fail_steps"] += steps
        if hit_wall_ep: tg["wall_hit_episodes"] += 1
        if stuck_ep: tg["stuck_episodes"] += 1
        if is_timeout: tg["timeout_episodes"] += 1

        # 진행 상황 출력
        if (ep + 1) % 50 == 0:
            print(f"   -> {ep + 1}/{num_episodes} 완료 (현재 성공률: {total_success / (ep + 1) * 100:.1f}%)")

    # ----- 결과 요약 출력 -----
    success_rate = total_success / num_episodes * 100.0
    avg_steps_all = total_steps / num_episodes if num_episodes > 0 else 0.0
    avg_steps_success = total_success_steps / total_success if total_success > 0 else 0.0

    wall_rate = wall_hit_total / num_episodes * 100.0
    stuck_rate = stuck_total / num_episodes * 100.0
    timeout_rate = timeout_total / num_episodes * 100.0

    print(f"\n🏆 총 성공률: {total_success}/{num_episodes} ({success_rate:.1f}%)")
    print(f"📏 평균 스텝 (성공 시): {avg_steps_success:.1f}")
    print(f"🧱 Wall-hit Rate: {wall_rate:.1f}%")
    print(f"🔁 Stuck Rate: {stuck_rate:.1f}%")
    print(f"⏱ Timeout Rate: {timeout_rate:.1f}%")

    print("\n🔎 타겟별 상세 통계:")
    for k in target_keys:
        tg = per_target[k]
        tot = tg["total"]
        suc = tg["success"]
        rate = suc / tot * 100.0 if tot > 0 else 0.0
        avg_s = tg["steps"] / tot if tot > 0 else 0.0
        print(f"  - Target {k}: {suc}/{tot} ({rate:.1f}%) | AvgStep: {avg_s:.1f} | Stuck: {tg['stuck_episodes']}")

    print("==================================\n")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    # 재현성 시드
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = LibraryShelfEnvAG()
    policy = load_policy(env)

    if policy:
        # 1. 고정 출발 (S)
        eval_mode(env, policy, num_episodes=200, random_start=False)

        # 2. 랜덤 출발 (Random)
        eval_mode(env, policy, num_episodes=200, random_start=True)


if __name__ == "__main__":
    main()