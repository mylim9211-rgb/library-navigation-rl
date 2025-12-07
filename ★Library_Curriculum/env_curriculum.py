# env_curriculum.py
import random
import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from matplotlib import colors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------
# 환경: A~F 서가 + 랜덤 시작 옵션 + 장애물 충돌 처리
# ------------------------------------------
class LibraryShelfEnvAG:
    """
    A~F 서가가 있는 도서관 환경.
    state: (x, y) 정규화 좌표 + 타겟 One-hot
    action: 0=위, 1=아래, 2=왼쪽, 3=오른쪽
    """

    def __init__(self):
        self.ascii_map = [
            "#################################################",
            "# S                 B       B                   #",
            "#                                               #",
            "#   AAA   ███   ███   BBB   ███   ███   CCC     #",
            "#   AAA   ███   ███   BBB   ███   ███   CCC     #",
            "#   AAA   ███   ███   BBB   ███   ███   CCC     #",
            "#                                               #",
            "#                   B        B                  #",
            "#                                               #",
            "#                                               #",
            "#   DDD   ███   ███   EEE   ███   ███   FFF     #",
            "#   DDD   ███   ███   EEE   ███   ███   FFF     #",
            "#   DDD   ███   ███   EEE   ███   ███   FFF     #",
            "#                                               #",
            "#                                               #",
            "#################################################",
        ]

        self.height = len(self.ascii_map)
        self.width = len(self.ascii_map[0])

        # 0: 바닥, 1: 벽/장애물, 2: 서가, 3: 벤치
        self.base_map = np.zeros((self.height, self.width), dtype=np.int32)

        self.targets = {}  # "A" -> [(x,y), ...]
        self.target_keys = []  # ["A","B","C","D","E","F"]
        self.start_pos = None

        self._parse_ascii_map()

        # 현재 타겟
        self.current_target_idx = 0
        self.current_target_key = self.target_keys[self.current_target_idx]
        self.goal_poses = []  # 해당 서가 셀들

        # 에피소드 관리
        self.max_steps = 200
        self.steps = 0
        self.pos = None

        # 👉 왕복(stuck) 탐지용: 최근 위치 기록
        self.last_positions = []

        # 액션 개수
        self.n_actions = 4

        # 상태 차원 (더미 reset 한 번 돌려서 계산)
        dummy_state = self.reset()
        self.state_dim = len(dummy_state)

    # -------------------- 맵 파싱 --------------------
    def _parse_ascii_map(self):
        for y, row in enumerate(self.ascii_map):
            for x, ch in enumerate(row):
                if ch == "#":
                    self.base_map[y, x] = 1  # 벽
                elif ch == "█":
                    self.base_map[y, x] = 1  # 장애물(벽 취급)
                elif ch == "S":
                    self.base_map[y, x] = 0
                    self.start_pos = (x, y)
                elif ch == "B":
                    # 주변 체크해서 서가인지 벤치인지 구분
                    left_same = (x > 0 and row[x - 1] == "B")
                    right_same = (x < len(row) - 1 and row[x + 1] == "B")
                    if left_same or right_same:
                        self._register_shelf_cell("B", x, y)
                    else:
                        self.base_map[y, x] = 3  # 벤치
                elif ch in "ACDEF":
                    self._register_shelf_cell(ch, x, y)
                else:
                    self.base_map[y, x] = 0

        self.target_keys = sorted(self.targets.keys())

    def _register_shelf_cell(self, key, x, y):
        if key not in self.targets:
            self.targets[key] = []
        self.targets[key].append((x, y))
        self.base_map[y, x] = 2  # 서가

    # -------------------- 유틸 --------------------
    def _sample_random_start(self):
        # 시작 위치는 오직 '바닥(0)'에서만 가능
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.base_map[y, x] == 0:
                return (x, y)

    def _compute_goal_poses_for_current_target(self):
        # 👉 해당 서가 셀들만 목표로 사용 (단순 버전)
        self.goal_poses = list(self.targets[self.current_target_key])

    def _build_state(self):
        # 좌표 정규화 (0~1)
        x_norm = self.pos[0] / (self.width - 1)
        y_norm = self.pos[1] / (self.height - 1)

        # 타겟 One-hot
        target_oh = np.zeros(len(self.target_keys), dtype=np.float32)
        target_oh[self.current_target_idx] = 1.0

        return np.concatenate([[x_norm, y_norm], target_oh], axis=0)

    def get_min_dist(self):
        """현재 위치와 타겟 서가들 사이의 최소 Manhattan 거리."""
        return min(
            abs(self.pos[0] - gx) + abs(self.pos[1] - gy)
            for gx, gy in self.goal_poses
        )

    # -------------------- Gym 스타일 API --------------------
    def reset(self, target_idx=None, random_start=False):
        if target_idx is None:
            self.current_target_idx = random.randint(0, len(self.target_keys) - 1)
        else:
            self.current_target_idx = target_idx

        self.current_target_key = self.target_keys[self.current_target_idx]
        self._compute_goal_poses_for_current_target()

        if random_start:
            self.pos = self._sample_random_start()
        else:
            self.pos = self.start_pos

        self.steps = 0

        # 👉 왕복 패턴 탐지 초기화
        self.last_positions = [self.pos]

        return self._build_state()

    def step(self, action: int):
        # 0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽
        dx, dy = 0, 0
        if action == 0:
            dy = -1
        elif action == 1:
            dy = 1
        elif action == 2:
            dx = -1
        elif action == 3:
            dx = 1

        x, y = self.pos
        nx = x + dx
        ny = y + dy

        # 이전 거리 (delta 보상을 위해)
        prev_dist = self.get_min_dist()

        # 1. 맵 범위 체크
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            nx, ny = x, y
        else:
            # 2. 장애물 / 서가 충돌 체크
            cell_type = self.base_map[ny, nx]

            if cell_type == 1 or cell_type == 3:
                # 벽/벤치는 절대 못 지나감
                nx, ny = x, y
            elif cell_type == 2:
                # 서가: 내가 찾아야 할 서가면 OK, 아니면 장애물 취급
                if (nx, ny) not in self.goal_poses:
                    nx, ny = x, y
                # 목표 서가인 경우에는 진입 허용
            # 0이면 그냥 이동

        # 위치 갱신
        self.pos = (nx, ny)
        self.steps += 1

        # 새 거리
        new_dist = self.get_min_dist()
        reached = (self.pos in self.goal_poses)

        # 기본 시간 페널티
        reward = -0.01

        # ---------------- delta(거리 차) 보상 ----------------
        delta = prev_dist - new_dist  # 가까워지면 > 0
        if delta > 0:
            reward += 0.3 + delta * 0.05
        elif delta < 0:
            reward += -0.3 + delta * 0.05
        else:
            reward += -0.01  # 제자리면 약한 패널티

        done = False
        info = {"reached_goal": False, "stuck": False}

        # 👉 최근 위치 기록 (왕복 패턴 ABAB 탐지)
        self.last_positions.append(self.pos)

        # [수정된 부분] 🚨 Stuck 감지 로직 완화
        # 4번 반복(A-B-A-B)해도 죽이지 않고(done=False) 감점만 줌
        if len(self.last_positions) >= 4 and not reached:
            a1, a2, a3, a4 = self.last_positions[-4:]
            if a1 == a3 and a2 == a4:
                reward -= 0.5  # 패널티를 조금 더 강화 (-0.3 -> -0.5)
                # done = True  <--- 주석 처리! (이제 안 죽음)
                info["stuck"] = True

        # 목표 도달 우선 처리
        if reached:
            reward += 1.0
            done = True
            info["reached_goal"] = True
        elif self.steps >= self.max_steps and not done:
            done = True

        return self._build_state(), reward, done, info

    # -------------------- 시각화 (원하면 테스트용) --------------------
    def visualize_episode(self, traj, title="Trajectory"):
        fig, ax = plt.subplots(figsize=(12, 5))

        visual = self.base_map.copy()

        # 타겟 서가 강조 (색상값 4)
        key = self.current_target_key
        for tx, ty in self.targets[key]:
            visual[ty, tx] = 4

        cmap = colors.ListedColormap([
            "#e0e0e0",  # 0 바닥
            "#000000",  # 1 벽/장애물
            "#8B4513",  # 2 일반 서가 (갈색)
            "#d17f00",  # 3 벤치 (주황)
            "#4B0082",  # 4 타겟 서가 (보라)
        ])

        ax.imshow(visual, cmap=cmap, origin="upper", vmin=0, vmax=4)

        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]

        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label="Path")
        ax.scatter(xs[0], ys[0], c="green", s=100, label="Start", zorder=5)
        ax.scatter(xs[-1], ys[-1], c="blue", s=100, label="End", zorder=5)

        ax.set_title(title)
        ax.axis("off")
        ax.legend()
        plt.tight_layout()
        plt.show()