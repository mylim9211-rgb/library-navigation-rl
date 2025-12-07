# env_advanced.py
import random
import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from matplotlib import colors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------
# DQN / DuelingDQN
# ------------------------------------------
class DQN(nn.Module):
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
        v = self.value_stream(f)  # (B, 1)
        a = self.adv_stream(f)    # (B, n_actions)
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)
        return q


# ------------------------------------------
# 환경: A~G 서가 + 랜덤 시작 + 장애물 충돌 처리 강화
# ------------------------------------------
class LibraryShelfEnvAG:
    """
    A~G 서가가 있는 도서관 환경.
    state: (x, y) 정규화 좌표 + 타겟 One-hot
    action: 상하좌우
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
            "#    B        ████       B       ████           #",
            "#             ████               ████           #",
            "#                                               #",
            "#   DDD   ███   ███   EEE   ███   ███   FFF     #",
            "#   DDD   ███   ███   EEE   ███   ███   FFF     #",
            "#   DDD   ███   ███   EEE   ███   ███   FFF     #",
            "#                     ███                       #",
            "#                     ███   GGGGG               #",
            "#                           GGGGG               #",
            "#################################################",
        ]

        self.height = len(self.ascii_map)
        self.width = len(self.ascii_map[0])

        # 0: 바닥, 1: 벽, 2: 서가, 3: 벤치
        self.base_map = np.zeros((self.height, self.width), dtype=np.int32)

        self.targets = {}       # "A" -> [(x,y), ...]
        self.target_keys = []   # ["A","B","C","D","E","F","G"]
        self.start_pos = None

        self._parse_ascii_map()

        # 현재 타겟
        self.current_target_idx = 0
        self.current_target_key = self.target_keys[self.current_target_idx]
        self.goal_poses = []

        # 에피소드 관리
        self.max_steps = 200
        self.steps = 0
        self.pos = None

        # 상태 차원 자동 계산
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
                elif ch in "ACDEFG":
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
        # 해당 서가의 모든 좌표를 정답으로 설정
        self.goal_poses = self.targets[self.current_target_key]

    def _build_state(self):
        # 좌표 정규화 (0~1)
        x_norm = self.pos[0] / (self.width - 1)
        y_norm = self.pos[1] / (self.height - 1)

        # 타겟 One-hot
        target_oh = np.zeros(len(self.target_keys), dtype=np.float32)
        target_oh[self.current_target_idx] = 1.0

        return np.concatenate([[x_norm, y_norm], target_oh], axis=0)

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

        nx = self.pos[0] + dx
        ny = self.pos[1] + dy

        # 1. 맵 범위 체크
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            nx, ny = self.pos
        else:
            # 2. 장애물 충돌 체크
            cell_type = self.base_map[ny, nx]

            # (1) 벽(1)이나 벤치(3)는 절대 못 지나감
            if cell_type == 1 or cell_type == 3:
                nx, ny = self.pos

            # (2) 서가(2)인 경우
            elif cell_type == 2:
                # 내가 찾아야 할 '목표 서가'라면 진입 허용
                if (nx, ny) in self.goal_poses:
                    pass
                # 남의 서가라면 장애물 취급
                else:
                    nx, ny = self.pos
            # (3) 바닥(0)이면 이동 가능

        self.pos = (nx, ny)
        self.steps += 1

        # 거리 보상 (Manhattan Distance)
        dist = min([
            abs(self.pos[0] - gx) + abs(self.pos[1] - gy)
            for gx, gy in self.goal_poses
        ])
        max_dist = self.width + self.height

        reached = (self.pos in self.goal_poses)

        # 보상 체계: 스텝 페널티 + 거리 페널티
        reward = -0.01 - (dist / max_dist) * 0.1

        done = False
        info = {}

        if reached:
            reward = 1.0
            done = True
            info["reached_goal"] = True
        elif self.steps >= self.max_steps:
            done = True
            info["reached_goal"] = False

        return self._build_state(), reward, done, info

    # -------------------- 시각화 (완결 경로용) --------------------
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
