# library_env_random_start.py
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ==========================================================
# 1. 도서관 서가 환경 (랜덤 시작점 지원)
# ==========================================================
class LibraryShelfEnv:
    def __init__(self):
        ascii_map = [
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
            "#                                               #",
            "#################################################",
        ]

        self.height = len(ascii_map)
        self.width = len(ascii_map[0])

        # 0: 바닥, 1: 벽, 2: 서가, 3: 벤치
        self.base_map = np.zeros((self.height, self.width), dtype=int)
        self.start = None
        self.targets = {}   # {'A': [...], 'B': [...], 'C': [...]}

        for y, row in enumerate(ascii_map):
            for x, ch in enumerate(row):
                if ch == '#':
                    self.base_map[y, x] = 1

                elif ch == '█':
                    self.base_map[y, x] = 2  # 일반 서가

                elif ch == 'B':
                    # 가운데 3줄(y=3~5)의 B는 B-서가, 나머지는 벤치
                    if 3 <= y <= 5:
                        self.base_map[y, x] = 2
                        self.targets.setdefault('B', []).append((x, y))
                    else:
                        self.base_map[y, x] = 3  # 벤치

                elif ch == 'S':
                    self.start = (x, y)
                    self.base_map[y, x] = 0

                elif ch in ['A', 'C']:
                    self.base_map[y, x] = 2
                    self.targets.setdefault(ch, []).append((x, y))

        self.target_keys = sorted(list(self.targets.keys()))  # ['A','B','C'] 기대
        print("타겟 서가 목록:", self.target_keys)

        self.current_target_idx = 0
        self.pos = self.start
        self.goal_poses = []
        self.max_steps = 300
        self.steps = 0

        # 🔥 랜덤 시작점 후보 (바닥 칸)
        self.free_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.base_map[y, x] == 0
        ]

    # ---------- 유틸 ----------
    def _get_access_points(self, target_cells):
        """서가 주변에서 로봇이 설 수 있는 빈칸 좌표"""
        access_points = []
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for tx, ty in target_cells:
            for dx, dy in dirs:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.base_map[ny, nx] == 0 and (nx, ny) not in access_points:
                        access_points.append((nx, ny))
        return access_points

    def _nearest_target_cell(self):
        """현재 위치에서 가장 가까운 서가 칸 하나 선택"""
        key = self.target_keys[self.current_target_idx]
        cells = self.targets[key]
        ax, ay = self.pos
        tx, ty = min(cells, key=lambda p: abs(p[0] - ax) + abs(p[1] - ay))
        return tx, ty

    def _get_state(self):
        """[agent_x, agent_y, target_x, target_y] (0~1 정규화)"""
        ax, ay = self.pos
        tx, ty = self._nearest_target_cell()
        return np.array([
            ax / self.width,
            ay / self.height,
            tx / self.width,
            ty / self.height
        ], dtype=np.float32)

    # ---------- Gym 스타일 API ----------
    def reset(self, target_idx=None, random_start=False):
        """
        target_idx: None이면 A/B/C 중 랜덤, 아니면 해당 인덱스
        random_start: True면 free_cells 중 랜덤 위치에서 시작
        """
        self.steps = 0

        if random_start and self.free_cells:
            self.pos = random.choice(self.free_cells)
        else:
            self.pos = self.start

        if target_idx is None:
            self.current_target_idx = np.random.randint(0, len(self.target_keys))
        else:
            self.current_target_idx = target_idx

        key = self.target_keys[self.current_target_idx]
        self.goal_poses = self._get_access_points(self.targets[key])
        return self._get_state()

    def step(self, action):
        """0:상, 1:하, 2:좌, 3:우"""
        self.steps += 1
        x, y = self.pos
        tx, ty = self._nearest_target_cell()
        old_dist = abs(x - tx) + abs(y - ty)

        if action == 0:
            ny, nx = y - 1, x
        elif action == 1:
            ny, nx = y + 1, x
        elif action == 2:
            ny, nx = y, x - 1
        else:
            ny, nx = y, x + 1

        reward = -0.02
        done = False
        hit_wall = False
        reached_goal = False

        # 벽/서가/벤치 or 맵 밖
        if (nx < 0 or nx >= self.width or
                ny < 0 or ny >= self.height or
                self.base_map[ny, nx] != 0):
            reward -= 0.3
            hit_wall = True
            # 위치는 그대로 (벽에 박힘)
        else:
            self.pos = (nx, ny)

        # 이동 후 거리
        tx, ty = self._nearest_target_cell()
        new_dist = abs(self.pos[0] - tx) + abs(self.pos[1] - ty)
        reward += 0.01 * (old_dist - new_dist)  # 가까워지면 +, 멀어지면 -

        if self.pos in self.goal_poses:
            reward += 2.0
            done = True
            reached_goal = True

        if self.steps >= self.max_steps:
            done = True

        info = {"hit_wall": hit_wall, "reached_goal": reached_goal}
        return self._get_state(), reward, done, info


# ==========================================================
# 2. DQN / Dueling DQN 네트워크
# ==========================================================
class DQN(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        f = self.feature(x)
        value = self.value_stream(f)
        adv = self.adv_stream(f)
        adv_mean = adv.mean(dim=1, keepdim=True)
        return value + adv - adv_mean
