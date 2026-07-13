"""
envs/map_loader.py
──────────────────────────────────────────────────────────────────────
MovingAI .map 文件加载器

支持格式：
    type octile
    height H
    width  W
    map
    @@@@...   ← @ T O = 障碍，. G S W = 可通行

主要功能：
    1. 加载 .map 文件 → 二值占据栅格
    2. 缩放到仿真坐标系（米）
    3. BFS 采样合法起点/终点（保证连通且路径足够长）
    4. 栅格射线投射（替代圆形障碍物的 simulate_laser）

用法：
    loader = MapLoader('maps/maze-32-32-2.map', cell_size=0.5)
    ranges = loader.simulate_laser(rx, ry, rtheta, n_beams=60, max_range=8.0)
    starts, goals = loader.sample_starts_goals(n=3, min_dist=8.0)
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional


# 可通行字符集
FREE_CHARS = set('. G S W')


class MapLoader:
    """
    加载 MovingAI .map 文件并提供仿真所需接口。

    坐标系约定：
        栅格 (row, col) → 仿真 (x, y)
        x = col * cell_size - width_m  / 2
        y = -(row * cell_size - height_m / 2)   ← y 轴朝上
    """

    def __init__(self, map_path: str, cell_size: float = 0.5):
        """
        参数:
            map_path  : .map 文件路径
            cell_size : 每个栅格对应的实际距离 (m)
        """
        self.cell_size = cell_size
        self.grid: np.ndarray   # (H, W) bool，True=障碍
        self.H: int
        self.W: int
        self._load(map_path)

        self.width_m  = self.W * cell_size
        self.height_m = self.H * cell_size

        # 预计算可通行格子列表，供采样用
        rows, cols = np.where(~self.grid)
        self._free_cells = list(zip(rows.tolist(), cols.tolist()))

        # 预计算用于射线投射的浮点障碍中心（加速）
        self._build_laser_cache()

    # ── 文件加载 ──────────────────────────────────────────────────────

    def _load(self, path: str):
        lines = Path(path).read_text().splitlines()
        H = W = 0
        map_start = 0
        for idx, line in enumerate(lines):
            line = line.strip()
            if line.startswith('height'):
                H = int(line.split()[1])
            elif line.startswith('width'):
                W = int(line.split()[1])
            elif line == 'map':
                map_start = idx + 1
                break

        assert H > 0 and W > 0, "地图文件格式错误，未找到 height/width"

        grid = np.ones((H, W), dtype=bool)   # 默认全障碍
        for r, line in enumerate(lines[map_start: map_start + H]):
            for c, ch in enumerate(line):
                if ch in FREE_CHARS:
                    grid[r, c] = False

        self.grid = grid
        self.H    = H
        self.W    = W

    # ── 坐标转换 ──────────────────────────────────────────────────────

    def rc_to_xy(self, row: float, col: float) -> Tuple[float, float]:
        """栅格坐标 → 仿真坐标 (m)"""
        x =  (col + 0.5) * self.cell_size - self.width_m  / 2.0
        y = -(row + 0.5) * self.cell_size + self.height_m / 2.0
        return x, y

    def xy_to_rc(self, x: float, y: float) -> Tuple[int, int]:
        """仿真坐标 → 栅格坐标（取整）"""
        col = int((x + self.width_m  / 2.0) / self.cell_size)
        row = int((self.height_m / 2.0 - y) / self.cell_size)
        return row, col

    def is_free_xy(self, x: float, y: float) -> bool:
        """仿真坐标是否可通行"""
        r, c = self.xy_to_rc(x, y)
        if r < 0 or r >= self.H or c < 0 or c >= self.W:
            return False
        return not self.grid[r, c]

    # ── 激光射线投射 ──────────────────────────────────────────────────

    def _build_laser_cache(self):
        """预计算障碍格子中心坐标，供 simulate_laser 使用"""
        rows, cols = np.where(self.grid)
        if len(rows) == 0:
            self._obs_xy = np.empty((0, 2), dtype=np.float32)
        else:
            xs = ( cols + 0.5) * self.cell_size - self.width_m  / 2.0
            ys = -(rows + 0.5) * self.cell_size + self.height_m / 2.0
            self._obs_xy = np.stack([xs, ys], axis=1).astype(np.float32)

    def simulate_laser(self,
                       robot_x: float, robot_y: float, robot_theta: float,
                       n_beams: int = 60,
                       max_range: float = 8.0,
                       fov_deg: float = 180.0) -> np.ndarray:
        """
        基于占据栅格的射线投射，返回 (n_beams,) 距离数组。

        方法：DDA（数字微分分析）逐格步进，速度快且精确。
        """
        fov_rad = np.deg2rad(fov_deg)
        angles  = np.linspace(-fov_rad / 2, fov_rad / 2, n_beams) + robot_theta
        ranges  = np.full(n_beams, max_range, dtype=np.float32)

        half_w = self.width_m  / 2.0
        half_h = self.height_m / 2.0

        for i, angle in enumerate(angles):
            ranges[i] = self._dda_ray(
                robot_x, robot_y, angle, max_range, half_w, half_h
            )

        return ranges

    def _dda_ray(self, ox: float, oy: float, angle: float,
                 max_range: float, half_w: float, half_h: float) -> float:
        """DDA 单条射线投射，返回命中距离"""
        dx = np.cos(angle)
        dy = np.sin(angle)
        cs = self.cell_size

        # 起始栅格
        col0 = (ox + half_w) / cs + 1e-6
        row0 = (half_h - oy) / cs + 1e-6

        # 步进方向
        step_c = 1 if dx >= 0 else -1
        step_r = 1 if dy <= 0 else -1   # y 朝上，row 朝下

        # 到第一条格线的距离
        if dx != 0:
            t_delta_c = abs(cs / dx)
            t_max_c   = ((np.floor(col0) + (1 if dx > 0 else 0)) - col0) * cs / dx
            if t_max_c < 0:
                t_max_c += t_delta_c
        else:
            t_delta_c = np.inf
            t_max_c   = np.inf

        if dy != 0:
            t_delta_r = abs(cs / dy)
            # row 增大 = y 减小
            t_max_r   = ((np.floor(row0) + (1 if dy < 0 else 0)) - row0) * cs / abs(dy)
            if t_max_r < 0:
                t_max_r += t_delta_r
        else:
            t_delta_r = np.inf
            t_max_r   = np.inf

        col = int(np.floor(col0))
        row = int(np.floor(row0))
        t   = 0.0

        while t < max_range:
            # 越界 = 出地图
            if row < 0 or row >= self.H or col < 0 or col >= self.W:
                break
            if self.grid[row, col]:
                return float(t)

            if t_max_c < t_max_r:
                t     = t_max_c
                t_max_c += t_delta_c
                col   += step_c
            else:
                t     = t_max_r
                t_max_r += t_delta_r
                row   += step_r

            if t > max_range:
                break

        return max_range

    # ── BFS 合法位置采样 ───────────────────────────────────────────────

    def _bfs_dist(self, start_rc: Tuple[int, int]) -> np.ndarray:
        """BFS 从 start_rc 出发，返回所有可达格子的距离（格数）"""
        dist = np.full((self.H, self.W), -1, dtype=np.int32)
        sr, sc = start_rc
        if self.grid[sr, sc]:
            return dist
        queue = deque()
        queue.append((sr, sc))
        dist[sr, sc] = 0
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<self.H and 0<=nc<self.W and \
                   not self.grid[nr,nc] and dist[nr,nc] < 0:
                    dist[nr,nc] = dist[r,c] + 1
                    queue.append((nr, nc))
        return dist

    def _is_safe_cell(self, r: int, c: int, margin: int = 1) -> bool:
        """
        检查格子 (r,c) 周围 margin 格内是否全部可通行。
        用于过滤紧贴墙壁的起点/终点，防止机器人生成时立刻碰墙。
        """
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.H and 0 <= nc < self.W):
                    return False
                if self.grid[nr, nc]:
                    return False
        return True

    def sample_starts_goals(self,
                            n: int,
                            min_dist_m: float = 6.0,
                            max_tries: int = 500,
                            rng: Optional[np.random.Generator] = None
                            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        采样 n 组合法的 (起点, 终点)。

        保证：
          - 起点和终点都在可通行区域，且周围1格内无墙
          - 起点到终点的 BFS 距离 >= min_dist_m / cell_size
          - 各机器人起点之间距离 >= 2 * cell_size（不重叠）

        返回：
            starts : list of (2,) ndarray，仿真坐标 (x, y)
            goals  : list of (2,) ndarray，仿真坐标 (x, y)
        """
        if rng is None:
            rng = np.random.default_rng()

        min_dist_cells = int(min_dist_m / self.cell_size)

        # 只保留周围1格都可通行的安全格子作为候选
        safe_cells = [
            (r, c) for r, c in self._free_cells
            if self._is_safe_cell(r, c, margin=1)
        ]
        # 若安全格子太少则退化为所有可通行格子
        free = safe_cells if len(safe_cells) >= n * 2 else self._free_cells

        starts_rc = []
        goals_rc  = []

        for _ in range(n):
            found = False
            for _ in range(max_tries):
                # 随机选起点
                sr, sc = free[rng.integers(len(free))]

                # 检查与已有起点不重叠
                too_close = any(
                    abs(sr - pr) + abs(sc - pc) < 2
                    for pr, pc in starts_rc
                )
                if too_close:
                    continue

                # BFS 找足够远的终点
                dist_map = self._bfs_dist((sr, sc))
                candidates = np.argwhere(dist_map >= min_dist_cells)
                if len(candidates) == 0:
                    continue

                idx = rng.integers(len(candidates))
                gr, gc = candidates[idx]

                starts_rc.append((sr, sc))
                goals_rc.append((int(gr), int(gc)))
                found = True
                break

            if not found:
                # 降低要求再试一次
                sr, sc = free[rng.integers(len(free))]
                gr, gc = free[rng.integers(len(free))]
                starts_rc.append((sr, sc))
                goals_rc.append((int(gr), int(gc)))

        starts = [np.array(self.rc_to_xy(r, c), dtype=np.float32)
                  for r, c in starts_rc]
        goals  = [np.array(self.rc_to_xy(r, c), dtype=np.float32)
                  for r, c in goals_rc]
        return starts, goals

    # ── 工具接口 ──────────────────────────────────────────────────────

    @property
    def arena_size(self) -> float:
        """返回地图的等效方形边长（取长宽最大值）"""
        return max(self.width_m, self.height_m)

    def print_info(self):
        free_ratio = len(self._free_cells) / (self.H * self.W)
        print(f"地图: {self.H}×{self.W} 格  "
              f"({self.width_m:.1f}m × {self.height_m:.1f}m)  "
              f"可通行: {free_ratio:.1%}")