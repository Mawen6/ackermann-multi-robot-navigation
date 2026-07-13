"""
envs/laser_processor.py
──────────────────────────────────────────────────────────────────────
激光点云 → 聚类 → 圆形障碍物 → VO 向量序列

流程:
  1. 180° 激光 N 射线 → 命中点 (x, y) 集合
  2. 相邻点按距离阈值聚类（简单 1D 扫描聚类）
  3. 每簇用最小外接圆近似 → CircleObstacle
  4. compute_all_vo() → (K, 5) VO 向量数组

仿真中激光由物理引擎直接返回距离数组，
部署时对接 ROS sensor_msgs/LaserScan。
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from configs import config as C
from algos.vo_rvo import CircleObstacle, compute_all_vo


# ══════════════════════════════════════════════════════════════════════
# 激光点云生成（仿真内部使用）
# ══════════════════════════════════════════════════════════════════════

def simulate_laser(robot_x: float, robot_y: float, robot_theta: float,
                   obstacles: List[CircleObstacle],
                   arena_size: float,
                   n_beams: int = C.LOW_LASER_BEAMS,
                   max_range: float = C.LOW_LASER_RANGE,
                   rect_walls=None) -> np.ndarray:          # ← Bug1: 加参数
    """向量化射线投射 + 矩形障碍支持。"""
    fov_rad = np.deg2rad(C.LASER_FOV)
    angles  = np.linspace(-fov_rad / 2, fov_rad / 2, n_beams) + robot_theta
    dx = np.cos(angles)
    dy = np.sin(angles)
    ranges = np.full(n_beams, max_range, dtype=np.float32)
    half = arena_size / 2.0

    # ── 圆形障碍物 (向量化) ──────────────────────────────────────────
    if obstacles:
        ox = np.array([o.x - robot_x for o in obstacles], dtype=np.float64)
        oy = np.array([o.y - robot_y for o in obstacles], dtype=np.float64)
        or_ = np.array([o.r         for o in obstacles], dtype=np.float64)
        b_c = -2.0 * (ox[:, None] * dx[None, :] + oy[:, None] * dy[None, :])
        c_c = (ox[:, None]**2 + oy[:, None]**2) - or_[:, None]**2
        disc = b_c**2 - 4 * c_c
        valid = disc >= 0
        t1 = np.where(valid, (-b_c - np.sqrt(np.maximum(disc, 0))) / 2.0, np.inf)
        t1 = np.where(t1 > 1e-6, t1, np.inf)
        min_t = t1.min(axis=0).astype(np.float32)
        ranges = np.minimum(ranges, min_t)

    # ── 矩形障碍物 (用已有的 ray_intersect) ──────────────────────────
    if rect_walls:
        for b in range(n_beams):
            for rect in rect_walls:
                t = rect.ray_intersect(robot_x, robot_y,
                                       float(dx[b]), float(dy[b]),  # ← Bug2/3: 用数组的第b个
                                       max_range)
                if t < ranges[b]:
                    ranges[b] = t

    # ── 四面墙 (解析式) ──────────────────────────────────────────────
    eps = 1e-9
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(dx >  eps, (half  - robot_x) / dx, np.inf)
        hit = np.abs(robot_y + t * dy) <= half
        ranges = np.minimum(ranges, np.where((t > 1e-6) & hit, t, np.inf).astype(np.float32))
        t = np.where(dx < -eps, (-half - robot_x) / dx, np.inf)
        hit = np.abs(robot_y + t * dy) <= half
        ranges = np.minimum(ranges, np.where((t > 1e-6) & hit, t, np.inf).astype(np.float32))
        t = np.where(dy >  eps, (half  - robot_y) / dy, np.inf)
        hit = np.abs(robot_x + t * dx) <= half
        ranges = np.minimum(ranges, np.where((t > 1e-6) & hit, t, np.inf).astype(np.float32))
        t = np.where(dy < -eps, (-half - robot_y) / dy, np.inf)
        hit = np.abs(robot_x + t * dx) <= half
        ranges = np.minimum(ranges, np.where((t > 1e-6) & hit, t, np.inf).astype(np.float32))

    return np.clip(ranges, 0.0, max_range).astype(np.float32)
# ══════════════════════════════════════════════════════════════════════
# 聚类算法
# ══════════════════════════════════════════════════════════════════════

def ranges_to_points(robot_x: float, robot_y: float, robot_theta: float,
                     ranges: np.ndarray,
                     max_range: float) -> np.ndarray:
    """
    将距离数组转换为全局坐标系下的命中点 (N, 2)。
    超出 max_range 的点丢弃。
    """
    n_beams  = len(ranges)
    fov_rad  = np.deg2rad(C.LASER_FOV)
    angles   = np.linspace(-fov_rad / 2, fov_rad / 2, n_beams) + robot_theta

    valid = ranges < max_range * 0.99
    pts   = []
    for i in range(n_beams):
        if valid[i]:
            px = robot_x + ranges[i] * np.cos(angles[i])
            py = robot_y + ranges[i] * np.sin(angles[i])
            pts.append([px, py])

    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)


def cluster_points(points: np.ndarray,
                   dist_thresh: float = C.CLUSTER_DIST_THRESH,
                   min_pts: int = C.MIN_CLUSTER_POINTS) -> List[np.ndarray]:
    """
    简单 1D 扫描聚类（按激光角度顺序，相邻点距离 < dist_thresh 则归同一簇）。

    返回: List[ndarray(M, 2)]，每个元素是一个簇的点集
    """
    if len(points) == 0:
        return []

    clusters = []
    current  = [points[0]]

    for i in range(1, len(points)):
        dist = float(np.linalg.norm(points[i] - points[i-1]))
        if dist < dist_thresh:
            current.append(points[i])
        else:
            if len(current) >= min_pts:
                clusters.append(np.array(current))
            current = [points[i]]

    if len(current) >= min_pts:
        clusters.append(np.array(current))

    return clusters


def cluster_to_circle(cluster: np.ndarray) -> CircleObstacle:
    """
    将点簇近似为最小外接圆。
    使用简单实现：圆心 = 质心，半径 = 最大点到质心距离 + 裕量。
    """
    center = cluster.mean(axis=0)
    radius = float(np.max(np.linalg.norm(cluster - center, axis=1)))
    # 加一个最小半径保证
    radius = max(radius, C.ROBOT_RADIUS * 0.5)
    return CircleObstacle(x=float(center[0]), y=float(center[1]), r=radius)


# ══════════════════════════════════════════════════════════════════════
# 主接口：激光 → VO 向量
# ══════════════════════════════════════════════════════════════════════

class LaserProcessor:
    """
    激光处理器：封装完整的 laser → VO 向量 pipeline。

    仿真中：直接传入 ranges 数组（由 simulate_laser 生成）
    部署中：从 ROS LaserScan message 解析 ranges 后传入
    """

    def __init__(self, n_beams: int = C.LOW_LASER_BEAMS,
                 max_range: float = C.LOW_LASER_RANGE):
        self.n_beams   = n_beams
        self.max_range = max_range

    def process(self,
                robot_x: float, robot_y: float, robot_theta: float,
                ranges: np.ndarray,
                robot_radius: float = C.ROBOT_RADIUS) -> Tuple[np.ndarray, List[CircleObstacle]]:
        """
        输入: 激光距离数组 (n_beams,)
        输出:
            vo_vecs    : (K, 5) 数组，K = MAX_VO_OBSTACLES
            obs_circles: 聚类后的圆形障碍物列表（供可视化/调试用）
        """
        # 1. 距离 → 全局坐标点
        points = ranges_to_points(robot_x, robot_y, robot_theta,
                                  ranges, self.max_range)

        # 2. 聚类
        clusters = cluster_points(points)

        # 3. 每簇 → 圆形障碍
        obs_circles = [cluster_to_circle(c) for c in clusters]

        # 4. 计算 VO 向量
        robot_pos = np.array([robot_x, robot_y], dtype=np.float32)
        vo_vecs   = compute_all_vo(robot_pos, robot_radius, obs_circles)

        return vo_vecs, obs_circles

    def high_level_ranges(self,
                          robot_x: float, robot_y: float, robot_theta: float,
                          obstacles: List[CircleObstacle],
                          arena_size: float) -> np.ndarray:
        """
        生成上层高精度激光数据 (60 beams)，用于建图和通道估计。
        """
        return simulate_laser(
            robot_x, robot_y, robot_theta,
            obstacles, arena_size,
            n_beams=C.HIGH_LASER_BEAMS,
            max_range=C.HIGH_LASER_RANGE,
        )


# ══════════════════════════════════════════════════════════════════════
# 通道宽度估计（上层使用）
# ══════════════════════════════════════════════════════════════════════

def estimate_corridor_width(robot_x: float, robot_y: float, robot_theta: float,
                            goal_x: float, goal_y: float,
                            ranges_high: np.ndarray,
                            sector_half_angle: float = np.deg2rad(30)) -> float:
    """
    估计朝目标方向的通道宽度（m）。

    方法：
      1. 计算 robot → goal 方向角
      2. 在该方向 ±30° 扇区内，取左侧最近距离 + 右侧最近距离
      3. 通道宽 ≈ left_clear + right_clear（简化，不减机器人宽度）

    返回归一化值 ∈ [0, 1]（除以 MAX_NEIGHBORS × ROBOT_RADIUS）
    """
    n     = C.HIGH_LASER_BEAMS
    fov   = np.deg2rad(C.LASER_FOV)
    angles = np.linspace(-fov/2, fov/2, n) + robot_theta

    goal_angle = float(np.arctan2(goal_y - robot_y, goal_x - robot_x))

    left_min  = C.HIGH_LASER_RANGE
    right_min = C.HIGH_LASER_RANGE

    for i, a in enumerate(angles):
        diff = (a - goal_angle + np.pi) % (2 * np.pi) - np.pi
        if abs(diff) > sector_half_angle:
            continue
        r = float(ranges_high[i])
        if diff > 0:
            left_min  = min(left_min, r)
        else:
            right_min = min(right_min, r)

    width = (left_min + right_min) / (2.0 * C.HIGH_LASER_RANGE)
    return float(np.clip(width, 0.0, 1.0))
