"""
envs/low_env.py
──────────────────────────────────────────────────────────────────────
下层 TD3 环境（10 Hz）

职责:
  - 管理 N 台阿克曼机器人的物理仿真
  - 每步：接收 (v_cmd, δ_cmd) → 更新运动学 → 激光 → VO/RVO → reward
  - 提供 Gym 风格接口（reset/step）

新增（窄道路权机制）:
  - 激光检测窄道（左右扇区自由空间之和 < 阈值 → 窄道）
  - entry_depth: 进入窄道的深度（先进者更深）
  - 路权奖励: 两车都在窄道时，进得深(先进)的鼓励前进，浅(后进)的鼓励后退

下层观测（per robot）:
  laser(36) + rvo(M,7) + wp(2) + vel(4) + goal(2)
下层动作:
  (v_cmd, δ_cmd) ∈ [-1, 1]²（per robot）
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from envs.ackermann_robot import AckermannRobot
from envs.comm_interface import SimCommInterface
from algos.vo_rvo import (CircleObstacle, compute_all_vo,
                           compute_all_rvo, rvo_reward)
from envs.rect_obstacle import RectObstacle
from envs.laser_processor import simulate_laser
from configs import config as C
from envs.frontier_module import LowLevelFrontierModule, SLAMFrontierModule
from envs.occupancy_map import OccupancyMap


# ══════════════════════════════════════════════════════════════════════
# 朝向奖励辅助函数
# ══════════════════════════════════════════════════════════════════════

def _parse_laser_to_vo(
        ranges: np.ndarray,
        robot_x: float, robot_y: float, robot_theta: float,
        robot_radius: float,
        arena_size: float,
) -> tuple:
    """
    把激光扫描数据拆分成：墙的 VO 向量 + 圆形障碍物列表。
    """
    from envs.laser_processor import (ranges_to_points, cluster_points,
                                       cluster_to_circle, simulate_laser)
    from algos.vo_rvo import CircleObstacle

    half      = arena_size / 2.0
    wall_tol  = 0.15   # 命中点距墙面 < 15cm 判定为墙

    fov    = np.deg2rad(C.LASER_FOV)
    angles = np.linspace(-fov / 2, fov / 2, len(ranges)) + robot_theta

    obs_pts = []   # 非墙点

    for k in range(len(ranges)):
        r = float(ranges[k])
        if r >= C.LOW_LASER_RANGE * 0.99:
            continue   # 无命中，跳过

        px = robot_x + r * np.cos(angles[k])
        py = robot_y + r * np.sin(angles[k])

        is_wall = (abs(abs(px) - half) < wall_tol or
                   abs(abs(py) - half) < wall_tol)

        if not is_wall:
            obs_pts.append([px, py])

    obs_circles = []
    if obs_pts:
        pts      = np.array(obs_pts, dtype=np.float32)
        clusters = cluster_points(pts)
        obs_circles = [cluster_to_circle(c) for c in clusters]

    wall_vo = compute_wall_vo_vectors(robot_x, robot_y, robot_radius, arena_size)

    return wall_vo, obs_circles


def _wall_virtual_obstacles(robot_x: float, robot_y: float,
                             arena_size: float) -> List[CircleObstacle]:
    """已废弃，墙壁改用 compute_wall_vo_vectors 直接计算 VO 向量。"""
    return []


def compute_wall_vo_vectors(robot_x: float, robot_y: float,
                             robot_radius: float,
                             arena_size: float) -> np.ndarray:
    """
    直接计算四面墙的 VO 向量，不通过圆形近似。
    返回 (4, 5) 数组，4 面墙各一个向量；超出感知范围的用 padding(ttc=1)。
    """
    half     = arena_size / 2.0
    result   = np.zeros((4, C.VO_DIM), dtype=np.float32)
    result[:, 4] = 1.0   # 默认 padding

    walls = [
        (half,     robot_y, +1.0,  0.0),
        (-half,    robot_y, -1.0,  0.0),
        (robot_x,  half,    0.0, +1.0),
        (robot_x, -half,    0.0, -1.0),
    ]

    for k, (wx, wy, cd, sd) in enumerate(walls):
        dist     = float(np.sqrt((wx - robot_x)**2 + (wy - robot_y)**2))
        surf     = max(dist - robot_radius, 0.0)

        if surf > C.LOW_LASER_RANGE:
            continue

        ttc_norm = float(np.clip(
            surf / max(C.MAX_SPEED * C.VO_TIME_HORIZON, 1e-3),
            0.0, 1.0
        ))

        comb_r   = robot_radius

        result[k] = [
            cd,
            sd,
            min(dist, C.LOW_LASER_RANGE) / C.LOW_LASER_RANGE,
            min(comb_r, C.LOW_LASER_RANGE) / C.LOW_LASER_RANGE,
            ttc_norm,
        ]

    return result   # (4, 5)


def _proximity_penalty(laser_min_dist: float, car_v: float = 0.0) -> float:
    """Penalty for driving too close to static obstacles, walls, or reached robots seen by laser."""
    safe = float(getattr(C, "OBS_CLEARANCE_SAFE", 1.10))
    critical = float(getattr(C, "OBS_CLEARANCE_CRITICAL", 0.55))

    if laser_min_dist >= safe:
        return 0.0

    danger = (safe - laser_min_dist) / max(safe, 1e-6)
    reward = C.LOW_REW_OBS_CLEARANCE * (danger ** 2)

    if laser_min_dist < critical:
        deep = (critical - laser_min_dist) / max(critical, 1e-6)
        reward += C.LOW_REW_OBS_CRITICAL * (deep ** 2)

    # If the robot is already close to an obstacle, forward speed should become expensive.
    if car_v > 0.0:
        speed_ratio = min(car_v / max(C.MAX_SPEED, 1e-6), 1.0)
        reward += C.LOW_REW_OBS_SPEED * danger * speed_ratio

    return float(reward)

def _wall_proximity_penalty(car) -> float:
    """保留兼容，内部调用新函数（如有其他地方引用）。"""
    half = C.ARENA_SIZE / 2.0
    dist_right = (half - car.radius) - car.x
    dist_left  = car.x - (-half + car.radius)
    dist_top   = (half - car.radius) - car.y
    dist_bot   = car.y - (-half + car.radius)
    min_wall   = max(min(dist_right, dist_left, dist_top, dist_bot), 0.0)
    return _proximity_penalty(min_wall)


def _heading_reward(car_theta: float,
                    wp_dir: np.ndarray,
                    vo_v: np.ndarray,
                    rvo_v: np.ndarray) -> float:
    """
    朝向奖励：当机器人朝向指向 waypoint 且不在任何 VO/RVO 锥内时，给小正奖励。
    """
    heading = np.array([np.cos(car_theta), np.sin(car_theta)], dtype=np.float32)
    alignment = float(np.dot(heading, wp_dir))
    if alignment <= 0.5:
        return 0.0

    for v in vo_v:
        dist_n = float(v[2])
        rad_n  = float(v[3])
        if dist_n < 1e-3 or rad_n < 1e-3:
            continue
        vo_dir = np.array([float(v[0]), float(v[1])], dtype=np.float32)
        cos_to_center = float(np.dot(heading, vo_dir))
        sin_half = min(rad_n / max(dist_n, 1e-3), 1.0)
        cos_half = float(np.sqrt(max(1.0 - sin_half**2, 0.0)))
        if cos_to_center > cos_half:
            return 0.0

    for v in rvo_v:
        dist_n = float(v[2])
        rad_n  = float(v[3])
        if dist_n < 1e-3 or rad_n < 1e-3:
            continue
        vo_dir = np.array([float(v[0]), float(v[1])], dtype=np.float32)
        cos_to_center = float(np.dot(heading, vo_dir))
        sin_half = min(rad_n / max(dist_n, 1e-3), 1.0)
        cos_half = float(np.sqrt(max(1.0 - sin_half**2, 0.0)))
        if cos_to_center > cos_half:
            return 0.0

    return float(C.LOW_REW_HEADING * alignment)


# ══════════════════════════════════════════════════════════════════════
# 静态障碍物采样工具
# ══════════════════════════════════════════════════════════════════════

def sample_obstacles(rng: np.random.Generator,
                     n: int = None,
                     arena: float = C.ARENA_SIZE) -> List[CircleObstacle]:
    """
    采样障碍物，支持圆形和矩形混合，数量和大小随地图自动缩放。
    """
    if n is None:
        n = C.N_STATIC_OBS

    scale   = arena / 12.0
    n_total = max(n, int(round(n * scale * scale)))

    r_min = C.OBS_RADIUS_MIN * (scale ** 0.5)
    r_max = min(C.OBS_RADIUS_MAX * (scale ** 0.5), arena * 0.08)

    min_passage = 2 * C.ROBOT_RADIUS + C.PASSAGE_MARGIN
    half = arena / 2.0
    obs_list: List[CircleObstacle] = []

    n_rect   = int(n_total * 0.4) if arena >= 20 else 0
    n_circle = n_total - n_rect

    for _ in range(n_circle):
        for _ in range(500):
            r = float(rng.uniform(r_min, r_max))
            wall_margin = r + min_passage
            lo = -half + wall_margin
            hi =  half - wall_margin
            if lo >= hi:
                break
            x = float(rng.uniform(lo, hi))
            y = float(rng.uniform(lo, hi))

            ok = True
            for o in obs_list:
                gap = float(np.sqrt((x-o.x)**2 + (y-o.y)**2)) - r - o.r
                if gap < min_passage:
                    ok = False; break
            if not ok:
                continue

            obs_list.append(CircleObstacle(x, y, r))
            break

    rect_list = []

    for _ in range(n_rect):
        if rng.random() < 0.5:
            w = float(rng.uniform(arena * 0.15, arena * 0.30))
            h = float(rng.uniform(r_min * 1.5, r_max * 1.5))
        else:
            w = float(rng.uniform(r_min * 1.5, r_max * 1.5))
            h = float(rng.uniform(arena * 0.15, arena * 0.30))
        r_equiv = max(w, h) * 0.6

        wall_margin = max(w, h) / 2 + min_passage
        lo = -half + wall_margin
        hi =  half - wall_margin
        if lo >= hi:
            continue

        for _ in range(500):
            cx = float(rng.uniform(lo, hi))
            cy = float(rng.uniform(lo, hi))
            ok = True
            for o in obs_list:
                gap = float(np.sqrt((cx-o.x)**2 + (cy-o.y)**2)) - r_equiv - o.r
                if gap < min_passage:
                    ok = False; break
            if ok:
                for rr in rect_list:
                    if (abs(cx - rr.cx) < (w + rr.w)/2 + min_passage and
                        abs(cy - rr.cy) < (h + rr.h)/2 + min_passage):
                        ok = False; break
            if not ok:
                continue
            rect_list.append(RectObstacle(cx=cx, cy=cy, w=w, h=h))
            break

    return obs_list, rect_list


def make_corridor_scene(wall_thickness: float, gap_width: float,
                        arena: float, rng) -> tuple:
    """
    会车死锁场景：横墙(上下分区) + 中央开口 + 对向交叉起终点(N=2)。
    返回 (rect_walls, starts, goals)
    """
    half     = arena / 2.0
    gap_half = gap_width / 2.0

    left_w  = half - gap_half
    right_w = half - gap_half
    left_cx  = -half + left_w / 2.0
    right_cx =  half - right_w / 2.0

    rect_walls = [
        RectObstacle(cx=left_cx,  cy=0.0, w=left_w,  h=wall_thickness),
        RectObstacle(cx=right_cx, cy=0.0, w=right_w, h=wall_thickness),
    ]

    margin = half * 0.7
    x_jit  = arena * 0.12
    ax = float(rng.uniform(-x_jit, x_jit))
    bx = float(rng.uniform(-x_jit, x_jit))
    starts = [
        np.array([ax,  margin], dtype=np.float32),
        np.array([bx, -margin], dtype=np.float32),
    ]
    goals = [
        np.array([bx, -margin], dtype=np.float32),
        np.array([ax,  margin], dtype=np.float32),
    ]
    return rect_walls, starts, goals


def sample_free_point(rng, obstacles, clearance, arena=C.ARENA_SIZE,
                      avoid=None, avoid_dist=0.0,
                      min_from=None, min_dist=0.0,
                      rect_walls=None):
    half = arena / 2.0 - clearance
    avoid = avoid or []
    rect_walls = rect_walls or []
    for _ in range(500):
        p = rng.uniform(-half, half, 2).astype(np.float32)
        if any(np.linalg.norm(p - o.pos) < o.r + clearance for o in obstacles):
            continue
        if any(w.collides_circle(float(p[0]), float(p[1]), clearance)
               for w in rect_walls):
            continue
        if any(np.linalg.norm(p - q) < avoid_dist for q in avoid):
            continue
        if min_from is not None and np.linalg.norm(p - min_from) < min_dist:
            continue
        return p
    return p


# ══════════════════════════════════════════════════════════════════════
# 下层环境
# ══════════════════════════════════════════════════════════════════════

class LowLevelEnv:
    """
    下层多机器人环境。
    """

    def __init__(self, n_robots: int = 1, seed: Optional[int] = None,
                 maze_level: int = 0, map_loader=None):
        self.n          = n_robots
        self._rng       = np.random.default_rng(seed)
        self._comm      = SimCommInterface()
        self.maze_level = maze_level
        self.map_loader = map_loader
        if map_loader is not None:
            C.ARENA_SIZE = max(map_loader.width_m, map_loader.height_m)

        # 仿真状态
        self.cars:       List[AckermannRobot]  = []
        self.goals:      List[np.ndarray]      = []
        self.waypoints:  List[np.ndarray]      = []
        self.obstacles:  List[CircleObstacle]  = []
        self.rect_walls: List[RectObstacle]    = []
        self._active:    List[bool]            = []
        self._step_count:  int                 = 0
        self._last_actions: np.ndarray         = np.zeros((n_robots, 2), dtype=np.float32)
        self._prev_wp_dists: List[float]       = [0.0] * n_robots
        # 死路检测：连续帧计数
        self._deadend_count: List[int]         = [0] * n_robots
        self._wait_count:    List[int]         = [0] * n_robots
        # ── 窄道路权机制状态 ──────────────────────────────────────────
        self._entry_depth:    List[float] = [0.0]   * n_robots  # 进入窄道深度
        self._in_corridor:    List[bool]  = [False] * n_robots  # 是否在窄道
        self._corridor_entry: List        = [None]  * n_robots  # 进窄道时的位置
        # LSP Frontier 长期规划模块（启发式，始终可用）
        self.lsp = LowLevelFrontierModule()
        # SLAM 地图（每个机器人独立）
        self.slam_maps: List[OccupancyMap] = []
        # GNN LSP 规划器（可选）
        self.gnn_planner = None
        # 会车压力场景
        self.corridor_mode   = False
        self.wall_thickness  = 0.5
        self.gap_width       = 1.5
        # 调试开关
        self.debug_waypoint = False
        self._prev_rvo_risk = [0.0] * self.n

    # ── reset ────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None) -> List[dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count   = 0
        self._last_actions = np.zeros((self.n, 2), dtype=np.float32)
        self._deadend_count = [0] * self.n
        self._wait_count    = [0] * self.n
        # 窄道路权状态重置
        self._entry_depth    = [0.0]   * self.n
        self._in_corridor    = [False] * self.n
        self._corridor_entry = [None]  * self.n

        # ── 障碍物 / 起终点设置 ────────────────────────────────────────
        if self.corridor_mode:
            self.rect_walls, starts, self.goals = make_corridor_scene(
                self.wall_thickness, self.gap_width, C.ARENA_SIZE, self._rng)
            self.obstacles = []
        elif self.map_loader is not None:
            self.rect_walls = []
            self.obstacles  = []
            starts_g, goals_g = self.map_loader.sample_starts_goals(
                n=self.n, min_dist_m=C.MIN_START_GOAL, rng=self._rng)
            starts = [s.astype(np.float32) for s in starts_g]
            self.goals = [g.astype(np.float32) for g in goals_g]
        elif self.maze_level >= 10:
            from envs.room_scenes import get_scene, get_valid_positions
            self.rect_walls = get_scene(self.maze_level)
            self.obstacles  = []
            positions = get_valid_positions(
                self.maze_level, self.n * 2,
                rng=self._rng, min_dist=C.MIN_START_GOAL
            )
            starts = [np.array(positions[i], dtype=np.float32)
                      for i in range(self.n)]
            self.goals = [np.array(positions[self.n + i], dtype=np.float32)
                          for i in range(self.n)]
        elif self.maze_level >= 2:
            from envs.maze_scenes import get_scene, get_valid_positions
            self.rect_walls = get_scene(self.maze_level)
            self.obstacles  = []
            positions = get_valid_positions(
                self.maze_level, self.n * 2,
                rng=self._rng, min_dist=C.MIN_START_GOAL
            )
            starts = [np.array(positions[i], dtype=np.float32)
                      for i in range(self.n)]
            self.goals = [np.array(positions[self.n + i], dtype=np.float32)
                          for i in range(self.n)]
        else:
            self.obstacles, self.rect_walls = sample_obstacles(self._rng, arena=C.ARENA_SIZE)
            starts = []
            for _ in range(self.n):
                p = sample_free_point(
                    self._rng, self.obstacles,
                    clearance=C.ROBOT_RADIUS + 0.2,
                    arena=C.ARENA_SIZE,
                    avoid=starts, avoid_dist=C.MIN_ROBOT_SPACING,
                    rect_walls=self.rect_walls,
                )
                starts.append(p)
            self.goals = []
            min_sg = C.MIN_START_GOAL * (C.ARENA_SIZE / 12.0)
            min_sg = float(np.clip(min_sg, C.MIN_START_GOAL,
                                   C.ARENA_SIZE * 0.4))
            for i in range(self.n):
                g = sample_free_point(
                    self._rng, self.obstacles,
                    clearance=C.GOAL_TOL + 0.2,
                    arena=C.ARENA_SIZE,
                    avoid=starts + self.goals, avoid_dist=C.GOAL_TOL,
                    min_from=starts[i], min_dist=min_sg,
                )
                self.goals.append(g)

        self.cars = []
        for i, s in enumerate(starts):
            to_g   = self.goals[i] - s
            theta0 = float(np.arctan2(to_g[1], to_g[0]))
            theta0 += float(self._rng.uniform(-0.3, 0.3))
            self.cars.append(AckermannRobot(x=float(s[0]), y=float(s[1]),
                                            theta=theta0))

        self._active = [True] * self.n
        self.waypoints = [g.copy() for g in self.goals]
        self._prev_wp_dists = [
            float(np.linalg.norm(self.cars[i].pos - self.waypoints[i]))
            for i in range(self.n)
        ]

        self.lsp.reset(self.n)
        self.slam_maps = [
            OccupancyMap(resolution=0.3, map_size_m=C.ARENA_SIZE * 1.5)
            for _ in range(self.n)
        ]
        return self._make_obs_list()

    # ── waypoint 设置（上层调用）─────────────────────────────────────
    def set_waypoints(self, waypoints: List[np.ndarray]) -> None:
        assert len(waypoints) == self.n
        self.waypoints = [wp.copy() for wp in waypoints]
        self._prev_wp_dists = [
            float(np.linalg.norm(self.cars[i].pos - self.waypoints[i]))
            for i in range(self.n)
        ]

    def set_debug_waypoint(self, enabled: bool = True) -> None:
        self.debug_waypoint = enabled
        print(f"✓ Waypoint 调试: {'开启' if enabled else '关闭'}")

    def set_gnn_planner(self, planner) -> None:
        self.gnn_planner = planner
        mode = "GNN学习模式" if planner is not None else "启发式模式"
        print(f"✓ LSP 规划器切换: {mode}")

    # ── step ─────────────────────────────────────────────────────────
    def step(self, actions: np.ndarray) -> Tuple[List[dict], List[float],
                                                  List[bool], List[bool],
                                                  List[dict]]:
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n, 2)
        self._step_count += 1
        self._last_actions = actions.copy()

        # 1. 移动
        for i in range(self.n):
            if self._active[i]:
                self.cars[i].step(float(actions[i, 0]), float(actions[i, 1]))

        # 1.5 更新 SLAM 地图
        for i in range(self.n):
            if self._active[i]:
                car = self.cars[i]
                if self.map_loader is not None:
                    ranges_hi_step = self.map_loader.simulate_laser(
                        car.x, car.y, car.theta,
                        n_beams=C.HIGH_LASER_BEAMS,
                        max_range=C.HIGH_LASER_RANGE,
                        fov_deg=C.LASER_FOV,
                    )
                else:
                    ranges_hi_step = simulate_laser(
                        car.x, car.y, car.theta,
                        self.obstacles, C.ARENA_SIZE,
                        n_beams=C.HIGH_LASER_BEAMS,
                        max_range=C.HIGH_LASER_RANGE,
                        rect_walls=self.rect_walls,
                    )
                self.slam_maps[i].update(
                    car.x, car.y, car.theta,
                    ranges_hi_step,
                    max_range=C.HIGH_LASER_RANGE,
                )
        # GNN 选择新的 waypoint（降频）
        if self._step_count % C.HIGH_LEVEL_INTERVAL == 0:
            self._update_gnn_waypoints()
            for i in range(self.n):
                if self._active[i]:
                    self._prev_wp_dists[i] = float(np.linalg.norm(
                        self.cars[i].pos - self.waypoints[i]))
        # 2. 更新通信状态
        all_states = {}
        for i in range(self.n):
            all_states[i] = {
                'pos':         self.cars[i].pos.copy(),
                'vel':         self.cars[i].vel_vec.copy(),
                'radius':      self.cars[i].radius,
                'goal':        self.goals[i].copy(),
                'last_action': actions[i].copy(),
            }
        self._comm.update_state(all_states)

        # 3. 碰撞检测
        wall_col   = [self._wall_col(i) if self._active[i] else False
                      for i in range(self.n)]
        obs_col    = [self._obs_col(i)  if self._active[i] else False
                      for i in range(self.n)]
        robot_col  = self._robot_col()
        collided   = [wall_col[i] or obs_col[i] or robot_col[i]
                      for i in range(self.n)]

        # 4. 到达检测
        wp_dists   = [float(np.linalg.norm(self.cars[i].pos - self.waypoints[i]))
                      for i in range(self.n)]
        goal_dists = [float(np.linalg.norm(self.cars[i].pos - self.goals[i]))
                      for i in range(self.n)]
        wp_reached   = [wp_dists[i] < C.GOAL_TOL for i in range(self.n)]
        goal_reached = [goal_dists[i] < C.GOAL_TOL for i in range(self.n)]

        for i in range(self.n):
            if collided[i] or goal_reached[i]:
                self._active[i] = False

        # 5. 计算奖励 + 构造返回
        obs_list = self._make_obs_list()

        # ── 更新每台车的窄道状态（用于路权奖励）──────────────────────
        for i in range(self.n):
            if self._active[i]:
                laser_i = obs_list[i]['laser'] * C.LOW_LASER_RANGE
                self._update_corridor_state(i, laser_i)

        rews, infos = [], []
        for i in range(self.n):
            car   = self.cars[i]
            rvo_v = obs_list[i]['rvo']

            laser_min = float(obs_list[i]['laser'].min()) * C.LOW_LASER_RANGE

            to_wp  = self.waypoints[i] - car.pos
            rho    = float(np.linalg.norm(to_wp))
            wp_dir = (to_wp / rho).astype(np.float32) if rho > 1e-6 \
                     else np.array([1.0, 0.0], dtype=np.float32)

            rew = self._reward(i, collided[i], wp_reached[i], goal_reached[i],
                               wp_dists[i], rvo_v, car.theta, wp_dir,
                               laser_min, car.v,
                action=actions[i])
            rews.append(rew)

            laser_ranges_i = np.array([
                float(obs_list[i]['laser'][k]) * C.LOW_LASER_RANGE
                for k in range(C.LOW_LASER_BEAMS)
            ], dtype=np.float32)
            is_deadend = self._detect_deadend(i, laser_ranges_i)
            if is_deadend:
                self._deadend_count[i] += 1
            else:
                self._deadend_count[i] = 0
            deadend_confirmed = (self._deadend_count[i] >= C.DEADEND_CONFIRM)

            infos.append({
                'reached':   goal_reached[i],
                'wp_reached':wp_reached[i],
                'collided':  collided[i],
                'dist_goal': goal_dists[i],
                'dist_wp':   wp_dists[i],
                'step':      self._step_count,
                'deadend':   deadend_confirmed,
            })

        self._prev_wp_dists = wp_dists

        deadend_term = [infos[i]['deadend'] and self.maze_level >= 2
                        for i in range(self.n)]

        terms  = [collided[i] or goal_reached[i] or deadend_term[i]
                  for i in range(self.n)]
        truncs = [(not terms[i]) and (self._step_count >= C.MAX_LOW_STEPS)
                  for i in range(self.n)]

        return obs_list, rews, terms, truncs, infos

    # ── 窄道检测 + entry_depth 更新 ──────────────────────────────────
    def _update_corridor_state(self, i: int, laser_ranges: np.ndarray) -> None:
        """
        激光检测窄道，更新 in_corridor 和 entry_depth。
        窄道判据：左侧扇区最近距 + 右侧扇区最近距 < CORRIDOR_WIDTH_THRESH
        entry_depth = 离进入点的距离（先进者更深）
        """
        car = self.cars[i]
        n_beams = len(laser_ranges)
        fov_half = np.deg2rad(C.LASER_FOV / 2.0)
        angles = np.linspace(-fov_half, fov_half, n_beams)

        # 左侧 +60~+120°，右侧 -120~-60°（车体系）
        left_mask  = (angles >=  np.deg2rad(60)) & (angles <=  np.deg2rad(120))
        right_mask = (angles <= -np.deg2rad(60)) & (angles >= -np.deg2rad(120))

        def smin(mask):
            return float(np.min(laser_ranges[mask])) if mask.any() else C.LOW_LASER_RANGE

        width  = smin(left_mask) + smin(right_mask)
        thresh = getattr(C, 'CORRIDOR_WIDTH_THRESH', 1.1)
        now_in = (width < thresh)

        if now_in and not self._in_corridor[i]:
            # 刚进入窄道
            self._in_corridor[i]    = True
            self._corridor_entry[i] = car.pos.copy()
            self._entry_depth[i]    = 0.0
        elif now_in and self._in_corridor[i]:
            # 在窄道内：更新深度
            self._entry_depth[i] = float(np.linalg.norm(
                car.pos - self._corridor_entry[i]))
        else:
            # 回到宽阔区：清零
            self._in_corridor[i]    = False
            self._entry_depth[i]    = 0.0
            self._corridor_entry[i] = None

    def _update_gnn_waypoints(self) -> None:
        for i in range(self.n):
            if not self._active[i]:
                continue
            try:
                car      = self.cars[i]
                goal_pos = self.goals[i]

                dist_to_goal = float(np.linalg.norm(car.pos - goal_pos))
                if dist_to_goal < C.LOW_LASER_RANGE * 1.5:
                    self.waypoints[i] = self._select_visible_waypoint(i, goal_pos.copy())
                    continue

                frontiers_world = self.slam_maps[i].get_frontiers(
                    car.pos[0], car.pos[1],
                    max_range=C.LOW_LASER_RANGE * 2.0,
                )
                if len(frontiers_world) == 0:
                    self.waypoints[i] = self._select_visible_waypoint(i, goal_pos.copy())
                    continue
                n_f = min(len(frontiers_world), 8)
                fps = [frontiers_world[j] for j in range(n_f)]

                if self.gnn_planner is not None:
                    best = self.gnn_planner.plan(
                        frontiers   = fps,
                        widths      = [0.3] * n_f,
                        robot_pos   = car.pos,
                        robot_theta = car.theta,
                        robot_vel   = car.vel_vec,
                        goal_pos    = goal_pos,
                        slam_map    = self.slam_maps[i],
                    )
                    self.waypoints[i] = self._select_visible_waypoint(i, fps[best].copy())
                    continue

                best_idx   = 0
                best_score = -float('inf')

                to_goal    = goal_pos - car.pos
                goal_dist  = float(np.linalg.norm(to_goal))
                goal_angle = float(np.arctan2(to_goal[1], to_goal[0]))

                for j, fp in enumerate(fps):
                    diff       = fp - car.pos
                    dist       = float(np.linalg.norm(diff))
                    fp_angle   = float(np.arctan2(diff[1], diff[0]))

                    import math
                    l1 = C.ROBOT_RADIUS * 3
                    l2 = C.LOW_LASER_RANGE
                    dist_score = math.tanh(
                        (dist - l1) / max(l2 - l1, 1e-3)
                    ) * l2

                    angle_diff = abs(float(
                        (fp_angle - goal_angle + np.pi) % (2*np.pi) - np.pi
                    ))
                    goal_align = (np.pi - angle_diff) / np.pi

                    d_fg       = float(np.linalg.norm(goal_pos - fp))
                    dist_to_goal_score = -d_fg / max(goal_dist, 1e-3)

                    info_score = self._frontier_info_score(
                        fp, self.slam_maps[i]
                    )

                    score = (0.3 * dist_score / max(l2, 1e-3)
                           + 0.3 * goal_align
                           + 0.2 * dist_to_goal_score
                           + 0.2 * info_score)

                    if self.debug_waypoint and self._step_count % 50 == 0:
                        print(f"  [Robot{i} step{self._step_count} F{j}] "
                              f"fp={fp.round(1)}  dist={dist:.1f}  "
                              f"dist_score={dist_score:.2f}  "
                              f"goal_align={goal_align:.2f}  "
                              f"d2g_score={dist_to_goal_score:.2f}  "
                              f"info={info_score:.2f}  → score={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_idx   = j

                self.waypoints[i] = self._select_visible_waypoint(i, fps[best_idx].copy())

                if self.debug_waypoint and self._step_count % 50 == 0:
                    print(f"  [Robot{i}] ★ 选 F{best_idx} "
                          f"wp={fps[best_idx].round(1)}  score={best_score:.3f}\n")

            except Exception:
                pass


    def _current_laser_ranges_for_waypoint(self, i: int) -> np.ndarray:
        """Use the same low-level observation laser to judge local waypoint visibility."""
        obs_i = self._make_obs_list()[i]
        return np.asarray(obs_i["laser"], dtype=np.float32) * C.LOW_LASER_RANGE

    def _waypoint_visible_from_laser(self, i: int, wp: np.ndarray,
                                     laser_ranges: np.ndarray) -> bool:
        car = self.cars[i]
        rel = np.asarray(wp, dtype=np.float32) - car.pos
        dist = float(np.linalg.norm(rel))

        if dist < 1e-6:
            return True

        if dist > float(getattr(C, "WP_VISIBLE_MAX_DIST", 2.2)):
            return False

        bearing = float(np.arctan2(rel[1], rel[0]) - car.theta)
        bearing = (bearing + np.pi) % (2.0 * np.pi) - np.pi

        fov_half = np.deg2rad(C.LASER_FOV / 2.0)
        if abs(bearing) > fov_half:
            return False

        n = len(laser_ranges)
        idx = int(round((bearing + fov_half) / max(2.0 * fov_half, 1e-6) * (n - 1)))
        idx = max(0, min(n - 1, idx))

        clearance = float(getattr(C, "WP_VISIBLE_CLEARANCE", 0.45))
        edge_margin = float(getattr(C, "WP_VISIBLE_EDGE_MARGIN", 0.20))
        visible_dist = float(laser_ranges[idx]) - clearance - edge_margin

        return dist <= visible_dist

    def _select_visible_waypoint(self, i: int, candidate_wp: np.ndarray) -> np.ndarray:
        """Keep waypoint inside current observable free space.

        Original high-level selector can propose goal/frontier/GNN waypoint.
        This function accepts it only if current laser can see it. Otherwise it
        samples visible laser directions and picks a reachable local waypoint.
        """
        car = self.cars[i]
        goal = self.goals[i]
        laser_ranges = self._current_laser_ranges_for_waypoint(i)

        candidate_wp = np.asarray(candidate_wp, dtype=np.float32)
        if self._waypoint_visible_from_laser(i, candidate_wp, laser_ranges):
            return candidate_wp.copy()

        to_goal = goal - car.pos
        old_goal_dist = float(np.linalg.norm(to_goal))
        if old_goal_dist < 1e-6:
            return goal.copy()

        goal_dir = to_goal / max(old_goal_dist, 1e-6)

        fov_half = np.deg2rad(C.LASER_FOV / 2.0)
        angles = np.linspace(-fov_half, fov_half, len(laser_ranges))

        clearance = float(getattr(C, "WP_VISIBLE_CLEARANCE", 0.45))
        edge_margin = float(getattr(C, "WP_VISIBLE_EDGE_MARGIN", 0.20))
        min_dist = float(getattr(C, "WP_VISIBLE_MIN_DIST", 0.45))
        max_dist = float(getattr(C, "WP_VISIBLE_MAX_DIST", 2.2))

        best_wp = car.pos.copy()
        best_score = -1e18

        for a, r in zip(angles, laser_ranges):
            usable = min(float(r) - clearance - edge_margin, max_dist)
            if usable < min_dist:
                continue

            heading = car.theta + float(a)
            direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)

            for frac in (0.55, 0.75, 1.0):
                d = max(min_dist, usable * frac)
                cand = car.pos + direction * d

                if np.any(np.abs(cand) > C.ARENA_SIZE):
                    continue

                new_goal_dist = float(np.linalg.norm(goal - cand))
                progress = old_goal_dist - new_goal_dist
                align = float(np.dot(direction, goal_dir))
                clear_bonus = min(float(r), C.LOW_LASER_RANGE) / max(C.LOW_LASER_RANGE, 1e-6)
                turn_cost = abs(float(a))

                score = (
                    float(getattr(C, "WP_VISIBLE_GOAL_BONUS", 3.0)) * progress
                    + float(getattr(C, "WP_VISIBLE_ALIGN_BONUS", 1.2)) * align
                    + float(getattr(C, "WP_VISIBLE_CLEAR_BONUS", 0.35)) * clear_bonus
                    - float(getattr(C, "WP_VISIBLE_TURN_PENALTY", 0.15)) * turn_cost
                )

                if score > best_score:
                    best_score = score
                    best_wp = cand.astype(np.float32)

        return best_wp.astype(np.float32)

    def _frontier_info_score(self, frontier_pos: np.ndarray,
                              slam_map) -> float:
        if not slam_map._initialized:
            return 1.0

        k    = 8
        half = k // 2
        cr, cc = slam_map.world_to_grid(
            float(frontier_pos[0]), float(frontier_pos[1])
        )

        r0 = max(0, cr - half); r1 = min(slam_map.n_cells, cr + half)
        c0 = max(0, cc - half); c1 = min(slam_map.n_cells, cc + half)

        if r1 <= r0 or c1 <= c0:
            return 0.0

        patch   = slam_map.grid[r0:r1, c0:c1]
        n_total = patch.size
        n_unknown = int((patch == 0).sum())

        return n_unknown / max(n_total, 1)

    # ── 观测构造 ─────────────────────────────────────────────────────
    def _make_obs_list(self) -> List[dict]:
        obs_list = []

        for i in range(self.n):
            car = self.cars[i]
            pos = car.pos

            if self.map_loader is not None:
                ranges = self.map_loader.simulate_laser(
                    car.x, car.y, car.theta,
                    n_beams=C.LOW_LASER_BEAMS,
                    max_range=C.LOW_LASER_RANGE,
                    fov_deg=C.LASER_FOV,
                )
            else:
                laser_obstacles = list(self.obstacles)
                for j in range(self.n):
                    if j == i:
                        continue
                    laser_obstacles.append(CircleObstacle(
                        float(self.cars[j].x),
                        float(self.cars[j].y),
                        float(self.cars[j].radius),
                    ))

                ranges = simulate_laser(
                    car.x, car.y, car.theta,
                    laser_obstacles, C.ARENA_SIZE,
                    n_beams=C.LOW_LASER_BEAMS,
                    max_range=C.LOW_LASER_RANGE,
                    rect_walls=self.rect_walls,
                )
            laser_obs = (ranges / C.LOW_LASER_RANGE).astype(np.float32)

            _, _, nbr_info = self._comm.get_neighbors(i, pos)
            rvo_vecs = compute_all_rvo(pos, car.vel_vec, car.radius, nbr_info)

            wp    = self.waypoints[i]
            to_wp = wp - pos
            rho   = float(np.linalg.norm(to_wp))
            if rho > 1e-6:
                theta_wp = float(np.arctan2(to_wp[1], to_wp[0]) - car.theta)
                theta_wp = (theta_wp + np.pi) % (2 * np.pi) - np.pi
            else:
                theta_wp = 0.0
            wp_obs = np.array([
                min(rho, C.ARENA_SIZE) / C.ARENA_SIZE,
                theta_wp / np.pi,
            ], dtype=np.float32)

            vel_obs = np.array([
                car.v / C.MAX_SPEED,
                float(np.clip(car.omega / C.MAX_OMEGA, -1.0, 1.0)),
                float(np.cos(car.theta)),
                float(np.sin(car.theta)),
            ], dtype=np.float32)

            goal = self.goals[i]
            to_goal = goal - pos
            rho_g   = float(np.linalg.norm(to_goal))
            if rho_g > 1e-6:
                theta_g = float(np.arctan2(to_goal[1], to_goal[0]) - car.theta)
                theta_g = (theta_g + np.pi) % (2 * np.pi) - np.pi
            else:
                theta_g = 0.0
            goal_obs = np.array([
                min(rho_g, C.ARENA_SIZE) / C.ARENA_SIZE,
                theta_g / np.pi,
            ], dtype=np.float32)

            obs_list.append({
                'laser': laser_obs,
                'rvo':   rvo_vecs,
                'wp':    wp_obs,
                'vel':   vel_obs,
                'goal':  goal_obs,
            })

        return obs_list

    # ── 奖励 ─────────────────────────────────────────────────────────
    def _reward(self, i: int, collided: bool, wp_reached: bool,
                goal_reached: bool, wp_dist: float,
                rvo_v: np.ndarray,
                car_theta: float, wp_dir: np.ndarray,
                laser_min: float,
                car_v: float,
                  action=None) -> float:
        if collided:
            pen = C.LOW_REW_COLLISION
            if abs(car_v) > 0.5 * C.MAX_SPEED:
                pen += C.LOW_REW_COLLISION_SPEED
            return pen
        if goal_reached:
            return C.LOW_REW_WP_REACH * 2

        r = 0.0
        if wp_reached:
            r += C.LOW_REW_WP_REACH

        # 进度奖励（只奖励正向进度）
        prev = self._prev_wp_dists[i]
        delta = prev - wp_dist
        if delta > 0:
            r += delta * C.LOW_REW_APPROACH
        elif car_v >= 0:
            r += delta * C.LOW_REW_APPROACH * 0.3
        # else: 倒车远离 → 不惩罚

        # RVO 惩罚
        rvo_pen = rvo_reward(
            np.zeros((0, 5), dtype=np.float32),
            rvo_v,
            action=action,
            car_theta=car_theta,
        )
        r += rvo_pen

        # Action-level RVO cone penalty:
        # This block is for the original compact RVO feature only. In cone mode,
        # rvo_reward() already handles cone geometry, so we skip this old shaping.
        if getattr(C, "RVO_OBS_MODE", "compact") != "cone" and action is not None and len(action) >= 2:
            speed_cmd = float(np.clip(action[0], -1.0, 1.0))
            steer_cmd = float(np.clip(action[1], -1.0, 1.0))

            # Approximate Ackermann action as a short-horizon velocity direction.
            cand_dir = car_theta + steer_cmd * C.MAX_STEER
            if speed_cmd < 0.0:
                cand_dir += np.pi

            def _wrap_angle(a):
                return (a + np.pi) % (2.0 * np.pi) - np.pi

            action_rvo_pen = 0.0
            action_rvo_out = 0.0
            for v in rvo_v:
                ttc = float(v[4])
                if ttc >= 0.99:
                    continue

                nbr_ang = float(np.arctan2(float(v[1]), float(v[0])))
                dist = max(float(v[2]) * C.COMM_RANGE, 1e-3)
                radius = max(float(v[3]) * C.COMM_RANGE, 1e-3)

                half_angle = float(np.arcsin(np.clip(radius / dist, 0.0, 0.99)))
                margin = float(getattr(C, "RVO_ACTION_MARGIN", 0.15))
                ang_err = abs(_wrap_angle(cand_dir - nbr_ang))

                danger = float(np.exp(-C.RVO_DANGER_DECAY * float(np.clip(ttc, 0.0, 1.0))))

                # Inside or near the cone: chosen action is unsafe.
                if ang_err < half_angle + margin:
                    inside = (half_angle + margin - ang_err) / max(half_angle + margin, 1e-6)
                    action_rvo_pen += danger * inside

                    # Deep center of cone: stronger penalty.
                    if ang_err < half_angle * 0.5:
                        deep = (half_angle * 0.5 - ang_err) / max(half_angle * 0.5, 1e-6)
                        action_rvo_pen += (-C.LOW_REW_RVO_ACTION_DEEP / max(-C.LOW_REW_RVO_ACTION_IN, 1e-6)) * danger * deep
                else:
                    # Clearly outside cone while danger exists: small positive guidance.
                    clearance = min((ang_err - half_angle) / max(np.pi - half_angle, 1e-6), 1.0)
                    action_rvo_out += danger * clearance

            r += C.LOW_REW_RVO_ACTION_IN * action_rvo_pen
            r += C.LOW_REW_RVO_ACTION_OUT * action_rvo_out

        # Temporal RVO shaping: reward starting to leave the danger field.
        # Compact RVO only. Cone mode has a different feature layout, so v[4]
        # is not TTC and must not be used here.
        if getattr(C, "RVO_OBS_MODE", "compact") != "cone":
            valid_rvo_for_field = [v for v in rvo_v if float(v[4]) < 0.99]
            if valid_rvo_for_field:
                cur_rvo_danger = max(float(np.exp(-C.RVO_DANGER_DECAY * float(np.clip(v[4], 0.0, 1.0))))
                                     for v in valid_rvo_for_field)
            else:
                cur_rvo_danger = 0.0

            if not hasattr(self, "_prev_rvo_danger") or len(self._prev_rvo_danger) != self.n:
                self._prev_rvo_danger = [0.0] * self.n

            prev_rvo_danger = float(self._prev_rvo_danger[i])
            if cur_rvo_danger < prev_rvo_danger:
                r += C.LOW_REW_RVO_EXIT * (prev_rvo_danger - cur_rvo_danger)
            self._prev_rvo_danger[i] = cur_rvo_danger

        # Ackermann reachable-area shaping:
        # In open space, reverse/stop is discouraged.
        # If the forward reachable region is blocked, backing out is rewarded.
        if action is not None and len(action) >= 1:
            speed_cmd = float(np.clip(action[0], -1.0, 1.0))

            # Static obstacle/wall blocking in front, approximated by minimum low laser range.
            front_blocked_by_laser = laser_min < C.ACK_BLOCK_LASER_THRESH

            # Dynamic blocking: a neighbor is close to the front cone or has dangerous TTC.
            front_blocked_by_rvo = False
            for rv in rvo_v:
                ttc = float(rv[4])
                if ttc >= 0.99:
                    continue
                side = abs(float(rv[1]))
                dist = float(rv[2]) * C.COMM_RANGE
                if side < C.ACK_BLOCK_SIDE_THRESH and (
                    dist < C.ACK_BLOCK_RVO_DIST or ttc < C.ACK_BLOCK_RVO_TTC
                ):
                    front_blocked_by_rvo = True
                    break

            forward_blocked = bool(front_blocked_by_laser or front_blocked_by_rvo)

            if forward_blocked:
                # Forward into a blocked Ackermann reachable area is bad.
                if speed_cmd > 0.0:
                    r += C.LOW_REW_BLOCKED_FORWARD * speed_cmd

                # Backing out is useful only when forward is actually blocked.
                if speed_cmd < 0.0:
                    r += C.LOW_REW_BACKOUT * abs(speed_cmd)

                # Stopping is acceptable in a blocked situation, so do not add stop penalty here.
            else:
                # Open space: reverse/stop should be rare. High RVO danger softens the penalty.
                relief = min(float(cur_rvo_danger), float(getattr(C, "RVO_DANGER_RELIEF", 0.6)))
                penalty_scale = max(0.0, 1.0 - relief)

                if speed_cmd < 0.0:
                    r += C.LOW_REW_REVERSE * abs(speed_cmd) * penalty_scale

                if abs(speed_cmd) < C.STOP_SPEED_THRESH:
                    stop_ratio = 1.0 - abs(speed_cmd) / max(C.STOP_SPEED_THRESH, 1e-6)
                    r += C.LOW_REW_STOP * stop_ratio * penalty_scale

        # RVO 危险时的减速奖励
        valid_rvo = [v for v in rvo_v if float(v[4]) < 0.99]
        if valid_rvo:
            min_ttc = min(float(v[4]) for v in valid_rvo)
            danger  = max(0.0, 0.5 - min_ttc) / 0.5
            if danger > 0:
                speed_norm = abs(car_v) / C.MAX_SPEED
                r += C.LOW_REW_SLOW * danger * (1.0 - speed_norm)
                if min_ttc < 0.3:
                    r -= C.LOW_REW_SPEED_DANGER * speed_norm
        
        # 多机器人间距惩罚（越近越痛，与速度无关）
        for j in range(self.n):
            if j == i or not self._active[j]:
                continue
            d_ij = float(np.linalg.norm(self.cars[i].pos - self.cars[j].pos))
            min_safe = self.cars[i].radius + self.cars[j].radius + 0.5
            if d_ij < min_safe:
                # 距离越近惩罚越重，速度越快惩罚越重
                closeness = (min_safe - d_ij) / min_safe
                speed_norm = abs(car_v) / C.MAX_SPEED
                r -= 0.5 * closeness * (0.5 + 0.5 * speed_norm)

        # RVO 等待奖励：只奖励短时让行；超时躺平转为惩罚
        has_rvo_danger = any(float(v[4]) < 0.5 for v in rvo_v)
        if has_rvo_danger and abs(car_v) < 0.3 * C.MAX_SPEED:
            self._wait_count[i] += 1
            if self._wait_count[i] <= C.WAIT_PATIENCE:
                r += C.LOW_REW_WAIT
            else:
                r += C.LOW_REW_STUCK
        elif abs(car_v) < 0.1 * C.MAX_SPEED and not has_rvo_danger:
            self._wait_count[i] += 1
            if self._wait_count[i] > C.WAIT_PATIENCE:
                r += C.LOW_REW_STUCK
        else:
            self._wait_count[i] = 0

        # 朝向奖励
        if car_v >= 0:
            r += _heading_reward(car_theta, wp_dir,
                                 np.zeros((0, 5), dtype=np.float32), rvo_v)

        # 近障/近墙预警
        r += _proximity_penalty(laser_min, car_v)

        # ── 窄道路权奖励 ──────────────────────────────────
        # 两车都在窄道且接近时：进得深(先进)的鼓励前进，浅(后进)的鼓励后退
        if self._in_corridor[i]:
            for j in range(self.n):
                if j == i or not self._active[j] or not self._in_corridor[j]:
                    continue
                d_ij = float(np.linalg.norm(self.cars[i].pos - self.cars[j].pos))
                if d_ij > C.LOW_LASER_RANGE:
                    continue
                if self._entry_depth[i] > self._entry_depth[j]:
                    # 我先进(更深) → 我有路权 → 前进给奖励
                    if car_v > 0:
                        r += C.LOW_REW_YIELD
                else:
                    # 我后进(更浅) → 我该让 → 后退给奖励
                    if car_v < 0:
                        r += C.LOW_REW_YIELD
                break

        # 时间惩罚
        r += C.LOW_REW_TIME
        # LSP 长期规划奖励
        r += C.LOW_REW_LSP * (0.5 - self.lsp.get_cost(i))
        return float(r)

    # ── 碰撞检测 ─────────────────────────────────────────────────────
    def _wall_col(self, i: int) -> bool:
        if self.map_loader is not None:
            return False
        c    = self.cars[i]
        half = C.ARENA_SIZE / 2.0
        r    = c.radius
        return (c.x < -half+r or c.x > half-r or
                c.y < -half+r or c.y > half-r)

    def _obs_col(self, i: int) -> bool:
        c = self.cars[i]
        if self.map_loader is not None:
            r  = c.radius
            ml = self.map_loader
            cell = ml.cell_size
            H, W = ml.grid.shape
            if (c.x - r < -ml.width_m/2 or c.x + r > ml.width_m/2 or
                c.y - r < -ml.height_m/2 or c.y + r > ml.height_m/2):
                return True
            r_tl, c_tl = ml.xy_to_rc(c.x - r, c.y + r)
            r_br, c_br = ml.xy_to_rc(c.x + r, c.y - r)
            rmin = max(0, min(r_tl, r_br)); rmax = min(H-1, max(r_tl, r_br))
            cmin = max(0, min(c_tl, c_br)); cmax = min(W-1, max(c_tl, c_br))
            for gr in range(rmin, rmax + 1):
                for gc in range(cmin, cmax + 1):
                    if not ml.grid[gr, gc]:
                        continue
                    gx, gy = ml.rc_to_xy(gr, gc)
                    nx = min(max(c.x, gx - cell/2), gx + cell/2)
                    ny = min(max(c.y, gy - cell/2), gy + cell/2)
                    if (c.x - nx)**2 + (c.y - ny)**2 < r*r:
                        return True
            return False
        if any(float(np.linalg.norm(c.pos - o.pos)) < c.radius + o.r
               for o in self.obstacles):
            return True
        if any(w.collides_circle(c.x, c.y, c.radius)
               for w in self.rect_walls):
            return True
        return False

    def _robot_col(self) -> List[bool]:
        cols = [False] * self.n
        for i in range(self.n):
            # 已完成机器人不再被判失败，但仍作为实体阻挡其他未完成机器人
            if not self._active[i]:
                continue
            for j in range(self.n):
                if i == j:
                    continue
                d = float(np.linalg.norm(self.cars[i].pos - self.cars[j].pos))
                if d < self.cars[i].radius + self.cars[j].radius:
                    cols[i] = True
                    break
        return cols

    def _detect_deadend(self, i: int, laser_ranges: np.ndarray) -> bool:
        n_beams  = len(laser_ranges)
        fov_half = np.deg2rad(C.LASER_FOV / 2.0)
        angles   = np.linspace(-fov_half, fov_half, n_beams)

        R_turn   = C.WHEELBASE / np.tan(C.MAX_STEER)
        thresh   = R_turn + C.ROBOT_RADIUS

        a = angles
        front_mask = np.abs(a) < np.deg2rad(45)
        left_mask  = (a >=  np.deg2rad(45))  & (a < np.deg2rad(135))
        right_mask = (a <= -np.deg2rad(45))  & (a > -np.deg2rad(135))
        back_mask  = np.abs(a) >= np.deg2rad(135)

        def smin(mask):
            return float(np.min(laser_ranges[mask])) if mask.any() else C.LOW_LASER_RANGE
        front_min = smin(front_mask)
        left_min  = smin(left_mask)
        right_min = smin(right_mask)
        back_min  = smin(back_mask)

        return (front_min < thresh and
                left_min  < R_turn and
                right_min < R_turn and
                back_min  < thresh)