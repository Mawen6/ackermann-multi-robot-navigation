"""
envs/frontier_module.py
──────────────────────────────────────────────────────────────────────
Frontier 提取 + waypoint 吸附（原有上层功能保留）
+ 下层 LSP Frontier 模块（SLAM地图版本）

原有上层函数（高层激光60束）：完全不变
新增下层类：SLAMFrontierModule
  - 基于 OccupancyMap 的真实 frontier 检测
  - 以 frontier 为中心的地图 patch 作为 LSP 输入
  - P_S 可切换：启发式 或 学习版（LSPNet）
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from configs import config as C


# ══════════════════════════════════════════════════════════════════════
# 原有上层函数（完全不变）
# ══════════════════════════════════════════════════════════════════════

def extract_frontiers(robot_x, robot_y, robot_theta, ranges,
                      max_range=C.HIGH_LASER_RANGE,
                      n_beams=C.HIGH_LASER_BEAMS,
                      frontier_ratio=0.92):
    fov_rad = np.deg2rad(C.LASER_FOV)
    angles  = np.linspace(-fov_rad/2, fov_rad/2, n_beams) + robot_theta
    thresh  = max_range * frontier_ratio
    is_open = ranges > thresh
    frontiers = []
    for i in range(n_beams):
        if not is_open[i]:
            continue
        if i == 0 or i == n_beams - 1:
            has_boundary = True
        else:
            has_boundary = (ranges[i-1] < thresh) or (ranges[i+1] < thresh)
        if has_boundary:
            fx = robot_x + ranges[i] * np.cos(angles[i])
            fy = robot_y + ranges[i] * np.sin(angles[i])
            frontiers.append([fx, fy])
    return np.array(frontiers, dtype=np.float32) if frontiers \
           else np.empty((0, 2), dtype=np.float32)


def merge_frontiers(frontiers, merge_dist=0.5):
    if len(frontiers) == 0:
        return frontiers
    merged = [frontiers[0]]
    for pt in frontiers[1:]:
        dists = np.linalg.norm(np.array(merged) - pt, axis=1)
        if dists.min() > merge_dist:
            merged.append(pt)
    return np.array(merged, dtype=np.float32)


def snap_to_frontier(frontiers, robot_pos, target_angle,
                     max_angle_diff=np.deg2rad(50)):
    if len(frontiers) == 0:
        return None
    fx = frontiers[:, 0] - robot_pos[0]
    fy = frontiers[:, 1] - robot_pos[1]
    f_angles = np.arctan2(fy, fx)
    angle_diff = np.abs(np.arctan2(
        np.sin(f_angles - target_angle),
        np.cos(f_angles - target_angle)
    ))
    valid = angle_diff < max_angle_diff
    if not valid.any():
        return None
    best = int(np.argmin(np.where(valid, angle_diff, np.inf)))
    return frontiers[best].copy()


def get_frontier_waypoint(robot_x, robot_y, robot_theta,
                          ranges_hi, target_angle, fallback_wp,
                          max_angle_diff=np.deg2rad(50)):
    robot_pos = np.array([robot_x, robot_y], dtype=np.float32)
    frontiers = extract_frontiers(robot_x, robot_y, robot_theta, ranges_hi)
    frontiers = merge_frontiers(frontiers)
    snapped   = snap_to_frontier(frontiers, robot_pos, target_angle, max_angle_diff)
    return snapped if snapped is not None else fallback_wp


# ══════════════════════════════════════════════════════════════════════
# 旧版下层模块（启发式，保留用于对比实验）
# ══════════════════════════════════════════════════════════════════════

LSP_FRONTIER_DIM  = 7
LSP_MAX_FRONTIERS = 5
LSP_OBS_DIM       = LSP_MAX_FRONTIERS * LSP_FRONTIER_DIM + 1  # = 36


@dataclass
class _FrontierFeat:
    cos_angle:     float
    sin_angle:     float
    width_norm:    float
    goal_align:    float
    p_success:     float
    p_occupied:    float
    expected_cost: float

    def to_vec(self) -> np.ndarray:
        return np.array([
            self.cos_angle, self.sin_angle,
            self.width_norm, self.goal_align,
            self.p_success, self.p_occupied,
            self.expected_cost,
        ], dtype=np.float32)


class LowLevelFrontierModule:
    """旧版启发式 frontier 模块（保留，用于对比实验）"""

    N_BEAMS    = C.LOW_LASER_BEAMS
    FOV_DEG    = C.LASER_FOV
    MAX_RANGE  = C.LOW_LASER_RANGE
    OPEN_RATIO = 0.92
    MIN_BEAMS  = 2
    OCC_RADIUS = 1.5

    def __init__(self):
        fov_rad = np.deg2rad(self.FOV_DEG)
        self._beam_angles = np.linspace(-fov_rad/2, fov_rad/2, self.N_BEAMS)
        self._beam_step   = fov_rad / max(self.N_BEAMS - 1, 1)
        self._open_thresh = self.MAX_RANGE * self.OPEN_RATIO
        self._cur_frontier: dict = {}
        self._cur_cost:     dict = {}

    def reset(self, n_robots: int) -> None:
        self._cur_frontier = {i: None for i in range(n_robots)}
        self._cur_cost     = {i: 0.5  for i in range(n_robots)}

    def get_obs(self, robot_idx, robot_pos, robot_theta,
                goal_pos, laser_obs, n_robots):
        raw = self._detect(laser_obs, robot_theta)
        if not raw:
            self._cur_frontier[robot_idx] = None
            self._cur_cost[robot_idx]     = 1.0
            return np.zeros(LSP_OBS_DIM, dtype=np.float32)

        to_goal    = goal_pos - robot_pos
        goal_dist  = float(np.linalg.norm(to_goal))
        goal_angle = float(np.arctan2(to_goal[1], to_goal[0])) \
                     if goal_dist > 1e-6 else robot_theta
        others = [self._cur_frontier.get(j)
                  for j in range(n_robots) if j != robot_idx]

        feats = [self._feat(rel, world, w, robot_pos, goal_pos,
                            goal_dist, goal_angle, others)
                 for (rel, world, w) in raw]
        feats.sort(key=lambda f: f.expected_cost)
        feats = feats[:LSP_MAX_FRONTIERS]

        best  = feats[0]
        b_ang = np.arctan2(best.sin_angle, best.cos_angle) + robot_theta
        self._cur_frontier[robot_idx] = robot_pos + self.MAX_RANGE * np.array([
            np.cos(b_ang), np.sin(b_ang)
        ])
        self._cur_cost[robot_idx] = best.expected_cost

        parts = [f.to_vec() for f in feats]
        while len(parts) < LSP_MAX_FRONTIERS:
            parts.append(np.zeros(LSP_FRONTIER_DIM, dtype=np.float32))
        n_norm = min(len(raw), LSP_MAX_FRONTIERS) / LSP_MAX_FRONTIERS
        return np.concatenate(parts + [np.array([n_norm])]).astype(np.float32)

    def get_cost(self, robot_idx):
        return float(self._cur_cost.get(robot_idx, 0.5))

    def _detect(self, laser_obs, robot_theta):
        raw_ranges = laser_obs * self.MAX_RANGE
        open_mask  = raw_ranges > self._open_thresh
        results = []
        i = 0
        while i < self.N_BEAMS:
            if open_mask[i]:
                start = i
                while i < self.N_BEAMS and open_mask[i]:
                    i += 1
                end = i - 1
                n_open = end - start + 1
                if n_open >= self.MIN_BEAMS:
                    center_idx = (start + end) / 2.0
                    rel_angle  = float(np.interp(
                        center_idx, [0, self.N_BEAMS-1],
                        [self._beam_angles[0], self._beam_angles[-1]]
                    ))
                    world_angle = float((rel_angle + robot_theta + np.pi)
                                        % (2*np.pi) - np.pi)
                    width = n_open * self._beam_step
                    results.append((rel_angle, world_angle, width))
            else:
                i += 1
        return results

    def _feat(self, rel_angle, world_angle, width,
              robot_pos, goal_pos, goal_dist, goal_angle, others):
        fp = robot_pos + self.MAX_RANGE * np.array([
            np.cos(world_angle), np.sin(world_angle)
        ])
        diff       = abs(_adiff(world_angle, goal_angle))
        goal_align = float((np.pi - diff) / np.pi)
        d_fg       = float(np.linalg.norm(goal_pos - fp))
        dist_norm  = min(d_fg / max(goal_dist*2, 1e-3), 1.0)
        w_norm     = float(min(width / np.pi, 1.0))
        p_success  = float(np.clip(
            0.4*goal_align + 0.4*w_norm + 0.2*(1.0-dist_norm), 0.0, 1.0
        ))
        p_occ = 0.0
        for f in others:
            if f is not None and float(np.linalg.norm(fp-f)) < self.OCC_RADIUS:
                p_occ = 1.0
                break
        d_to_f = self.MAX_RANGE
        V_fail = goal_dist
        cost   = (p_success*(1-p_occ)*(d_to_f+d_fg)
                  + p_success*p_occ*(d_to_f+self.MAX_RANGE+d_fg)
                  + (1-p_success)*V_fail)
        cost_norm = float(min(cost/max(2.0*goal_dist, 1.0), 1.0))
        return _FrontierFeat(
            cos_angle=float(np.cos(rel_angle)),
            sin_angle=float(np.sin(rel_angle)),
            width_norm=w_norm, goal_align=goal_align,
            p_success=p_success, p_occupied=p_occ,
            expected_cost=cost_norm,
        )


# ══════════════════════════════════════════════════════════════════════
# 新版：基于 SLAM 地图的 LSP Frontier 模块
# ══════════════════════════════════════════════════════════════════════

# SLAM LSP 的观测维度
SLAM_PATCH_SIZE   = 16          # patch 边长（格子数），16×16
SLAM_PATCH_DIM    = SLAM_PATCH_SIZE * SLAM_PATCH_SIZE  # = 256
SLAM_SCALAR_DIM   = 4           # [goal_rel(2), sg_rel(2)]
SLAM_LSP_INPUT_DIM = SLAM_PATCH_DIM + SLAM_SCALAR_DIM  # = 260（和原论文一致）

# 给 RL 的观测：每个 frontier 的精简特征（不含 patch，patch 只给 LSP 网络用）
SLAM_FRONTIER_DIM  = 6          # [cos,sin,dist,goal_align,p_success,p_occupied]
SLAM_MAX_FRONTIERS = 5
SLAM_LSP_OBS_DIM   = SLAM_MAX_FRONTIERS * SLAM_FRONTIER_DIM + 1  # = 31


class SLAMFrontierModule:
    """
    基于 SLAM 占用栅格地图的 LSP Frontier 模块。

    和旧版的区别：
      frontier 检测：基于累积地图（free/unknown边界），不重复探索
      P_S 预测：   基于地图 patch（CNN），而非启发式规则
      历史信息：   走过的地方不再是 frontier

    在 LowLevelEnv 里用法：
      self.slam_maps = [OccupancyMap(...) for _ in range(n)]
      self.slam_lsp  = SLAMFrontierModule()

      # reset() 里
      for m in self.slam_maps: m.reset()
      self.slam_lsp.reset(n)

      # _make_obs_list() 里，每步更新地图后
      self.slam_maps[i].update(car.x, car.y, car.theta, ranges_hi)
      lsp_obs = self.slam_lsp.get_obs(
          robot_idx=i,
          robot_pos=car.pos,
          robot_theta=car.theta,
          goal_pos=self.goals[i],
          slam_map=self.slam_maps[i],
          n_robots=self.n,
      )
    """

    OCC_RADIUS    = 2.0   # 其他机器人占用 frontier 的判定半径（米）
    MAX_FRONTIERS = SLAM_MAX_FRONTIERS

    def __init__(self, predictor=None):
        """
        predictor: LSPNet 实例（可选）
          None    → 使用启发式 P_S（对比基准）
          LSPNet  → 使用学习的 P_S（论文方法）
        """
        self.predictor = predictor
        self._cur_frontier: dict = {}
        self._cur_cost:     dict = {}

    def reset(self, n_robots: int) -> None:
        self._cur_frontier = {i: None for i in range(n_robots)}
        self._cur_cost     = {i: 0.5  for i in range(n_robots)}

    def set_predictor(self, predictor) -> None:
        """训练好 LSPNet 后调用此接口切换到学习版本"""
        self.predictor = predictor

    # ── 主接口 ────────────────────────────────────────────────────────

    def get_obs(
        self,
        robot_idx:   int,
        robot_pos:   np.ndarray,
        robot_theta: float,
        goal_pos:    np.ndarray,
        slam_map,                    # OccupancyMap 实例
        n_robots:    int,
        other_maps:  Optional[List] = None,  # 其他机器人的地图（通信）
    ) -> np.ndarray:
        """
        返回 (SLAM_LSP_OBS_DIM=31,) float32
        直接拼接到 obs dict 里作为 'lsp' 键。
        """
        # 1. 从 SLAM 地图检测真实 frontier
        frontiers = slam_map.get_frontiers(
            robot_pos[0], robot_pos[1],
            max_range=C.LOW_LASER_RANGE * 2.0,  # 用更大范围检测 frontier
        )

        if len(frontiers) == 0:
            self._cur_frontier[robot_idx] = None
            self._cur_cost[robot_idx]     = 1.0
            return np.zeros(SLAM_LSP_OBS_DIM, dtype=np.float32)

        # 2. 其他机器人的当前 frontier（p_occupied）
        others = [self._cur_frontier.get(j)
                  for j in range(n_robots) if j != robot_idx]

        # 3. 对每个 frontier 计算特征
        to_goal    = goal_pos - robot_pos
        goal_dist  = float(np.linalg.norm(to_goal))
        goal_angle = float(np.arctan2(to_goal[1], to_goal[0])) \
                     if goal_dist > 1e-6 else robot_theta

        feats = []
        for fp_world in frontiers:
            feat = self._compute_frontier_feat(
                frontier_world=fp_world,
                robot_pos=robot_pos,
                robot_theta=robot_theta,
                goal_pos=goal_pos,
                goal_dist=goal_dist,
                goal_angle=goal_angle,
                slam_map=slam_map,
                others=others,
            )
            feats.append(feat)

        # 4. 按期望代价排序，取最优 MAX_FRONTIERS 个
        feats.sort(key=lambda x: x[6])   # x[6] = expected_cost
        feats = feats[:self.MAX_FRONTIERS]

        # 5. 更新当前最优 frontier
        best_pos = feats[0][7]   # frontier 世界坐标
        self._cur_frontier[robot_idx] = best_pos
        self._cur_cost[robot_idx]     = feats[0][6]

        # 6. 构造 RL 观测向量（精简特征，不含 patch）
        parts = []
        for feat in feats:
            parts.append(np.array(feat[:6], dtype=np.float32))
        while len(parts) < self.MAX_FRONTIERS:
            parts.append(np.zeros(SLAM_FRONTIER_DIM, dtype=np.float32))

        n_norm = min(len(frontiers), self.MAX_FRONTIERS) / self.MAX_FRONTIERS
        return np.concatenate(parts + [np.array([n_norm])]).astype(np.float32)

    def get_cost(self, robot_idx: int) -> float:
        return float(self._cur_cost.get(robot_idx, 0.5))

    def get_best_frontier(self, robot_idx: int) -> Optional[np.ndarray]:
        """返回最优 frontier 的世界坐标（用于 waypoint 设置）"""
        return self._cur_frontier.get(robot_idx)

    # ── Patch 提取 ────────────────────────────────────────────────────

    def get_patch(self, frontier_world: np.ndarray,
                  slam_map,
                  patch_size: int = SLAM_PATCH_SIZE) -> np.ndarray:
        """
        提取以 frontier 为中心的地图 patch。

        返回: (patch_size, patch_size) float32
          -1.0 = unknown
           0.0 = free
           1.0 = occupied
        """
        if not slam_map._initialized:
            return np.full((patch_size, patch_size), -1.0, dtype=np.float32)

        half = patch_size // 2
        fr_row, fr_col = slam_map.world_to_grid(
            float(frontier_world[0]), float(frontier_world[1])
        )

        patch = np.full((patch_size, patch_size), -1.0, dtype=np.float32)
        for dr in range(-half, half):
            for dc in range(-half, half):
                r = fr_row + dr
                c = fr_col + dc
                pr = dr + half
                pc = dc + half
                if slam_map._in_bounds(r, c):
                    val = slam_map.grid[r, c]
                    if val == 0:    # UNKNOWN
                        patch[pr, pc] = -1.0
                    elif val == 1:  # FREE
                        patch[pr, pc] =  0.0
                    else:           # OCCUPIED
                        patch[pr, pc] =  1.0

        # 归一化到 [0, 1]（-1→0, 0→0.5, 1→1）
        patch = (patch + 1.0) / 2.0
        return patch

    def get_lsp_network_input(
        self,
        frontier_world: np.ndarray,
        robot_pos:      np.ndarray,
        robot_theta:    float,
        goal_pos:       np.ndarray,
        slam_map,
        patch_size:     int = SLAM_PATCH_SIZE,
    ) -> np.ndarray:
        """
        构造 LSPNet 的输入向量（260维，和原论文一致）：
          patch (256维) + goal_rel (2维) + sg_rel (2维)

        用于：
          1. 数据收集时记录样本
          2. 推理时输入 LSPNet 预测 P_S
        """
        # 地图 patch（flatten）
        patch = self.get_patch(frontier_world, slam_map, patch_size)
        patch_flat = patch.flatten()   # (256,)

        # 坐标归一化
        norm = C.LOW_LASER_RANGE

        # goal 在机器人坐标系中的位置
        goal_rel = _world_to_robot(goal_pos, robot_pos, robot_theta) / norm

        # frontier 在机器人坐标系中的位置
        sg_rel = _world_to_robot(frontier_world, robot_pos, robot_theta) / norm

        return np.concatenate([patch_flat, goal_rel, sg_rel]).astype(np.float32)

    # ── 内部计算 ──────────────────────────────────────────────────────

    def _compute_frontier_feat(
        self,
        frontier_world: np.ndarray,
        robot_pos:      np.ndarray,
        robot_theta:    float,
        goal_pos:       np.ndarray,
        goal_dist:      float,
        goal_angle:     float,
        slam_map,
        others:         list,
    ) -> tuple:
        """
        计算单个 frontier 的完整特征。

        返回 (cos_angle, sin_angle, dist_norm, goal_align,
               p_success, p_occupied, expected_cost, frontier_world)
        """
        # 方向和距离
        diff_vec    = frontier_world - robot_pos
        world_angle = float(np.arctan2(diff_vec[1], diff_vec[0]))
        rel_angle   = _adiff(world_angle, robot_theta)
        dist        = float(np.linalg.norm(diff_vec))
        dist_norm   = min(dist / max(C.LOW_LASER_RANGE, 1.0), 1.0)

        # 与目标方向对齐度
        angle_diff = abs(_adiff(world_angle, goal_angle))
        goal_align = float((np.pi - angle_diff) / np.pi)

        # P_S 预测
        if self.predictor is not None:
            # 学习版：用 LSPNet
            import torch
            x = self.get_lsp_network_input(
                frontier_world, robot_pos, robot_theta, goal_pos, slam_map
            )
            with torch.no_grad():
                x_t = torch.FloatTensor(x).unsqueeze(0)
                ps_t, _, _ = self.predictor(x_t)
                p_success = float(ps_t[0])
        else:
            # 启发式版：和旧版一样
            d_fg      = float(np.linalg.norm(goal_pos - frontier_world))
            dnorm     = min(d_fg / max(goal_dist*2, 1e-3), 1.0)
            w_norm    = min(dist / (C.LOW_LASER_RANGE * 2), 1.0)
            p_success = float(np.clip(
                0.4*goal_align + 0.3*w_norm + 0.3*(1-dnorm), 0.0, 1.0
            ))

        # p_occupied
        p_occ = 0.0
        for f in others:
            if f is not None and \
               float(np.linalg.norm(frontier_world - f)) < self.OCC_RADIUS:
                p_occ = 1.0
                break

        # 期望代价（贝尔曼分解）
        d_fg      = float(np.linalg.norm(goal_pos - frontier_world))
        V_fail    = goal_dist
        cost      = (p_success*(1-p_occ)*(dist+d_fg)
                     + p_success*p_occ*(dist+2*C.LOW_LASER_RANGE+d_fg)
                     + (1-p_success)*V_fail)
        cost_norm = float(min(cost/max(2.0*goal_dist, 1.0), 1.0))

        return (float(np.cos(rel_angle)),  # 0: cos_angle
                float(np.sin(rel_angle)),  # 1: sin_angle
                dist_norm,                 # 2: dist_norm
                goal_align,                # 3: goal_align
                p_success,                 # 4: p_success
                p_occ,                     # 5: p_occupied
                cost_norm,                 # 6: expected_cost
                frontier_world.copy())     # 7: world pos（不进入观测向量）


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _adiff(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2*np.pi) - np.pi)


def _world_to_robot(world_pt: np.ndarray,
                    robot_pos: np.ndarray,
                    robot_theta: float) -> np.ndarray:
    """世界坐标 → 机器人坐标系"""
    delta = world_pt - robot_pos
    c, s  = np.cos(-robot_theta), np.sin(-robot_theta)
    x_r   = c*delta[0] - s*delta[1]
    y_r   = s*delta[0] + c*delta[1]
    return np.array([x_r, y_r], dtype=np.float32)
