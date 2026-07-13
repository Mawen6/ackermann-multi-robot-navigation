"""
algos/vo_rvo.py
──────────────────────────────────────────────────────────────────────
VO（Velocity Obstacle）和 RVO（Reciprocal VO）几何计算

参考: rl_rvo_nav (RA-Letter 2022)

核心概念:
  VO(A|B) = 使机器人 A 在时间窗口 T 内与障碍物 B 碰撞的速度集合
  RVO(A|B) = VO(A|B) 平移 (v_A + v_B)/2（互惠：双方各承担一半）

输出向量格式:
  VO  向量 (5维): [cos_α, sin_α, dist_norm, radius_norm, ttc_norm]
  RVO 向量 (7维): [cos_α, sin_α, dist_norm, radius_norm, ttc_norm,
                   rel_vx_norm, rel_vy_norm]

全部归一化到 [-1, 1]，供 BiRNN 直接处理。
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from configs import config as C


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

class CircleObstacle:
    """圆形障碍物（静态或动态）"""
    def __init__(self, x: float, y: float, r: float):
        self.x = x
        self.y = y
        self.r = r

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# VO 计算（静态障碍物）
# ══════════════════════════════════════════════════════════════════════

def compute_vo_vector(robot_pos: np.ndarray,
                      robot_radius: float,
                      obs: CircleObstacle,
                      time_horizon: float = C.VO_TIME_HORIZON) -> np.ndarray:
    """
    计算机器人相对于静态圆形障碍物的 VO 向量（5维）。

    VO 的几何含义：
      从机器人位置出发，VO 是一个圆锥形速度禁区。
      锥顶在原点，轴线指向障碍物，半角 = arcsin(combined_r / dist)。

    输出（归一化）:
      [cos_α, sin_α,        ← VO 轴线方向（指向障碍物）
       dist_norm,           ← 中心距离 / OBS_RANGE
       radius_norm,         ← 组合半径 / OBS_RANGE
       ttc_norm]            ← 估计碰撞时间 / time_horizon

    若障碍物超出感知范围，返回 None（由调用方过滤）。
    """
    delta   = obs.pos - robot_pos
    dist    = float(np.linalg.norm(delta))
    comb_r  = robot_radius + obs.r
    surf    = max(dist - comb_r, 0.0)

    # 超出感知范围
    if surf > C.LOW_LASER_RANGE:
        return None

    if dist < 1e-6:
        cos_a, sin_a = 1.0, 0.0
    else:
        cos_a = float(delta[0] / dist)
        sin_a = float(delta[1] / dist)

    # 到碰撞的估计时间（机器人以最大速度直冲障碍）
    ttc_raw  = max(surf, 0.0) / max(C.MAX_SPEED, 1e-3)
    ttc_norm = float(np.clip(ttc_raw / time_horizon, 0.0, 1.0))

    return np.array([
        cos_a,
        sin_a,
        min(dist,   C.LOW_LASER_RANGE) / C.LOW_LASER_RANGE,
        min(comb_r, C.LOW_LASER_RANGE) / C.LOW_LASER_RANGE,
        ttc_norm,
    ], dtype=np.float32)


def compute_all_vo(robot_pos: np.ndarray,
                   robot_radius: float,
                   obstacles: List[CircleObstacle]) -> np.ndarray:
    """
    计算所有静态障碍物的 VO 向量，按距离排序，返回 (K, 5) 数组。
    K ≤ MAX_VO_OBSTACLES，不足时 padding 为 0（ttc_norm=1 表示"安全"）。
    """
    vecs = []
    for obs in obstacles:
        v = compute_vo_vector(robot_pos, robot_radius, obs)
        if v is not None:
            surf = max(float(np.linalg.norm(obs.pos - robot_pos))
                       - robot_radius - obs.r, 0.0)
            vecs.append((surf, v))

    # 按距离升序
    vecs.sort(key=lambda x: x[0])
    K = C.MAX_VO_OBSTACLES

    result = np.zeros((K, C.VO_DIM), dtype=np.float32)
    result[:, 4] = 1.0   # ttc_norm=1 padding（远处安全）
    for i, (_, v) in enumerate(vecs[:K]):
        result[i] = v

    return result   # (K, 5)


# ══════════════════════════════════════════════════════════════════════
# RVO 计算（动态邻居机器人）
# ══════════════════════════════════════════════════════════════════════

def compute_rvo_vector(ego_pos: np.ndarray,
                       ego_vel: np.ndarray,
                       ego_radius: float,
                       nbr_pos: np.ndarray,
                       nbr_vel: np.ndarray,
                       nbr_radius: float,
                       time_horizon: float = C.VO_TIME_HORIZON) -> np.ndarray:
    """
    计算相对于邻居机器人的 RVO 向量（7维）。

    RVO 修正：VO 轴线从 nbr_pos 平移到 nbr_pos + (ego_vel + nbr_vel)/2
    → 互惠：双方各承担一半避让责任

    输出（归一化）:
      [cos_α, sin_α,        ← 相对位置方向
       dist_norm,           ← 中心距 / COMM_RANGE
       radius_norm,         ← 组合半径 / COMM_RANGE
       ttc_norm,            ← 估计碰撞时间 / time_horizon
       rel_vx_norm,         ← 相对速度 x / MAX_SPEED
       rel_vy_norm]         ← 相对速度 y / MAX_SPEED
    """
    delta    = nbr_pos - ego_pos
    dist     = float(np.linalg.norm(delta))
    comb_r   = ego_radius + nbr_radius

    if dist < 1e-6:
        cos_a, sin_a = 1.0, 0.0
    else:
        cos_a = float(delta[0] / dist)
        sin_a = float(delta[1] / dist)

    # 相对速度（RVO 修正：减去平均速度）
    rel_vel  = ego_vel - nbr_vel   # 未经 RVO 修正的相对速度
    # RVO 轴线：从 ego 看 nbr，在速度空间中的参考点 = (v_ego + v_nbr)/2
    # 这里将相对速度用于 ttc 估计
    closing  = float(np.dot(rel_vel, delta / max(dist, 1e-6)))  # 接近速度（正=靠近）
    ttc_raw  = (max(dist - comb_r, 0.0) / max(closing, 1e-3)) if closing > 0 else time_horizon
    ttc_norm = float(np.clip(ttc_raw / time_horizon, 0.0, 1.0))

    # 相对速度归一化
    rel_vx_n = float(np.clip(rel_vel[0] / C.MAX_SPEED, -1.0, 1.0))
    rel_vy_n = float(np.clip(rel_vel[1] / C.MAX_SPEED, -1.0, 1.0))

    return np.array([
        cos_a,
        sin_a,
        min(dist,   C.COMM_RANGE) / C.COMM_RANGE,
        min(comb_r, C.COMM_RANGE) / C.COMM_RANGE,
        ttc_norm,
        rel_vx_n,
        rel_vy_n,
    ], dtype=np.float32)


def compute_all_rvo(ego_pos: np.ndarray,
                    ego_vel: np.ndarray,
                    ego_radius: float,
                    neighbors: List[dict]) -> np.ndarray:
    """
    计算所有邻居的 RVO 向量，按距离排序，返回 (M, 7) 数组。
    M ≤ MAX_NEIGHBORS，不足时 padding（ttc_norm=1）。

    neighbors: List[{pos, vel, radius}]
    """
    vecs = []
    for nbr in neighbors:
        dist = float(np.linalg.norm(nbr['pos'] - ego_pos))
        if dist > C.COMM_RANGE:
            continue
        v = compute_rvo_vector(
            ego_pos, ego_vel, ego_radius,
            nbr['pos'], nbr['vel'], nbr['radius']
        )
        vecs.append((dist, v))

    vecs.sort(key=lambda x: x[0])
    M = C.MAX_NEIGHBORS

    result = np.zeros((M, C.RVO_DIM), dtype=np.float32)
    result[:, 4] = 1.0   # ttc_norm=1 padding
    for i, (_, v) in enumerate(vecs[:M]):
        result[i] = v

    return result   # (M, 7)


# ══════════════════════════════════════════════════════════════════════
# VO/RVO 奖励计算（供下层 reward shaping 使用）
# ══════════════════════════════════════════════════════════════════════

def rvo_reward(vo_vecs: np.ndarray,
               rvo_vecs: np.ndarray,
               action: Optional[np.ndarray] = None) -> float:
    """
    RVO danger-field reward.

    Desired behavior:
      - entering RVO danger: uncomfortable penalty
      - deep inside RVO: strong penalty
      - steering away from the dangerous side: reward
      - fully clear: tiny comfort reward

    rvo_vec fields:
      [cos(angle), sin(angle), dist_norm, radius_norm, ttc_norm, rel_vx, rel_vy]
    """
    reward = 0.0
    active = []

    for v in rvo_vecs:
        ttc = float(v[4])
        dist_n = float(v[2])
        rad_n = float(v[3])

        # padding / no danger
        if ttc >= 0.99 and dist_n <= 1e-6:
            continue
        if ttc >= 0.99:
            continue

        side = float(v[1])
        ttc = float(np.clip(ttc, 0.0, 1.0))

        # danger: 0=safe, 1=very dangerous
        danger = float(np.exp(-C.RVO_DANGER_DECAY * ttc))
        active.append((danger, ttc, side, dist_n, rad_n))

        # entering / staying near RVO
        if ttc < C.RVO_TTC_ENTER_THRESH:
            enter = (C.RVO_TTC_ENTER_THRESH - ttc) / max(C.RVO_TTC_ENTER_THRESH, 1e-6)
            reward += C.LOW_REW_RVO_ENTER * enter

        # deep inside RVO
        if ttc < C.RVO_TTC_DEEP_THRESH:
            deep = (C.RVO_TTC_DEEP_THRESH - ttc) / max(C.RVO_TTC_DEEP_THRESH, 1e-6)
            reward += C.LOW_REW_RVO_DEEP * (deep ** 2)

        # geometric cone pressure: closer and wider cones are worse
        if dist_n > 1e-3:
            cone_pressure = min(rad_n / dist_n, 1.0) ** 2
            reward += C.LOW_REW_RVO_AREA * cone_pressure

    # Steering-away reward.
    # Neighbor on left (side > 0) -> reward right steering (action[1] < 0).
    # Neighbor on right (side < 0) -> reward left steering (action[1] > 0).
    if action is not None and len(action) >= 2:
        steer = float(np.clip(action[1], -1.0, 1.0))
        steer_reward = 0.0
        for danger, ttc, side, _, _ in active:
            if abs(side) < 0.05:
                continue
            desired_turn = -np.sign(side)
            steer_reward += danger * desired_turn * steer
        reward += C.LOW_REW_RVO_STEER * steer_reward

    if not active:
        reward += C.LOW_REW_RVO_CLEAR

    return float(reward)


# ===== FULL RVO CONE ABLATION PATCH =====
# This patch keeps the original compact RVO implementation as default.
# When C.RVO_OBS_MODE == "cone", each neighbor is represented by:
#   c = [v_apex_x, v_apex_y, v_left_x, v_left_y, v_right_x, v_right_y]
# where v_apex is the RVO cone apex and v_left/v_right are unit boundary
# directions in velocity space.

_ORIG_compute_all_rvo = compute_all_rvo
_ORIG_rvo_reward = rvo_reward


def _rvo_get(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _rvo_norm(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32), 0.0
    return (v / n).astype(np.float32), n


def _rvo_rot(v, ang):
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)


def _rvo_cone_feature(self_pos, self_vel, self_radius, nbr):
    p_i = np.asarray(self_pos, dtype=np.float32)
    v_i = np.asarray(self_vel, dtype=np.float32)

    p_j = np.asarray(_rvo_get(nbr, "pos", _rvo_get(nbr, "position", np.zeros(2))), dtype=np.float32)
    v_j = np.asarray(_rvo_get(nbr, "vel", _rvo_get(nbr, "velocity", np.zeros(2))), dtype=np.float32)
    r_j = float(_rvo_get(nbr, "radius", getattr(C, "ROBOT_RADIUS", self_radius)))

    p_ij = p_j - p_i
    axis, d = _rvo_norm(p_ij)

    combined_r = float(self_radius) + r_j

    if d < 1e-6:
        axis = np.array([1.0, 0.0], dtype=np.float32)
        half_angle = np.pi / 2.0
    else:
        half_angle = float(np.arcsin(np.clip(combined_r / max(d, combined_r + 1e-6), 0.0, 0.99)))

    v_left = _rvo_rot(axis, half_angle)
    v_right = _rvo_rot(axis, -half_angle)

    # Standard reciprocal velocity obstacle apex. This is the midpoint of the
    # two agents' velocities in velocity space.
    v_apex = 0.5 * (v_i + v_j)

    vmax = max(float(getattr(C, "MAX_SPEED", 1.0)), 1e-6)
    feat = np.array([
        v_apex[0] / vmax,
        v_apex[1] / vmax,
        v_left[0],
        v_left[1],
        v_right[0],
        v_right[1],
    ], dtype=np.float32)

    return feat


def compute_all_rvo(pos, vel, radius, nbr_info):
    if getattr(C, "RVO_OBS_MODE", "compact") != "cone":
        return _ORIG_compute_all_rvo(pos, vel, radius, nbr_info)

    # Replay buffer requires fixed-shape RVO observations.
    max_n = int(getattr(
        C,
        "MAX_RVO_NEIGHBORS",
        getattr(C, "RVO_MAX_NEIGHBORS", getattr(C, "MAX_NEIGHBORS", 5))
    ))

    feats = []
    for nbr in nbr_info:
        feats.append(_rvo_cone_feature(pos, vel, radius, nbr))

    out = np.zeros((max_n, 6), dtype=np.float32)

    if len(feats) > 0:
        arr = np.asarray(feats, dtype=np.float32)
        k = min(max_n, arr.shape[0])
        out[:k, :] = arr[:k, :]

    return out


def _inside_rvo_cone(v_action, cone):
    apex = np.asarray(cone[0:2], dtype=np.float32) * max(float(getattr(C, "MAX_SPEED", 1.0)), 1e-6)
    vl = np.asarray(cone[2:4], dtype=np.float32)
    vr = np.asarray(cone[4:6], dtype=np.float32)

    vl, _ = _rvo_norm(vl)
    vr, _ = _rvo_norm(vr)

    axis = vl + vr
    axis, axis_norm = _rvo_norm(axis)
    if axis_norm < 1e-6:
        return False, 1.0

    w = np.asarray(v_action, dtype=np.float32) - apex
    w_dir, w_norm = _rvo_norm(w)
    if w_norm < 1e-6:
        return True, -1.0

    # If velocity is behind the apex relative to the cone axis, it is outside.
    if float(np.dot(w_dir, axis)) <= 0.0:
        return False, 1.0

    half = float(np.arccos(np.clip(np.dot(axis, vl), -1.0, 1.0)))
    ang = float(np.arccos(np.clip(np.dot(axis, w_dir), -1.0, 1.0)))

    margin = ang - half
    inside = margin < 0.0
    return inside, margin


def rvo_reward(vo_vecs, rvo_vecs, action=None, car_theta=None, **kwargs):
    if getattr(C, "RVO_OBS_MODE", "compact") != "cone":
        try:
            return _ORIG_rvo_reward(vo_vecs, rvo_vecs, action=action)
        except TypeError:
            return _ORIG_rvo_reward(vo_vecs, rvo_vecs)

    if rvo_vecs is None or len(rvo_vecs) == 0:
        return 0.0

    # If action/theta is unavailable, only return a mild risk prior.
    if action is None or car_theta is None:
        return 0.0

    speed = float(action[0]) * float(getattr(C, "MAX_SPEED", 1.0))
    v_action = np.array([
        speed * np.cos(float(car_theta)),
        speed * np.sin(float(car_theta)),
    ], dtype=np.float32)

    reward = 0.0

    for cone in np.asarray(rvo_vecs, dtype=np.float32):
        if cone.shape[0] < 6:
            continue

        # Ignore padded empty cone rows. Without this, N=1 or missing-neighbor
        # cases receive artificial RVO clear rewards from all-zero padding.
        if float(np.linalg.norm(cone)) < 1e-6:
            continue

        inside, margin = _inside_rvo_cone(v_action, cone)

        if inside:
            depth = float(max(0.0, -margin))
            reward += float(getattr(C, "LOW_REW_RVO_CONE_IN", -1.2))
            reward += float(getattr(C, "LOW_REW_RVO_CONE_DEEP", -2.5)) * min(depth, 1.5) ** 2
        else:
            clear = float(np.clip(margin, 0.0, 1.0))
            reward += float(getattr(C, "LOW_REW_RVO_CONE_CLEAR", 0.15)) * clear

    return float(reward)
