"""
envs/ackermann_robot.py
──────────────────────────────────────────────────────────────────────
阿克曼运动学模型

状态: (x, y, θ, v, δ)
  x, y  : 全局位置 (m)
  θ     : 航向角 (rad), 范围 [-π, π]
  v     : 当前线速度 (m/s)
  δ     : 当前前轮转角 (rad)

动作: (v_cmd, δ_cmd) ∈ [-1, 1]²
  v_cmd  → 缩放到 [MIN_SPEED, MAX_SPEED]
  δ_cmd  → 缩放到 [-MAX_STEER, MAX_STEER]

运动学方程（离散 Euler 积分，步长 DT_LOW）:
  x'  = x + v·cos(θ)·dt
  y'  = y + v·sin(θ)·dt
  θ'  = θ + (v/L)·tan(δ)·dt
  ω   = v·tan(δ)/L   （当前角速度，导出量）
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from configs import config as C


def _wrap(angle: float) -> float:
    """将角度归一化到 [-π, π]"""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class AckermannRobot:
    x:      float
    y:      float
    theta:  float = 0.0
    v:      float = 0.0
    delta:  float = 0.0
    radius: float = field(default_factory=lambda: C.ROBOT_RADIUS)

    # ── 运动学步进 ───────────────────────────────────────────────────
    def step(self, v_cmd: float, delta_cmd: float) -> None:
        """
        v_cmd, delta_cmd ∈ [-1, 1]
        内部缩放后更新状态，步长 DT_LOW。
        """
        # 缩放：v_cmd ∈ [-1,1] 映射到 [-MAX_SPEED, MAX_SPEED]（含倒车）
        v     = float(np.clip(v_cmd,    -1.0,  1.0)) * C.MAX_SPEED
        delta = float(np.clip(delta_cmd, -1.0, 1.0)) * C.MAX_STEER

        dt = C.DT_LOW
        self.x     += v * np.cos(self.theta) * dt
        self.y     += v * np.sin(self.theta) * dt
        self.theta  = _wrap(self.theta + (v / C.WHEELBASE) * np.tan(delta) * dt)
        self.v      = v
        self.delta  = delta

    # ── 导出量 ──────────────────────────────────────────────────────
    @property
    def omega(self) -> float:
        """角速度 ω = v·tan(δ)/L (rad/s)"""
        return float(self.v * np.tan(self.delta) / C.WHEELBASE)

    @property
    def vx(self) -> float:
        return float(self.v * np.cos(self.theta))

    @property
    def vy(self) -> float:
        return float(self.v * np.sin(self.theta))

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    @property
    def vel_vec(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=np.float32)

    # ── 归一化状态（供观测使用）──────────────────────────────────────
    @property
    def obs_state(self) -> np.ndarray:
        """
        5 维归一化自身状态：
        [cos(θ), sin(θ), v/MAX_SPEED, δ/MAX_STEER, ω/MAX_OMEGA]
        全部 ∈ [-1, 1]
        """
        return np.array([
            np.cos(self.theta),
            np.sin(self.theta),
            self.v / C.MAX_SPEED,
            self.delta / C.MAX_STEER,
            np.clip(self.omega / C.MAX_OMEGA, -1.0, 1.0),
        ], dtype=np.float32)

    # ── 通信消息（发给邻居）──────────────────────────────────────────
    def comm_msg(self, observer_pos: np.ndarray,
                 goal_pos: np.ndarray,
                 last_action: np.ndarray) -> np.ndarray:
        """
        生成发给位于 observer_pos 的邻居的 9 维通信消息：
          [Δx_n, Δy_n, dist_n,          ← 相对位置（以观察者为原点）
           goal_cos, goal_sin,           ← 自己的目标方向
           last_v_n, last_δ_n,           ← 上一步动作（归一化）
           v_n, ttc_n]                   ← 自身速度 + 时间到碰撞估计
        全部 ∈ [-1, 1]
        """
        delta_xy = self.pos - observer_pos
        dist     = float(np.linalg.norm(delta_xy))
        dist_clip = min(dist, C.COMM_RANGE)

        if dist < 1e-6:
            dx_n, dy_n = 0.0, 0.0
        else:
            dx_n = float(delta_xy[0] / dist)
            dy_n = float(delta_xy[1] / dist)

        # 自己到目标的方向（全局）
        to_goal  = goal_pos - self.pos
        g_dist   = float(np.linalg.norm(to_goal))
        if g_dist < 1e-6:
            goal_cos, goal_sin = 1.0, 0.0
        else:
            goal_cos = float(to_goal[0] / g_dist)
            goal_sin = float(to_goal[1] / g_dist)

        # 时间到碰撞（简化估计：dist / max(relative_closing_speed, ε)）
        ttc_raw  = dist_clip / max(self.v + 1e-3, 1e-3)
        ttc_norm = float(np.clip(ttc_raw / C.VO_TIME_HORIZON, 0.0, 1.0))

        return np.array([
            dx_n,
            dy_n,
            dist_clip / C.COMM_RANGE,
            goal_cos,
            goal_sin,
            float(last_action[0]),   # v_cmd（已归一化）
            float(last_action[1]),   # δ_cmd（已归一化）
            self.v / C.MAX_SPEED,
            ttc_norm,
        ], dtype=np.float32)
