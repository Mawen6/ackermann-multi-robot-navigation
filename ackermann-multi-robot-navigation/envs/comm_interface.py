"""
envs/comm_interface.py
──────────────────────────────────────────────────────────────────────
通信抽象层：隔离仿真和 ROS 部署的差异

训练时：SimCommInterface 直接读取仿真状态
部署时：RosCommInterface 订阅 ROS topic（接口相同）

邻居消息格式（9维，见 ackermann_robot.py comm_msg()）:
  [Δx_n, Δy_n, dist_n, goal_cos, goal_sin,
   last_v_cmd, last_δ_cmd, v_norm, ttc_norm]
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional
from configs import config as C


# ══════════════════════════════════════════════════════════════════════
# 基类接口
# ══════════════════════════════════════════════════════════════════════

class CommInterface:
    """
    通信接口基类。子类实现 get_neighbors()。
    """

    def get_neighbors(self, robot_id: int,
                      ego_pos: np.ndarray) -> tuple:
        """
        获取 robot_id 的通信邻居信息。

        返回:
            msgs      : np.ndarray (MAX_NEIGHBORS, NEIGHBOR_MSG_DIM)
            mask      : np.ndarray (MAX_NEIGHBORS,) bool, True=padding
            nbr_info  : List[dict] {pos, vel, radius}，用于 RVO 计算
        """
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════
# 仿真通信接口
# ══════════════════════════════════════════════════════════════════════

class SimCommInterface(CommInterface):
    """
    仿真环境通信接口。
    直接从仿真状态读取邻居信息，无通信延迟。

    在每个仿真步骤开始前调用 update_state() 注入当前所有机器人状态。
    """

    def __init__(self, comm_range: float = C.COMM_RANGE,
                 max_neighbors: int = C.MAX_NEIGHBORS):
        self.comm_range   = comm_range
        self.max_neighbors = max_neighbors
        # 当前所有机器人状态，由 env.step() 在每步更新
        self._all_states: Dict[int, dict] = {}   # {robot_id: state_dict}

    def update_state(self, all_states: Dict[int, dict]) -> None:
        """
        注入所有机器人当前状态。
        state_dict 格式:
          {
            'pos': np.ndarray(2,),
            'vel': np.ndarray(2,),     # (vx, vy)
            'radius': float,
            'goal': np.ndarray(2,),
            'last_action': np.ndarray(2,),  # (v_cmd, δ_cmd)
          }
        """
        self._all_states = all_states

    def get_neighbors(self, robot_id: int,
                      ego_pos: np.ndarray) -> tuple:
        """
        返回在通信范围内的邻居消息，按距离升序，最多 MAX_NEIGHBORS 个。
        """
        K    = self.max_neighbors
        msgs = np.zeros((K, C.NEIGHBOR_MSG_DIM), dtype=np.float32)
        mask = np.ones(K, dtype=bool)   # True = padding
        nbr_info = []

        candidates = []
        for jid, state in self._all_states.items():
            if jid == robot_id:
                continue
            dist = float(np.linalg.norm(state['pos'] - ego_pos))
            if dist <= self.comm_range:
                candidates.append((dist, jid, state))

        candidates.sort(key=lambda x: x[0])

        for i, (_, jid, state) in enumerate(candidates[:K]):
            # 生成 9 维通信消息
            delta_xy = state['pos'] - ego_pos
            dist     = float(np.linalg.norm(delta_xy))
            dc       = min(dist, self.comm_range)
            if dist < 1e-6:
                dx_n, dy_n = 0.0, 0.0
            else:
                dx_n = float(delta_xy[0] / dist)
                dy_n = float(delta_xy[1] / dist)

            to_goal = state['goal'] - state['pos']
            gd = float(np.linalg.norm(to_goal))
            gc = float(to_goal[0]/gd) if gd > 1e-6 else 1.0
            gs = float(to_goal[1]/gd) if gd > 1e-6 else 0.0

            v_spd = float(np.linalg.norm(state['vel']))
            ttc_r = dc / max(v_spd + 1e-3, 1e-3)
            ttc_n = float(np.clip(ttc_r / C.VO_TIME_HORIZON, 0.0, 1.0))

            la = state['last_action']
            msgs[i] = np.array([
                dx_n, dy_n,
                dc / self.comm_range,
                gc, gs,
                float(la[0]), float(la[1]),
                float(np.clip(v_spd / C.MAX_SPEED, 0, 1)),
                ttc_n,
            ], dtype=np.float32)
            mask[i] = False   # 有效消息

            nbr_info.append({
                'pos':    state['pos'].copy(),
                'vel':    state['vel'].copy(),
                'radius': state['radius'],
            })

        return msgs, mask, nbr_info
