"""
algos/birnn_td3/networks.py
──────────────────────────────────────────────────────────────────────
下层 TD3 网络

观测设计：
  静态障碍（墙+圆形障碍）→ 激光20维直接归一化 → LaserEncoder(MLP) → 64维
  动态障碍（邻居机器人）  → RVO序列(通信获取) → BiRNNEncoder(GRU) → 128维
  自身状态               → wp(2) + vel(4)                         → 6维
  ─────────────────────────────────────────────────────────────────
  拼接特征: 64 + 128 + 2 + 4 = 198 维
  → Actor MLP(256,256) → tanh → (v_cmd, δ_cmd)
  → Critic MLP(256,256) → Q值

vel 4维: (v_norm, ω_norm, cos_θ, sin_θ)
  加入朝向让网络感知阿克曼可达弧的方向
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from configs import config as C


# ══════════════════════════════════════════════════════════════════════
# 激光编码器（静态障碍感知）
# ══════════════════════════════════════════════════════════════════════

class LaserEncoder(nn.Module):
    """
    激光 20 维归一化距离 → 64 维特征。
    固定维度用 MLP，不需要 RNN。

    输入:  (B, LOW_LASER_DIM=20)  归一化距离 ∈ [0,1]
    输出:  (B, 64)
    """
    def __init__(self, laser_dim: int = C.LOW_LASER_DIM,
                 out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(laser_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, laser: torch.Tensor) -> torch.Tensor:
        return self.net(laser)


# ══════════════════════════════════════════════════════════════════════
# RVO BiRNN 编码器（动态障碍感知）
# ══════════════════════════════════════════════════════════════════════

class BiRNNEncoder(nn.Module):
    """
    处理 RVO 序列，输出固定维度特征。

    序列长度固定为 MAX_NEIGHBORS=4（不足用 padding zeros）。
    不用 pack_padded_sequence：序列短（4个），pack 的开销比收益大。
    直接全量前向，网络自己学会忽略 padding 位（padding 全0）。

    输入:
        x : (B, MAX_NEIGHBORS=4, RVO_DIM=7)
    输出:
        (B, hidden*2=128)
    """
    def __init__(self, input_dim: int = C.RVO_DIM,
                 hidden_dim: int = C.BIRNN_HIDDEN):
        super().__init__()
        self.hidden = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, M, RVO_DIM)
        lengths: 保留参数兼容旧接口，不再使用
        """
        B, M, _ = x.shape
        x_proj = self.proj(x.reshape(B * M, -1)).reshape(B, M, self.hidden)
        _, h_n = self.rnn(x_proj)
        return torch.cat([h_n[0], h_n[1]], dim=-1)   # (B, hidden*2)

    def forward_no_pack(self, x: torch.Tensor) -> torch.Tensor:
        """推理时调用，和 forward 相同（保持接口兼容）。"""
        return self.forward(x)


# ══════════════════════════════════════════════════════════════════════
# 特征提取器（Actor / Critic 共用结构，权重独立）
# ══════════════════════════════════════════════════════════════════════

class LowLevelFeatureExtractor(nn.Module):
    """
    输入（4 个分支）:
        laser : (B, 20)      归一化激光
        rvo   : (B, M, 7)    RVO 序列
        wp    : (B, 2)       waypoint 极坐标
        vel   : (B, 4)       (v_norm, ω_norm, cos_θ, sin_θ)

    输出:
        feat  : (B, LOW_FEAT_DIM=198)
    """
    def __init__(self):
        super().__init__()
        self.laser_enc = LaserEncoder()                    # 36 → 64
        self.rvo_rnn   = BiRNNEncoder(C.RVO_DIM,
                                      C.BIRNN_HIDDEN)     # 7  → 128
        # 下层简洁观测: laser + rvo + wp + vel + goal (无lsp/无ref_path)
        self.feat_dim  = C.LOW_FEAT_DIM                   # 200

    def forward(self, laser: torch.Tensor,
                rvo:   torch.Tensor,
                wp:    torch.Tensor,
                vel:   torch.Tensor,
                goal:  torch.Tensor) -> torch.Tensor:
        h_laser = self.laser_enc(laser)
        h_rvo   = self.rvo_rnn(rvo)
        return torch.cat([h_laser, h_rvo, wp, vel, goal], dim=-1)

    @torch.no_grad()
    def forward_single(self, laser_np, rvo_np, wp_np, vel_np, goal_np):
        """推理时 batch=1 快速调用。"""
        laser = torch.FloatTensor(laser_np).unsqueeze(0)
        rvo   = torch.FloatTensor(rvo_np).unsqueeze(0)
        wp    = torch.FloatTensor(wp_np).unsqueeze(0)
        vel   = torch.FloatTensor(vel_np).unsqueeze(0)
        goal  = torch.FloatTensor(goal_np).unsqueeze(0)
        h_laser = self.laser_enc(laser)
        h_rvo   = self.rvo_rnn.forward_no_pack(rvo)
        return torch.cat([h_laser, h_rvo, wp, vel, goal], dim=-1)


# ══════════════════════════════════════════════════════════════════════
# Actor
# ══════════════════════════════════════════════════════════════════════

class LowLevelActor(nn.Module):
    """
    输出 (v_cmd, δ_cmd)：
      v_cmd  ∈ [0,1]  （阿克曼不倒车，select_action 里 clip）
      δ_cmd  ∈ [-1,1]

    关键设计：v_cmd 和 δ_cmd 用独立的输出层，
    v_cmd 的 bias 初始化为 +0.5（tanh(0.5)≈0.46），
    让网络一开始就倾向于前进，避免冷启动停滞。
    δ_cmd 的 bias 初始化为 0（不偏向任何方向）。
    """
    def __init__(self, feat_dim: int = C.LOW_FEAT_DIM,
                 hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # 独立输出头
        self.v_head     = nn.Linear(hidden, 1)   # v_cmd
        self.delta_head = nn.Linear(hidden, 1)   # δ_cmd

        # v_cmd: 小权重 + 正 bias → 初始输出 tanh(0.5) ≈ 0.46
        nn.init.orthogonal_(self.v_head.weight, gain=0.01)
        nn.init.constant_(self.v_head.bias, 0.0)   # 含倒车, 不预设前进

        # δ_cmd: 小权重 + 零 bias → 初始输出接近 0（不偏向转向）
        nn.init.orthogonal_(self.delta_head.weight, gain=0.01)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.trunk(feat)
        v     = torch.tanh(self.v_head(h))      # ∈ [-1,1]，后续 clip 到 [0,1]
        delta = torch.tanh(self.delta_head(h))  # ∈ [-1,1]
        return torch.cat([v, delta], dim=-1)    # (B, 2)


# ══════════════════════════════════════════════════════════════════════
# Twin Critic
# ══════════════════════════════════════════════════════════════════════

class LowLevelCritic(nn.Module):
    """
    TD3 Twin Critic。
    输入: feat(198) + action(2) → Q值(1)
    两个独立 Q 网络，取 min 防止过估计。
    """
    def __init__(self, feat_dim: int = C.LOW_FEAT_DIM,
                 action_dim: int = C.LOW_ACTION_DIM,
                 hidden: int = 256):
        super().__init__()
        in_dim = feat_dim + action_dim

        def make_q():
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, feat: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([feat, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, feat: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([feat, action], dim=-1))