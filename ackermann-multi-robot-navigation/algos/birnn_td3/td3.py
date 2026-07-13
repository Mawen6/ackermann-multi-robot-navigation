"""
algos/birnn_td3/td3.py
──────────────────────────────────────────────────────────────────────
下层 TD3 训练逻辑

obs 改动：
  'vo'    → 'laser'  (20维归一化激光，固定大小)
  'rvo'   → 'rvo'    (可变长RVO序列，保留)
  'vel'   → 'vel'    (4维，加了cos_θ, sin_θ)
  vo_len 已删除（激光固定维度不需要length）
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from collections import deque
import random

from algos.birnn_td3.networks import (
    LowLevelFeatureExtractor, LowLevelActor, LowLevelCritic
)
from configs import config as C


# ══════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════

class LowReplayBuffer:
    """
    存储每条 transition:
      laser(20), rvo(M,7), wp(2), vel(4), action(2),
      reward, next_laser, next_rvo, next_wp, next_vel, done
    """

    def __init__(self, capacity: int = C.TD3_BUFFER_SIZE):
        self.capacity = capacity
        self.buf: deque = deque(maxlen=capacity)
        self.total_transition = 0
        self.collision_transition = 0

    def push(self, trans: dict) -> None:
        """
        trans keys:
          laser, rvo, wp, vel, goal         当前 obs
          action, reward, done
          next_laser, next_rvo, next_wp, next_vel, next_goal
        """
        self.buf.append(trans)

    def sample(self, batch_size: int) -> dict:
        batch = random.sample(self.buf, batch_size)

        # 激光：固定 20 维
        laser  = torch.FloatTensor(np.array([b['laser']      for b in batch]))
        nlaser = torch.FloatTensor(np.array([b['next_laser'] for b in batch]))

        # RVO：固定 (MAX_NEIGHBORS, RVO_DIM)，padding 已在 env 里完成
        rvo  = torch.FloatTensor(np.array([b['rvo']      for b in batch]))
        nrvo = torch.FloatTensor(np.array([b['next_rvo'] for b in batch]))

        wp   = torch.FloatTensor(np.array([b['wp']       for b in batch]))
        vel  = torch.FloatTensor(np.array([b['vel']      for b in batch]))
        nwp  = torch.FloatTensor(np.array([b['next_wp']  for b in batch]))
        nvel = torch.FloatTensor(np.array([b['next_vel'] for b in batch]))
        goal  = torch.FloatTensor(np.array([b['goal']      for b in batch]))
        ngoal = torch.FloatTensor(np.array([b['next_goal'] for b in batch]))
        act  = torch.FloatTensor(np.array([b['action']   for b in batch]))
        rew  = torch.FloatTensor(np.array([b['reward']   for b in batch])).unsqueeze(1)
        done = torch.FloatTensor(np.array([float(b['done']) for b in batch])).unsqueeze(1)

        return {
            'laser': laser, 'next_laser': nlaser,
            'rvo':   rvo,   'next_rvo':   nrvo,
            'wp': wp, 'vel': vel,
            'next_wp': nwp, 'next_vel': nvel,
            'goal': goal,   'next_goal': ngoal,
            'action': act, 'reward': rew, 'done': done,
        }

    def __len__(self) -> int:
        return len(self.buf)


# ══════════════════════════════════════════════════════════════════════
# TD3 Agent
# ══════════════════════════════════════════════════════════════════════

class LowLevelTD3:
    """
    下层 TD3（参数共享：多机训练时所有机器人共用一个实例）。
    actor_feat 和 critic_feat 独立，不共享优化器，防止梯度冲突。
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

        # Actor 侧
        self.actor_feat  = LowLevelFeatureExtractor().to(self.device)
        self.actor       = LowLevelActor(self.actor_feat.feat_dim).to(self.device)

        # Critic 侧（独立 feat，不共享）
        self.critic_feat = LowLevelFeatureExtractor().to(self.device)
        self.critic      = LowLevelCritic(self.critic_feat.feat_dim).to(self.device)

        # Target 网络
        self.actor_feat_tgt  = LowLevelFeatureExtractor().to(self.device)
        self.actor_tgt       = LowLevelActor(self.actor_feat.feat_dim).to(self.device)
        self.critic_feat_tgt = LowLevelFeatureExtractor().to(self.device)
        self.critic_tgt      = LowLevelCritic(self.critic_feat.feat_dim).to(self.device)
        self._hard_copy()

        self.actor_opt = torch.optim.Adam(
            list(self.actor_feat.parameters()) + list(self.actor.parameters()),
            lr=C.TD3_ACTOR_LR
        )
        self.critic_opt = torch.optim.Adam(
            list(self.critic_feat.parameters()) + list(self.critic.parameters()),
            lr=C.TD3_CRITIC_LR
        )

        self.buffer      = LowReplayBuffer()
        self.total_steps = 0
        self.n_updates   = 0
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        self.last_q = 0.0

    # ── 推理 ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, obs: dict,
                      noise_std: float = 0.0) -> np.ndarray:
        """
        obs: {'laser':(20,), 'rvo':(M,7), 'wp':(2,), 'vel':(4,)}
        返回 (v_cmd, δ_cmd): v_cmd ∈ [0,1], δ_cmd ∈ [-1,1]
        """
        feat = self.actor_feat.forward_single(
            obs['laser'], obs['rvo'], obs['wp'], obs['vel'], obs['goal'],
        ).to(self.device)
        action = self.actor(feat).squeeze(0).cpu().numpy()
        # 含倒车: v_cmd ∈ [-1,1] 不再 clip 成 ≥0
        if noise_std > 0:
            action[0] = float(np.clip(
                action[0] + np.random.normal(0, noise_std), -1.0, 1.0
            ))
            action[1] = float(np.clip(
                action[1] + np.random.normal(0, noise_std * 2.0), -1.0, 1.0
            ))
        return action.astype(np.float32)

    # ── 存储 ─────────────────────────────────────────────────────────
    def store(self, trans: dict) -> None:
        self.buffer.push(trans)
        self.total_steps += 1

    # ── 训练 ─────────────────────────────────────────────────────────
    def train_step(self) -> Optional[dict]:
        if len(self.buffer) < C.TD3_LEARNING_STARTS:
            return None

        batch = self.buffer.sample(C.TD3_BATCH_SIZE)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        # ── Critic 更新 ──────────────────────────────────────────────
        with torch.no_grad():
            tgt_feat = self.actor_feat_tgt.forward(
                batch['next_laser'], batch['next_rvo'],
                batch['next_wp'],    batch['next_vel'], batch['next_goal'],
            )
            tgt_a = self.actor_tgt(tgt_feat)
            tgt_a = torch.cat([tgt_a[:, :1].clamp(-1, 1),
                                tgt_a[:, 1:].clamp(-1, 1)], dim=1)
            noise = (torch.randn_like(tgt_a) * C.TD3_TARGET_NOISE
                     ).clamp(-C.TD3_NOISE_CLIP, C.TD3_NOISE_CLIP)
            tgt_a = torch.cat([
                (tgt_a[:, :1] + noise[:, :1]).clamp(-1.0, 1.0),
                (tgt_a[:, 1:] + noise[:, 1:]).clamp(-1.0, 1.0),
            ], dim=1)

            tgt_cfeat = self.critic_feat_tgt.forward(
                batch['next_laser'], batch['next_rvo'],
                batch['next_wp'],    batch['next_vel'], batch['next_goal'],
            )
            q1t, q2t = self.critic_tgt(tgt_cfeat, tgt_a)
            q_tgt = batch['reward'] + C.TD3_GAMMA * (1 - batch['done']) * torch.min(q1t, q2t)

        cur_cfeat = self.critic_feat.forward(
            batch['laser'], batch['rvo'],
            batch['wp'],    batch['vel'], batch['goal'],
        )
        q1, q2 = self.critic(cur_cfeat, batch['action'])
        critic_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_feat.parameters()) + list(self.critic.parameters()), 1.0
        )
        self.critic_opt.step()

        stats = {
            'critic_loss': float(critic_loss.detach()),
            'Q_mean': float(torch.min(q1, q2).mean().detach()),
        }

        # ── Actor 延迟更新 ───────────────────────────────────────────
        self.n_updates += 1
        if self.n_updates % C.TD3_POLICY_DELAY == 0:
            cur_afeat = self.actor_feat.forward(
                batch['laser'], batch['rvo'],
                batch['wp'],    batch['vel'], batch['goal'],
            )
            a_pred = self.actor(cur_afeat)
            a_pred = torch.cat([a_pred[:, :1].clamp(-1, 1),
                                 a_pred[:, 1:]], dim=1)
            with torch.no_grad():
                eval_cfeat = self.critic_feat.forward(
                    batch['laser'], batch['rvo'],
                    batch['wp'],    batch['vel'], batch['goal'],
                )
            actor_loss = -self.critic.q1_only(eval_cfeat, a_pred).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_feat.parameters()) + list(self.actor.parameters()), 1.0
            )
            self.actor_opt.step()
            self._soft_update()
            stats['actor_loss'] = float(actor_loss.detach())
        if 'actor_loss' in locals():
            self.last_actor_loss = float(actor_loss.item())
        self.last_critic_loss = float(critic_loss.item())

        self.last_q = float(q1.mean().item())
        return stats

    # ── 网络复制 ─────────────────────────────────────────────────────
    def _hard_copy(self) -> None:
        self.actor_feat_tgt.load_state_dict(self.actor_feat.state_dict())
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_feat_tgt.load_state_dict(self.critic_feat.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

    def _soft_update(self) -> None:
        tau = C.TD3_TAU
        for p, pt in zip(self.actor_feat.parameters(), self.actor_feat_tgt.parameters()):
            pt.data.copy_(tau * p.data + (1-tau) * pt.data)
        for p, pt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
            pt.data.copy_(tau * p.data + (1-tau) * pt.data)
        for p, pt in zip(self.critic_feat.parameters(), self.critic_feat_tgt.parameters()):
            pt.data.copy_(tau * p.data + (1-tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.copy_(tau * p.data + (1-tau) * pt.data)

    # ── 存档/恢复 ─────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor_feat':      self.actor_feat.state_dict(),
            'actor':           self.actor.state_dict(),
            'critic_feat':     self.critic_feat.state_dict(),
            'critic':          self.critic.state_dict(),
            'actor_feat_tgt':  self.actor_feat_tgt.state_dict(),
            'actor_tgt':       self.actor_tgt.state_dict(),
            'critic_feat_tgt': self.critic_feat_tgt.state_dict(),
            'critic_tgt':      self.critic_tgt.state_dict(),
            'total_steps':     self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)

        def _load(module, key):
            """
            兼容旧 checkpoint 的加载：
              1. 形状完全匹配 → 直接复制
              2. 新增键（如 lsp_enc）→ 保持随机初始化
              3. 形状不匹配（如 trunk.0.weight 因 feat_dim 变大）
                 → 把旧权重填入新权重的前 N 列，其余保持随机初始化
                 → 这样旧的避碰能力完全保留，LSP 部分从零学起
            """
            import torch
            old_state  = ckpt[key]
            new_state  = module.state_dict()
            patched    = {}
            for k, new_val in new_state.items():
                if k not in old_state:
                    # 新增的键（如 lsp_enc）：保持随机初始化
                    patched[k] = new_val
                    print(f"  ⚠ {key}/{k}: 新增，随机初始化")
                elif old_state[k].shape == new_val.shape:
                    # 形状相同：直接复制
                    patched[k] = old_state[k]
                else:
                    # 形状不同：把旧权重填入新权重的对应位置
                    patched[k] = new_val.clone()
                    old_v = old_state[k]
                    # 对每个维度取 min，把旧权重填入新权重的左上角
                    slices = tuple(slice(0, min(n, o))
                                   for n, o in zip(new_val.shape, old_v.shape))
                    patched[k][slices] = old_v[slices]
                    print(f"  ⚠ {key}/{k}: 形状 {tuple(old_v.shape)}→{tuple(new_val.shape)}，旧权重已填充")
            module.load_state_dict(patched, strict=True)

        _load(self.actor_feat,      'actor_feat')
        _load(self.actor,           'actor')
        _load(self.critic_feat,     'critic_feat')
        _load(self.critic,          'critic')
        _load(self.actor_feat_tgt,  'actor_feat_tgt')
        _load(self.actor_tgt,       'actor_tgt')
        _load(self.critic_feat_tgt, 'critic_feat_tgt')
        _load(self.critic_tgt,      'critic_tgt')
        self.total_steps = ckpt.get('total_steps', 0)
        print(f"  ✓ TD3 加载自 {path}（steps={self.total_steps}）")