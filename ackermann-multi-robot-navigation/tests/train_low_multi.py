"""
tests/train_low_multi.py
──────────────────────────────────────────────────────────────────────
Stage 1: 多机下层 TD3（参数共享 + RVO 互惠避碰）

双重课程学习:
  机器人数量: 3 → 4 → 5 → 6 → 7 → 8
  障碍物数量: 固定 0（无障碍纯多机协作阶段）

推进条件: all_reach_rate（全员同时到达率）

用法:
    python -m tests.train_low_multi
    python -m tests.train_low_multi --stage0-ckpt checkpoints/low_multi/td3_N3_obs0.pth
    python -m tests.train_low_multi --resume
    python -m tests.train_low_multi --n-robots 3 --n-obstacles 0 --resume
"""

import sys, time, argparse, signal, json
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.low_env import LowLevelEnv
from algos.birnn_td3.td3 import LowLevelTD3
from configs import config as C

LOG_DIR  = ROOT / "logs"        / "low_multi"
CKPT_DIR = ROOT / "checkpoints" / "low_multi"
LOG_DIR.mkdir(parents=True,  exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_S0 = ROOT / "checkpoints" / "low_single" / "td3_final.pth"

# 机器人数量课程: (n_robots, all_reach 推进阈值, 评估窗口)
# 从 N=3 开始（加载 td3_N3_obs0.pth），逐步推进到 N=8
ROBOT_CURRICULUM = [
    (3,  0.85, 150),
    (4,  0.85, 150),
    (5,  0.85, 200),
    (6,  0.82, 200),
    (7,  0.80, 250),
    (8,  0.80, 300),
]

# 障碍物课程: 固定 0 障碍，阈值设极高永不推进
OBS_CURRICULUM = [
    (0, 0.99, 100),
]

# ── 走廊模式课程（暂时不用，保留备用） ──────────────────────────────
# WALL_CURRICULUM = [
#     (0.5, 0.70, 100),
#     (1.0, 0.70, 100),
#     (1.5, 0.65, 100),
#     (2.0, 0.60, 100),
# ]
# ────────────────────────────────────────────────────────────────────


def run_episode(env: LowLevelEnv, agent: LowLevelTD3,
                noise_std: float, random_phase: bool) -> dict:
    obs_list   = env.reset()
    n          = env.n
    done       = [False] * n
    ep_rews    = [0.0]  * n
    final_info = [{} for _ in range(n)]
    update_stats = []
    collision_transitions = 0
    total_transitions = 0
    steer_abs_sum = 0.0

    while not all(done):
        actions = []
        for i in range(n):
            if done[i]:
                actions.append(np.zeros(2, dtype=np.float32))
            elif random_phase:
                actions.append(np.array([
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, 1.0),
                ], dtype=np.float32))
            else:
                actions.append(agent.select_action(obs_list[i], noise_std=noise_std))

        obs_next, rews, terms, truncs, infos = env.step(np.stack(actions))

        for i in range(n):
            if done[i]:
                continue
            agent.store({
                'laser':      obs_list[i]['laser'],
                'rvo':        obs_list[i]['rvo'],
                'wp':         obs_list[i]['wp'],
                'vel':        obs_list[i]['vel'],
                'goal':       obs_list[i]['goal'],
                'action':     actions[i],
                'reward':     rews[i],
                'next_laser': obs_next[i]['laser'],
                'next_rvo':   obs_next[i]['rvo'],
                'next_wp':    obs_next[i]['wp'],
                'next_vel':   obs_next[i]['vel'],
                'next_goal':  obs_next[i]['goal'],
                'done':       terms[i] or truncs[i],
                'collision_transition': bool(infos[i].get('collided', False)),
            })
            total_transitions += 1
            collision_transitions += int(bool(infos[i].get('collided', False)))
            steer_abs_sum += abs(float(actions[i][1]))
            ep_rews[i]  += rews[i]
            final_info[i] = infos[i]
            if terms[i] or truncs[i]:
                done[i] = True

        if agent.total_steps % 8 == 0:
            stats = agent.train_step()
            if stats is not None:
                update_stats.append(stats)

        obs_list = obs_next

    return {
        'reached':   [final_info[i].get('reached',  False) for i in range(n)],
        'collided':  [final_info[i].get('collided', False) for i in range(n)],
        'all_reach': all(final_info[i].get('reached', False) for i in range(n)),
        'any_crash': any(final_info[i].get('collided', False) for i in range(n)),
        'avg_reward': float(np.mean(ep_rews)),
        'collision_transition': collision_transitions / max(total_transitions, 1),
        'avg_abs_steer': steer_abs_sum / max(total_transitions, 1),
        'critic_loss': float(np.mean([s['critic_loss'] for s in update_stats])) if update_stats else np.nan,
        'Q_mean': float(np.mean([s['Q_mean'] for s in update_stats if 'Q_mean' in s])) if update_stats else np.nan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-ckpt",  default=str(DEFAULT_S0))
    parser.add_argument("--total-steps",  type=int, default=2_000_000)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--n-robots",     type=int, default=None)
    parser.add_argument("--n-obstacles",  type=int, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    agent = LowLevelTD3(device=args.device)

    CURRICULUM_STATE_FILE = CKPT_DIR / "curriculum_state.json"

    # 把 stage0_ckpt 转成绝对路径
    explicit_ckpt = Path(args.stage0_ckpt)
    if not explicit_ckpt.is_absolute():
        explicit_ckpt = ROOT / explicit_ckpt

    # 是否用户明确指定了非默认文件
    is_explicit = (explicit_ckpt.exists() and
                   explicit_ckpt.resolve() != DEFAULT_S0.resolve())

    if args.resume and is_explicit:
        agent.load(str(explicit_ckpt))
        print(f"✓ 加载指定文件: {explicit_ckpt.name}")
    elif args.resume:
        ckpts = sorted(CKPT_DIR.glob("td3_*.pth"))
        if ckpts:
            agent.load(str(ckpts[-1]))
        else:
            args.resume = False

    if not args.resume:
        if explicit_ckpt.exists():
            agent.load(str(explicit_ckpt))
            agent.buffer.buf.clear()
            print(f"✓ 加载 Stage 0: {explicit_ckpt.name}（已清空 buffer）")
        else:
            print(f"⚠ 找不到 {explicit_ckpt.name}，从头训练")

    if args.resume and agent.total_steps > 0:
        if agent.total_steps >= args.total_steps:
            args.total_steps = agent.total_steps + args.total_steps
            print(f"  自动延长到 {args.total_steps} 步")
        if CURRICULUM_STATE_FILE.exists() and args.n_robots is None:
            state = json.loads(CURRICULUM_STATE_FILE.read_text())
            args.n_robots    = state.get('n_robots', 3)
            args.n_obstacles = state.get('n_obstacles', 0)
            print(f"  自动恢复课程阶段: N={args.n_robots} obs={args.n_obstacles}")

    # 确定起始课程阶段
    robot_stage = next(
        (i for i, (n, _, _) in enumerate(ROBOT_CURRICULUM) if n == args.n_robots),
        0
    ) if args.n_robots else 0
    obs_stage = next(
        (i for i, (n, _, _) in enumerate(OBS_CURRICULUM) if n == args.n_obstacles),
        0
    ) if args.n_obstacles else 0

    n_robots,  rr_thresh, rr_win  = ROBOT_CURRICULUM[robot_stage]
    n_obs,     _,         _       = OBS_CURRICULUM[obs_stage]
    C.N_STATIC_OBS = 0   # 本阶段固定无障碍
    env = LowLevelEnv(n_robots=n_robots, seed=args.seed)

    # ── 走廊模式（暂时注释，保留备用） ──────────────────────────────
    # n_robots = 2
    # env = LowLevelEnv(n_robots=2, seed=args.seed)
    # env.corridor_mode = True
    # wall_stage = 0
    # wall_thickness, wall_thresh, wall_win = WALL_CURRICULUM[wall_stage]
    # env.wall_thickness = wall_thickness
    # env.gap_width = 1.5
    # C.N_STATIC_OBS = 0
    # ────────────────────────────────────────────────────────────────

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(LOG_DIR / f"run_{ts}")

    all_reach_buf, any_crash_buf = [], []
    avg_reward_buf, critic_loss_buf, q_mean_buf, collision_transition_buf, avg_abs_steer_buf = [], [], [], [], []
    ep = 0
    t0 = time.time()

    def _save_and_exit(sig, frame):
        print(f"\n⚠ 中断，保存中...")
        p = CKPT_DIR / f"td3_interrupt_{agent.total_steps}.pth"
        agent.save(str(p))
        CURRICULUM_STATE_FILE.write_text(json.dumps({
            'n_robots': n_robots, 'n_obstacles': n_obs,
            'total_steps': agent.total_steps,
        }))
        print(f"✓ 已保存: {p}")
        print(f"✓ 课程状态: N={n_robots} obs={n_obs}")
        writer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    print(f"\n{'='*65}")
    print(f" Stage 1: 多机 TD3 + RVO  ({args.total_steps:,} steps)")
    print(f" 机器人课程: {[c[0] for c in ROBOT_CURRICULUM]}")
    print(f" 障碍物: 固定 0（纯多机协作阶段）")
    print(f" 当前起点: {n_robots}台 + 0个障碍物")
    print(f"{'='*65}\n")

    while agent.total_steps < args.total_steps:
        random_phase = (agent.total_steps < C.TD3_LEARNING_STARTS)
        progress     = agent.total_steps / args.total_steps
        noise_std    = C.TD3_NOISE_STD * max(1.0 - progress * 1.5, 0.05)

        result = run_episode(env, agent, noise_std, random_phase)
        ep += 1

        all_reach_buf.append(float(result['all_reach']))
        any_crash_buf.append(float(result['any_crash']))
        avg_reward_buf.append(float(result['avg_reward']))
        collision_transition_buf.append(float(result['collision_transition']))
        avg_abs_steer_buf.append(float(result['avg_abs_steer']))
        if not np.isnan(result['critic_loss']):
            critic_loss_buf.append(float(result['critic_loss']))
        if not np.isnan(result['Q_mean']):
            q_mean_buf.append(float(result['Q_mean']))

        for buf in [all_reach_buf, any_crash_buf, avg_reward_buf,
                    collision_transition_buf, avg_abs_steer_buf, critic_loss_buf, q_mean_buf]:
            if len(buf) > 200: buf.pop(0)

        # 打印
        if ep % 10 == 0:
            ar = np.mean(all_reach_buf[-100:])
            ac = np.mean(any_crash_buf[-100:])
            avg_r = float(np.mean(result['reached']))
            avg_reward = np.mean(avg_reward_buf[-100:])
            collision_transition = np.mean(collision_transition_buf[-100:])
            avg_abs_steer = np.mean(avg_abs_steer_buf[-100:])
            critic_loss = np.mean(critic_loss_buf[-100:]) if critic_loss_buf else np.nan
            q_mean = np.mean(q_mean_buf[-100:]) if q_mean_buf else np.nan
            elapsed = (time.time() - t0) / 60
            print(f"[ep {ep:5d} | {agent.total_steps:7d} steps | {elapsed:.1f}min]"
                  f"  N={n_robots} obs={n_obs}"
                  f"  all_reach={ar:.1%}  any_crash={ac:.1%}"
                  f"  indiv={avg_r:.1%}"
                  f"  avg_reward={avg_reward:.2f}"
                  f"  critic_loss={critic_loss:.3g}"
                  f"  Q_mean={q_mean:.3f}"
                  f"  collision_transition={collision_transition:.2%}"
                  f"  avg_abs_steer={avg_abs_steer:.3f}")
            writer.add_scalar(f'N{n_robots}_obs{n_obs}/all_reach', ar, agent.total_steps)
            writer.add_scalar(f'N{n_robots}_obs{n_obs}/any_crash', ac, agent.total_steps)
            writer.add_scalar(f'N{n_robots}_obs{n_obs}/avg_reward', avg_reward, agent.total_steps)
            writer.add_scalar(f'N{n_robots}_obs{n_obs}/collision_transition', collision_transition, agent.total_steps)
            if not np.isnan(critic_loss):
                writer.add_scalar(f'N{n_robots}_obs{n_obs}/critic_loss', critic_loss, agent.total_steps)
            if not np.isnan(q_mean):
                writer.add_scalar(f'N{n_robots}_obs{n_obs}/Q_mean', q_mean, agent.total_steps)

        # 机器人课程推进（障碍物固定为0，仅按机器人数量推进）
        if (len(all_reach_buf) >= rr_win
                and robot_stage < len(ROBOT_CURRICULUM) - 1):
            ar = np.mean(all_reach_buf[-rr_win:])
            ac = np.mean(any_crash_buf[-rr_win:])
            if ar >= rr_thresh and ac <= 0.08:
                sp = CKPT_DIR / f"td3_N{n_robots}_obs{n_obs}.pth"
                agent.save(str(sp))
                robot_stage += 1
                n_robots, rr_thresh, rr_win = ROBOT_CURRICULUM[robot_stage]
                C.N_STATIC_OBS = 0
                env = LowLevelEnv(n_robots=n_robots, seed=args.seed)
                # 清空 buffer：新机器人数量下 RVO 分布完全不同
                agent.buffer.buf.clear()
                all_reach_buf.clear(); any_crash_buf.clear()
                # 保存课程状态
                CURRICULUM_STATE_FILE.write_text(json.dumps({
                    'n_robots': n_robots, 'n_obstacles': n_obs,
                    'total_steps': agent.total_steps,
                }))
                print(f"\n{'═'*65}")
                print(f"  ✓ 机器人推进 → {n_robots}台  目标≥{rr_thresh:.0%}")
                print(f"  已清空 Replay Buffer（RVO分布已变化）")
                print(f"{'═'*65}\n")
                continue

        # ── 障碍物课程推进（本阶段禁用，保留备用） ──────────────────
        # if (len(all_reach_buf) >= obs_win
        #         and obs_stage < len(OBS_CURRICULUM) - 1):
        #     ar = np.mean(all_reach_buf[-obs_win:])
        #     if ar >= obs_thresh:
        #         sp = CKPT_DIR / f"td3_N{n_robots}_obs{n_obs}.pth"
        #         agent.save(str(sp))
        #         obs_stage += 1
        #         n_obs, obs_thresh, obs_win = OBS_CURRICULUM[obs_stage]
        #         C.N_STATIC_OBS = n_obs
        #         all_reach_buf.clear(); any_crash_buf.clear()
        #         print(f"\n{'─'*65}")
        #         print(f"  ✓ 障碍物推进 → {n_obs}个  目标≥{obs_thresh:.0%}")
        #         print(f"{'─'*65}\n")
        #         continue
        # ────────────────────────────────────────────────────────────

        # ── 墙厚课程推进（走廊模式，暂时注释） ──────────────────────
        # if (len(all_reach_buf) >= wall_win
        #         and wall_stage < len(WALL_CURRICULUM) - 1):
        #     ar = np.mean(all_reach_buf[-wall_win:])
        #     if ar >= wall_thresh:
        #         sp = CKPT_DIR / f"td3_corridor_wall{wall_thickness:.1f}.pth"
        #         agent.save(str(sp))
        #         wall_stage += 1
        #         wall_thickness, wall_thresh, wall_win = WALL_CURRICULUM[wall_stage]
        #         env.wall_thickness = wall_thickness
        #         agent.buffer.buf.clear()
        #         all_reach_buf.clear(); any_crash_buf.clear()
        #         print(f"\n{'═'*65}")
        #         print(f"  ✓ 墙厚推进 → {wall_thickness}m  目标≥{wall_thresh:.0%}")
        #         print(f"{'═'*65}\n")
        #         continue
        # ────────────────────────────────────────────────────────────

        # 定期保存
        if agent.total_steps % 100_000 < n_robots * C.MAX_LOW_STEPS:
            agent.save(str(CKPT_DIR / f"td3_{agent.total_steps}.pth"))

    print(f"\n训练完成，总耗时: {(time.time()-t0)/60:.1f} 分钟")
    final = CKPT_DIR / "td3_final.pth"
    agent.save(str(final))
    print(f"最终模型: {final}")
    writer.close()


if __name__ == "__main__":
    main()