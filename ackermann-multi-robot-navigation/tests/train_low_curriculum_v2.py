"""
Progressive multi-agent TD3 curriculum.

Flow:
  N=2: obs0 -> obs1 -> obs2 -> obs3 -> obs4 -> corridor
  N=3: obs0 -> obs1 -> obs2 -> obs3 -> obs4 -> corridor
  ...
  N=8: obs0 -> obs1 -> obs2 -> obs3 -> obs4 -> corridor
"""

import sys, time, argparse, signal, json
from pathlib import Path
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.low_env import LowLevelEnv
from algos.birnn_td3.td3 import LowLevelTD3
from configs import config as C

LOG_DIR  = ROOT / "logs" / "low_multi_curriculum_v2"
CKPT_DIR = ROOT / "checkpoints" / "low_multi_curriculum_v2"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = CKPT_DIR / "curriculum_state_v2.json"
DEFAULT_S0 = ROOT / "checkpoints" / "low_multi" / "td3_final.pth"


def build_curriculum():
    """Build curriculum from N=2..13 and obs=0..5.

    Each robot count first learns no-obstacle navigation, then progressively
    increases obstacle count from 1 to 5.
    """
    stages = []

    for n in range(2, 14):
        # Reach threshold is gradually relaxed as multi-agent density increases.
        if n <= 3:
            base_thr = 0.88
        elif n <= 5:
            base_thr = 0.85
        elif n <= 8:
            base_thr = 0.80
        elif n <= 10:
            base_thr = 0.78
        else:
            base_thr = 0.75

        # Crash tolerance is also slightly relaxed for larger teams.
        if n <= 5:
            crash_thr = 0.08
        elif n <= 8:
            crash_thr = 0.10
        else:
            crash_thr = 0.12

        if n <= 5:
            win = 180
        elif n <= 8:
            win = 220
        elif n <= 10:
            win = 260
        else:
            win = 300

        for obs in range(0, 6):
            stages.append(dict(
                name=f"N{n}_obs{obs}",
                n_robots=n,
                n_obs=obs,
                mode="normal",
                reach_thr=max(base_thr - 0.02 * obs, 0.68),
                crash_thr=crash_thr + (0.02 if obs > 0 else 0.0),
                win=win,
            ))

    return stages


CURRICULUM = build_curriculum()


def make_env(stage, seed):
    # Force obstacle count for this curriculum stage.
    C.N_STATIC_OBS = int(stage["n_obs"])

    env = LowLevelEnv(n_robots=int(stage["n_robots"]), seed=seed)

    # Keep an explicit copy on env as well, in case reset() reads env-level fields.
    env.n_static_obs = int(stage["n_obs"])
    env.n_obstacles = int(stage["n_obs"])

    if stage["mode"] == "corridor":
        env.corridor_mode = True
        env.wall_thickness = float(stage.get("wall_thickness", 0.5))
        env.gap_width = float(stage.get("gap_width", 1.5))
        # Corridor stage uses walls, not random circular obstacles by default.
        C.N_STATIC_OBS = 0
        env.n_static_obs = 0
        env.n_obstacles = 0
    else:
        env.corridor_mode = False

    return env


def run_episode(env: LowLevelEnv, agent: LowLevelTD3,
                noise_std: float, random_phase: bool) -> dict:
    obs_list = env.reset()
    n = env.n
    done = [False] * n
    ep_rews = [0.0] * n
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

        actions = np.stack(actions)
        obs_next, rews, terms, truncs, infos = env.step(actions)

        for i in range(n):
            if done[i]:
                continue

            agent.store({
                "laser": obs_list[i]["laser"],
                "rvo": obs_list[i]["rvo"],
                "wp": obs_list[i]["wp"],
                "vel": obs_list[i]["vel"],
                "goal": obs_list[i]["goal"],
                "action": actions[i],
                "reward": rews[i],
                "next_laser": obs_next[i]["laser"],
                "next_rvo": obs_next[i]["rvo"],
                "next_wp": obs_next[i]["wp"],
                "next_vel": obs_next[i]["vel"],
                "next_goal": obs_next[i]["goal"],
                "done": terms[i] or truncs[i],
                "collision_transition": bool(infos[i].get("collided", False)),
            })

            total_transitions += 1
            collision_transitions += int(bool(infos[i].get("collided", False)))
            steer_abs_sum += abs(float(actions[i][1]))

            ep_rews[i] += float(rews[i])
            final_info[i] = infos[i]

            if terms[i] or truncs[i]:
                done[i] = True

        if agent.total_steps % 8 == 0:
            stats = agent.train_step()
            if stats is not None:
                update_stats.append(stats)

        obs_list = obs_next

    q_vals = [s["Q_mean"] for s in update_stats if "Q_mean" in s]
    critic_vals = [s["critic_loss"] for s in update_stats if "critic_loss" in s]

    return {
        "reached": [final_info[i].get("reached", False) for i in range(n)],
        "collided": [final_info[i].get("collided", False) for i in range(n)],
        "all_reach": all(final_info[i].get("reached", False) for i in range(n)),
        "any_crash": any(final_info[i].get("collided", False) for i in range(n)),
        "avg_reward": float(np.mean(ep_rews)),
        "collision_transition": collision_transitions / max(total_transitions, 1),
        "avg_abs_steer": steer_abs_sum / max(total_transitions, 1),
        "critic_loss": float(np.mean(critic_vals)) if critic_vals else np.nan,
        "Q_mean": float(np.mean(q_vals)) if q_vals else np.nan,
    }


def find_stage_index(n_robots=None, mode=None, n_obs=None):
    if n_robots is None:
        return 0
    for idx, s in enumerate(CURRICULUM):
        if s["n_robots"] != n_robots:
            continue
        if mode is not None and s["mode"] != mode:
            continue
        if n_obs is not None and s["n_obs"] != n_obs:
            continue
        return idx
    return 0


def find_stage_index_by_name(stage_name):
    for idx, s in enumerate(CURRICULUM):
        if s.get("name") == stage_name:
            return idx
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-ckpt", default=str(DEFAULT_S0))
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage-index", type=int, default=None)
    parser.add_argument("--n-robots", type=int, default=None)
    parser.add_argument("--n-obstacles", type=int, default=None)
    parser.add_argument("--mode", choices=["normal", "corridor"], default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    agent = LowLevelTD3(device=args.device)

    explicit_ckpt = Path(args.stage0_ckpt)
    if not explicit_ckpt.is_absolute():
        explicit_ckpt = ROOT / explicit_ckpt

    stage_idx = 0

    if args.resume:
        ckpts = sorted(CKPT_DIR.glob("td3_*.pth"))
        if ckpts:
            agent.load(str(ckpts[-1]))
            print(f"resume checkpoint: {ckpts[-1]}")
        if STATE_FILE.exists():
            state = json.loads(STATE_FILE.read_text())

            # The curriculum may have changed, so do not blindly trust old
            # numeric stage_idx. Prefer remapping by saved stage name.
            saved_stage = state.get("stage", {})
            saved_name = saved_stage.get("name")
            remapped = find_stage_index_by_name(saved_name) if saved_name else None

            if remapped is not None:
                stage_idx = int(remapped)
                print(f"resume curriculum stage by name: {saved_name} -> index {stage_idx}")
            else:
                stage_idx = int(state.get("stage_idx", 0))
                print(f"resume curriculum stage by old index: {stage_idx}")
    else:
        if explicit_ckpt.exists():
            agent.load(str(explicit_ckpt))
            agent.buffer.buf.clear()
            print(f"loaded stage0 checkpoint and cleared buffer: {explicit_ckpt}")
        else:
            print(f"warning: stage0 checkpoint not found: {explicit_ckpt}")

    if args.stage_index is not None:
        stage_idx = int(args.stage_index)
    elif args.n_robots is not None:
        stage_idx = find_stage_index(args.n_robots, args.mode, args.n_obstacles)

    stage_idx = max(0, min(stage_idx, len(CURRICULUM) - 1))
    stage = CURRICULUM[stage_idx]
    env = make_env(stage, args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(LOG_DIR / f"run_{ts}")

    all_reach_buf, any_crash_buf = [], []
    avg_reward_buf, critic_loss_buf, q_mean_buf = [], [], []
    collision_transition_buf, avg_abs_steer_buf = [], []

    ep = 0
    t0 = time.time()

    def save_state(tag):
        p = CKPT_DIR / f"td3_{tag}_{agent.total_steps}.pth"
        agent.save(str(p))
        STATE_FILE.write_text(json.dumps({
            "stage_idx": stage_idx,
            "stage": stage,
            "total_steps": agent.total_steps,
        }, indent=2))
        print(f"saved: {p}")

    def _save_and_exit(sig, frame):
        print("\ninterrupt: saving...")
        save_state("interrupt")
        writer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    print("\n" + "=" * 72)
    print(f"Multi-agent curriculum v2 ({args.total_steps:,} steps)")
    print(f"Stages: N=2..13, each obs0 -> obs1..5")
    print(f"Current stage {stage_idx + 1}/{len(CURRICULUM)}: {stage}")
    print("=" * 72 + "\n")

    while agent.total_steps < args.total_steps:
        stage = CURRICULUM[stage_idx]
        random_phase = agent.total_steps < C.TD3_LEARNING_STARTS
        progress = agent.total_steps / max(args.total_steps, 1)
        noise_std = C.TD3_NOISE_STD * max(1.0 - progress * 1.5, 0.05)

        result = run_episode(env, agent, noise_std, random_phase)
        ep += 1

        all_reach_buf.append(float(result["all_reach"]))
        any_crash_buf.append(float(result["any_crash"]))
        avg_reward_buf.append(float(result["avg_reward"]))
        collision_transition_buf.append(float(result["collision_transition"]))
        avg_abs_steer_buf.append(float(result["avg_abs_steer"]))

        if not np.isnan(result["critic_loss"]):
            critic_loss_buf.append(float(result["critic_loss"]))
        if not np.isnan(result["Q_mean"]):
            q_mean_buf.append(float(result["Q_mean"]))

        for buf in [
            all_reach_buf, any_crash_buf, avg_reward_buf,
            collision_transition_buf, avg_abs_steer_buf,
            critic_loss_buf, q_mean_buf,
        ]:
            if len(buf) > 300:
                buf.pop(0)

        if ep % 10 == 0:
            ar = float(np.mean(all_reach_buf[-100:]))
            ac = float(np.mean(any_crash_buf[-100:]))
            indiv = float(np.mean(result["reached"]))
            avg_reward = float(np.mean(avg_reward_buf[-100:]))
            collision_transition = float(np.mean(collision_transition_buf[-100:]))
            avg_abs_steer = float(np.mean(avg_abs_steer_buf[-100:]))
            critic_loss = float(np.mean(critic_loss_buf[-100:])) if critic_loss_buf else np.nan
            q_mean = float(np.mean(q_mean_buf[-100:])) if q_mean_buf else np.nan
            elapsed = (time.time() - t0) / 60.0

            print(
                f"[ep {ep:5d} | {agent.total_steps:7d} steps | {elapsed:.1f}min]"
                f" stage={stage_idx+1}/{len(CURRICULUM)} {stage['name']}"
                f" N={stage['n_robots']} obs={stage['n_obs']} actual_obs={len(env.obstacles)} mode={stage['mode']}"
                f" all_reach={ar:.1%} any_crash={ac:.1%} indiv={indiv:.1%}"
                f" avg_reward={avg_reward:.2f} critic_loss={critic_loss:.3g}"
                f" Q_mean={q_mean:.3f} collision_transition={collision_transition:.2%}"
                f" avg_abs_steer={avg_abs_steer:.3f}"
            )

            tag = stage["name"]
            writer.add_scalar(f"{tag}/all_reach", ar, agent.total_steps)
            writer.add_scalar(f"{tag}/any_crash", ac, agent.total_steps)
            writer.add_scalar(f"{tag}/avg_reward", avg_reward, agent.total_steps)
            writer.add_scalar(f"{tag}/collision_transition", collision_transition, agent.total_steps)
            writer.add_scalar(f"{tag}/avg_abs_steer", avg_abs_steer, agent.total_steps)
            if not np.isnan(critic_loss):
                writer.add_scalar(f"{tag}/critic_loss", critic_loss, agent.total_steps)
            if not np.isnan(q_mean):
                writer.add_scalar(f"{tag}/Q_mean", q_mean, agent.total_steps)

        # Advance curriculum only when both reach and crash are stable.
        win = int(stage["win"])
        if len(all_reach_buf) >= win and stage_idx < len(CURRICULUM) - 1:
            ar = float(np.mean(all_reach_buf[-win:]))
            ac = float(np.mean(any_crash_buf[-win:]))
            if ar >= float(stage["reach_thr"]) and ac <= float(stage["crash_thr"]):
                save_state(stage["name"])
                stage_idx += 1
                stage = CURRICULUM[stage_idx]
                env = make_env(stage, args.seed)

                agent.buffer.buf.clear()
                all_reach_buf.clear()
                any_crash_buf.clear()
                avg_reward_buf.clear()
                critic_loss_buf.clear()
                q_mean_buf.clear()
                collision_transition_buf.clear()
                avg_abs_steer_buf.clear()

                STATE_FILE.write_text(json.dumps({
                    "stage_idx": stage_idx,
                    "stage": stage,
                    "total_steps": agent.total_steps,
                }, indent=2))

                print("\n" + "=" * 72)
                print(f"advance to stage {stage_idx + 1}/{len(CURRICULUM)}: {stage}")
                print("cleared replay buffer")
                print("=" * 72 + "\n")
                continue

        # Periodic save
        if agent.total_steps % 100_000 < stage["n_robots"] * C.MAX_LOW_STEPS:
            save_state("periodic")

    print(f"\ntraining done, elapsed {(time.time() - t0) / 60:.1f} min")
    final = CKPT_DIR / "td3_final.pth"
    agent.save(str(final))
    STATE_FILE.write_text(json.dumps({
        "stage_idx": stage_idx,
        "stage": CURRICULUM[stage_idx],
        "total_steps": agent.total_steps,
    }, indent=2))
    print(f"final model: {final}")
    writer.close()


if __name__ == "__main__":
    main()
