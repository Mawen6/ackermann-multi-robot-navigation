# configs/config.py
# ──────────────────────────────────────────────────────────────────────
# 所有超参数集中管理，按模块分组，修改这里即可
# ──────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════
# 1. 机器人 / 运动学
# ═══════════════════════════════════════════
ROBOT_RADIUS     = 0.25
WHEELBASE        = 0.50
MAX_SPEED        = 1.5
MIN_SPEED        = 0.0
MAX_STEER        = 0.55
MAX_OMEGA        = MAX_SPEED * 1.5 / WHEELBASE
DT_LOW           = 0.1
DT_HIGH          = 0.5
LOW_STEPS_PER_HIGH = int(DT_HIGH / DT_LOW)

# ═══════════════════════════════════════════
# 2. 场地
# ═══════════════════════════════════════════
ARENA_SIZE       = 12.0
N_STATIC_OBS     = 2
OBS_RADIUS_MIN   = 0.25
OBS_RADIUS_MAX   = 0.55
GOAL_TOL         = 0.35
MIN_START_GOAL   = 3.0
MIN_ROBOT_SPACING= 0.8
PASSAGE_MARGIN   = 0.3

MAZE_MIN_WIDTH   = 1.0
DEADEND_CONFIRM  = 8
DEADEND_FRONT_ANGLE = 50.0
DEADEND_SIDE_ANGLE  = 45.0

# ═══════════════════════════════════════════
# 3. 激光参数
# ═══════════════════════════════════════════
LASER_FOV        = 360.0
LASER_MAX_RANGE  = 8.0
LASER_MIN_RANGE  = 0.05

LOW_LASER_BEAMS  = 36
LOW_LASER_RANGE  = 4.0

HIGH_LASER_BEAMS = 72
HIGH_LASER_RANGE = 8.0

# ═══════════════════════════════════════════
# 4. 激光聚类 → VO 参数
# ═══════════════════════════════════════════
CLUSTER_DIST_THRESH = 0.4
MIN_CLUSTER_POINTS  = 2
MAX_VO_OBSTACLES    = 8
VO_TIME_HORIZON     = 3.0

# ═══════════════════════════════════════════
# 5. 通信参数
# ═══════════════════════════════════════════
COMM_RANGE       = 5.0
MAX_NEIGHBORS    = 4
NEIGHBOR_MSG_DIM = 9

# ═══════════════════════════════════════════
# 6. 局部占据栅格
# ═══════════════════════════════════════════
OCC_MAP_SIZE     = 25
OCC_MAP_RES      = 0.2
OCC_MAP_CHANNELS = 3
OCC_CNN_FEAT_DIM = 64

# ═══════════════════════════════════════════
# 7. 下层 TD3
# ═══════════════════════════════════════════
LOW_LASER_DIM    = 36
RVO_DIM          = 7
BIRNN_HIDDEN     = 64
LOW_WP_DIM       = 2
LOW_VEL_DIM      = 4
LOW_GOAL_DIM     = 2
LOW_ACTION_DIM   = 2

# ── LSP Frontier 观测（新增）────────────────────────────────────────
LSP_FRONTIER_DIM  = 7
LSP_MAX_FRONTIERS = 5
LSP_OBS_DIM       = LSP_MAX_FRONTIERS * LSP_FRONTIER_DIM + 1   # = 36

# feat_dim = LaserMLP(64) + RVO_BiRNN(128) + wp(2) + vel(4) + LSP_enc(32) = 230
LOW_FEAT_DIM     = 64 + BIRNN_HIDDEN * 2 + LOW_WP_DIM + LOW_VEL_DIM + LOW_GOAL_DIM

TD3_ACTOR_LR     = 1e-4
TD3_CRITIC_LR    = 1e-4
TD3_GAMMA        = 0.99
TD3_TAU          = 0.005
TD3_BUFFER_SIZE  = 300_000
TD3_BATCH_SIZE   = 256
TD3_POLICY_DELAY = 2
TD3_NOISE_STD    = 0.05
TD3_TARGET_NOISE = 0.1
TD3_NOISE_CLIP   = 0.2
TD3_LEARNING_STARTS = 2_000

CORRIDOR_WIDTH_THRESH = 1.1    # 左右自由空间之和 < 此值 → 窄道
LOW_REW_YIELD         = 0.3    # 路权奖励系数
# 下层奖励系数
LOW_REW_WP_REACH        = 80.0
LOW_REW_COLLISION       = -50.0
LOW_REW_COLLISION_SPEED = -10.0
LOW_REW_APPROACH        = 4.0
LOW_REW_RVO_AREA        = -0.8
LOW_REW_TTC             = -1.0
LOW_REW_RVO_STEER       = 1.5
LOW_REW_TIME            = -0.05
LOW_REW_HEADING         = 0.12
LOW_REW_WALL_PROX       =  -0.5
LOW_REW_WAIT            = -0.02
LOW_REW_SLOW            = 0.0
LOW_REW_SPEED_DANGER    =   0.02
LOW_REW_LSP             =   0.3    # ← 新增：LSP 长期规划奖励权重
WAIT_PATIENCE           = 15
LOW_REW_STUCK           = -1.0
# 静态障碍 碰前减速势场（教"近且快=痛、近且慢=可接受"）
PROX_DANGER_ZONE        = 0.8     # 危险区放宽到1.2m，给Ackermann足够反应提前量
LOW_REW_PROX_SPEED      = -3.0
LOW_REW_PROX_SLOW       =  0.1    # 危险区内低速通过 轻奖（鼓励主动减速而非硬冲）

WALL_VO_RADIUS    = 0.8

# ═══════════════════════════════════════════
# 8. 上层 IPPO
# ═══════════════════════════════════════════
HIGH_EGO_DIM      = 70
HIGH_NBR_MSG_DIM  = 10
HIGH_MAX_NEIGHBORS= 4
HIGH_HIDDEN_DIM   = 256
HIGH_NUM_HEADS    = 4
HIGH_ACTION_DIM   = 5

HIGH_ACTIONS = {
    0: ( 0.0,   0),
    1: ( 1.0,   0),
    2: ( 1.0, +45),
    3: ( 1.0, -45),
    4: (-0.6,   0),
}
HIGH_MASK_BACK    = True

IPPO_LR_ACTOR     = 3e-4
IPPO_LR_CRITIC    = 3e-4
IPPO_GAMMA        = 0.99
IPPO_LAMBDA       = 0.95
IPPO_EPS_CLIP     = 0.2
IPPO_EPOCHS       = 4
IPPO_ENTROPY      = 0.01
IPPO_UPDATE_FREQ  = 8

MAX_N_AGENTS      = 10
PER_AGENT_STATE   = 9
GLOBAL_STATE_DIM  = MAX_N_AGENTS * PER_AGENT_STATE

HIGH_REW_GOAL        =  20.0
HIGH_REW_LOW_PASS    =   1.0
HIGH_REW_SMART_WAIT  =   0.1
HIGH_REW_TIME        =  -0.5
HIGH_REW_TIMEOUT     = -20.0
HIGH_LEVEL_INTERVAL = 5 

MAX_LOW_STEPS    = 300
MAX_HIGH_STEPS   = 100

STUCK_WINDOW     = 15
STUCK_DIST_THRESH= 0.3

HIGH_CURRICULUM = [
    (1, 0, 0.90, 1000),
    (2, 0, 0.85, 2000),
    (2, 2, 0.80, 2000),
    (3, 0, 0.80, 2000),
    (3, 4, 0.75, 3000),
    (5, 4, 0.70, 4000),
    (5, 8, 0.65, 5000),
]

LOW_REW_RVO_ENTER       = -1.5
LOW_REW_RVO_DEEP        = -6.0
LOW_REW_RVO_EXIT        = 2.5
LOW_REW_RVO_CLEAR       = 0.05
RVO_TTC_ENTER_THRESH    = 0.9
RVO_TTC_DEEP_THRESH     = 0.45
RVO_DANGER_DECAY        = 6.0
LOW_REW_RVO_ACTION_IN   = -2.0
LOW_REW_RVO_ACTION_DEEP = -5.0
LOW_REW_RVO_ACTION_OUT  = 0.8
RVO_ACTION_MARGIN       = 0.15
LOW_REW_REVERSE         = -0.8
LOW_REW_STOP            = -0.4
STOP_SPEED_THRESH       = 0.08
LOW_REW_BACKOUT         = 0.9
LOW_REW_BLOCKED_FORWARD = -1.2
ACK_BLOCK_LASER_THRESH = 1.10
ACK_BLOCK_RVO_DIST      = 1.2
ACK_BLOCK_RVO_TTC       = 0.65
ACK_BLOCK_SIDE_THRESH   = 0.45
RVO_DANGER_RELIEF       = 0.6
OBS_CLEARANCE_SAFE = 1.10

OBS_CLEARANCE_CRITICAL = 0.55

LOW_REW_OBS_CLEARANCE = -3.0

LOW_REW_OBS_CRITICAL = -8.0

LOW_REW_OBS_SPEED = -2.0

WP_VISIBLE_CLEARANCE = 0.45

WP_VISIBLE_EDGE_MARGIN = 0.20

WP_VISIBLE_MAX_DIST = 2.2

WP_VISIBLE_MIN_DIST = 0.45

WP_VISIBLE_GOAL_BONUS = 3.0

WP_VISIBLE_ALIGN_BONUS = 1.2

WP_VISIBLE_CLEAR_BONUS = 0.35

WP_VISIBLE_TURN_PENALTY = 0.15


# ===== RVO cone ablation =====
# compact: use the original compact RVO risk feature.
# cone: use full RVO cone observation c=[apex, left boundary, right boundary].
RVO_OBS_MODE = "compact"
RVO_CONE_DIM = 6
LOW_REW_RVO_CONE_IN = -1.2
LOW_REW_RVO_CONE_DEEP = -2.5
LOW_REW_RVO_CONE_CLEAR = 0.15

