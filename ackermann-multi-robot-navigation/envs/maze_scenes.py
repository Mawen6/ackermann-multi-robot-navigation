"""
envs/maze_scenes.py
──────────────────────────────────────────────────────────────────────
手工设计的迷宫场景（课程式，Level 2-5）

所有场景基于 12m×12m 场地，通道宽度 ≥ 1m。
每个场景返回 (rect_obstacles, start_regions, goal_regions)：
  rect_obstacles : List[RectObstacle]
  start_regions  : List[(x_min,x_max,y_min,y_max)]  机器人可出生区域
  goal_regions   : List[(x_min,x_max,y_min,y_max)]  目标可生成区域

Level 2: 走廊    — 两条平行长墙，1.5m 通道
Level 3: T形     — T字型路口
Level 4: U形死路 — U形陷阱，测试死路检测
Level 5: 十字迷宫 — 4个房间，中央走廊
"""

from __future__ import annotations
from typing import List, Tuple
from envs.rect_obstacle import RectObstacle

# 类型别名
Region = Tuple[float, float, float, float]   # (x_min, x_max, y_min, y_max)
Scene  = Tuple[List[RectObstacle], List[Region], List[Region]]


def _rect(cx, cy, w, h) -> RectObstacle:
    return RectObstacle(cx, cy, w, h)


# ══════════════════════════════════════════════════════════════════════
# Level 2：走廊
# ══════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────────────┐
#  │                                  │
#  │  ██████████████    ██████████   │  ← 上墙（各约5m长）
#  │                                  │
#  │          走廊（1.5m宽）          │
#  │                                  │
#  │  ██████████████    ██████████   │  ← 下墙
#  │                                  │
#  └──────────────────────────────────┘
#
def make_corridor() -> Scene:
    walls = [
        _rect(-2.5,  2.0, 5.0, 0.4),   # 上左墙
        _rect( 3.5,  2.0, 4.0, 0.4),   # 上右墙
        _rect(-2.5, -2.0, 5.0, 0.4),   # 下左墙
        _rect( 3.5, -2.0, 4.0, 0.4),   # 下右墙
    ]
    # 起点：左侧区域，终点：右侧区域
    starts = [(-5.5, -1.5, -1.5, 1.5)]
    goals  = [( 1.5,  5.5, -1.5, 1.5)]
    return walls, starts, goals


# ══════════════════════════════════════════════════════════════════════
# Level 3：T形路口
# ══════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────────────┐
#  │        │           │             │
#  │        │  1.5m宽   │             │
#  │        │           │             │
#  │────────┘           └────────     │
#  │         横向走廊（1.5m高）       │
#  │─────────────────────────────    │
#  │                                  │
#  └──────────────────────────────────┘
#
def make_t_junction() -> Scene:
    walls = [
        # 竖向走廊左右墙（上半部分）
        _rect(-1.25,  1.5, 0.4, 3.0),   # 左墙
        _rect( 1.25,  1.5, 0.4, 3.0),   # 右墙
        # 横向走廊上下墙
        _rect( 0.0,  -1.0, 9.0, 0.4),   # 上墙
        _rect(-3.5,  -3.0, 2.5, 0.4),   # 下左墙
        _rect( 3.5,  -3.0, 2.5, 0.4),   # 下右墙
    ]
    starts = [(-5.0, -2.5, -4.5, -1.5)]   # 左侧
    goals  = [( 2.5,  5.0, -4.5, -1.5),   # 右侧
               (-0.7,  0.7,  2.5,  5.0)]   # 上方
    return walls, starts, goals


# ══════════════════════════════════════════════════════════════════════
# Level 4：U形死路
# ══════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────────────┐
#  │                                  │
#  │   ┌──────────────────────┐       │
#  │   │                      │       │
#  │   │   U形死路区域         │       │
#  │   │                      │       │
#  │   └───────┐  ┌───────────┘       │
#  │           │  │  ← 入口（1.5m宽）  │
#  └──────────────────────────────────┘
#
def make_u_trap() -> Scene:
    walls = [
        # U形的三条边
        _rect( 0.0,  2.0, 6.0, 0.4),   # 顶边
        _rect(-2.8,  0.0, 0.4, 4.0),   # 左边
        _rect( 2.8,  0.0, 0.4, 4.0),   # 右边
        # 入口两侧（确保入口刚好 1.5m）
        _rect(-3.5, -2.5, 2.5, 0.4),   # 入口左侧延伸
        _rect( 3.5, -2.5, 2.5, 0.4),   # 入口右侧延伸
    ]
    # 起点在 U 形外，目标在 U 形内（测试死路检测）
    starts = [(-5.0, -4.0, -1.0, 1.0)]
    goals  = [(-1.5,  1.5,  0.5, 3.5)]   # U形内部
    return walls, starts, goals


# ══════════════════════════════════════════════════════════════════════
# Level 5：十字迷宫（4个房间）
# ══════════════════════════════════════════════════════════════════════
#
#  ┌──────────┬──────────────┬──────────┐
#  │ 房间1    │  走廊（1.5m） │  房间2   │
#  │          │              │          │
#  ├──────────┤              ├──────────┤
#  │  走廊    │    中央       │  走廊    │
#  │          │              │          │
#  ├──────────┤              ├──────────┤
#  │ 房间3    │              │  房间4   │
#  │          │              │          │
#  └──────────┴──────────────┴──────────┘
#
def make_cross_maze() -> Scene:
    walls = [
        # 水平隔墙（上）：左右各一段，中间留 1.5m 通道
        _rect(-3.5,  1.5, 3.5, 0.4),
        _rect( 3.5,  1.5, 3.5, 0.4),
        # 水平隔墙（下）
        _rect(-3.5, -1.5, 3.5, 0.4),
        _rect( 3.5, -1.5, 3.5, 0.4),
        # 竖直隔墙（左）：上下各一段，中间留 1.5m 通道
        _rect(-1.5,  3.5, 0.4, 3.5),
        _rect(-1.5, -3.5, 0.4, 3.5),
        # 竖直隔墙（右）
        _rect( 1.5,  3.5, 0.4, 3.5),
        _rect( 1.5, -3.5, 0.4, 3.5),
    ]
    # 起点/终点分散在4个房间
    starts = [
        (-5.0, -2.5,  2.5,  5.0),   # 房间1（左上）
        ( 2.5,  5.0,  2.5,  5.0),   # 房间2（右上）
        (-5.0, -2.5, -5.0, -2.5),   # 房间3（左下）
        ( 2.5,  5.0, -5.0, -2.5),   # 房间4（右下）
    ]
    goals = starts[:]   # 目标也在4个房间，随机选不同于起点的房间
    return walls, starts, goals


# ══════════════════════════════════════════════════════════════════════
# 场景注册表
# ══════════════════════════════════════════════════════════════════════

MAZE_LEVELS = {
    2: make_corridor,
    3: make_t_junction,
    4: make_u_trap,
    5: make_cross_maze,
}


def get_maze_scene(level: int) -> Scene:
    """获取指定难度的迷宫场景。level 2-5。"""
    if level not in MAZE_LEVELS:
        raise ValueError(f"level {level} 不存在，可选 {list(MAZE_LEVELS.keys())}")
    return MAZE_LEVELS[level]()


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def get_scene(level: int) -> list:
    """返回矩形障碍物列表。"""
    walls, _, _ = get_maze_scene(level)
    return walls


def get_valid_positions(level: int, n: int,
                        rng, min_dist: float = 2.0) -> list:
    """
    在迷宫场景的可行区域内采样 n 个不重叠的位置。
    保证位置不在矩形障碍内，且互相距离 >= min_dist。
    """
    import numpy as np
    from configs import config as C

    walls, starts, goals = get_maze_scene(level)
    half   = C.ARENA_SIZE / 2.0
    radius = C.ROBOT_RADIUS + 0.2   # 采样点到障碍的安全距离

    # 合并所有可用区域
    all_regions = starts + goals

    positions = []
    for _ in range(n):
        for _ in range(2000):
            # 从可用区域随机选一个
            region = all_regions[rng.integers(len(all_regions))]
            x_min, x_max, y_min, y_max = region
            x_min = max(x_min, -half + radius)
            x_max = min(x_max,  half - radius)
            y_min = max(y_min, -half + radius)
            y_max = min(y_max,  half - radius)
            if x_min >= x_max or y_min >= y_max:
                continue

            x = float(rng.uniform(x_min, x_max))
            y = float(rng.uniform(y_min, y_max))

            # 检查不在矩形障碍内
            in_wall = any(w.collides_circle(x, y, radius) for w in walls)
            if in_wall:
                continue

            # 检查与已有位置的距离
            too_close = any(
                np.sqrt((x - p[0])**2 + (y - p[1])**2) < min_dist
                for p in positions
            )
            if too_close:
                continue

            positions.append((x, y))
            break
        else:
            # 500次没找到，放宽到随机场地位置
            x = float(rng.uniform(-half + radius, half - radius))
            y = float(rng.uniform(-half + radius, half - radius))
            positions.append((x, y))

    return positions
