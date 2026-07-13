"""
envs/room_scenes.py
──────────────────────────────────────────────────────────────────────
50m × 50m 多房间场景

设计原则：
  - 6个房间 + 走廊网络
  - 门口宽度 ≥ 2m（保证阿克曼机器人能通过）
  - 每个房间有多个进入方向
  - 从某个房间出发，只有部分走廊通向目标
  - P_S=0 的 frontier 比例预期 30-50%（适合 LSP 训练）

场景变体：
  Level 10：标准6房间（对称布局）
  Level 11：非对称6房间（更难）
  Level 12：长走廊+4房间（需要规划更长路径）
  Level 13：随机障碍物 + 走廊混合
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
from envs.rect_obstacle import RectObstacle

Region = Tuple[float, float, float, float]
Scene  = Tuple[List[RectObstacle], List[Region], List[Region]]

ARENA = 50.0
HALF  = ARENA / 2.0      # 25.0
WALL  = 0.8               # 墙厚 0.8m
DOOR  = 2.5               # 门口宽 2.5m（阿克曼转弯半径约0.9m，2.5m足够）


def _rect(cx, cy, w, h) -> RectObstacle:
    return RectObstacle(cx, cy, w, h)


def _hwall(x_center, y_center, length) -> RectObstacle:
    """水平墙段"""
    return _rect(x_center, y_center, length, WALL)


def _vwall(x_center, y_center, length) -> RectObstacle:
    """竖直墙段"""
    return _rect(x_center, y_center, WALL, length)


# ══════════════════════════════════════════════════════════════════════
# Level 10：标准6房间（对称十字走廊）
# ══════════════════════════════════════════════════════════════════════
#
#  ┌──────────┬───┬──────────┐
#  │  房间1   │ 2 │  房间3   │
#  │          ┤   ├          │
#  ├──────┬───┘   └───┬──────┤
#  │      │   走廊    │      │
#  ├──────┴───┐   ┌───┴──────┤
#  │  房间4   ┤   ├  房间6   │
#  │          │ 5 │          │
#  └──────────┴───┴──────────┘
#
# 走廊宽 = 4m，门口宽 = 2.5m
# 场地 50m×50m，房间约 18m×18m
#
def make_room_scene_6() -> Scene:
    """
    6房间标准布局。

    坐标系：中心(0,0)，范围[-25,25]

    水平走廊：y ∈ [-2, 2]（宽4m）
    竖直走廊：x ∈ [-2, 2]（宽4m）

    水平隔墙：
      上隔墙 y=2，左段 x=[-25, -2-DOOR/2]，右段 x=[2+DOOR/2, 25]
      下隔墙 y=-2，同上

    竖直隔墙：
      左隔墙 x=-2，上段 y=[2+DOOR/2, 25]，下段 y=[-25, -2-DOOR/2]
      右隔墙 x=2，同上
    """
    c = 2.0      # 走廊半宽
    d = DOOR / 2  # 门口半宽

    walls = [
        # ── 水平上隔墙（y=c） ──────────────────────────────────────
        # 左段：从左墙到左走廊（带门口）
        # 中间在 x=[-d, d] 处有门
        _hwall((-HALF + (- c - d)) / 2,    c, (-c - d) - (-HALF)),
        # 右段：从右走廊到右墙
        _hwall(( c + d + HALF) / 2,         c, HALF - (c + d)),

        # ── 水平下隔墙（y=-c） ─────────────────────────────────────
        _hwall((-HALF + (- c - d)) / 2,   -c, (-c - d) - (-HALF)),
        _hwall(( c + d + HALF) / 2,        -c, HALF - (c + d)),

        # ── 竖直左隔墙（x=-c） ─────────────────────────────────────
        # 上段：从上走廊到上墙
        _vwall(-c, ( c + d + HALF) / 2,  HALF - (c + d)),
        # 下段：从下墙到下走廊
        _vwall(-c, (-HALF + (- c - d)) / 2, (-c - d) - (-HALF)),

        # ── 竖直右隔墙（x=c） ──────────────────────────────────────
        _vwall( c, ( c + d + HALF) / 2,  HALF - (c + d)),
        _vwall( c, (-HALF + (- c - d)) / 2, (-c - d) - (-HALF)),
    ]

    # 6个房间的区域（起点/终点可在任意房间）
    margin = 1.5   # 离墙的安全距离
    rooms = [
        (-HALF+margin, -c-margin,  c+margin, HALF-margin),   # 房间1：左上
        (-c+margin,    -c-margin,  c+margin,  c-margin),     # 房间2：中央（走廊交叉口，可以用）
        ( c+margin,    -c-margin, HALF-margin, HALF-margin),  # 房间3：右上
        (-HALF+margin, -HALF+margin, -c-margin, -c+margin),  # 房间4：左下（注意坐标顺序）
        (-c+margin,   -HALF+margin,  c-margin,  -c-margin),  # 房间5：中下
        ( c+margin,   -HALF+margin, HALF-margin, -c+margin),  # 房间6：右下
    ]

    # 修正：确保 x_min < x_max, y_min < y_max
    corrected = []
    for r in rooms:
        x_min = min(r[0], r[2])
        x_max = max(r[0], r[2])
        y_min = min(r[1], r[3])
        y_max = max(r[1], r[3])
        if x_max - x_min > 1.0 and y_max - y_min > 1.0:
            corrected.append((x_min, x_max, y_min, y_max))

    return walls, corrected, corrected


# ══════════════════════════════════════════════════════════════════════
# Level 11：非对称6房间（更有挑战性）
# ══════════════════════════════════════════════════════════════════════
#
#  ┌────────────────┬──┬──────┐
#  │    大房间1     │2 │ 房3  │
#  │                ┤  ├──────┤
#  ├────────┬───────┘  └─┬────┤
#  │        │    走廊     │    │
#  ├────────┴──┐  ┌──────┴────┤
#  │  房间4    │  │   房间5   │
#  │           ┤  ├──┬────────┤
#  └───────────┴──┘  │ 房间6  │
#                     └────────┘
#
def make_room_scene_asymmetric() -> Scene:
    """非对称布局，迫使机器人做出更多规划决策"""
    walls = [
        # 水平主走廊 y=3（上半部）
        _hwall(-16.0,  3.0, 18.0),   # 左段
        _hwall( 14.0,  3.0, 22.0),   # 右段

        # 水平主走廊 y=-3（下半部）
        _hwall(-12.0, -3.0, 26.0),   # 左段
        _hwall( 16.0, -3.0, 18.0),   # 右段

        # 竖直左走廊 x=-3
        _vwall(-3.0,  16.0, 19.0),   # 上段
        _vwall(-3.0, -16.0, 19.0),   # 下段

        # 竖直右走廊 x=5（非对称）
        _vwall( 5.0,  15.0, 17.0),
        _vwall( 5.0, -15.0, 17.0),

        # 额外隔墙：把右上分成两个小房间
        _hwall(15.0, 12.0, 15.0),    # 右上水平隔墙
    ]

    rooms = [
        (-25.0+1.5, -3.0-1.5,  3.0-1.5, 25.0-1.5),   # 左上大房间
        (-3.0+1.5,   3.0+1.5,  5.0-1.5, 25.0-1.5),   # 中上走廊区（可做起点）
        ( 5.0+1.5,  12.0+1.5, 25.0-1.5, 25.0-1.5),   # 右上小房间A
        ( 5.0+1.5, -25.0+1.5, 25.0-1.5, 12.0-1.5),   # 右上小房间B
        (-25.0+1.5, -25.0+1.5, -3.0-1.5, -3.0+1.5),  # 左下房间
        (-3.0+1.5,  -25.0+1.5,  5.0-1.5, -3.0-1.5),  # 中下房间
        ( 5.0+1.5,  -25.0+1.5, 25.0-1.5, -3.0+1.5),  # 右下房间
    ]

    corrected = []
    for r in rooms:
        x_min, x_max = min(r[0],r[2]), max(r[0],r[2])
        y_min, y_max = min(r[1],r[3]), max(r[1],r[3])
        if x_max-x_min > 2.0 and y_max-y_min > 2.0:
            corrected.append((x_min, x_max, y_min, y_max))

    return walls, corrected, corrected


# ══════════════════════════════════════════════════════════════════════
# Level 12：长走廊 + 4个大房间（测试长距离规划）
# ══════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────┐
#  │        大房间1           │
#  ├────────┬─────────┬───────┤
#  │        │  走廊   │       │
#  │  房间2 ├─────────┤ 房间3 │
#  │        │  走廊   │       │
#  ├────────┴─────────┴───────┤
#  │        大房间4           │
#  └──────────────────────────┘
#
def make_room_scene_corridor() -> Scene:
    """长走廊强迫机器人规划绕路"""
    walls = [
        # 上下水平隔墙（距上下各 15m）
        _hwall(-10.0,  10.0, 30.0),   # 上隔墙左段
        _hwall( 15.0,  10.0, 20.0),   # 上隔墙右段
        _hwall(-10.0, -10.0, 30.0),   # 下隔墙左段
        _hwall( 15.0, -10.0, 20.0),   # 下隔墙右段

        # 竖直中间隔墙（把走廊分成两条）
        _vwall(-5.0,  0.0, 16.0),     # 左竖墙
        _vwall( 5.0,  0.0, 16.0),     # 右竖墙（中间走廊4m宽）

        # 两侧房间的内墙
        _vwall(-15.0,  0.0, 16.0),    # 左房间右墙（留门）
        _vwall( 15.0,  0.0, 16.0),    # 右房间左墙（留门）
    ]

    rooms = [
        (-25.0+1.5, -25.0+1.5, 25.0-1.5, 10.0-1.5),   # 大房间1（上）
        (-25.0+1.5, -10.0+1.5,-15.0-1.5, 10.0-1.5),   # 房间2（左中）
        ( 15.0+1.5, -10.0+1.5, 25.0-1.5, 10.0-1.5),   # 房间3（右中）
        (-25.0+1.5,-25.0+1.5,  25.0-1.5,-10.0+1.5),   # 大房间4（下）
    ]

    corrected = []
    for r in rooms:
        x_min, x_max = min(r[0],r[2]), max(r[0],r[2])
        y_min, y_max = min(r[1],r[3]), max(r[1],r[3])
        if x_max-x_min > 2.0 and y_max-y_min > 2.0:
            corrected.append((x_min, x_max, y_min, y_max))

    return walls, corrected, corrected


# ══════════════════════════════════════════════════════════════════════
# Level 13：简单单门房间（最简单）
# ══════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────┐
#  │                          │
#  │          房间1           │
#  │                          │
#  ├─────────┐    ┌──────────┤
#  │         │ 门 │           │
#  │ 房间2   │    │   房间3   │
#  │         │    │           │
#  └─────────┴────┴──────────┘
#
def make_simple_two_room() -> Scene:
    """两个大房间 + 一个门口，最简单的场景"""
    walls = [
        # 中间水平隔墙（y=0），左右两段，中间留 5m 门口
        _hwall(-15.0,  0.0, 20.0),    # 左段
        _hwall( 15.0,  0.0, 20.0),    # 右段
    ]
    rooms = [
        (-23.5, 23.5,   2.0, 23.5),   # 上房间
        (-23.5, 23.5, -23.5, -2.0),   # 下房间
    ]
    return walls, rooms, rooms


# ══════════════════════════════════════════════════════════════════════
# Level 14：L形房间（中等难度）
# ══════════════════════════════════════════════════════════════════════
#
#  ┌────────────────┬─────────┐
#  │                │         │
#  │    房间1       │  房间3  │
#  │                │         │
#  ├──────┐         │         │
#  │      │         │         │
#  │房间2 │   通道  │         │
#  │      │         │         │
#  └──────┴─────────┴─────────┘
#
def make_l_shape_3room() -> Scene:
    """L形布局，3个房间，需要绕一下"""
    walls = [
        # 右边竖墙（x=10），分隔房间3和通道，留 6m 上方门口
        _vwall( 10.0, -7.0, 30.0),   # 下段
        # 左下小房间的墙（隔出房间2）
        _hwall(-15.0, -5.0, 20.0),   # 上墙（与中央通道分隔）
        _vwall( -5.0, -15.0, 20.0),  # 右墙
    ]
    rooms = [
        (-23.5, 9.5,   -5.0, 23.5),   # 房间1（左上 + 通道）
        (-23.5, -5.5, -23.5, -5.5),   # 房间2（左下，小）
        ( 10.5, 23.5, -23.5, 23.5),   # 房间3（右）
    ]
    return walls, rooms, rooms


# ══════════════════════════════════════════════════════════════════════
# Level 15：直走廊（最简单的迷宫式）
# ══════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────┐
#  │                          │
#  │ ████████      ██████████ │  ← 两段墙形成走廊
#  │                          │
#  │     长走廊（8m宽）       │
#  │                          │
#  │ ████████      ██████████ │
#  │                          │
#  └──────────────────────────┘
#
def make_straight_corridor() -> Scene:
    """简单直走廊，类似 12m maze level 2 但放大到 50m"""
    walls = [
        # 上墙左右段（y=8），中间留 8m 走廊
        _hwall(-15.0,  8.0, 18.0),
        _hwall( 15.0,  8.0, 18.0),
        # 下墙左右段（y=-8）
        _hwall(-15.0, -8.0, 18.0),
        _hwall( 15.0, -8.0, 18.0),
    ]
    rooms = [
        (-23.5, -7.0, -7.5, 7.5),     # 左房间
        ( 7.0,  23.5, -7.5, 7.5),     # 右房间
    ]
    return walls, rooms, rooms


# ══════════════════════════════════════════════════════════════════════
# 场景注册
# ══════════════════════════════════════════════════════════════════════

ROOM_LEVELS = {
    10: make_room_scene_6,
    11: make_room_scene_asymmetric,
    12: make_room_scene_corridor,
    13: make_simple_two_room,        # 最简单
    14: make_l_shape_3room,          # 中等
    15: make_straight_corridor,      # 简单直走廊
}


def get_room_scene(level: int) -> Scene:
    if level not in ROOM_LEVELS:
        raise ValueError(f"level {level} 不存在，可选 {list(ROOM_LEVELS.keys())}")
    return ROOM_LEVELS[level]()


def get_scene(level: int) -> list:
    walls, _, _ = get_room_scene(level)
    return walls


def get_valid_positions(level: int, n: int,
                        rng, min_dist: float = 3.0) -> list:
    """
    在房间场景的可行区域内采样 n 个不重叠的位置。
    """
    from configs import config as C
    walls, starts, goals = get_room_scene(level)
    half   = C.ARENA_SIZE / 2.0
    radius = C.ROBOT_RADIUS + 0.3

    all_regions = starts + goals
    positions   = []

    for _ in range(n):
        for _ in range(2000):
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

            if any(w.collides_circle(x, y, radius) for w in walls):
                continue
            if any(np.sqrt((x-p[0])**2 + (y-p[1])**2) < min_dist
                   for p in positions):
                continue

            positions.append((x, y))
            break
        else:
            x = float(rng.uniform(-half+radius, half-radius))
            y = float(rng.uniform(-half+radius, half-radius))
            positions.append((x, y))

    return positions