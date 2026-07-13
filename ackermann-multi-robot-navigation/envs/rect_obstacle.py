"""
envs/rect_obstacle.py
──────────────────────────────────────────────────────────────────────
轴对齐矩形障碍物（AABB）

提供：
  - 圆形机器人碰撞检测（最近点法）
  - 激光射线相交检测（slab法）
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RectObstacle:
    """
    轴对齐矩形障碍物。
    (cx, cy) = 中心，w = 宽(x方向)，h = 高(y方向)
    """
    cx: float
    cy: float
    w:  float
    h:  float

    @property
    def x_min(self): return self.cx - self.w / 2
    @property
    def x_max(self): return self.cx + self.w / 2
    @property
    def y_min(self): return self.cy - self.h / 2
    @property
    def y_max(self): return self.cy + self.h / 2

    def collides_circle(self, px: float, py: float, radius: float) -> bool:
        """圆形机器人与矩形碰撞检测（最近点法）。"""
        # 矩形内最近点
        nearest_x = float(np.clip(px, self.x_min, self.x_max))
        nearest_y = float(np.clip(py, self.y_min, self.y_max))
        dx = px - nearest_x
        dy = py - nearest_y
        return (dx*dx + dy*dy) < radius*radius

    def ray_intersect(self, ox: float, oy: float,
                      dx: float, dy: float,
                      max_range: float) -> float:
        """
        射线与矩形相交检测（slab法）。
        返回命中距离，未命中返回 max_range。

        ox,oy: 射线起点
        dx,dy: 射线方向（单位向量）
        """
        t_min = 0.0
        t_max = max_range

        # x slab
        if abs(dx) > 1e-9:
            tx1 = (self.x_min - ox) / dx
            tx2 = (self.x_max - ox) / dx
            t_min = max(t_min, min(tx1, tx2))
            t_max = min(t_max, max(tx1, tx2))
        else:
            if ox < self.x_min or ox > self.x_max:
                return max_range

        # y slab
        if abs(dy) > 1e-9:
            ty1 = (self.y_min - oy) / dy
            ty2 = (self.y_max - oy) / dy
            t_min = max(t_min, min(ty1, ty2))
            t_max = min(t_max, max(ty1, ty2))
        else:
            if oy < self.y_min or oy > self.y_max:
                return max_range

        if t_max >= t_min and t_min >= 0:
            return float(t_min)
        return max_range
