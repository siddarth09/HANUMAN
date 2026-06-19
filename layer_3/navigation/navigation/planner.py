"""A* global planner over the geometric cost grid.

8-connected, cost-weighted: step cost = move_dist * (1 + cost_penalty * avg_cell_cost).
Lethal cells are impassable.
"""
from __future__ import annotations
import heapq
import numpy as np

# 8-connectivity (di, dj, step length)
_NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, 1.41421356), (-1, 1, 1.41421356),
       (1, -1, 1.41421356), (1, 1, 1.41421356)]


def astar(cost: np.ndarray, lethal: np.ndarray, start, goal,
          cost_penalty: float = 4.0):
    """A* on a (H, W) grid. start/goal are (i, j) = (col, row).

    Returns a list of (i, j) cells from start to goal, or None if unreachable.
    """
    H, W = cost.shape
    si, sj = start
    gi, gj = goal
    if not (0 <= si < W and 0 <= sj < H and 0 <= gi < W and 0 <= gj < H):
        return None
    if lethal[sj, si] or lethal[gj, gi]:
        return None

    def h(i, j):
        return np.hypot(i - gi, j - gj)

    open_heap = [(h(si, sj), 0.0, (si, sj))]
    came_from = {}
    g_score = {(si, sj): 0.0}
    closed = set()

    while open_heap:
        _, g, (i, j) = heapq.heappop(open_heap)
        if (i, j) == (gi, gj):
            path = [(i, j)]
            while (i, j) in came_from:
                i, j = came_from[(i, j)]
                path.append((i, j))
            return path[::-1]
        if (i, j) in closed:
            continue
        closed.add((i, j))
        for di, dj, step in _NB:
            ni, nj = i + di, j + dj
            if not (0 <= ni < W and 0 <= nj < H) or lethal[nj, ni]:
                continue
            avg_c = 0.5 * (cost[j, i] + cost[nj, ni])
            tentative = g + step * (1.0 + cost_penalty * avg_c)
            if tentative < g_score.get((ni, nj), np.inf):
                came_from[(ni, nj)] = (i, j)
                g_score[(ni, nj)] = tentative
                heapq.heappush(open_heap, (tentative + h(ni, nj), tentative, (ni, nj)))
    return None
