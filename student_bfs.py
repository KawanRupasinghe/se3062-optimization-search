# student_bfs.py
# ============================================================
# TASK
#   Implement Breadth-First Search that returns a SHORTEST path
#   (by number of steps) from start to goal on an UNWEIGHTED grid.
#
# SIGNATURE (do not change):
#   bfs(start, goal, neighbors_fn, trace) -> List[Coord]
#
# PARAMETERS
#   start: (r, c)      tuple for start cell
#   goal:  (r, c)      tuple for goal cell
#   neighbors_fn(u):   function returning valid 4-neighbors of u
#   trace:             object with method trace.expand(node)
#                      YOU MUST call trace.expand(u) each time you
#                      pop/remove u from the FRONTIER to expand it.
#
# RETURN
#   A list of coordinates [(r0,c0), (r1,c1), ..., goal].
#   If no path is found, return [].
#
# NOTES
# - Use a QUEUE (FIFO).
# - Keep a parent map: parent[child] = node we came from.
# - Reconstruct path when you first reach goal.
# - You may print debug info; the runner will still grade correctly.
# ============================================================

from typing import List, Tuple, Callable, Dict
from collections import deque

Coord = Tuple[int, int]

def bfs(start: Coord,
        goal: Coord,
        neighbors_fn: Callable[[Coord], List[Coord]],
        trace) -> List[Coord]:
    """
    Implement classic BFS on an unweighted grid/graph.
    REQUIRED: call trace.expand(u) when you pop u from the queue.
    """
    # TODO: BFS steps
    # 1) Initialize a queue with start; parent = {start: None}
    # 2) While queue non-empty:
    #       u = queue.popleft(); trace.expand(u)
    #       if u == goal -> reconstruct and return path
    #       for v in neighbors_fn(u): if unseen -> parent[v]=u; enqueue v
    # 3) If not found, return []
    if start == goal:
        return [start]

    q = deque([start])
    parent: Dict[Coord, Coord | None] = {start: None}

    while q:
        u = q.popleft()
        # Mark expansion
        try:
            trace.expand(u)
        except Exception:
            # Be robust if trace is not provided or raises
            pass

        if u == goal:
            # Reconstruct
            path = [u]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.reverse()
            return path

        for v in neighbors_fn(u):
            if v not in parent:
                parent[v] = u
                q.append(v)

    return []

# --- (ONLY IF YOUR RUNNER PASSES A Graph INSTEAD OF neighbors_fn) ---
# def bfs_graph(graph, start, goal, trace):
#     return bfs(start, goal, graph.neighbors, trace)
