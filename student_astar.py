# student_astar.py
# ============================================================
# TASK
#   Implement A* search that returns (path, cost).
#
# SIGNATURE (do not change):
#   astar(start, goal, neighbors_fn, heuristic_fn, trace) -> (List[Coord], float)
#
# PARAMETERS
#   start, goal:           grid coordinates
#   neighbors_fn(u):       returns valid 4-neighbors of u
#   heuristic_fn(u, goal): returns a non-negative estimate to goal
#   trace:                 MUST call trace.expand(u) whenever you pop u
#                         from the PRIORITY QUEUE to expand it.
#
# EDGE COSTS
#   Assume unit step cost (=1) unless your runner specifies otherwise.
#   (If your runner supplies a graph.cost(u,v), adapt here if needed.)
#
# RETURN
#   (path, cost) where path is the list of coordinates from start to goal,
#   and cost is the sum of step costs along that path (float).
#   If no path exists, return ([], 0.0).
#
# IMPLEMENTATION HINT
# - Use min-heap over f = g + h.
# - Keep g[u] (cost from start), parent map, and a closed set.
# - On goal, reconstruct path and also compute cost (sum of steps).
# ============================================================

from typing import List, Tuple, Callable, Dict
import heapq

Coord = Tuple[int, int]

def astar(start: Coord,
          goal: Coord,
          neighbors_fn: Callable[[Coord], List[Coord]],
          heuristic_fn: Callable[[Coord, Coord], float],
          trace) -> Tuple[List[Coord], float]:
    """
    REQUIRED: call trace.expand(u) when you pop u from the PQ to expand.
    """
    # TODO: A* outline
    # 1) Initialize: g[start]=0, f[start]=h(start), push (f,g,node), parent, closed
    # 2) Loop: pop best f; skip if in closed; trace.expand(u); if goal -> reconstruct path
    #          else relax neighbors with unit cost, updating g/parent and pushing new (f,g,v)
    # 3) If not found: return []
    if start == goal:
        return [start]  # runner expects just a path list

    g: Dict[Coord, float] = {start: 0.0}
    parent: Dict[Coord, Coord | None] = {start: None}
    # Use a deterministic tie-breaker by including g in the heap tuple
    f0 = float(heuristic_fn(start, goal))
    pq: List[Tuple[float, float, Coord]] = [(f0, 0.0, start)]
    closed: set[Coord] = set()

    while pq:
        f_u, g_u, u = heapq.heappop(pq)
        if u in closed:
            continue

        try:
            trace.expand(u)
        except Exception:
            pass

        if u == goal:
            # Reconstruct path
            path: List[Coord] = [u]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.reverse()
            # Runner only uses path; cost can be derived if needed
            return path

        closed.add(u)

        for v in neighbors_fn(u):
            tg = g[u] + 1.0  # unit edge cost
            if (v not in g) or (tg < g[v] - 1e-12):
                g[v] = tg
                parent[v] = u
                fv = tg + float(heuristic_fn(v, goal))
                heapq.heappush(pq, (fv, tg, v))

    return []

# --- (ONLY IF YOUR RUNNER PASSES A Graph INSTEAD OF neighbors_fn) ---
# def astar_graph(graph, start, goal, heuristic_fn, trace):
#     return astar(start, goal, graph.neighbors, heuristic_fn, trace)
