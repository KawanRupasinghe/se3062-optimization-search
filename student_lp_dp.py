# student_lp_dp.py
from __future__ import annotations
from typing import List, Tuple, Optional
from functools import lru_cache
import math

"""
===========================================================
Overall Pseudocode & Study Guide (LP + DP)
===========================================================

A) Linear Programming in 2 variables (vertex enumeration)
   Goal: maximize Z = c1*x + c2*y subject to a1*x + a2*y <= b (and x>=0, y>=0)

   1) Model the feasible region:
      - Collect all given constraints (<= type).
      - Add non-negativity constraints: x>=0, y>=0.

   2) Enumerate candidate vertices:
      - Intersect every pair of constraint boundary lines (treat each as equality).
      - Keep only well-defined intersections (ignore parallel lines).
      - (Optionally) include the origin explicitly.

   3) Feasibility test:
      - For each candidate (x,y), check all constraints (<= type) with a small numeric tolerance.

   4) Objective evaluation:
      - Evaluate Z at each feasible vertex.
      - Select the best according to Z; tie-break deterministically if needed.

B) 0/1 Knapsack (Dynamic Programming)
   Problem: given values[i], weights[i], capacity C, pick subset to maximize total value without
            exceeding C.

   1) Bottom-Up Table (iterative):
      - Define dp[i][cap] = best value using items from i..n-1 with remaining capacity 'cap'.
      - Fill the table in an order that ensures subproblems are ready (e.g., i from n-1→0).
      - Transition: choose between skipping item i or taking it (if it fits), then record the best.

   2) Top-Down with Memoization (recursive):
      - Define f(i, cap): best value using items from i..n-1 with capacity 'cap'.
      - Base cases: end of items or cap==0 -> return 0.
      - Transition: if item i doesn’t fit, skip; else max(skip, take).
      - Cache results to avoid recomputation.

Notes:
- Use a small tolerance EPS for LP comparisons with floats.
- Keep implementations simple, readable, and consistent with the above plan.
"""

# ---------- LP (12.5% of total grade) ----------
Constraint = Tuple[float, float, float]  # a1, a2, b  meaning  a1*x + a2*y <= b
EPS = 1e-9

def _intersect(c1: Constraint, c2: Constraint) -> Optional[Tuple[float, float]]:
    """
    Compute the intersection point of two *boundary lines* obtained from constraints.
    Each constraint (a1, a2, b) corresponds to a boundary line a1*x + a2*y = b.

    Detailed steps (do NOT paste final formulae; write the algebra yourself):
      1) Unpack both constraints into coefficients.
      2) Treat them as a 2x2 linear system in variables x and y.
      3) Compute the determinant of the 2x2 coefficient matrix.
         - If it's (near) zero, lines are parallel/ill-conditioned → return None.
      4) Otherwise, solve the system for (x, y) using your preferred method for 2x2 systems.
      5) Return (x, y) as floats.

    Return:
      (x, y) if a unique intersection exists and is well-conditioned; otherwise None.
    """
    # TODO: Solve the 2x2 linear system for intersection; guard against near-zero determinant
    a1, a2, b1 = c1
    c1_, c2_, b2 = c2
    # Solve: a1*x + a2*y = b1;  c1_*x + c2_*y = b2
    det = a1 * c2_ - a2 * c1_
    if abs(det) < 1e-12:
        return None
    x = (b1 * c2_ - a2 * b2) / det
    y = (a1 * b2 - b1 * c1_) / det
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (float(x), float(y))


def _is_feasible(pt: Tuple[float, float], constraints: List[Constraint]) -> bool:
    """
    Check whether point (x,y) satisfies ALL constraints a1*x + a2*y <= b (with tolerance).

    Detailed steps:
      1) For each constraint (a1, a2, b), compute the left-hand side at (x,y).
      2) Compare LHS to RHS with a small EPS slack to account for floating-point rounding.
      3) If any constraint is violated beyond tolerance, return False.
      4) If all pass, return True.
    """
    # TODO: Check all constraints a1*x + a2*y <= b with an EPS tolerance
    x, y = pt
    for a1, a2, b in constraints:
        lhs = a1 * x + a2 * y
        if lhs > b + EPS:
            return False
    return True


def feasible_vertices(constraints: List[Constraint]) -> List[Tuple[float, float]]:
    """
    (6%) Enumerate and return all *feasible* vertices (x,y) of the polygonal feasible region.

    Detailed steps:
      1) Copy input constraints and append non-negativity:
         - Represent x>=0 and y>=0 as <=-type constraints suitable for your intersection logic.
           (Hint: you'll add two extra constraints to the list.)
      2) For every unordered pair of constraints, compute the intersection of their *boundary lines*.
         - Skip pairs that do not produce a unique intersection.
      3) Collect all intersection points plus the origin (as a simple additional candidate).
      4) Run the feasibility test on each candidate using _is_feasible.
      5) De-duplicate points robustly (e.g., rounding to fixed decimals or using a tolerance-based key).
      6) Return the list of unique feasible vertices.
    """
    # TODO: Build candidates by intersecting pairs + non-negativity, then filter and dedup
    # Copy constraints and add non-negativity (x >= 0, y >= 0)
    cons = list(constraints)
    # Represent x >= 0 as -1*x <= 0 and y >= 0 as -1*y <= 0
    cons.extend([(-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)])

    # Generate all pairwise intersections among boundary lines
    n = len(cons)
    candidates: List[Tuple[float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pt = _intersect(cons[i], cons[j])
            if pt is not None:
                candidates.append(pt)

    # Include the origin explicitly
    candidates.append((0.0, 0.0))

    # Filter feasible and de-duplicate with rounding key
    unique = {}
    for x, y in candidates:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        # Quick reject large negative numbers before feasibility test
        if x < -1e6 or y < -1e6:
            continue
        if _is_feasible((x, y), cons):
            key = (round(x, 10), round(y, 10))
            if key not in unique:
                unique[key] = (float(key[0]), float(key[1]))

    return list(unique.values())


def maximize_objective(vertices: List[Tuple[float, float]], c1: float, c2: float) -> Tuple[Tuple[float, float], float]:
    """
    (6.5%) Evaluate Z = c1*x + c2*y over feasible vertices and return (best_point, best_value).

    Detailed steps:
      1) Handle edge case: if vertices is empty, return a sensible default ((0.0, 0.0), 0.0).
      2) Initialize "best" with the first vertex and its objective value.
      3) Scan through remaining vertices:
         - Compute Z at each vertex.
         - If strictly better (beyond EPS), update best.
         - If tied within EPS, resolve deterministically (e.g., prefer larger x; if x ties, larger y).
      4) Return the best vertex and its value as a float.
    """
    # TODO: Scan vertices, compute Z=c1*x+c2*y, use deterministic tie-breaking
    if not vertices:
        return (0.0, 0.0), 0.0

    def obj(pt: Tuple[float, float]) -> float:
        return c1 * pt[0] + c2 * pt[1]

    best_pt = vertices[0]
    best_val = obj(best_pt)
    for pt in vertices[1:]:
        val = obj(pt)
        if val > best_val + EPS:
            best_pt, best_val = pt, val
        elif abs(val - best_val) <= EPS:
            # Tie-break: prefer larger x; if tie, larger y
            if (pt[0] > best_pt[0] + EPS) or (abs(pt[0] - best_pt[0]) <= EPS and pt[1] > best_pt[1] + EPS):
                best_pt, best_val = pt, val

    return best_pt, float(best_val)


# ---------- DP (12.5% of total grade) ----------
def knapsack_bottom_up(values: List[int], weights: List[int], capacity: int) -> int:
    """
    (6.5%) Bottom-up 0/1 knapsack. Return the optimal value (int).

    Table design (choose one and stick to it):
      Option A (common): dp[i][cap] = best value using items i..n-1 with remaining capacity 'cap'.
        - Dimensions: (n+1) x (capacity+1), initialized to 0.
        - Fill order: i from n-1 down to 0; cap from 0 to capacity.
        - Transition:
            skip = dp[i+1][cap]
            take = values[i] + dp[i+1][cap - weights[i]]  (only if it fits)
            dp[i][cap] = max(skip, take)

      Option B: dp[i][cap] = best value using first i items (0..i-1).
        - Dimensions: (n+1) x (capacity+1).
        - Fill order: i from 1 to n; cap from 0 to capacity.
        - Transition mirrors Option A but with shifted indices.

    Detailed steps:
      1) Validate input lengths and capacity.
      2) Allocate and initialize the 2D table to zeros.
      3) Implement your chosen formulation consistently, filling the table.
      4) Return the appropriate cell as the answer (depends on formulation).
    """
    # TODO: Bottom-up DP (Option A): dp[i][cap] = best using items i..n-1
    n = len(values)
    if n != len(weights) or capacity < 0:
        return 0
    # Option A formulation: dp[i][cap] = best using items i..n-1 with capacity cap
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        v, w = values[i], weights[i]
        for cap in range(0, capacity + 1):
            best = dp[i + 1][cap]  # skip
            if w <= cap:
                best = max(best, v + dp[i + 1][cap - w])
            dp[i][cap] = best
    return dp[0][capacity]


def knapsack_top_down(values: List[int], weights: List[int], capacity: int) -> int:
    """
    (6%) Top-down (memoized) 0/1 knapsack. Return optimal value (int).

    Recurrence (typical):
      f(i, cap) = 0                                     if i==n or cap==0
      f(i, cap) = f(i+1, cap)                           if weights[i] > cap
      f(i, cap) = max(
                      f(i+1, cap),                      # skip item i
                      values[i] + f(i+1, cap - w[i])    # take item i
                   )                                    otherwise

    Detailed steps:
      1) Define an inner function f(i, cap) and decorate with @lru_cache(None).
      2) Implement the base cases (past last item or capacity empty).
      3) Implement the recurrence using the rule above.
      4) Return f(0, capacity).
    """
    n = len(values)
    if n != len(weights) or capacity < 0:
        return 0

    # TODO: Memoized recursion f(i,cap) with base/fit checks and max(skip,take)
    @lru_cache(maxsize=None)
    def f(i: int, cap: int) -> int:
        if i >= n or cap <= 0:
            return 0
        w = weights[i]
        v = values[i]
        # If item doesn't fit, must skip
        if w > cap:
            return f(i + 1, cap)
        # Otherwise choose best of skip or take
        return max(f(i + 1, cap), v + f(i + 1, cap - w))

    return f(0, capacity)


# ------------- Optional local smoke test -------------
if __name__ == "__main__":
    # Minimal checks that won't reveal answers; just ensures your functions run.
    cons = [
        (1.0, 1.0, 6.0),
        (1.0, 0.0, 4.0),
        (0.0, 1.0, 5.0),
        (2.0, 1.0, 8.0),
    ]
    try:
        V = feasible_vertices(cons)
        print(f"[LP] #vertices found: {len(V)}")
        if V:
            bp, bv = maximize_objective(V, 3.0, 5.0)
            print(f"[LP] best vertex (masked): {bp}, value={bv:.2f}")
    except NotImplementedError:
        print("[LP] TODOs not yet implemented")

    vals = [6,5,18,15,10]
    wts  = [2,2,6,5,4]
    cap  = 10
    try:
        print("[DP] bottom-up (masked run):", knapsack_bottom_up(vals, wts, cap))
    except NotImplementedError:
        print("[DP] bottom-up TODO not implemented")
    try:
        print("[DP] top-down  (masked run):", knapsack_top_down(vals, wts, cap))
    except NotImplementedError:
        print("[DP] top-down  TODO not implemented")
