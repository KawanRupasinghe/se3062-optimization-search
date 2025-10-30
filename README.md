# ASE Testing Tool (Assignment)

A small Python project for classic search and optimization tasks:
BFS, A*, IDS, Simulated Annealing, Linear Programming (LP), Dynamic Programming (DP), and custom heuristics.
A single runner generates problems, grades results, and writes JSON outputs for a browser summary.

---

## Features

- **BFS / A\***: shortest-path on grid (with admissible heuristics)
- **IDS**: iterative deepening with depth-limited search
- **Simulated Annealing**: path smoothing under `length + 0.2 × turns`
- **LP (corner-point)** and **DP (0/1 knapsack)** reference implementations
- **Runner + HTML summary**: writes `problem.json` and `results.json`; `index.html` renders grids, plots, and scores

---

## Quick start

**Requirements:** Python 3.9+

```bash
# (optional) create a virtual env
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### run the default problem
```bash
python runner.py --student_id ITxxxxxxxx
```

### view on port
```bash
python -m http.server 8000      
```

### open the summary 
#### macOS
```bash
open index.html
```

#### Windows
```bash
start index.html
```

#### Linux
```bash
xdg-open index.html
```
