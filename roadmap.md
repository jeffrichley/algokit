Perfect, Jeff. Based on the full roadmap conversation you uploaded and your request to **ensure all implementation is under `src/`**, I’ve put together a comprehensive **project plan and directory structure** for **AlgoKit**, with packages organized by algorithm families, each housing implementations of their specific algorithms. Testing will live under a top-level `tests/` directory and mirror the structure of `src/`.

---

## ✅ ALGORITHMIC IMPLEMENTATION PLAN — AlgoKit

### 🧱 Base Project Structure

```
algokit/
├── src/
│   ├── classic_dp/                  # 📘 Dynamic Programming (Non-RL)
│   │   ├── linear_dp/
│   │   ├── grid_dp/
│   │   ├── knapsack/
│   │   ├── string_dp/
│   │   ├── interval_dp/
│   │   ├── bitmask_dp/
│   │   └── tree_dp/
│   ├── reinforcement_learning/      # 🧪 Model-Free RL
│   │   ├── tabular/
│   │   ├── deep_q/
│   │   └── policy_gradient/
│   ├── hrl/                         # 🧠 Hierarchical RL
│   │   ├── options_framework/
│   │   ├── feudal_rl/
│   │   └── hiro/
│   ├── dmps/                        # 🤖 Dynamic Movement Primitives
│   ├── gaussian_process/           # 📈 Gaussian Process Modeling
│   ├── real_time_control/          # 🕒 Real-Time Control
│   ├── mpc/                         # 🔁 Model Predictive Control
│   └── classical_planning/         # 🧭 Optional: A*, STRIPS, RRT, etc.
├── tests/
│   ├── classic_dp/
│   ├── reinforcement_learning/
│   ├── hrl/
│   ├── dmps/
│   ├── gaussian_process/
│   ├── real_time_control/
│   ├── mpc/
│   └── classical_planning/
├── notebooks/                      # Teaching/demo notebooks
├── assets/                         # GIFs, images, visualizations
├── README.md
├── pyproject.toml                  # With uv, ruff, black, pytest
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions (test, lint, mypy)
└── .pre-commit-config.yaml
```

---

### 🧭 Roadmap Phases

#### **🔹 Phase 1: Initialization**

* [x] Create GitHub repo
* [x] Scaffold project directory structure
* [x] Add `pyproject.toml` with `uv`, `pytest`, `ruff`, `mypy`
* [x] Add CI via GitHub Actions
* [x] Configure pre-commit: `black`, `ruff`, `mypy`, `trailing-whitespace`

#### **🔹 Phase 2: Implementation Modules**

Each major folder in `src/` will have:

* `__init__.py`
* One subpackage per concept category (e.g., `linear_dp`, `tabular`)
* Algorithm implementations with consistent naming:

  * `fibonacci.py`
  * `q_learning.py`
  * `dmp_encoder.py`
  * `pid_controller.py`
* Optional `visualize.py` or `demo_*.py` files for visualization

##### 🧩 Packages & Modules Breakdown

| **Family**               | **Submodules**                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------- |
| `classic_dp`             | `linear_dp`, `grid_dp`, `knapsack`, `string_dp`, `interval_dp`, `bitmask_dp`, `tree_dp` |
| `reinforcement_learning` | `tabular` (Q, SARSA), `deep_q` (DQN), `policy_gradient` (PPO, A2C)                      |
| `hrl`                    | `options_framework`, `feudal_rl`, `hiro`                                                |
| `dmps`                   | `dmp_core.py`, `imitation_learning.py`                                                  |
| `gaussian_process`       | `gp_regression.py`, `bayesian_optimization.py`                                          |
| `real_time_control`      | `pid.py`, `kalman_filter.py`, `bang_bang.py`                                            |
| `mpc`                    | `linear_mpc.py`, `nonlinear_mpc.py`                                                     |
| `classical_planning`     | `astar.py`, `dijkstra.py`, `rrt.py`, `strips.py`                                        |

#### **🔹 Phase 3: Testing**

* Each `tests/<family>/` mirrors the structure of `src/`
* Use `pytest` as test runner
* One `test_*.py` file per algorithm
* Example: `tests/classic_dp/linear_dp/test_fibonacci.py`

#### **🔹 Phase 4: Teaching & Demos**

* Jupyter notebooks in `notebooks/` for each family
* Use Matplotlib / Pygame for visualizations
* Include demo runners for live CLI interaction

---

### 🧠 Best Practices

* Type hint everything
* Add Google-style docstrings to all public methods
* Use `__all__` in `__init__.py` for clean API
* Keep algorithm logic separate from I/O and visualization
* Use `@dataclass` where useful (e.g. DMP params, MPC config)
