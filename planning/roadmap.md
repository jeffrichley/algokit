# 📘 Roadmap: AI, Planning, and Control Algorithms

This document outlines a comprehensive roadmap for implementing and teaching a variety of foundational and advanced algorithms in AI, control, planning, and machine learning. It is designed to support instructional use, demos for students, and personal mastery. This project is named **AlgoKit**.

---

## 🔁 Step-by-Step Plan for AlgoKit

### Phase 1: Setup and Scaffolding

1. ✅ Initialize GitHub repository: `algokit`
2. ✅ Create top-level folder structure (see end of doc)
3. ✅ Create template README and CONTRIBUTING docs
4. ✅ Define testing framework (e.g. `pytest`, `unittest`)
5. ✅ Set up CI (GitHub Actions or similar)

### Phase 2: Core Implementations

6. 🧠 Implement Classic DP problems (week-by-week)
7. 🧪 Implement core RL algorithms (Q, SARSA, PPO, etc.)
8. 🧱 Build HRL scaffolding (options, subgoal modeling)
9. 🧬 Add DMP encoding/decoding + demo tasks
10. 📈 Integrate GP regression and Bayesian opt
11. ⏱ Build PID + Kalman controller demos
12. 📉 Implement MPC with linear and nonlinear examples
13. 🧭 Add optional planning tools (A\*, RRT, etc.)

### Phase 3: Visualization and Teaching Support

14. 📝 Write Jupyter notebooks for each algorithm
15. 📊 Add animated visualizations (matplotlib, gifs)
16. 🧑‍🏫 Create sample lecture slides (reveal.js or pptx)
17. 🎓 Add student exercises and challenge problems

### Phase 4: Final Polish and Deployment

18. 🧹 Clean and lint all code (ruff, black, mypy)
19. 🚀 Publish as a JupyterBook and/or GitHub Pages
20. 🔁 Add plug-and-play `main.py` launcher for CLI use

---

## 1. Classic Dynamic Programming (DP)

### Goal:

Model problems with optimal substructure and overlapping subproblems.

### Categories & Algorithms:

#### 1.1 Linear DP

* Fibonacci (memoized/tabulated)
* Longest Increasing Subsequence (LIS)
* Maximum Subarray (Kadane’s Algorithm)
* House Robber
* Climbing Stairs
* Jump Game I & II

#### 1.2 2D / Grid-Based DP

* Unique Paths
* Minimum Path Sum
* Longest Common Subsequence (LCS)
* Edit Distance
* Maximum Square Submatrix

#### 1.3 Knapsack Variants

* 0/1 Knapsack
* Unbounded Knapsack
* Partition Equal Subset Sum
* Target Sum

#### 1.4 String DP

* LCS
* Edit Distance
* Regular Expression Matching
* Wildcard Matching
* Palindromic Substrings

#### 1.5 Interval DP

* Matrix Chain Multiplication
* Burst Balloons
* Palindrome Partitioning

#### 1.6 Bitmask / Digit DP

* Traveling Salesman Problem (TSP)
* Assignment Problem
* Digit Sum Count

#### 1.7 Tree/DAG DP

* Diameter of Tree
* House Robber III
* Longest Path in DAG

---

## 2. Reinforcement Learning (Model-Free)

### Goal:

Teach agents to learn optimal behavior through interaction with an environment.

### Algorithms:

* Q-Learning
* SARSA
* Deep Q-Network (DQN)
* Proximal Policy Optimization (PPO)
* A2C / A3C

### Concepts:

* Exploration vs exploitation
* Discounted return
* Temporal difference learning
* Experience replay and target networks

---

## 3. Hierarchical Reinforcement Learning (HRL)

### Goal:

Introduce temporal abstraction and multi-level decision-making.

### Algorithms:

* Options Framework
* Feudal Reinforcement Learning
* MAXQ Decomposition
* HIRO (state-of-the-art)

### Concepts:

* Subgoals
* Macro-actions
* Inter-option policy training

---

## 4. Dynamic Movement Primitives (DMPs)

### Goal:

Encode reusable trajectories for motion planning.

### Algorithms:

* DMP encoding/decoding
* Imitation learning using DMPs

### Concepts:

* Nonlinear dynamical systems
* Attractor landscapes
* Temporal scaling

---

## 5. Gaussian Process Modeling

### Goal:

Model functions with uncertainty using Bayesian regression.

### Algorithms:

* GP Regression
* Sparse GPs
* Bayesian Optimization
* PILCO (Model-based RL)

### Concepts:

* Kernel functions
* Posterior predictive distribution
* Acquisition functions (UCB, EI)

---

## 6. Real-Time Control

### Goal:

Implement fast, reactive control systems with feedback.

### Algorithms:

* PID Controller (P/PI/PD/PID)
* Bang-bang control
* Kalman Filter

### Concepts:

* Feedback vs feedforward
* Tuning gains
* Real-time latency constraints

---

## 7. Model Predictive Control (MPC)

### Goal:

Use optimization to plan control actions over a horizon.

### Algorithms:

* Finite Horizon MPC
* Nonlinear MPC
* Learning-based MPC

### Concepts:

* Cost functions
* State and control constraints
* Receding horizon optimization

---

## 8. (Optional) Classical Planning Algorithms

### Goal:

Solve planning problems in state or configuration space.

### Algorithms:

* A\*
* Dijkstra’s
* RRT / PRM (sampling-based)
* STRIPS / PDDL (symbolic AI)

### Concepts:

* Heuristics
* Search graph expansion
* Plan execution and validation

---

## Suggested Structure

```bash
algokit/
├── classic_dp/
├── reinforcement_learning/
├── hrl/
├── dmps/
├── gaussian_process/
├── real_time_control/
├── mpc/
├── classical_planning/
├── notebooks/
└── README.md
```

Each folder should include:

* Standalone implementations
* Visualization and test cases
* Notebook versions for demo/teaching
* Annotated examples with didactic comments

---

Ready to start building out each module or publishing this as a GitHub repo or JupyterBook.
