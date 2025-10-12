import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Pfad zu src hinzufügen, damit Imports funktionieren
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.benchmark_functions.benchmark import quadratic, sinusoidal, ackley
from src.hillClimbing_algorithm.main import (
    hill_climbing,
    steepest_hill_climbing,
    continuous_neighborhood,
)

# -----------------------------
# 1️⃣ Introduction
# -----------------------------
st.title("🧗 Hill Climbing Algorithm – Interactive Documentation")

st.markdown("""
### 1. Introduction

**Hill Climbing (HC)** is a simple local search optimization algorithm.  
It starts from an initial solution candidate and iteratively moves to neighboring solutions if they improve the objective function (for minimization, moves toward lower values).

**How it works (basics):**
- Start with an initial solution `x0 ∈ R^n`.
- Generate a neighbor using a **neighborhood function**.
- Move to the neighbor if `f(x_new) < f(x_current)`.
- Repeat until termination criterion is met (max iterations, no improvement, tolerance, etc.)

**Strengths:**
- Simple to implement and understand
- Efficient for smooth or low-dimensional landscapes

**Weaknesses:**
- Easily trapped in **local minima**
- Sensitive to step size and initialization
- No global search mechanism

**Computational complexity:**
- O(iterations × evaluations_per_iteration)
- Simple HC: one evaluation per iteration
- Steepest HC: multiple evaluations per iteration
""")

# -----------------------------
# 2️⃣ Methods
# -----------------------------
st.header("2. Methods – Implementation Details")

st.markdown("""
We implemented **two variants** of Hill Climbing:

1. **Simple Hill Climbing**
   - Samples **one random neighbor** per iteration.
   - Pros: fast, few evaluations.
   - Cons: easily trapped in local minima.

2. **Steepest Descent Hill Climbing**
   - Samples **multiple neighbors** per iteration and moves to the best.
   - Pros: better exploration, less likely to get stuck.
   - Cons: more evaluations per iteration, slower.

**Parameters for both variants:**
- Initial solution (`x0`), which can be 1D or n-dimensional
- **Fitness function** (`f`): benchmark functions (Quadratic, Sinusoidal, Ackley)
- **Neighborhood function** (`continuous_neighborhood`): defines how neighbors are generated
- **Termination criterion**: max iterations and tolerance for improvement
- Step size: controls how far the neighbor is from current solution
- Samples (Steepest variant): number of neighbors per iteration
""")

# -----------------------------
# 3️⃣ Interactive Results
# -----------------------------
st.header("3. Run Hill Climbing – Interactive Playground")

# Sidebar parameters
st.sidebar.header("Algorithm Settings")
algo_choice = st.sidebar.selectbox("Algorithm", ["Simple Hill Climbing", "Steepest Hill Climbing"])
func_choice = st.sidebar.selectbox("Benchmark Function", ["Quadratic (1D)", "Sinusoidal (1D)", "Ackley (2D)"])
max_iter = st.sidebar.slider("Max Iterations", 50, 2000, 300, 50)
step_size = st.sidebar.slider("Step Size", 0.01, 1.0, 0.1, 0.01)
samples = st.sidebar.slider("Neighbors per Iteration (Steepest)", 2, 50, 10)
tol = st.sidebar.number_input("Improvement Tolerance", 1e-8, 1e-2, 1e-6, format="%.1e")
seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

np.random.seed(seed)

# Select function and initial solution
if func_choice == "Quadratic (1D)":
    f = quadratic
    x0 = np.array([np.random.uniform(-5, 5)])
    dim = 1
elif func_choice == "Sinusoidal (1D)":
    f = sinusoidal
    x0 = np.array([np.random.uniform(-5, 5)])
    dim = 1
else:
    f = lambda x: ackley(x)
    x0 = np.random.uniform(-5, 5, size=2)
    dim = 2

# Run selected algorithm
if algo_choice == "Simple Hill Climbing":
    x_best, f_best, traj, evals = hill_climbing(
        f, x0, lambda x: continuous_neighborhood(x, step_size), max_iter=max_iter, tol=tol
    )
else:
    x_best, f_best, traj, evals = steepest_hill_climbing(
        f, x0, lambda x: continuous_neighborhood(x, step_size),
        max_iter=max_iter, samples=samples, tol=tol
    )

# Format f_best safely
f_best_display = f_best.item() if isinstance(f_best, np.ndarray) and f_best.size == 1 else f_best

# Display results
st.subheader("Optimization Result")
st.write(f"**Best Solution:** {x_best}")
st.write(f"**Function Value:** {f_best_display}")
st.write(f"**Objective Function Evaluations:** {evals}")

# -----------------------------
# Visualization
# -----------------------------
st.subheader("Search Trajectory Visualization")
fig, ax = plt.subplots(figsize=(8, 5))

if dim == 1:
    x = np.linspace(-5, 5, 400)
    y = f(x)
    ax.plot(x, y, label="f(x)")
    ax.scatter(traj[:, 0], f(np.array(traj[:, 0])), color="red", s=40, label="Trajectory")
    ax.set_title(f"{algo_choice} on {func_choice}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f(np.array([xx, yy])) for xx, yy in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.plot(traj[:, 0], traj[:, 1], 'r.-', label="Search Path")
    plt.title(f"{algo_choice} on {func_choice}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    st.pyplot(plt)

# -----------------------------
# 4️⃣ Discussion
# -----------------------------
st.header("4. Discussion / Insights")
st.markdown("""
**Observations:**
- Simple Hill Climbing is fast but may get trapped in local minima.
- Steepest variant explores better but requires more function evaluations.
- The trajectory shows how the algorithm navigates the search space.

**Parameter Sensitivity:**
- Step size heavily influences convergence and ability to escape shallow local minima.
- Maximum iterations and tolerance control the stopping behavior.
- Random seed affects initialization, especially for multimodal functions.

**Potential Improvements:**
- Adaptive step size
- Random restarts
- Hybrid algorithms (e.g., combine with Simulated Annealing)
""")

st.markdown("---")
st.markdown(
    "<small>Note: ChatGPT was used to create the visuals for this page.</small>",
    unsafe_allow_html=True
)