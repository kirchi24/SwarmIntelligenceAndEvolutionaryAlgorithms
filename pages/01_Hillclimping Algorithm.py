from importlib import import_module
import numpy as np
import streamlit as st

# setup imports
from src.benchmark import (
    quadratic,
    sinusoidal,
    ackley,
    rosenbrock,
    rastrigin,
    visualize_1d_input,
    visualize_2d_input_surface,
    add_trajectory_1d_input,
    add_trajectory_2d_input_surface,
)

from src.HillClimbingAlgorithm.algorithm import (
    hill_climbing,
    steepest_hill_climbing,
    continuous_neighborhood_batch as continuous_neighborhood,
)

# Streamlit setup
# 
st.set_page_config(page_title="Hill Climbing Algorithm", layout="wide")
st.title("Hill Climbing Algorithm Documentation")

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs(["Introduction", "Methods", "Results", "Discussion"])

# Einleitung
with tabs[0]:
    st.header("Introduction")
    st.markdown(
        """
        Hill Climbing is a **local search optimization algorithm** that iteratively improves a solution 
        by exploring its neighboring solutions. It is widely used for its simplicity and efficiency 
        in low-dimensional problems.

        ### How It Works
        1. Start with an initial solution.
        2. Evaluate neighboring solutions.
        3. Move to the neighbor with the best improvement.
        4. Repeat until no further improvement is possible or max iterations (m) is reached.

        ### Variants
        - **Simple Hill Climbing**: Examines one neighbor at a time and moves if it improves the solution.
        - **Steepest Ascent Hill Climbing**: Evaluates all neighbors and selects the one with the maximum improvement.

        ### Advantages
        - Easy to implement and understand.
        - Efficient for **unimodal functions** with a single global optimum.
        - Low memory requirements.

        ### Limitations
        - Can get stuck in **local maxima or plateaus**.  
        - Sensitive to the choice of **initial solution** and **step size**.
        - Not suitable for **high-dimensional or complex landscapes**.

        ### Use Cases
        - Function optimization in low-dimensional spaces.
        - Problems where a **good-enough solution** is acceptable.
        - Situations requiring **fast, lightweight algorithms**.
        """
    )

# Methoden
with tabs[1]:
    st.header("Implementation Overview")
    st.markdown(
        """
        This module provides **two variants of the Hill Climbing algorithm** for continuous optimization problems:

        ### 1. First-Improvement Hill Climbing (`hill_climbing`)
        - Evaluates **one neighbor at a time** and moves to the first improving solution.
        - Lightweight and memory-efficient (stores only one neighbor).
        - Parameters: `x0`, `step_size`, `max_iter`, `tol`, `patience`, and a **batch-capable neighborhood function**.

        ### 2. Steepest-Ascent Hill Climbing (`steepest_hill_climbing`)
        - Evaluates a **batch of neighbors** per iteration and selects the best one.
        - More robust but computationally heavier.
        - Parameters: same as above, plus `samples` for neighbors per iteration.

        ### Key Features
        - **Customizable Neighborhoods**: `continuous_neighborhood_batch` generates perturbed solutions for exploration.
        - **Stopping Criteria**: Supports `max_iter`, minimum improvement (`tol`), and lack of progress (`patience`).
        - **Trajectory Tracking**: Returns the full path of solutions, best solution, its value, and total evaluations.
        """
    )

# Results
with tabs[2]:
    st.header("Interactive Playground - Run Hill Climbing")

    # Sidebar parameters
    st.sidebar.header("Algorithm Settings")
    algo_choice = st.sidebar.selectbox("Algorithm", ["Simple Hill Climbing", "Steepest Hill Climbing"])
    func_choice = st.sidebar.selectbox(
        "Benchmark Function",
        ["Quadratic (1D)", "Sinusoidal (1D)", "Ackley (2D)", "Rosenbrock (2D)", "Rastrigin (2D)"],
    )
    max_iter = st.sidebar.slider("Max Iterations", 50, 2000, 300, 50)
    step_size = st.sidebar.slider("Step Size", 0.01, 1.0, 0.1, 0.01)
    samples = st.sidebar.slider("Neighbors per Iteration (Steepest)", 1, 50, 10)
    patience = st.sidebar.slider("Patience (early stopping)", 1, 50, 10)
    tol = 10 ** st.sidebar.slider(
        "log10(improvement tolerance)", min_value=-6.0, max_value=0.0, value=-3.0, step=0.1
    )
    seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)
    np.random.seed(seed)

    # Select benchmark
    if func_choice == "Quadratic (1D)":
        f, x0, dim = quadratic, np.array([np.random.uniform(-5, 5)]), 1
    elif func_choice == "Sinusoidal (1D)":
        f, x0, dim = sinusoidal, np.array([np.random.uniform(-5, 5)]), 1
    elif func_choice == "Ackley (2D)":
        f, x0, dim = ackley, np.random.uniform(-5, 5, 2), 2
    elif func_choice == "Rosenbrock (2D)":
        f, x0, dim = rosenbrock, np.random.uniform(-2, 2, 2), 2
    elif func_choice == "Rastrigin (2D)":
        f, x0, dim = rastrigin, np.random.uniform(-5, 5, 2), 2

    # Run algorithm
    if algo_choice == "Simple Hill Climbing":
        x_best, f_best, traj, evals = hill_climbing(
            f,
            x0,
            continuous_neighborhood,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            patience=patience,
        )
    else:
        x_best, f_best, traj, evals = steepest_hill_climbing(
            f,
            x0,
            continuous_neighborhood,
            step_size=step_size,
            max_iter=max_iter,
            samples=samples,
            tol=tol,
            patience=patience,
        )

    f_best_display = f_best.item() if isinstance(f_best, np.ndarray) else f_best

    # Results
    st.subheader("Optimization Result")
    st.write(f"**Best Solution:** {x_best}")
    st.write(f"**Function Value:** {f_best_display:.6f}")
    st.write(f"**Function Evaluations:** {evals}")

    # Visualization
    st.subheader("Search Trajectory Visualization")
    if dim == 1:
        fig = visualize_1d_input(f, name=func_choice, xlim=(-5, 5))
        fig = add_trajectory_1d_input(fig, traj, f)
        st.plotly_chart(fig, use_container_width=True)
    elif dim == 2:
        fig = visualize_2d_input_surface(f, name=func_choice, xlim=(-5, 5))
        fig = add_trajectory_2d_input_surface(fig, traj, f)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Visualization not available for this function.")

# Discussion
with tabs[3]:
    st.header("Discussion")
    st.markdown(
        """
        **Performance:**
        - Simple Hill Climbing is faster but may get stuck in local minima.
        - Steepest Hill Climbing explores more neighbors, often finding better optima but at higher cost.

        **Parameter Sensitivity:**
        - Step size, tolerance, and initial solution strongly affect results.
        - Too small a step size slows progress; too large may skip optimal regions.

        **Limitations:**
        - Struggles with multimodal and high-dimensional functions.
        - Deterministic unless restarted from multiple random seeds.

        **Potential Improvements:**
        - Adaptive or decreasing step size over iterations.
        - Random restarts to escape local minima.
        - Hybridization with simulated annealing or GA for global search.
        """
    )
    st.markdown(
        "<small>Note: This documentation was generated with the assistance of ChatGPT.</small>",
        unsafe_allow_html=True,
    )
