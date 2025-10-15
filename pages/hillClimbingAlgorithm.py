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
    visualize_1d_function,
    visualize_3d_function,
    add_trajectory_1d,
    add_trajectory_3d,
)

from src.HillClimbingAlgorithm.algorithm import (
    hill_climbing,
    steepest_hill_climbing,
    continuous_neighborhood_batch as continuous_neighborhood,
)


# title & introduction
st.title("Hill Climbing Algorithm - Interactive Documentation")

st.markdown(
    """
### Introduction
Hill Climbing (HC) is a local search optimization algorithm.
It iteratively moves to better neighboring solutions to minimize an objective function.

Key Points:
- Simple HC: samples one neighbor per iteration
- Steepest HC: samples multiple neighbors and moves to the best
- Strengths: simple, efficient in low dimensions
- Weaknesses: local minima, sensitive to initialization and step size
"""
)


# methods
st.header("Methods - Implementation Details")
st.markdown(
    """
Parameters for both variants:
- Initial solution x0 (1D or nD)
- Benchmark function f
- Neighborhood function continuous_neighborhood
- Max iterations, tolerance, step size
- Number of neighbors per iteration (Steepest HC)
"""
)


# interactive Playground
st.header("Run Hill Climbing - Interactive Playground")

# Sidebar parameters
st.sidebar.header("Algorithm Settings")
algo_choice = st.sidebar.selectbox(
    "Algorithm", ["Simple Hill Climbing", "Steepest Hill Climbing"]
)
func_choice = st.sidebar.selectbox(
    "Benchmark Function",
    [
        "Quadratic (1D)",
        "Sinusoidal (1D)",
        "Ackley (2D)",
        "Rosenbrock (2D)",
        "Rastrigin (2D)",
    ],
)
max_iter = st.sidebar.slider("Max Iterations", 50, 2000, 300, 50)
step_size = st.sidebar.slider("Step Size", 0.01, 1.0, 0.1, 0.01)
samples = st.sidebar.slider("Neighbors per Iteration (Steepest)", 2, 50, 10)
tol = st.sidebar.number_input("Improvement Tolerance", 1e-8, 1e-2, 1e-6, format="%.1e")
seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)
np.random.seed(seed)


# benchmark function selection
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


# run algorithm
if algo_choice == "Simple Hill Climbing":
    x_best, f_best, traj, evals = hill_climbing(
        f, x0, continuous_neighborhood, step_size=step_size, max_iter=max_iter, tol=tol
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
    )

f_best_display = f_best.item() if isinstance(f_best, np.ndarray) else f_best


# display results
st.subheader("Optimization Result")
st.write(f"Best Solution: {x_best}")
st.write(f"Function Value: {f_best_display}")
st.write(f"Function Evaluations: {evals}")


# visualization
st.subheader("Search Trajectory Visualization")
if dim == 1:
    fig = visualize_1d_function(f, name=func_choice, xlim=(-5, 5))
    fig = add_trajectory_1d(fig, traj, f)
    st.plotly_chart(fig, use_container_width=True)
elif dim == 2:
    fig = visualize_3d_function(f, name=func_choice, xlim=(-5, 5))
    fig= add_trajectory_3d(fig, traj, f)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Visualization not available.")


# discussion
st.header("Discussion / Insights")
st.markdown(
    """
Observations:
- Simple HC is fast but can get trapped in local minima.
- Steepest HC explores better but requires more evaluations.
- Trajectory shows the algorithm’s path in search space.

Parameter Sensitivity:
- Step size affects convergence.
- Max iterations and tolerance control stopping.
- Random seed affects initialization.

Potential Improvements:
- Adaptive step size
- Random restarts
- Hybrid algorithms (e.g., with Simulated Annealing)
"""
)

st.markdown("---")
st.markdown(
    "<small>Note: ChatGPT assisted in generating this page.</small>",
    unsafe_allow_html=True,
)
