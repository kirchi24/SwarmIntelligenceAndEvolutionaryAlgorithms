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
st.title("Hill Climbing Algorithm Documentation")

st.markdown(
    """
## Introduction
Hill Climbing is a fundamental optimization algorithm that iteratively improves a solution by exploring its neighbors. It is widely used due to its simplicity and efficiency in solving low-dimensional problems.

### Key Features:
- **Simple Hill Climbing**: Evaluates one neighbor per iteration.
- **Steepest Ascent Hill Climbing**: Considers multiple neighbors and selects the best.

#### Advantages:
- Easy to implement.
- Effective for unimodal functions.

#### Limitations:
- Prone to getting stuck in local minima.
- Performance depends on initialization and step size.
"""
)


# methods
st.header("Methods")
st.markdown(
    """
### Implementation Details
The Hill Climbing algorithm can be customized with the following parameters:
- **Initial Solution (x₀)**: Starting point for the search.
- **Benchmark Function (f)**: Objective function to minimize.
- **Neighborhood Function**: Defines the search space around the current solution.
- **Step Size**: Controls the magnitude of changes.
- **Max Iterations**: Limits the number of steps.
- **Tolerance**: Determines the stopping criterion.
- **Patience**: Early stopping if no improvement is observed.
- **Neighbors per Iteration**: (For Steepest HC) Number of neighbors evaluated.
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
samples = st.sidebar.slider("Neighbors per Iteration (Steepest)", 1, 50, 10)
patience = st.sidebar.slider("Patience (early stopping)", 1, 50, 10)
tol = 10 ** st.sidebar.slider(
    "log10 (improvement tolerance)", min_value=-6.0, max_value=-2.0, value=-3.0, step=0.1
)
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
    fig = add_trajectory_3d(fig, traj, f)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Visualization not available.")


# discussion
st.header("Discussion")
st.markdown(
    """
### Insights and Analysis
- **Performance**: Simple HC is faster but less thorough; Steepest HC is more robust but computationally expensive.
- **Parameter Sensitivity**: Results vary significantly with step size, tolerance, and initialization.
- **Limitations**: Struggles with multimodal functions and high-dimensional spaces.

### Potential Improvements
- Adaptive step size.
- Random restarts to escape local minima.
- Hybrid approaches combining HC with other algorithms.
"""
)

st.markdown("---")
st.markdown(
    "<small>Note: This documentation was generated with the assistance of ChatGPT.</small>",
    unsafe_allow_html=True,
)
