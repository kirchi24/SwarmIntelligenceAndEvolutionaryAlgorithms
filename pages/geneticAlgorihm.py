import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.GeneticAlgorithm.coffee_fitness import coffee_fitness_4d
from src.GeneticAlgorithm.population import Population  

st.set_page_config(page_title="Genetic Algorithm - Coffee Optimization", layout="wide")

st.title("Genetic Algorithm - Coffee Optimization")

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs(["Introduction", "Methods", "Results", "Discussion"])

# ------------------------
# Introduction
# ------------------------
with tabs[0]:
    st.markdown("""
    ## Introduction

    Genetic Algorithms (GAs) are **population-based optimization algorithms** inspired by natural selection.  
    They iteratively improve a population of candidate solutions using mechanisms such as:
    - **Selection**: Preferentially choose fitter individuals as parents.
    - **Crossover**: Combine genetic information from parents to produce offspring.
    - **Mutation**: Introduce small random variations to maintain diversity.

    **Strengths:**
    - Good for multimodal landscapes.
    - Can handle mixed-integer and continuous variables.
    - Avoids being trapped in local optima.

    **Limitations:**
    - Computationally expensive.
    - Convergence depends on population size and mutation/crossover rates.
    - No guarantee to find global optimum.
    """)

# ------------------------
# Methods
# ------------------------
with tabs[1]:
    st.header("Implementation Overview")
    st.markdown("""
    ### Encoding
    Each coffee candidate (chromosome) has 4 genes:
    - `roast` (int, 0-20)
    - `blend` (int, 0-100)
    - `grind` (int, 0-10)
    - `brew_time` (float, 0.0-5.0)

    Integer genes are handled with discrete mutation, while `brew_time` uses Gaussian noise.

    ### Genetic Operators
    - **Mutation**: ±1 or ±2 for integers; small Gaussian perturbation for `brew_time`.
    - **Crossover**: 
        - Integers randomly inherited from parents.
        - `brew_time` interpolated linearly between parents.

    ### Selection Methods
    - Tournament selection (default)
    - Roulette wheel (fitness-proportionate)

    ### Population
    - Default size: 30
    - Evolves via selection → crossover → mutation → evaluation
    """)

# ------------------------
# Results
# ------------------------
with tabs[2]:
    st.header("Results")

    # Sidebar parameters
    st.sidebar.header("GA Parameters")
    pop_size = st.sidebar.slider("Population Size", 10, 100, 30)
    generations = st.sidebar.slider("Generations", 10, 200, 50)
    crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2)
    selection_method = st.sidebar.selectbox("Selection Method", ["tournament", "roulette"])
    seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)
    np.random.seed(seed)

    # Initialize population
    population = Population(size=pop_size, selection_method=selection_method)
    population.evaluate()

    # Track best fitness
    best_fitness_history = []
    best_individual_history = []

    for gen in range(generations):
        population.evolve(crossover_rate=crossover_rate, mutation_rate=mutation_rate)
        best = population.best()
        best_fitness_history.append(best.fitness)
        best_individual_history.append(best.copy())

    # Best solution
    st.subheader("Best Solution Found")
    best_overall = max(
        (ind for ind in best_individual_history if ind is not None and ind.fitness is not None),
        key=lambda x: x.fitness,
        default=None
    )
    st.write(best_overall)

    # Fitness over generations (Plotly)
    st.subheader("Fitness Over Generations")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=best_fitness_history,
        mode='lines+markers',
        name='Best Fitness'
    ))
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Fitness",
        title="Best Fitness Across Generations",
        yaxis=dict(range=[0, 100]) 
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------
    # Fitness Landscape (Contour)
    # -------------------
    st.subheader("Fitness Landscape (Contour)")

    # --- Parameterwahl mit Dropdowns ---
    col1, col2 = st.columns(2)
    all_dims = ["roast", "blend", "grind", "brew_time"]

    with col1:
        fixed_dim1 = st.selectbox("First fixed parameter", all_dims, index=2)
    with col2:
        remaining_dims = [d for d in all_dims if d != fixed_dim1]
        fixed_dim2 = st.selectbox("Second fixed parameter", remaining_dims, index=2)

    fixed_dims = [fixed_dim1, fixed_dim2]
    variable_dims = [d for d in all_dims if d not in fixed_dims]

    # --- Eingabe der fixierten Werte ---
    st.markdown("#### Fixed Values")

    def get_input_for_param(param_name):
        """Erzeugt passenden Input je nach Parameter"""
        if param_name == "roast":
            return st.slider(f"Fixed value for {param_name}", 0, 20, 5, step=1)
        elif param_name == "blend":
            return st.slider(f"Fixed value for {param_name}", 0, 100, 50, step=1)
        elif param_name == "grind":
            return st.slider(f"Fixed value for {param_name}", 0, 10, 5, step=1)
        elif param_name == "brew_time":
            return st.slider(f"Fixed value for {param_name}", 0.0, 5.0, 2.5, step=0.1)

    fixed_values = {
        fixed_dims[0]: get_input_for_param(fixed_dims[0]),
        fixed_dims[1]: get_input_for_param(fixed_dims[1]),
    }

    # --- Grid-Erstellung (50 Punkte pro Achse) ---
    limits = {"roast": 20, "blend": 100, "grind": 10, "brew_time": 5.0}
    x_vals = np.linspace(0, limits[variable_dims[0]], 50)
    y_vals = np.linspace(0, limits[variable_dims[1]], 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(50):
        for j in range(50):
            args = {**fixed_values}
            args[variable_dims[0]] = int(X[i, j]) if variable_dims[0] in ["roast", "blend", "grind"] else X[i, j]
            args[variable_dims[1]] = int(Y[i, j]) if variable_dims[1] in ["roast", "blend", "grind"] else Y[i, j]
            Z[i, j] = coffee_fitness_4d(**args)

    # --- Plotly Contour Plot ---
    fig = go.Figure(data=
        go.Contour(
            z=Z,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            contours=dict(showlabels=True),
            colorbar=dict(title='Fitness')
        )
    )
    fig.update_layout(
        xaxis_title=variable_dims[0],
        yaxis_title=variable_dims[1],
        title="Fitness Landscape Contour"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Discussion
# ------------------------
with tabs[3]:
    st.header("Discussion & Analysis")
    st.markdown("""
    - GA successfully finds high-fitness coffee configurations.
    - Fitness improves steadily over generations, with occasional plateaus due to local optima.
    - Selection method, population size, mutation, and crossover rates significantly influence convergence speed.
    - Limitations: stochastic behavior; results can vary between runs.
    - Potential improvements:
        - Adaptive mutation or crossover rates.
        - Elitism to retain top individuals.
        - Hybrid approach with Hill Climbing for local refinement.
    """)
