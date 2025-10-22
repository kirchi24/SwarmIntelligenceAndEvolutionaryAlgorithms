import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.GeneticAlgorithm.coffee_fitness import coffee_fitness_4d
from src.GeneticAlgorithm.population import Population
from src.GeneticAlgorithm.chromosome import CoffeeChromosome

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
    st.markdown(
        """
    ## Introduction

    Genetic Algorithms (GAs) are **population-based optimization algorithms** inspired by natural selection.  
    They iteratively improve a population of candidate solutions using mechanisms such as:
    - **Selection**: Preferentially choose fitter individuals as parents.
    - **Crossover**: Combine genetic information from parents to produce offspring.
    - **Mutation**: Introduce small random variations to maintain diversity.
                
    The main challenge in applying genetic algorithms often lies in designing an appropriate fitness function, which quantitatively evaluates how well a candidate solution meets the problem's objectives.

    **Strengths:**
    - Good for multimodal landscapes.
    - Can handle mixed-integer and continuous variables.
    - Avoids being trapped in local optima.

    **Limitations:**
    - Computationally expensive.
    - Convergence depends on population size and mutation/crossover rates.
    - No guarantee to find global optimum.
    """
    )

# ------------------------
# Methods
# ------------------------
with tabs[1]:
    st.header("Implementation Overview")
    st.markdown(
        """
    ### Encoding

    Each coffee candidate (chromosome) consists of four genes:
    - `roast` (int, 0-20)  
    - `blend` (int, 0-100)  
    - `grind` (int, 0-10)  
    - `brew_time` (float, 0.0-5.0)  

    Integer genes are handled with **discrete mutation**, while `brew_time` uses **Gaussian noise** for smoother variation.

    ---

    ### Genetic Operators

    **Crossover (range: 0.0-1.0, standard 0.8)**  
    Crossover represents **recombination** — it combines genetic information from two parent individuals to create offspring.  
    In this implementation:
    - Integer genes (`roast`, `blend`, `grind`) are randomly inherited from either parent.  
    - The continuous gene (`brew_time`) is interpolated linearly between the two parent values.  
    A higher crossover rate (e.g., 0.8) increases the probability that offspring are formed by mixing traits from two parents, promoting exploration of new combinations.

    **Mutation (range: 0.0-1.0, int rate: 0.2, float rate: 0.2)**  
    Mutation introduces **small random changes** to maintain diversity and prevent premature convergence.  
    - Integer genes mutate by adding or subtracting small values (±1 or ±2).  
    - The `brew_time` gene receives a small Gaussian perturbation.  
    Separate mutation rates for integers and floats allow finer control over how often each type of gene changes.

    ---

    ### Selection Methods

    **1. Tournament Selection (default)**  
    - Randomly selects `k` individuals from the population (a “tournament”).  
    - The fittest among them is chosen as a parent.  
    - **Pros:** Strong selection pressure → fast convergence.  
    - **Cons:** Can reduce diversity and risk premature convergence.

    **2. Roulette Wheel Selection (Fitness-Proportionate)**  
    - Each individual's chance of selection is proportional to its fitness.  
    - **Pros:** Maintains population diversity → less risk of local optima.  
    - **Cons:** Progress can be slower and more stochastic.

    | Method     | Selection Pressure | Diversity | Convergence Speed |
    |------------|--------------------|------------|-------------------|
    | Tournament | High               | Low        | Fast              |
    | Roulette   | Medium             | High       | Slower            |

    ---

    ### Population

    The **population** is a collection of candidate coffee configurations that explore the search space together.  
    Each individual represents one potential solution, defined by the four coffee parameters.

    **Population Size**
    - Default: 30 individuals  
    - Configurable range: 10-100  

    **Effects of Size**
    - **Small populations (10-20):** Faster convergence but higher risk of local optima.  
    - **Large populations (50-100):** Better exploration but higher computational cost.
    """
    )

# ------------------------
# Results
# ------------------------
with tabs[2]:
    st.header("Results")

    # Sidebar parameters
    st.sidebar.header("GA Parameters")
    pop_size = st.sidebar.slider("Population Size", 10, 100, 20)
    generations = st.sidebar.slider("Generations", 10, 300, 50)
    crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
    mutation_rate_int = st.sidebar.slider("Mutation Rate (Int)", 0.0, 1.0, 0.2)
    mutation_rate_float = st.sidebar.slider("Mutation Rate (Float)", 0.0, 1.0, 0.2)
    selection_method = st.sidebar.selectbox(
        "Selection Method", ["tournament", "roulette"]
    )
    seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)
    np.random.seed(seed)

    # Initialize population
    population = Population(
        size=pop_size, selection_method=selection_method, fitness_fn=coffee_fitness_4d
    )
    population.evolve(
        crossover_rate=crossover_rate,
        mutation_int_prob=mutation_rate_int,
        mutation_float_prob=mutation_rate_float,
    )

    # Track best fitness
    best_fitness_history = []
    best_individual_history = []

    # Evolution loop
    progress_bar = st.progress(0)
    status_text = st.empty()

    # # Debug: Test the fitness function directly
    # st.sidebar.header("Debug Fitness Function")
    # test_roast = st.sidebar.slider("Test Roast", 0, 20, 10)
    # test_blend = st.sidebar.slider("Test Blend", 0, 100, 50)
    # test_grind = st.sidebar.slider("Test Grind", 0, 10, 5)
    # test_brew = st.sidebar.slider("Test Brew Time", 0.0, 5.0, 2.5)

    # test_fitness = coffee_fitness_4d(
    #     roast=test_roast,
    #     blend=test_blend,
    #     grind=test_grind,
    #     brew_time=test_brew
    # )
    # st.sidebar.metric("Test Fitness", f"{test_fitness:.2f}")

    for gen in range(generations):
        status_text.text(f"Generation {gen+1}/{generations}")
        progress_bar.progress((gen + 1) / generations)

        population.evolve(
            crossover_rate=crossover_rate,
            mutation_int_prob=mutation_rate_int,
            mutation_float_prob=mutation_rate_float,
        )

        best = population.best()
        if best and best.fitness is not None:
            best_fitness_history.append(best.fitness)
            best_individual_history.append(best.copy())
        else:
            # Fallback: nimm das erste Individuum
            if population.individuals and population.individuals[0].fitness is not None:
                best_fitness_history.append(population.individuals[0].fitness)
                best_individual_history.append(population.individuals[0].copy())

    # Best solution
    st.subheader("Best Solution Found")
    st.markdown(
        """
   Displays the **best coffee configuration** discovered after the genetic evolution — defined by the parameters **Roast**, **Blend**, **Grind**, and **Brew Time**.

    Includes:
    - **Best Fitness:** The highest fitness value achieved.  
    - **Parameter Table:** A summary of the four parameters of the best-performing solution.  
    - **Fitness Progress Plot:** A line chart showing how the best fitness evolved over generations.         
    """
    )

    if best_individual_history:
        # Finde das beste Individuum über alle Generationen
        valid_inds = [
            ind
            for ind in best_individual_history
            if ind and getattr(ind, "fitness", None) is not None
        ]

        if valid_inds:
            best_overall = max(valid_inds, key=lambda x: x.fitness)

            # Ausgabe der besten Lösung
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Best Fitness", f"{best_overall.fitness:.2f}")

            with col2:
                # Korrekte Datentypen für die Tabelle - alle Werte als String
                params = {
                    "Roast": str(getattr(best_overall, "roast", "N/A")),
                    "Blend": str(getattr(best_overall, "blend", "N/A")),
                    "Grind": str(getattr(best_overall, "grind", "N/A")),
                    "Brew Time": f"{getattr(best_overall, 'brew_time', 0):.2f}",
                }

                st.markdown("#### Parameters")
                # Verwende ein DataFrame mit korrekten Datentypen
                import pandas as pd

                df_params = pd.DataFrame(
                    {"Parameter": list(params.keys()), "Value": list(params.values())}
                )
                st.dataframe(df_params, use_container_width=True, hide_index=True)

            # Fitness-Verlauf plotten
            st.subheader("Fitness Progress")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=best_fitness_history, mode="lines+markers", name="Best Fitness"
                )
            )
            fig.update_layout(
                xaxis_title="Generation",
                yaxis_title="Fitness",
                title="Best Fitness per Generation",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("No valid individuals with fitness found in any generation.")
    else:
        st.error("No evolution history recorded.")

    # -------------------
    # Fitness Landscape (Contour)
    # -------------------
    st.subheader("Fitness Landscape (Contour)")
    st.markdown(
        """
    Allows exploration of the **fitness landscape** across any two chosen parameter axes.  
    For example: *Grind vs. Brew Time* while keeping *Roast* and *Blend* fixed.

    Usage:
    1. Select two **fixed parameters** (kept constant).  
    2. Set their **specific values** using sliders.      
                
    The algorithm evaluates a **50x50 grid** of fitness values using `coffee_fitness_4d`  
    and visualizes them in a **Plotly contour plot**, showing the fitness elevation across the selected dimensions.
    """
    )
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
            args[variable_dims[0]] = (
                int(X[i, j])
                if variable_dims[0] in ["roast", "blend", "grind"]
                else X[i, j]
            )
            args[variable_dims[1]] = (
                int(Y[i, j])
                if variable_dims[1] in ["roast", "blend", "grind"]
                else Y[i, j]
            )
            Z[i, j] = coffee_fitness_4d(**args)

    # --- Plotly Contour Plot ---
    fig = go.Figure(
        data=go.Contour(
            z=Z,
            x=x_vals,
            y=y_vals,
            colorscale="Viridis",
            contours=dict(showlabels=True),
            colorbar=dict(title="Fitness"),
        )
    )

    # --- Trajectory visualization of best individuals ---
    if best_individual_history:
        # Hole die x/y-Koordinaten basierend auf den aktuell gewählten variable_dims
        trajectory_x = []
        trajectory_y = []

        for ind in best_individual_history:
            trajectory_x.append(getattr(ind, variable_dims[0]))
            trajectory_y.append(getattr(ind, variable_dims[1]))

        # --- Trajectory line (Entwicklung über Generationen) ---
        fig.add_trace(
            go.Scatter(
                x=trajectory_x,
                y=trajectory_y,
                mode="lines+markers",
                marker=dict(size=6, color="red"),
                line=dict(color="red", dash="dash"),
                name="Trajectory",
            )
        )

        # --- Startpunkt (erste Generation) ---
        fig.add_trace(
            go.Scatter(
                x=[trajectory_x[0]],
                y=[trajectory_y[0]],
                mode="markers",
                marker=dict(size=10, color="blue"),
                name="Start",
            )
        )

        # --- Endpunkt (letzte Generation) ---
        fig.add_trace(
            go.Scatter(
                x=[trajectory_x[-1]],
                y=[trajectory_y[-1]],
                mode="markers",
                marker=dict(size=10, color="green"),
                name="End",
            )
        )

    fig.update_layout(
        xaxis_title=variable_dims[0],
        yaxis_title=variable_dims[1],
        title="Fitness Landscape Contour",
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Discussion
# ------------------------
with tabs[3]:
    st.header("Discussion & Analysis")
    st.markdown(
        """
    ## Discussion

    The Genetic Algorithm (GA) demonstrates its effectiveness in exploring the search space and identifying high-fitness coffee configurations.  
    Across generations, the **fitness typically improves steadily**, occasionally reaching **plateaus** when the population converges around local optima.

    ### Influence of Selection Methods
    - **Tournament Selection:**  
    This method consistently favors the fittest individuals from small random subsets of the population.  
    It promotes **strong selection pressure**, leading to **faster convergence** and more stable improvement in early generations.  
    However, excessive pressure can reduce diversity, causing the population to converge prematurely on suboptimal solutions.

    - **Roulette Wheel Selection:**  
    This probabilistic approach gives **every individual a chance** proportional to its fitness.  
    It maintains **higher diversity** and encourages broader exploration, which can help escape local optima.  
    However, progress per generation may be **slower** and more **stochastic**, leading to greater variability across runs.

    ### Observations from the Results
    - The GA generally finds **high-fitness configurations**, validating that the fitness function provides a meaningful gradient for optimization.  
    - **Tournament selection** tends to reach good solutions faster, while **roulette selection** explores more widely and can yield better results over longer runs.  
    - **Population size**, **mutation rate**, and **crossover rate** critically affect convergence speed and stability.  
    - Higher mutation maintains diversity but can destabilize convergence.  
    - Lower mutation speeds convergence but risks stagnation.  
    - Balanced crossover (around 0.7-0.9) helps combine traits effectively.

    ### Limitations and Future Improvements
    - Due to its **stochastic nature**, performance varies between runs, especially with small populations.  
    - Adaptive mechanisms could improve robustness:
    - **Adaptive mutation/crossover rates** to balance exploration and exploitation dynamically.  
    - **Elitism** to ensure top individuals persist across generations.  
    - **Hybrid approaches**, e.g., combining GA with **Hill Climbing**, could refine the best individuals locally for even higher fitness.

    In summary, both selection strategies are valuable:  
    **Tournament selection** excels at rapid optimization, while **Roulette selection** maintains genetic diversity for long-term improvement.  
    Choosing between them depends on whether **speed** or **robustness** is prioritized.
    """
    )
