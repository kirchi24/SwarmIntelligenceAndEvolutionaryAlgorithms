# code imported from https://github.com/hannahwimmer/SEA-demo/blob/master/sea_demo/codes/assignment02_fitness.py

import numpy as np
import matplotlib.pyplot as plt


def coffee_fitness_4d(roast: int, blend: int, grind: int, brew_time: float) -> float:
    """
    Fictional 4D coffee quality fitness function
    (ChatGPT-generated - sorry about that.. was just looking for a quick 4D dummy
    function...)
    ---------------------------------------------
    Parameters:
        roast (int):  [0, 20]
        blend (int):  [0, 50]
        grind (int):  [0, 10]
        brew_time (float): [0.0, 5.0] (minutes)

    Returns:
        float: quality score in [0, 100]

    The function is intentionally multimodal, with many local optima and
    a clear global optimum region around ideal values.
    """

    # --- normalize inputs to [0, 1] ---
    R = np.clip(roast / 20.0, 0, 1)
    B = np.clip(blend / 100.0, 0, 1)
    G = np.clip(grind / 10.0, 0, 1)
    T = np.clip(brew_time / 5.0, 0, 1)

    # --- sinusoidal landscape for local optima ---
    base_pattern = (
        np.sin(6 * np.pi * R) * np.cos(4 * np.pi * B)
        + np.sin(5 * np.pi * G) * np.cos(3 * np.pi * T)
        + 0.5 * np.sin(2 * np.pi * (R + B + G + T))
    )

    # --- smooth "global optimum" Gaussian region ---
    # Ideal combination: medium roast, balanced blend, mid grind, moderate brew
    ideal = np.exp(
        -((R - 0.6) ** 2 / 0.015)
        - ((B - 0.5) ** 2 / 0.02)
        - ((G - 0.5) ** 2 / 0.02)
        - ((T - 0.55) ** 2 / 0.015)
    )

    # --- cross-interaction term to couple dimensions (non-separable landscape) ---
    interactions = 0.2 * np.sin(3 * np.pi * R * B) + 0.15 * np.cos(4 * np.pi * G * T)

    # --- combine components ---
    score = 0.6 * ideal + 0.3 * base_pattern + interactions

    # --- add a small asymmetry (e.g., bitterness penalty) ---
    bitterness = 0.6 * R + 0.4 * T
    if bitterness > 0.7:
        score -= 0.2 * (bitterness - 0.7) ** 2

    # --- scale and clip to [0, 100] ---
    quality = np.clip(50 + 50 * score, 0, 100)

    return float(quality)


def plot_fitness_grid(fitness_function, fixed_dims, fixed_values, grid_points=100):
    """
    plots a contour of the fitness function by varying two dimensions while keeping the
    other two fixed.

    inputs:
    - fitness_function: callable, e.g., fitness_function(roast, blend, grind, brew_time)
    - fixed_dims: list of two strings, e.g., ['roast', 'blend'] (dims to keep fixed)
    - fixed_values: list of two floats, values for the fixed dimensions
    - grid_points: int, resolution of the grid
    output: contour plot
    """

    # all dimension names
    all_dims = ["roast", "blend", "grind", "brew_time"]
    variable_dims = [d for d in all_dims if d not in fixed_dims]
    limits = [20, 100, 10, 5.0]
    variable_lims = [l for (l, d) in zip(limits, all_dims) if d not in fixed_dims]

    # create grid
    X_vals = np.linspace(0, variable_lims[0], grid_points)
    Y_vals = np.linspace(0, variable_lims[1], grid_points)
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = np.zeros_like(X)

    # compute fitness
    for i in range(grid_points):
        for j in range(grid_points):
            args = dict(zip(fixed_dims, fixed_values))
            args[variable_dims[0]] = X[i, j]
            args[variable_dims[1]] = Y[i, j]
            Z[i, j] = fitness_function(**args)

    # plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Quality Score")
    plt.xlabel(variable_dims[0].capitalize())
    plt.ylabel(variable_dims[1].capitalize())
    plt.title(
        f"Fitness Landscape (fixed {fixed_dims[0]}={fixed_values[0]}, {fixed_dims[1]}={fixed_values[1]})"
    )
    plt.show()


# Example usage:
if __name__ == "__main__":
    plot_fitness_grid(
        coffee_fitness_4d, fixed_dims=["grind", "brew_time"], fixed_values=[5, 4.0]
    )
