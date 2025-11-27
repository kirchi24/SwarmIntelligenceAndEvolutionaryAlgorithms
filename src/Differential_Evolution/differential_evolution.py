import numpy as np
from copy import deepcopy

# imports from our radial encoding and utils
from radial_encoding import (
    generate_random_radii,
    smooth_radii,
    radii_to_polygon,
    validate_polygon,
    place_polygon_at_start,
)
from utils import (
    construct_corridor,
    objective_function,
    move_and_rotate_smooth,
    animate_shape,
)


def evaluate_candidate(radii, corridor, K, r_min, r_max, smooth_window=3, penalty=1e6):
    """Build polygon from radii, place at corridor start and evaluate objective.

    Returns scalar cost (lower is better).
    """
    # ensure within bounds
    radii = np.clip(radii, r_min, r_max)
    radii = smooth_radii(radii, window=smooth_window)
    poly = radii_to_polygon(radii)
    if not validate_polygon(poly):
        return penalty, None
    placed = place_polygon_at_start(poly, corridor, x_offset=0.02)
    cost = objective_function(corridor, placed)
    return float(cost), placed


def differential_evolution(
    K=24,
    r_min=0.1,
    r_max=1.5,
    popsize=20,
    F=0.6,
    CR=0.9,
    generations=100,
    smooth_window=3,
    seed=42,
):
    """Simple DE implementation optimizing radii vector.

    Returns best radii, best polygon (placed), history (best cost per generation).
    """
    rng = np.random.RandomState(seed)
    corridor = construct_corridor()

    # initialize population
    pop = np.array([generate_random_radii(k=K, r_min=r_min, r_max=r_max) for _ in range(popsize)])
    costs = np.zeros(popsize)
    placed_polys = [None] * popsize
    for i in range(popsize):
        costs[i], placed_polys[i] = evaluate_candidate(
            pop[i], corridor, K, r_min, r_max, smooth_window=smooth_window
        )

    best_idx = int(np.argmin(costs))
    best_cost = float(costs[best_idx])
    best_radii = pop[best_idx].copy()
    best_poly = placed_polys[best_idx]

    history = [best_cost]

    for gen in range(generations):
        for i in range(popsize):
            # mutation: pick three distinct indices
            idxs = [idx for idx in range(popsize) if idx != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            # ensure bounds
            mutant = np.clip(mutant, r_min, r_max)

            # crossover
            cross = rng.rand(K) < CR
            if not np.any(cross):
                cross[rng.randint(0, K)] = True
            trial = np.where(cross, mutant, pop[i])

            trial_cost, trial_poly = evaluate_candidate(
                trial, corridor, K, r_min, r_max, smooth_window=smooth_window
            )

            if trial_cost < costs[i]:
                pop[i] = trial
                costs[i] = trial_cost
                placed_polys[i] = trial_poly

                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_radii = trial.copy()
                    best_poly = trial_poly

        history.append(best_cost)
        # simple progress print
        if (gen + 1) % max(1, generations // 10) == 0:
            print(f"Gen {gen+1}/{generations} best_cost={best_cost:.3f}")

    return best_radii, best_poly, history


if __name__ == "__main__":
    # Quick run with small settings for demo / checking
    K = 50
    best_radii, best_poly, history = differential_evolution(
        K=K, r_min=0.1, r_max=2, popsize=32, F=0.6, CR=0.8, generations=1000, smooth_window=5, seed=1
    )

    print("Best cost history (last):", history[-5:])

    # show the best polygon moving through the corridor
    corridor = construct_corridor()
    placed = place_polygon_at_start(best_poly, corridor, x_offset=0.02) if best_poly is not None else None
    if placed is not None:
        possible, max_rot, path = move_and_rotate_smooth(corridor, placed)
        if path and len(path) > 0:
            print(f"Animating best candidate (max_rot={max_rot:.2f})")
            ani = animate_shape(corridor, placed, path, interval=60, repeat=False)
        else:
            print("Best candidate could not produce a path for animation. Showing static placement instead.")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.set_frame_on(False)

            cx, cy = corridor.exterior.xy
            ax.fill(cx, cy, color='lightgray', alpha=0.6)

            px, py = placed.exterior.xy
            ax.fill(px, py, color='tomato', alpha=0.8)

            ax.set_title('Best candidate placed in corridor (static)')
            plt.show()
    else:
        print("No valid best polygon found.")
