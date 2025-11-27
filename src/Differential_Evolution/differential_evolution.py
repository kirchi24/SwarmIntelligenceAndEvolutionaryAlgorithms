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

from radial_encoding import (
    rectangle_radii,
    circle_radii,
    ellipse_radii,
    regular_polygon_radii,
    star_radii,
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


def init_population_balanced_shapes(popsize, K, r_min, r_max, rng):
    """
    Erzeugt eine Population, bei der jede Shape-Klasse gleich stark vertreten ist.
    Keine if-Ketten: jede Shape-Art hat genau einen Generator.
    Balanced über alle Klassen verteilt.
    """

    generators = [
        lambda: rectangle_radii(
            rng.uniform(0.6, r_max * 0.9),
            rng.uniform(0.4, r_max * 0.7),
            K,
        ),
        lambda: circle_radii(
            rng.uniform(max(r_min, 0.3), r_max),
            K,
        ),
        lambda: ellipse_radii(
            rng.uniform(0.6, r_max),
            rng.uniform(0.3, r_max * 0.8),
            K,
        ),
        lambda: regular_polygon_radii(
            rng.integers(3, 9),
            rng.uniform(0.6, r_max),
            K,
            rotation=float(rng.uniform(0, 360)),
        ),
        lambda: star_radii(
            rng.integers(4, 9),
            rng.uniform(0.3, r_max * 0.5),
            rng.uniform(0.6, r_max),
            K,
        ),
        lambda: generate_random_radii(K, r_min, r_max),
    ]

    n_shapes = len(generators)
    base = popsize // n_shapes
    remainder = popsize % n_shapes

    population = []

    for gen in generators:
        for _ in range(base):
            population.append(gen())

    for i in range(remainder):
        population.append(generators[i]())

    pop = np.array(population, dtype=float)

    # noise + smoothing
    noise_scale = 0.04 * (r_max - r_min)
    pop += rng.normal(scale=noise_scale, size=pop.shape)
    pop = np.clip(pop, r_min, r_max)

    for i in range(popsize):
        pop[i] = smooth_radii(pop[i], window=3)

    return pop


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
    rng = np.random.default_rng(seed)
    corridor = construct_corridor()

    pop = init_population_balanced_shapes(popsize, K, r_min, r_max, rng)
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
            # mutation
            idxs = [x for x in range(popsize) if x != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)

            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = np.clip(mutant, r_min, r_max)

            # crossover
            cross = rng.random(K) < CR
            if not np.any(cross):
                cross[rng.integers(0, K)] = True

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
        if (gen + 1) % max(1, generations // 10) == 0:
            print(f"Gen {gen+1}/{generations} best_cost={best_cost:.3f}")

    return best_radii, best_poly, history


if __name__ == "__main__":
    K = 50
    best_radii, best_poly, history = differential_evolution(
        K=K,
        r_min=0.1,
        r_max=2,
        popsize=32,
        F=0.6,
        CR=0.8,
        generations=1000,
        smooth_window=5,
        seed=1,
    )

    print("Best cost history (last):", history[-5:])

    corridor = construct_corridor()
    placed = (
        place_polygon_at_start(best_poly, corridor, x_offset=0.02)
        if best_poly is not None
        else None
    )

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
