import numpy as np
from typing import Tuple, List
from src.GeneticAlgorithm.coffee_fitness import coffee_fitness_4d
from src.HillClimbingAlgorithm.algorithm import (
    steepest_hill_climbing,
    continuous_neighborhood_batch,
)

ALL_DIMS = ["roast", "blend", "grind", "brew_time"]

import numpy as np
from src.GeneticAlgorithm.coffee_fitness import coffee_fitness_4d


def coffee_obj_min(x: np.ndarray) -> np.ndarray:
    """
    Map 4D points to negated coffee fitness.

    Parameters
    ----------
    x : np.ndarray
        Shape (4,) for single point or (n,4) for batch of points.

    Returns
    -------
    float or np.ndarray
        Negated coffee fitness (minimization).
    """
    x = np.asarray(x, dtype=float)

    # Single point
    if x.ndim == 1:
        roast = int(np.clip(round(x[0]), 0, 20))
        blend = int(np.clip(round(x[1]), 0, 100))
        grind = int(np.clip(round(x[2]), 0, 10))
        brew_time = float(np.clip(x[3], 0.0, 5.0))
        return -coffee_fitness_4d(roast, blend, grind, brew_time)

    # Batch of points
    elif x.ndim == 2 and x.shape[1] == 4:
        results = []
        for row in x:
            roast = int(np.clip(round(row[0]), 0, 20))
            blend = int(np.clip(round(row[1]), 0, 100))
            grind = int(np.clip(round(row[2]), 0, 10))
            brew_time = float(np.clip(row[3], 0.0, 5.0))
            results.append(-coffee_fitness_4d(roast, blend, grind, brew_time))
        return np.array(results)

    else:
        raise ValueError(f"Input x must be shape (4,) or (n,4), got {x.shape}")


def run_single_restart(
    seed: int = 0,
    step_size: float = 2.0,
    samples: int = 40,
    max_iter: int = 500,
    patience: int = 50,
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """Run a single steepest-ascent hill-climbing restart."""
    np.random.seed(seed)
    x0 = np.array(
        [
            np.random.uniform(0, 20),
            np.random.uniform(0, 100),
            np.random.uniform(0, 10),
            np.random.uniform(0.0, 5.0),
        ]
    )
    return steepest_hill_climbing(
        coffee_obj_min,
        x0,
        continuous_neighborhood_batch,
        step_size=step_size,
        max_iter=max_iter,
        samples=samples,
        tol=1e-6,
        patience=patience,
    )


def run_hill_climb(
    generations: int = 50,
    samples: int = 40,
    step_size: float = 2.0,
    seed: int = 42,
) -> Tuple[dict, List[np.ndarray]]:
    """
    Run hill climbing for a number of generations (iterations) and return
    the best configuration and all trajectories.

    Returns:
        best_overall: dict with roast, blend, grind, brew_time, quality
        trajectories: list of numpy arrays with path of each iteration
    """
    best_overall = None
    best_quality = -np.inf
    trajectories = []

    for i in range(generations):
        x_best, f_best, traj, evals = run_single_restart(
            seed=seed + i,
            step_size=step_size,
            samples=samples,
            max_iter=1,  # 1 iteration per generation for stepping
            patience=1,
        )
        trajectories.append(traj)
        quality = -f_best
        if quality > best_quality:
            best_quality = quality
            best_overall = {
                "roast": int(np.clip(round(x_best[0]), 0, 20)),
                "blend": int(np.clip(round(x_best[1]), 0, 100)),
                "grind": int(np.clip(round(x_best[2]), 0, 10)),
                "brew_time": float(np.clip(x_best[3], 0.0, 5.0)),
                "quality": quality,
            }

    return best_overall, trajectories
