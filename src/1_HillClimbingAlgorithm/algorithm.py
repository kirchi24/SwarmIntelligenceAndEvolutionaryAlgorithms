import numpy as np
from typing import Callable, Iterable, Optional, Tuple, Union

from src.benchmark import (
    quadratic,
    sinusoidal,
    ackley,
    rosenbrock,
    rastrigin,
)


def continuous_neighborhood(
    x: Union[Iterable[float], np.ndarray], step_size: float = 0.1
) -> np.ndarray:
    """Sample a continuous neighbor of `x`.

    Parameters
    ----------
    x : array-like
        Current solution (vector-like).
    step_size : float, optional
        Maximum uniform perturbation per component (default 0.1).

    Returns
    -------
    np.ndarray
        Neighbor with same shape as `x`.
    """
    arr = np.array(x, dtype=float)
    return arr + np.random.uniform(-step_size, step_size, size=arr.shape)


def hill_climbing(
    f: Callable[[Union[np.ndarray, Iterable[float]]], float],
    x0: Union[np.ndarray, Iterable[float]],
    neighborhood_fn: Callable[[np.ndarray], np.ndarray],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """First-improvement hill-climbing optimizer.

    Parameters
    ----------
    f : callable
        Objective function to minimize; accepts a vector and returns a scalar.
    x0 : array-like
        Initial solution.
    neighborhood_fn : callable
        Function producing a single neighbor for a given solution.
    max_iter : int, optional
        Maximum iterations (default 1000).
    tol : float, optional
        Minimum improvement required to accept a move (default 1e-6).

    Returns
    -------
    x_best : np.ndarray
        Best found solution.
    f_best : float
        Objective value at `x_best`.
    trajectory : np.ndarray
        Array of visited solutions with shape (k, n).
    evaluations : int
        Number of objective evaluations performed.
    """
    x_current = np.array(x0, dtype=float)
    f_current = f(x_current)
    trajectory = [x_current.copy()]
    evaluations = 1
    for _ in range(max_iter):
        x_new = neighborhood_fn(x_current)
        f_new = f(x_new)
        evaluations += 1
        if f_new < f_current - tol:
            x_current, f_current = x_new, f_new
            trajectory.append(x_current.copy())
    return x_current, f_current, np.array(trajectory), evaluations


def steepest_hill_climbing(
    f: Callable[[Union[np.ndarray, Iterable[float]]], float],
    x0: Union[np.ndarray, Iterable[float]],
    neighborhood_fn: Callable[[np.ndarray], np.ndarray],
    max_iter: int = 1000,
    samples: int = 20,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """Steepest-ascent hill-climbing (best-of-sample).

    Parameters
    ----------
    f : callable
        Objective function to minimize.
    x0 : array-like
        Initial solution.
    neighborhood_fn : callable
        Function producing a single neighbor.
    max_iter : int, optional
        Maximum iterations (default 1000).
    samples : int, optional
        Number of neighbors sampled each iteration (default 20).
    tol : float, optional
        Minimum improvement required to accept a move (default 1e-6).

    Returns
    -------
    x_best, f_best, trajectory, evaluations
        Same as returned by :func:`hill_climbing`.
    """
    x_current = np.array(x0, dtype=float)
    f_current = f(x_current)
    trajectory = [x_current.copy()]
    evaluations = 1
    for _ in range(max_iter):
        neighbors = np.array([neighborhood_fn(x_current) for _ in range(samples)])
        f_values = np.array([f(x) for x in neighbors])
        evaluations += samples
        best_idx = np.argmin(f_values)
        if f_values[best_idx] < f_current - tol:
            x_current = neighbors[best_idx]
            f_current = f_values[best_idx]
            trajectory.append(x_current.copy())
    return x_current, f_current, np.array(trajectory), evaluations


def get_benchmark(name: str) -> Optional[Callable]:
    """Return a benchmark function by name.

    Parameters
    ----------
    name : str
        One of 'quadratic', 'sinusoidal', 'ackley', 'rosenbrock', 'rastrigin'.

    Returns
    -------
    callable or None
        The matching function or `None` if unknown.
    """
    name = name.lower()
    mapping = {
        "quadratic": quadratic,
        "sinusoidal": sinusoidal,
        "ackley": ackley,
        "rosenbrock": rosenbrock,
        "rastrigin": rastrigin,
    }
    return mapping.get(name)