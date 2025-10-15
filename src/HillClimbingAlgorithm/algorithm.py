import numpy as np
from typing import Callable, Iterable, Optional, Tuple, Union

from src.benchmark import (
    quadratic,
    sinusoidal,
    ackley,
    rosenbrock,
    rastrigin,
)

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


def continuous_neighborhood_batch(
    x: Union[Iterable[float], np.ndarray], 
    step_size: float = 0.1, 
    samples: int = 1
) -> np.ndarray:
    """
    Sample multiple continuous neighbors of `x`.

    Parameters
    ----------
    x : array-like
        Current solution (vector-like).
    step_size : float, optional
        Maximum uniform perturbation per component (default 0.1).
    samples : int, optional
        Number of neighbors to generate (default 1).

    Returns
    -------
    np.ndarray
        Array of shape (samples, n) where each row is a perturbed neighbor 
        of `x`. `n` is the dimensionality of `x`.
    """
    x = np.array(x, dtype=float)
    perturbations = np.random.uniform(-step_size, step_size, size=(samples, x.size))
    neighbors = x + perturbations
    return neighbors


def hill_climbing(
    f: Callable[[Union[np.ndarray, Iterable[float]]], float],
    x0: Union[np.ndarray, Iterable[float]],
    neighborhood_fn: Callable[[Union[np.ndarray, Iterable[float]], float, int], np.ndarray],
    step_size: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
    patience: Optional[int] = None,
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """
    First-improvement hill-climbing optimizer using a batch-capable neighbor function.

    Parameters
    ----------
    f : callable
        Objective function to minimize; accepts a vector and returns a scalar.
    x0 : array-like
        Initial solution vector.
    neighborhood_fn : callable
        Function producing neighbors, with signature 
        `neighborhood_fn(x, step_size, samples) -> np.ndarray`.
    step_size : float, optional
        Maximum perturbation applied to each component when generating neighbors (default 0.1).
    max_iter : int, optional
        Maximum number of iterations (default 1000).
    tol : float, optional
        Minimum improvement required to accept a move (default 1e-6).
    patience : int or None, optional
        Maximum consecutive non-improving iterations before stopping (default None).

    Returns
    -------
    x_best : np.ndarray
        Best solution found.
    f_best : float
        Objective value at `x_best`.
    trajectory : np.ndarray
        Array of visited solutions with shape `(num_steps, n_features)`.
    evaluations : int
        Number of function evaluations performed.
    """
    x_current = np.array(x0, dtype=float)
    f_current = f(x_current)
    trajectory = [x_current.copy()]
    evaluations = 1
    no_improve = 0

    for _ in range(max_iter):
        x_neighbor = neighborhood_fn(x_current, step_size, 1)[0]
        f_neighbor = f(x_neighbor)
        evaluations += 1

        if f_neighbor < f_current - tol:
            x_current, f_current = x_neighbor, f_neighbor
            trajectory.append(x_current.copy())
            no_improve = 0
        else:
            no_improve += 1

        if patience is not None and no_improve >= patience:
            break

    trajectory = np.array(trajectory)
    return (x_current, f_current, trajectory, evaluations)


def steepest_hill_climbing(
    f: Callable[[Union[np.ndarray, Iterable[float]]], float],
    x0: Union[np.ndarray, Iterable[float]],
    neighborhood_fn: Callable[[Union[np.ndarray, Iterable[float]], float, int], np.ndarray],
    step_size: float = 0.1,
    max_iter: int = 1000,
    samples: int = 20,
    tol: float = 1e-6,
    patience: Optional[int] = None
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """
    Steepest-ascent hill-climbing (best-of-sample) optimizer using a batch-capable neighbor function.

    Parameters
    ----------
    f : callable
        Objective function to minimize; accepts a vector and returns a scalar.
    x0 : array-like
        Initial solution vector.
    neighborhood_fn : callable
        Function producing neighbors, with signature 
        `neighborhood_fn(x, step_size, samples) -> np.ndarray`.
    step_size : float, optional
        Maximum perturbation applied to each component when generating neighbors (default 0.1).
    max_iter : int, optional
        Maximum number of iterations (default 1000).
    samples : int, optional
        Number of neighbors to generate per iteration (default 20).
    tol : float, optional
        Minimum improvement required to accept a move (default 1e-6).
    patience : int or None, optional
        Maximum consecutive non-improving iterations before stopping (default None).

    Returns
    -------
    x_best : np.ndarray
        Best solution found.
    f_best : float
        Objective value at `x_best`.
    trajectory : np.ndarray
        Array of visited solutions with shape `(num_steps, n_features)`.
    evaluations : int
        Number of function evaluations performed.
    """
    x_current = np.array(x0, dtype=float)
    f_current = f(x_current)
    trajectory = [x_current.copy()]
    evaluations = 1
    no_improve = 0

    for _ in range(max_iter):
        x_neighbors = neighborhood_fn(x_current, step_size, samples=samples)
        f_neighbors = f(x_neighbors)
        evaluations += samples
        best_idx = np.argmin(f_neighbors)

        if f_neighbors[best_idx] < f_current - tol:
            x_current = x_neighbors[best_idx]
            f_current = f_neighbors[best_idx]
            trajectory.append(x_current.copy())
            no_improve = 0
        else:
            no_improve += 1

        if patience is not None and no_improve >= patience:
            break

    trajectory = np.array(trajectory)
    return x_current, f_current, trajectory, evaluations
