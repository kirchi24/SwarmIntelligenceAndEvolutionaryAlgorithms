from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _as_array(x: Iterable) -> np.ndarray:
    """Convert input to numpy array with at least 1-D shape.

    Accepts scalars or sequences. Keeps last axis as the variable axis for n-D inputs.
    """
    return np.asarray(x, dtype=float)


def quadratic(x: Iterable) -> np.ndarray:
    """Quadratic function f(x) = x^2.

    Works element-wise for arrays and supports scalar or vector inputs.
    """
    arr = _as_array(x)
    return arr**2


def sinusoidal(x: Iterable) -> np.ndarray:
    """Sinusoidal function f(x) = sin(x).

    Works element-wise for arrays and supports scalar or vector inputs.
    """
    arr = _as_array(x)
    return np.sin(arr)


def ackley(
    x: Iterable, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi
) -> np.ndarray:
    """Ackley function.

    Standard form for x in R^n:
      f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2)))
             - exp(1/n * sum(cos(c * x_i))) + a + exp(1)

    Accepts 1-D vectors or an array of vectors (last axis is the variable axis).
    """
    arr = _as_array(x)
    orig_ndim = arr.ndim

    # ensure 2D (num_samples, num_features)
    x_values = np.atleast_2d(arr)

    # Ensure the last axis represents the variable dimension
    if x_values.shape[-1] == 1 and orig_ndim == 1:
        n = 1
    else:
        n = x_values.shape[-1]

    # Core Ackley computation
    sum_sq = np.sum(x_values**2, axis=1)
    sum_cos = np.sum(np.cos(c * x_values), axis=1)

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    result = term1 + term2 + a + np.e

    # return right shape depending on input
    if orig_ndim <= 1:
        return float(result[0])
    else:
        return result.reshape(arr.shape[:-1])



def rosenbrock(x):
    """Rosenbrock function"""
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin(x):
    """Rastrigin function"""
    x = np.asarray(x)
    n = x.size
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def visualize_1d():
    x = np.linspace(-5, 5, 400)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, quadratic(x))
    plt.title("Quadratic Function f(x)=x²")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, sinusoidal(x))
    plt.title("Sinusoidal Function f(x)=sin(x)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_2d(func, name):
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func(np.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())])
    Z = Z.reshape(X.shape)

    fig = plt.figure(figsize=(12, 5))

    # Surface plot
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap="viridis")
    ax1.set_title(f"{name} Surface")

    # Contour plot
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, cmap="viridis", levels=50)
    fig.colorbar(contour)
    ax2.set_title(f"{name} Contour")

    plt.show()
