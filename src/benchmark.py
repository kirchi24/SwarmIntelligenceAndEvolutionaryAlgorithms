from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _as_array(x: Iterable) -> np.ndarray:
    """Convert input to numpy array with at least 1-D shape.

    Accepts scalars or sequences. Keeps last axis as the variable axis for n-D inputs.
    """
    return np.asarray(x, dtype=float)


def _as_batch(x: Iterable) -> Tuple[np.ndarray, int, int]:
    """
    Convert input x into a 2D array (num_samples, num_features) for batch evaluation.

    Returns:
        xs : 2D np.ndarray of shape (num_samples, num_features)
        n : number of features per sample (last axis)
        orig_ndim : original number of dimensions of x
    """
    arr = np.asarray(x, dtype=float)
    orig_ndim = arr.ndim
    xs = np.atleast_2d(arr)
    n = xs.shape[-1] if not (xs.shape[-1] == 1 and orig_ndim == 1) else 1
    return xs, n, orig_ndim


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
    x_values, n, orig_ndim = _as_batch(x)

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
        return result.reshape(x_values.shape[:-1])


def rosenbrock(x: Iterable) -> float:
    """
    Rosenbrock function for vector x in R^n.
    f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    """
    xs, n, orig_ndim = _as_batch(x)

    if n < 2:
        result = np.zeros(xs.shape[0])
    else:
        vals = 100.0 * (xs[:, 1:] - xs[:, :-1] ** 2) ** 2 + (1.0 - xs[:, :-1]) ** 2
        result = np.sum(vals, axis=1)

    return float(result[0]) if orig_ndim <= 1 else result.reshape(xs.shape[:-1])


def rastrigin(x: Iterable) -> float:
    """
    Rastrigin function for vector x in R^n.
    f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    """
    xs, n, orig_ndim = _as_batch(x)
    result = 10.0 * n + np.sum(xs**2 - 10.0 * np.cos(2.0 * np.pi * xs), axis=1)
    return float(result[0]) if orig_ndim <= 1 else result.reshape(xs.shape[:-1])


def visualize_1d(xlim: Tuple[float, float] = (-5.0, 5.0), points: int = 400) -> None:
    """Quick 1-D visualization for quadratic and sinusoidal functions."""
    x = np.linspace(xlim[0], xlim[1], points)

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


def visualize_2d(
    func,
    name: str = "Function",
    xlim: Tuple[float, float] = (-5.0, 5.0),
    ylim: Optional[Tuple[float, float]] = None,
    points: int = 200,
) -> None:
    """Visualize a 2-D function with a surface and contour plot.

    The function must accept a length-2 vector and return a scalar.
    """
    if ylim is None:
        ylim = xlim

    x = np.linspace(xlim[0], xlim[1], points)
    y = np.linspace(ylim[0], ylim[1], points)
    X, Y = np.meshgrid(x, y)

    # Evaluate func on the grid efficiently
    pts = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([func(p) for p in pts])
    Z = Z.reshape(X.shape)

    fig = plt.figure(figsize=(12, 5))

    # Surface plot
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_title(f"{name} Surface")

    # Contour plot
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, cmap="viridis", levels=50)
    fig.colorbar(contour, ax=ax2)
    ax2.set_title(f"{name} Contour")

    plt.show()


if __name__ == "__main__":
    # Quick demo to verify functions run and plots render
    print("Quadratic(2) =", quadratic(2))
    print("Sinusoidal(pi/2) =", sinusoidal(np.pi / 2))

    # Ackley in 2D at origin (global minimum)
    print("Ackley([0,0]) =", ackley([0.0, 0.0]))

    # Rosenbrock and Rastrigin examples
    print("Rosenbrock([1,1]) =", rosenbrock([1.0, 1.0]))
    print("Rastrigin([0,0]) =", rastrigin([0.0, 0.0]))

    # Visualizations (uncomment to show)
    visualize_1d()
    visualize_2d(ackley, name="Ackley")
