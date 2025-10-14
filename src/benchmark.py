from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _as_array(x: Iterable) -> np.ndarray:
    """Convert input to a numpy array.

    Parameters
    ----------
    x : Iterable
        Scalar or array-like input.

    Returns
    -------
    np.ndarray
        Array view of `x` with dtype float.
    """
    return np.asarray(x, dtype=float)


def _as_batch(x: Iterable) -> Tuple[np.ndarray, int, int]:
    """Prepare input for batched evaluation.

    Converts `x` to a 2-D array `xs` with shape (K, n) where each row is
    an evaluation vector of length `n`. Also returns `n` and the original
    number of dimensions of `x`.

    Parameters
    ----------
    x : Iterable
        Scalar, vector, or array-like input. The last axis is the feature axis.

    Returns
    -------
    xs : np.ndarray
        2-D array of shape (K, n) for evaluation.
    n : int
        Number of features per sample.
    orig_ndim : int
        Original number of dimensions of `x`.
    """
    arr = np.asarray(x, dtype=float)
    orig_ndim = arr.ndim
    xs = np.atleast_2d(arr)
    n = xs.shape[-1] if not (xs.shape[-1] == 1 and orig_ndim == 1) else 1
    return xs, n, orig_ndim


def quadratic(x: Iterable) -> np.ndarray:
    """Quadratic function (element-wise).

    Formula
    -------
    f(x) = x^2

    Parameters
    ----------
    x : Iterable
        Scalar or array-like input.

    Returns
    -------
    np.ndarray
        Squared values, same shape as `x`.
    """
    arr = _as_array(x)
    return arr**2


def sinusoidal(x: Iterable) -> np.ndarray:
    """Sinusoidal function (element-wise).

    Formula
    -------
    f(x) = sin(x)

    Parameters
    ----------
    x : Iterable
        Scalar or array-like input.

    Returns
    -------
    np.ndarray
        Sine of `x`, same shape as input.
    """
    arr = _as_array(x)
    return np.sin(arr)


def ackley(
    x: Iterable, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi
) -> np.ndarray:
    """Ackley function for x ∈ R^n.

    Formula
    -------
    f(x) = -a * exp(-b * sqrt((1/n) * sum_{i=1}^n x_i^2))
           - exp((1/n) * sum_{i=1}^n cos(c * x_i)) + a + e

    Parameters
    ----------
    x : Iterable
        1-D array-like of length `n` or array with last axis `n`. Leading
        axes are treated as batch dimensions.
    a : float, optional
        Scaling constant (default 20.0).
    b : float, optional
        Exponential decay constant (default 0.2).
    c : float, optional
        Frequency multiplier for cosine term (default 2*pi).

    Returns
    -------
    float or np.ndarray
        Ackley value(s). Scalar for single-vector input, array for batched input.
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
    """Rosenbrock function for x ∈ R^n.

    Formula
    -------
    f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Parameters
    ----------
    x : Iterable
        1-D array-like of length `n` or array with last axis `n`.

    Returns
    -------
    float or np.ndarray
        Rosenbrock value(s) for the input vector(s).
    """
    xs, n, orig_ndim = _as_batch(x)

    if n < 2:
        result = np.zeros(xs.shape[0])
    else:
        vals = 100.0 * (xs[:, 1:] - xs[:, :-1] ** 2) ** 2 + (1.0 - xs[:, :-1]) ** 2
        result = np.sum(vals, axis=1)

    return float(result[0]) if orig_ndim <= 1 else result.reshape(xs.shape[:-1])


def rastrigin(x: Iterable) -> float:
    """Rastrigin function for x ∈ R^n.

    Formula
    -------
    f(x) = 10*n + sum_{i=1}^n [x_i^2 - 10*cos(2*pi*x_i)]

    Parameters
    ----------
    x : Iterable
        1-D array-like of length `n` or array with last axis `n`.

    Returns
    -------
    float or np.ndarray
        Rastrigin value(s) for the input vector(s).
    """
    xs, n, orig_ndim = _as_batch(x)
    result = 10.0 * n + np.sum(xs**2 - 10.0 * np.cos(2.0 * np.pi * xs), axis=1)
    return float(result[0]) if orig_ndim <= 1 else result.reshape(xs.shape[:-1])


def visualize_1d_functions(
    xlim: Tuple[float, float] = (-5.0, 5.0), points: int = 400
) -> Figure:
    """
    Visualize 1-D functions: quadratic and sinusoidal.

    Plots the quadratic function f(x) = x^2 and the sinusoidal function f(x) = sin(x)
    side by side over the specified range.

    Parameters
    ----------
    xlim : tuple of float, optional
        The (min, max) range for the x-axis. Default is (-5.0, 5.0).
    points : int, optional
        Number of points to evaluate for the plot. Default is 400.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the two subplots.
    """
    x = np.linspace(xlim[0], xlim[1], points)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x, quadratic(x))
    axes[0].set_title("Quadratic Function f(x)=x²")
    axes[0].grid(True)

    axes[1].plot(x, sinusoidal(x))
    axes[1].set_title("Sinusoidal Function f(x)=sin(x)")
    axes[1].grid(True)

    fig.tight_layout()
    return fig


def visualize_2d_function(
    func: Callable[[np.ndarray], float],
    name: str = "Function",
    xlim: Tuple[float, float] = (-5.0, 5.0),
    ylim: Optional[Tuple[float, float]] = None,
    points: int = 200,
) -> Figure:
    """
    Visualize a 2-D function with a surface and a contour plot.

    The function must accept a 2-element vector and return a scalar.

    Parameters
    ----------
    func : callable
        Function taking a 2-D vector (length-2) and returning a scalar.
    name : str, optional
        Name of the function, used in plot titles. Default is "Function".
    xlim : tuple of float, optional
        Range for x-axis. Default is (-5.0, 5.0).
    ylim : tuple of float, optional
        Range for y-axis. Default is equal to xlim.
    points : int, optional
        Number of points along each axis. Default is 200.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the surface plot (3D) and contour plot (2D).
    """
    if ylim is None:
        ylim = xlim

    x = np.linspace(xlim[0], xlim[1], points)
    y = np.linspace(ylim[0], ylim[1], points)
    X, Y = np.meshgrid(x, y)

    pts = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([func(p) for p in pts]).reshape(X.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Replace axes[0] with a 3D surface axes
    fig.delaxes(axes[0])
    axes[0] = fig.add_subplot(1, 2, 1, projection="3d")
    axes[0].plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
    axes[0].set_title(f"{name} Surface")

    # 2D contour plot
    contour = axes[1].contourf(X, Y, Z, cmap="viridis", levels=50)
    fig.colorbar(contour, ax=axes[1])
    axes[1].set_title(f"{name} Contour")

    fig.tight_layout()
    return fig


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
    visualize_1d_functions()
    visualize_2d_function(ackley, name="Ackley")
