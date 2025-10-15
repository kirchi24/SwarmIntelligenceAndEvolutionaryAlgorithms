from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


# ----------------------------------------------------------------------
# Benchmark functions
# ----------------------------------------------------------------------


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
    sum_sq = np.sum(x_values**2, axis=-1)
    sum_cos = np.sum(np.cos(c * x_values), axis=-1)

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
        result = np.sum(vals, axis=-1)

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
    result = 10.0 * n + np.sum(xs**2 - 10.0 * np.cos(2.0 * np.pi * xs), axis=-1)
    return float(result[0]) if orig_ndim <= 1 else result.reshape(xs.shape[:-1])


# ----------------------------------------------------------------------
# 1D Visualization
# ----------------------------------------------------------------------


def visualize_1d_function(
    func: Callable[[np.ndarray], np.ndarray],
    name: str = "Function",
    xlim: Tuple[float, float] = (-5.0, 5.0),
    points: int = 400,
) -> go.Figure:
    """
    Visualize a single 1-D function as an interactive Plotly line chart.

    Parameters
    ----------
    func : callable
        Function taking a 1D numpy array of x values and returning y values.
    name : str, optional
        Plot title. Default is "Function".
    xlim : tuple of float, optional
        Range of x values. Default is (-5.0, 5.0).
    points : int, optional
        Number of x points to evaluate. Default is 400.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive 1D function plot.
    """
    x = np.linspace(xlim[0], xlim[1], points)
    y = func(x)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color="royalblue"))
    )
    fig.update_layout(
        title=f"{name} (1D Function)",
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig


def add_trajectory_1d(
    fig: go.Figure,
    trajectory: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
) -> go.Figure:
    """
    Overlay a 1D optimization trajectory on a Plotly function plot.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure created by `visualize_1d_function`.
    trajectory : np.ndarray
        Optimization trajectory of shape (num_steps,) or (num_steps, 1).
    func : callable
        Function used to compute y-values along the trajectory.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Updated figure with the trajectory overlay.
    """
    trajectory = np.array(trajectory).reshape(-1)
    y_vals = func(trajectory)

    fig.add_trace(
        go.Scatter(
            x=trajectory,
            y=y_vals,
            mode="markers+lines",
            name="Trajectory",
            marker=dict(color="red", size=6),
            line=dict(color="red", dash="dash"),
        )
    )
    return fig


# ----------------------------------------------------------------------
# 3D Visualization
# ----------------------------------------------------------------------


def visualize_3d_function(
    func: Callable[[np.ndarray], float],
    name: str = "Function",
    xlim: Tuple[float, float] = (-5.0, 5.0),
    ylim: Optional[Tuple[float, float]] = None,
    points: int = 200,
) -> go.Figure:
    """
    Visualize a 2D function as an interactive 3D surface.

    Parameters
    ----------
    func : callable
        Function taking a 2D vector (x, y) and returning a scalar f(x, y).
    name : str, optional
        Plot title. Default is "Function".
    xlim, ylim : tuple of float, optional
        Axis limits for x and y. Default is (-5, 5).
    points : int, optional
        Grid resolution. Default is 200.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive 3D surface plot.
    """
    if ylim is None:
        ylim = xlim

    x = np.linspace(xlim[0], xlim[1], points)
    y = np.linspace(ylim[0], ylim[1], points)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([func(p) for p in pts]).reshape(X.shape)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Viridis",
                showscale=True,
                name=name,
            )
        ]
    )

    fig.update_layout(
        title=f"{name} (3D Surface)",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x, y)",
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def add_trajectory_3d(
    fig: go.Figure,
    trajectory: np.ndarray,
    func: Callable[[np.ndarray], float],
) -> go.Figure:
    """
    Overlay a 3D optimization trajectory on a Plotly surface plot.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure created by `visualize_3d_function`.
    trajectory : np.ndarray
        Optimization trajectory of shape (num_steps, 2).
    func : callable
        Function used to compute z-values for trajectory points.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Updated 3D figure with the trajectory overlay.
    """
    trajectory = np.array(trajectory)
    if trajectory.shape[1] != 2:
        raise ValueError("Trajectory must have shape (num_steps, 2).")

    z_vals = np.array([func(p) for p in trajectory])

    fig.add_trace(
        go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=z_vals,
            mode="markers+lines",
            name="Trajectory",
            marker=dict(size=4, color="red"),
            line=dict(color="red", width=3),
        )
    )
    return fig


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


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
    arr = _as_array(x)
    orig_ndim = arr.ndim
    x_values = np.atleast_2d(arr)
    n = x_values.shape[-1] if not (x_values.shape[-1] == 1 and orig_ndim == 1) else 1
    return x_values, n, orig_ndim


# ----------------------------------------------------------------------
# Sample usage / quick demo
# ----------------------------------------------------------------------


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
    visualize_1d_function()
    visualize_3d_function(ackley, name="Ackley")
