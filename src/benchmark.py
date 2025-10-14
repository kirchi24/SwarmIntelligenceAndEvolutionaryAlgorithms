import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def quadratic(x):
    """Quadratic function f(x) = x^2"""
    return x**2


def sinusoidal(x):
    """Sinusoidal function f(x) = sin(x)"""
    return np.sin(x)


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    """
    Ackley function
    f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2)))
           - exp(1/n * sum(cos(c * x_i))) + a + exp(1)
    """
    x = np.asarray(x)
    n = x.shape[-1] if x.ndim > 1 else 1
    sum_sq = np.sum(x**2, axis=-1)
    sum_cos = np.sum(np.cos(c * x), axis=-1)
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.e


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
