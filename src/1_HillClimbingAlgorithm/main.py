import numpy as np


def continuous_neighborhood(x, step_size=0.1):
    return x + np.random.uniform(-step_size, step_size, size=x.shape)


def hill_climbing(f, x0, neighborhood_fn, max_iter=1000, tol=1e-6):
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


def steepest_hill_climbing(f, x0, neighborhood_fn, max_iter=1000, samples=20, tol=1e-6):
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
