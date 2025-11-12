import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def construct_schedule(tau: np.array, alpha: float = 1, beta: float = 5) -> np.array:
    N, D, S = tau.shape
    schedule = np.full((N, D, S), np.nan)
    for d in range(D):
        for s in range(S):
            etas = np.array([eta_function(schedule, n, d, s) for n in range(N)])
            probs = (tau[:, d, s] ** alpha) * (etas**beta)
            probs /= probs.sum()

            # choose at least 2 nurses per shift
            chosen = np.random.choice(N, size=2, replace=False, p=probs)
            schedule[chosen, d, s] = 1
    return schedule


def heuristic_score(
    schedule, required_per_shift=2, penalties=[10000, 5000, 3000, 400, 300]
):
    """
    schedule: N x S x D matrix of 0s, 1s, or np.nan for undecided slots
    required_per_shift: amount of nurses necessary per shift
    penalties: penalties for violations of
        - coverage (less than two nurses per shift),
        - rest (morning shift following previous night shift),
        - maximum daily work (more than two shifts a day),
        - fairness (unfair distribution of shifts across nurses)
        - days off (less than one day of rest every week)
    """

    N, D, S = schedule.shape
    morning = 0
    night = S - 1

    # penalty for violation of coverage
    coverage = schedule.sum(axis=0)  # shape (D, S)
    coverage_deficit_matrix = np.maximum(0, required_per_shift - coverage)
    coverage_deficit = int(np.nansum(coverage_deficit_matrix))  # scalar count

    # penalty for rest violation
    rest_violations = 0
    # for days 0..D-2, if nurse works night d and morning d+1 -> violation
    if D >= 2:
        night_work = schedule[:, : D - 1, night]  # shape (N, D-1)
        next_morning = schedule[:, 1:D, morning]  # shape (N, D-1)
        rest_violations = int(
            np.nansum(
                np.logical_and(np.nan_to_num(night_work), np.nan_to_num(next_morning))
            )
        )

    # penalty for violation of maximum daily work
    shifts_per_nurse_day = np.nansum(schedule, axis=2)  # shape (N, D)
    daily_over_matrix = np.maximum(0, shifts_per_nurse_day - 2)
    daily_over = float(np.nansum(daily_over_matrix))

    # penalty for fairness violation
    shifts_per_nurse = np.nansum(schedule, axis=(1, 2))  # shape (N,)
    mean_shifts = np.nanmean(shifts_per_nurse) if N > 0 else 0.0
    fairness_penalty = float(np.nansum((shifts_per_nurse - mean_shifts) ** 2))

    # penalty for days-off violation
    days_off_per_nurse = np.nansum(
        shifts_per_nurse_day == 0, axis=1
    )  # days off per nurse
    dayoff_violations = int(np.nansum(days_off_per_nurse == 0))

    # sum up hard and soft penalties
    hard_penalty = (
        penalties[0] * coverage_deficit
        + penalties[1] * rest_violations
        + penalties[2] * daily_over
    )

    soft_penalty = penalties[3] * fairness_penalty + penalties[4] * dayoff_violations

    total_score = hard_penalty + soft_penalty

    # Return breakdown for analysis
    breakdown = {
        "coverage_deficit_count": coverage_deficit,
        "rest_violations": rest_violations,
        "daily_over_count": daily_over,
        "fairness_raw": fairness_penalty,
        "dayoff_violations": dayoff_violations,
        "hard_penalty": hard_penalty,
        "soft_penalty": soft_penalty,
        "total_score": total_score,
        "mean_shifts_per_nurse": float(mean_shifts),
        "shifts_per_nurse": shifts_per_nurse.tolist(),
    }

    return total_score, breakdown


def eta_function(partial_schedule, n, d, s):
    """
    1. assume you'd schedule nurse n for shift s on day d
    2. calculate the penalty of the schedule found so far
    3. undo the schedule assignment again (this was just for calculating the score)
    4. write everything into a heuristic desirability score (eta) in (0,1]
    """
    partial_schedule[n, d, s] = 1
    total_score, _ = heuristic_score(partial_schedule)
    partial_schedule[n, d, s] = np.nan
    eta_value = 1 / (1 + total_score)  # 0: worst, 1: best

    return eta_value


def evaporate_pheromones(tau: np.ndarray, rho: float):
    """Evaporate pheromones: tau <- (1-rho) * tau"""
    tau *= 1.0 - rho


def deposit_pheromones(
    tau: np.ndarray, schedule: np.ndarray, score: float, Q: float = 1.0
):
    """
    Deposit pheromone where schedule has assignments.
    Use amount proportional to Q / (1 + score) so better (lower) scores deposit more.
    """
    deposit_amount = Q / (1.0 + score)
    mask = ~np.isnan(schedule) & (schedule == 1)
    tau[mask] += deposit_amount


def update_pheromones(
    tau: np.ndarray,
    all_schedules: list,
    scores: list,
    rho: float = 0.1,
    Q: float = 1.0,
    pheromone_min=1e-6,
    pheromone_max=1e6,
):
    """Evaporate then deposit from each ant's schedule."""
    evaporate_pheromones(tau, rho)
    for sched, sc in zip(all_schedules, scores):
        deposit_pheromones(tau, sched, sc, Q=Q)

    # clamp to avoid numerical extremes
    np.clip(tau, pheromone_min, pheromone_max, out=tau)


def select_best_schedule(all_schedules: list, scores: list):
    """Return best schedule (lowest score) and its score & breakdown."""
    idx = int(np.argmin(scores))
    best = all_schedules[idx]
    best_score, breakdown = heuristic_score(best)
    return best, best_score, breakdown


# ---------------------------
# ACO main loop
# ---------------------------
def run_aco(
    tau_init,
    num_ants=30,
    max_iters=50,
    alpha=1.0,
    beta=5.0,
    rho=0.1,
    Q=1.0,
    verbose=True,
    seed=None,
    return_tau_history=False,
    progress_callback=None,
):
    """
    Führt die Ant Colony Optimization (ACO) aus.

    Parameters
    ----------
    tau_init : np.ndarray
        Anfangswerte der Pheromonmatrix (N, D, S)
    num_ants : int
        Anzahl der Ameisen pro Iteration
    max_iters : int
        Maximale Anzahl Iterationen
    alpha, beta, rho, Q : float
        Standardparameter der ACO
    verbose : bool
        Print-Ausgaben im Terminal
    seed : int
        Zufalls-Seed für Reproduzierbarkeit
    return_tau_history : bool
        Falls True, komplette Pheromonhistorie zurückgeben
    progress_callback : callable
        Optional. Wird nach jeder Iteration mit (current_iter, total_iters, best_score_so_far) aufgerufen.

    Returns
    -------
    best_schedule, best_score, breakdown, tau, tau_history, best_score_history
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    tau = tau_init.copy().astype(float)
    N, D, S = tau.shape

    # Sicherstellen, dass alle Werte positiv sind
    tau[tau <= 0] = 1.0

    best_overall_score = float("inf")
    best_overall_schedule = None
    best_breakdown = None

    tau_history = []
    best_score_history = []

    for itr in range(max_iters):
        # 1 Konstruktion von Lösungen
        all_schedules = [
            construct_schedule(tau, alpha=alpha, beta=beta) for _ in range(num_ants)
        ]
        scores = [heuristic_score(s)[0] for s in all_schedules]

        # 2 Pheromon-Update
        update_pheromones(tau, all_schedules, scores, rho=rho, Q=Q)

        # 3 Aktuellen besten Score bestimmen
        local_best_idx = int(np.argmin(scores))
        local_best_score = scores[local_best_idx]
        best_score_history.append(local_best_score)

        # 4 Global bestes Ergebnis aktualisieren
        if local_best_score < best_overall_score:
            best_overall_score = local_best_score
            best_overall_schedule = all_schedules[local_best_idx]
            _, best_breakdown = heuristic_score(best_overall_schedule)

        # 5 Fortschritt melden (optional an Streamlit)
        if progress_callback is not None:
            progress_callback(itr + 1, max_iters, best_overall_score)

        # 6 Optional: Pheromonhistorie speichern
        if return_tau_history:
            tau_history.append(tau.copy())

        if verbose and (itr % max(1, max_iters // 10) == 0 or itr == max_iters - 1):
            print(
                f"Iter {itr+1}/{max_iters} | best={best_overall_score:.2f} | local={local_best_score:.2f}"
            )

    # Rückgabe inkl. History
    if return_tau_history:
        return (
            best_overall_schedule,
            best_overall_score,
            best_breakdown,
            tau,
            tau_history,
            best_score_history,
        )
    else:
        return (
            best_overall_schedule,
            best_overall_score,
            best_breakdown,
            tau,
            None,
            best_score_history,
        )


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    # small demo problem
    N = 6  # nurses
    D = 7  # days
    S = 3  # shifts (morning, afternoon, night)

    # initialize pheromone (N,D,S) with small positive values
    tau0 = np.ones((N, D, S)) * 0.1

    # run ACO with tau and score history
    (
        best_schedule,
        best_score,
        breakdown,
        tau_final,
        tau_history,
        best_score_history,
    ) = run_aco(
        tau0,
        num_ants=30,
        max_iters=100,
        alpha=1.0,
        beta=5.0,
        rho=0.1,
        Q=100.0,
        verbose=True,
        seed=42,
        return_tau_history=True,
    )

    print("\nBest score:", best_score)
    print("Breakdown:", breakdown)
    print("Shifts per nurse:", np.nansum(best_schedule, axis=(1, 2)))
    print("Day-wise schedule (nurses x days x shifts):")
    print(best_schedule)

    # ========================
    # Score-History Plot
    # ========================
    fig_score = go.Figure()
    fig_score.add_trace(
        go.Scatter(
            y=best_score_history,
            mode="lines+markers",
            line=dict(color="green", width=3),
            name="Best Score per Iteration",
        )
    )
    fig_score.update_layout(
        title="Best Score over Iterations",
        xaxis_title="Iteration",
        yaxis_title="Best Score",
        template="plotly_white",
        width=800,
        height=400,
    )
    fig_score.show()

    # ========================
    # Pheromon-Historie Animation
    # ========================
    # flatten tau_history for animation
    N, D, S = tau_history[0].shape
    frames = []
    x, y, z, c, iteration = [], [], [], [], []

    for itr, tau_matrix in enumerate(tau_history):
        for n in range(N):
            for d in range(D):
                for s in range(S):
                    x.append(n)
                    y.append(d)
                    z.append(s)
                    c.append(tau_matrix[n, d, s])
                    iteration.append(itr + 1)  # iteration index

    # create a DataFrame for Plotly Express
    import pandas as pd

    df = pd.DataFrame(
        {"Nurse": x, "Day": y, "Shift": z, "Pheromone": c, "Iteration": iteration}
    )

    # 3D scatter with animation over iterations
    fig = px.scatter_3d(
        df,
        x="Nurse",
        y="Day",
        z="Shift",
        color="Pheromone",
        animation_frame="Iteration",
        color_continuous_scale="Viridis",
        range_color=[df["Pheromone"].min(), df["Pheromone"].max()],
    )

    fig.update_layout(
        scene=dict(xaxis_title="Nurse", yaxis_title="Day", zaxis_title="Shift"),
        title="Pheromone evolution over iterations",
    )
    fig.show()
