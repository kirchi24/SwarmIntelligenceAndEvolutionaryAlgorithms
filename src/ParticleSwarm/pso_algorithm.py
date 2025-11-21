import os
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ==========================================================
# GPU-Support optional
# ==========================================================
try:
    import cudf
    from cuml.ensemble import RandomForestClassifier as CuRF
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# ==========================================================
# Utility-Funktionen
# ==========================================================
def macro_class_accuracy(y_true, y_pred):
    labels = np.unique(y_true)
    accs = []
    per_class = {}
    for lab in labels:
        mask = (y_true == lab)
        if np.sum(mask) == 0:
            acc = 0
        else:
            acc = np.mean(y_pred[mask] == y_true[mask])
        per_class[lab] = float(acc)
        accs.append(acc)
    return float(np.mean(accs)), per_class


def quick_eda(X, y, top_n=10):
    print(f"Data shape: {X.shape}")
    print("\nClass distribution:")
    cnt = Counter(y)
    for k, v in sorted(cnt.items()):
        print(f"  class {k}: {v}")

    df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(X.shape[1])])
    desc = df.describe().T

    print(f"\nFeature stats (first {top_n} shown):")
    for i in range(min(top_n, X.shape[1])):
        s = desc.iloc[i]
        print(
            f"{desc.index[i]}: mean={s['mean']:.3f}, std={s['std']:.3f}, "
            f"min={s['min']:.3f}, 25={s['25%']:.3f}, 50={s['50%']:.3f}, "
            f"75={s['75%']:.3f}, max={s['max']:.3f}"
        )
    print()
    return desc


# ==========================================================
# PSO Lite Version – verwendet in Streamlit und im main
# ==========================================================
def run_pso(
    n_particles: int,
    iterations: int,
    w: float,
    c1: float,
    c2: float,
    alpha: float,
    n_estimators: int,
    progress_callback=None,
):
    """
    Führt PSO-Feature-Selection aus.
    Rückgabe:
        best_features (0/1 Maske)
        best_fitness
        fitness_history
    """
    # ------------------------------------
    # Load dataset
    # ------------------------------------
    data = fetch_covtype()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    n_features = X_train.shape[1]
    rng = np.random.RandomState(42)

    # ------------------------------------
    # Initialize Swarm
    # ------------------------------------
    positions = rng.uniform(-1, 1, (n_particles, n_features))
    velocities = rng.uniform(-0.5, 0.5, (n_particles, n_features))

    personal_best_pos = positions.copy()
    personal_best_scores = np.full(n_particles, -np.inf)

    global_best_pos = None
    global_best_score = -np.inf

    fitness_history = []

    # ------------------------------------
    # Fitness function
    # ------------------------------------
    def evaluate(position):
        mask = (1 / (1 + np.exp(-position)) > 0.5).astype(int)
        selected = np.where(mask == 1)[0]

        if len(selected) == 0:
            return -999, mask

        Xtr = X_train[:, selected]
        Xte = X_test[:, selected]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=1
        )
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xte)

        macro_acc, _ = macro_class_accuracy(y_test, pred)
        sparsity_penalty = 1 - len(selected) / n_features

        fitness = alpha * macro_acc + (1 - alpha) * sparsity_penalty
        return fitness, mask

    # ------------------------------------
    # PSO loop
    # ------------------------------------
    for it in range(iterations):

        for i in range(n_particles):
            fitness, mask = evaluate(positions[i])

            # update personal best
            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_pos[i] = positions[i].copy()

            # update global best
            if fitness > global_best_score:
                global_best_score = fitness
                global_best_pos = positions[i].copy()

        fitness_history.append(global_best_score)

        if progress_callback:
            progress_callback(it + 1, iterations, global_best_score)

        # velocity + position update
        for i in range(n_particles):
            r1 = rng.rand(n_features)
            r2 = rng.rand(n_features)

            cognitive = c1 * r1 * (personal_best_pos[i] - positions[i])
            social = c2 * r2 * (global_best_pos - positions[i])

            velocities[i] = w * velocities[i] + cognitive + social
            positions[i] += velocities[i]

    best_features = (1 / (1 + np.exp(-global_best_pos)) > 0.5).astype(int)
    return best_features, global_best_score, fitness_history


# ==========================================================
# MAIN for CLI Execution
# ==========================================================
def main():
    t0 = time.time()

    print("1) Loading dataset...")
    data = fetch_covtype()
    X, y = data.data, data.target
    print(f"Loaded: X={X.shape}, y={y.shape}")

    print("\n2) EDA")
    quick_eda(X, y)

    print("\n3) Train/Test Split + Scaling")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print(f"GPU available: {GPU_AVAILABLE}")

    print("\n4) Running PSO (run_pso)...")
    best_features, best_fitness, fitness_hist = run_pso(
        n_particles=20,
        iterations=10,
        w=0.7,
        c1=1.5,
        c2=1.5,
        alpha=0.7,
        n_estimators=20,
        progress_callback=None
    )

    selected = np.where(best_features == 1)[0]

    print("\nPSO DONE")
    print("Best fitness:", best_fitness)
    print("Selected features:", selected.tolist())
    print("Count:", len(selected))

    print("\n5) Final RandomForest Evaluation")

    X_tr_sel = X_train[:, selected]
    X_te_sel = X_test[:, selected]

    # CPU fallback is default
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr_sel, y_train)
    y_pred = rf.predict(X_te_sel)

    overall = accuracy_score(y_test, y_pred)
    macro, per_class = macro_class_accuracy(y_test, y_pred)

    print("Overall accuracy:", overall)
    print("Macro accuracy:", macro)
    print("Per-class:")
    for k in sorted(per_class.keys()):
        print(f"  class {k}: {per_class[k]:.4f}")

    print("\nDONE. Time: %.1f sec" % (time.time() - t0))


if __name__ == "__main__":
    main()
