import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings("ignore")

# ==========================================================
# GPU-Support optional
# ==========================================================
try:
    import cudf
    from cuml.ensemble import ExtraTreesClassifier as CuRF

    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# ==========================================================
# Utility-Funktionen
# ==========================================================
def fast_macro_f1(y_true, y_pred):
    """Optimized macro F1-score calculation"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.union1d(y_true, y_pred)
    idx_t = np.searchsorted(labels, y_true)
    idx_p = np.searchsorted(labels, y_pred)

    # Confusion matrix
    cm = np.zeros((labels.size, labels.size), int)
    np.add.at(cm, (idx_t, idx_p), 1)

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    denom = 2 * TP + FP + FN
    f1 = 2 * TP / np.where(denom == 0, 1, denom)

    return float(f1.mean())


def quick_exploratory_data_analysis(X, y, top_n=10):
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


# ------------------------------------
# Fitness function
# ------------------------------------
def evaluate(
    position, X_train, X_test, y_train, y_test, alpha=0.7, n_estimators=50, max_depth=5
):
    """
    Evaluates a particle's position.
    Position is a real-valued vector; apply sigmoid and threshold at 0.5 to get feature mask.
    Returns fitness score and binary mask.
    """
    mask = (1 / (1 + np.exp(-position)) > 0.5).astype(int)
    selected = np.where(mask == 1)[0]
    n_features = X_train.shape[1]

    if len(selected) == 0:
        return -999, mask

    Xtr = X_train[:, selected]
    Xte = X_test[:, selected]

    clf = ExtraTreesClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
    )
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    macro_f1 = fast_macro_f1(y_test, pred)
    sparsity_penalty = 1 - len(selected) / n_features

    fitness = alpha * macro_f1 + (1 - alpha) * sparsity_penalty
    return fitness, mask, macro_f1


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
    max_depth: int = 5,
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
    global_mask = None
    global_best_score = -np.inf

    fitness_history = []
    f1_history = []

    # ------------------------------------
    # PSO loop
    # ------------------------------------
    for it in range(iterations):
        # evaluate particles
        for i in range(n_particles):
            fitness, mask, f1 = evaluate(
                positions[i],
                X_train,
                X_test,
                y_train,
                y_test,
                alpha,
                n_estimators,
                max_depth,
            )

            # update personal best
            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_pos[i] = positions[i].copy()

            # update global best
            if fitness > global_best_score:
                global_best_score = fitness
                global_best_pos = positions[i].copy()
                global_mask = mask.copy()
                global_f1 = f1

        fitness_history.append(global_best_score)
        f1_history.append(global_f1)

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

    best_features = global_mask
    return best_features, global_best_score, global_f1, fitness_history, f1_history


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
    quick_exploratory_data_analysis(X, y)

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
        progress_callback=None,
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
    rf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr_sel, y_train)
    y_pred = rf.predict(X_te_sel)

    overall = accuracy_score(y_test, y_pred)
    macro = fast_macro_f1(y_test, y_pred)

    print("Overall accuracy:", overall)
    print("Macro F1-Score:", macro)
    print("\nDONE. Time: %.1f sec" % (time.time() - t0))


if __name__ == "__main__":
    main()
