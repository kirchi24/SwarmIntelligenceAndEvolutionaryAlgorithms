import os
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as SkRF

warnings.filterwarnings("ignore")

# try GPU RF (cuML)
try:
    import cudf
    from cuml.ensemble import RandomForestClassifier as CuRF
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# ======================================================================
# PSO Feature Selector
# ======================================================================

class PSOFeatureSelector:
    """
    Binary PSO for feature selection.
    Parallel particle evaluation with joblib.
    Fitness = alpha * macro_accuracy + (1 - alpha) * (1 - n_sel/N)
    Uses RandomForest (sklearn or cuML if available).
    """

    def __init__(
        self,
        n_particles=20,
        n_features=54,
        w=0.7,
        c1=1.5,
        c2=1.5,
        alpha=0.7,
        use_gpu=False,
        rf_params=None,
        random_state=42,
        n_jobs=-1,
    ):
        self.n_particles = n_particles
        self.n_features = n_features
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # RF-params: n_jobs = 1 (avoid nested parallel)
        if rf_params is None:
            rf_params = {"n_estimators": 100, "random_state": random_state, "n_jobs": 1}
        else:
            rf_params = dict(rf_params)
            rf_params["n_jobs"] = 1
        self.rf_params = rf_params

        rng = np.random.RandomState(random_state)
        self.positions = rng.uniform(-1, 1, (n_particles, n_features))
        self.velocities = rng.uniform(-0.5, 0.5, (n_particles, n_features))

        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(n_particles, -np.inf)
        self.global_best_position = None
        self.global_best_score = -np.inf

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def binarize(self, pos):
        return (self.sigmoid(pos) > 0.5).astype(int)

    @staticmethod
    def macro_class_accuracy(y_true, y_pred):
        labels = np.unique(y_true)
        per_class = {}
        accs = []
        for lab in labels:
            mask = (y_true == lab)
            acc = np.mean(y_pred[mask] == y_true[mask]) if mask.sum() else 0.0
            per_class[lab] = float(acc)
            accs.append(acc)
        return float(np.mean(accs)), per_class

    def _fit_predict_rf(self, X_train, y_train, X_test):
        """Fit RandomForest and return preds. GPU if possible."""
        if self.use_gpu:
            try:
                df_tr = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
                df_te = cudf.DataFrame.from_pandas(pd.DataFrame(X_test))
                s_y = cudf.Series(y_train)
                model = CuRF(
                    n_estimators=self.rf_params["n_estimators"],
                    random_state=self.rf_params["random_state"]
                )
                model.fit(df_tr, s_y)
                preds_gpu = model.predict(df_te)
                return preds_gpu.to_array()
            except Exception:
                pass  # fallback

        # CPU fallback
        clf = SkRF(**self.rf_params)
        clf.fit(X_train, y_train)
        return clf.predict(X_test)

    def evaluate_particle(self, binary_mask, X_train, X_test, y_train, y_test):
        selected = np.where(binary_mask == 1)[0]
        n_selected = len(selected)
        if n_selected == 0:
            return -999.0, 0.0, {}

        X_tr = X_train[:, selected]
        X_te = X_test[:, selected]

        # Avoid nested parallelism
        with threadpool_limits(limits=1):
            preds = self._fit_predict_rf(X_tr, y_train, X_te)

        macro_acc, per_class = self.macro_class_accuracy(y_test, preds)
        penalty = n_selected / float(self.n_features)

        fitness = self.alpha * macro_acc + (1 - self.alpha) * (1 - penalty)
        return fitness, macro_acc, per_class

    def _eval_particle_job(self, i, X_train, X_test, y_train, y_test):
        mask = self.binarize(self.positions[i])
        return (i, *self.evaluate_particle(mask, X_train, X_test, y_train, y_test))

    def optimize(self, X_train, X_test, y_train, y_test, iterations=30, verbose=True):
        rng = np.random.RandomState(self.random_state)

        for it in range(iterations):
            # parallel evaluation of all particles
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._eval_particle_job)(i, X_train, X_test, y_train, y_test)
                for i in range(self.n_particles)
            )

            # update pbest & gbest
            for i, fitness, macro_acc, per_class in results:
                if fitness > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i].copy()
                if fitness > self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i].copy()

            # update velocities + positions
            for i in range(self.n_particles):
                r1 = rng.rand(self.n_features)
                r2 = rng.rand(self.n_features)
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.positions[i] += self.velocities[i]

            if verbose:
                mask = self.binarize(self.global_best_position)
                sel = mask.sum()
                _, macro, _ = self.evaluate_particle(mask, X_train, X_test, y_train, y_test)
                print(f"[PSO] Iter {it+1}/{iterations} — gbest={self.global_best_score:.4f}, macro={macro:.4f}, n_features={sel}")

        return self.global_best_position, self.global_best_score


# ======================================================================
# EDA helpers
# ======================================================================

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


# ======================================================================
# Main
# ======================================================================

def main():
    t0 = time.time()

    print("1) Loading dataset…")
    data = fetch_covtype()
    X, y = data.data, data.target
    print(f"Loaded: X={X.shape}, y={y.shape}\n")

    print("2) EDA")
    quick_eda(X, y, top_n=10)

    print("3) Train/Test split + Scaling")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print(f"Scaled: X_train={X_train.shape}, X_test={X_test.shape}\n")

    print("GPU available:", GPU_AVAILABLE)

    # PSO config
    pso = PSOFeatureSelector(
        n_particles=20,
        n_features=X_train.shape[1],
        w=0.7,
        c1=1.5,
        c2=1.5,
        alpha=0.7,
        use_gpu=GPU_AVAILABLE,
        rf_params={"n_estimators": 20, "random_state": 42, "n_jobs": 1},
        random_state=42,
        n_jobs=-1
    )

    print("4) Running PSO…")
    best_pos, best_score = pso.optimize(
        X_train, X_test, y_train, y_test,
        iterations=10,
        verbose=True
    )

    mask = pso.binarize(best_pos)
    selected = np.where(mask == 1)[0]

    print("\nPSO DONE")
    print("Best fitness:", best_score)
    print("Selected features:", selected.tolist())
    print("Count:", len(selected))

    # final evaluation
    print("\n5) Final RandomForest evaluation")
    X_tr_sel = X_train[:, selected]
    X_te_sel = X_test[:, selected]

    if GPU_AVAILABLE:
        try:
            print("Using GPU cuML RF")
            df_tr = cudf.DataFrame.from_pandas(pd.DataFrame(X_tr_sel))
            df_te = cudf.DataFrame.from_pandas(pd.DataFrame(X_te_sel))
            s_y = cudf.Series(y_train)
            model = CuRF(n_estimators=200, random_state=42)
            model.fit(df_tr, s_y)
            preds_gpu = model.predict(df_te)
            y_pred = preds_gpu.to_array()
        except Exception:
            print("GPU failed, fallback to CPU")
            rf = SkRF(n_estimators=200, random_state=42, n_jobs=os.cpu_count())
            rf.fit(X_tr_sel, y_train)
            y_pred = rf.predict(X_te_sel)
    else:
        print("Using CPU sklearn RF")
        rf = SkRF(n_estimators=200, random_state=42, n_jobs=os.cpu_count())
        rf.fit(X_tr_sel, y_train)
        y_pred = rf.predict(X_te_sel)

    overall = accuracy_score(y_test, y_pred)
    macro, per_class = PSOFeatureSelector.macro_class_accuracy(y_test, y_pred)

    print("Overall accuracy:", overall)
    print("Macro accuracy:", macro)
    print("Per-class:")
    for k in sorted(per_class.keys()):
        print(f"  class {k}: {per_class[k]:.4f}")

    print("\nDONE. Time: %.1f sec" % (time.time() - t0))


if __name__ == "__main__":
    main()
