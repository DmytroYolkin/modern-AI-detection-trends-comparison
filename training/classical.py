"""Classical (non-neural) classifiers over the cached extractor features.

An alternative to the neural fusion model in `model.py`. Instead of the
`MultiFeatureFusion` backbone learning to combine the three feature streams, a
classical classifier (XGBoost, random forest, ...) operates directly on the
concatenated feature vector:

    [ NELA 87 | StyleDecipher 10 | TRACE 128 ]  ->  225-dim input

Each classical model does its own feature combination -- tree splits, linear
weights, or MLP layers -- so no fusion step is needed; the model *is* the
combiner. `train_classical.py` drives this.
"""

from __future__ import annotations

import inspect

import numpy as np

# --- supported backends ----------------------------------------------------
BACKENDS = ("xgboost", "random_forest", "logreg", "svm", "mlp",
            "hist_gbm", "gradient_boosting")


def flatten_features(nela: np.ndarray, style: np.ndarray, trace: np.ndarray) -> np.ndarray:
    """Concatenate the three feature blocks into one ``(N, 225)`` matrix."""
    return np.concatenate(
        [np.asarray(nela, np.float32),
         np.asarray(style, np.float32),
         np.asarray(trace, np.float32)],
        axis=1,
    ).astype(np.float32)


def block_importances(importances: np.ndarray, dims: dict) -> dict:
    """Sum a per-feature importance vector into per-extractor fractions.

    Answers "how much does each extractor contribute?" -- the comparison this
    whole repo is about.

    Tolerant of single-block ``dims`` (e.g. ``{"nela": 87}`` for a
    NELA-only classifier): missing blocks are skipped and the surviving
    fractions still sum to ~1.0 across the present blocks.
    """
    order = ("nela", "style", "trace")
    out, start = {}, 0
    total = float(np.sum(np.abs(importances))) or 1.0
    for name in order:
        if name not in dims:
            continue
        width = dims[name]
        out[name] = float(np.sum(np.abs(importances[start:start + width])) / total)
        start += width
    return out


def select_blocks(
    nela: np.ndarray, style: np.ndarray, trace: np.ndarray,
    blocks: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Concatenate just the requested feature blocks, in canonical order.

    Used to train and evaluate single-modality (or 2-of-3-modality) classifiers
    without padding the missing blocks with zeros -- the saved model's input
    width then genuinely matches its training dims.
    """
    table = {"nela": nela, "style": style, "trace": trace}
    parts: list[np.ndarray] = []
    for name in ("nela", "style", "trace"):
        if name in blocks:
            parts.append(np.asarray(table[name], np.float32))
    if not parts:
        raise ValueError(f"no blocks selected from {blocks!r}")
    return np.concatenate(parts, axis=1).astype(np.float32)


def _build_estimator(backend: str, seed: int, overrides: dict):
    """Construct an unfitted estimator for `backend` (imports done lazily)."""
    n_estimators = overrides.get("n_estimators")
    max_depth = overrides.get("max_depth")
    learning_rate = overrides.get("learning_rate")

    if backend == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=n_estimators or 400,
            max_depth=max_depth or 6,
            learning_rate=learning_rate or 0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    if backend == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=n_estimators or 400,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=seed,
        )
    if backend == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    if backend == "svm":
        from sklearn.svm import SVC

        return SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)
    if backend == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,                       # L2 regularisation
            learning_rate_init=learning_rate or 1e-3,
            max_iter=n_estimators or 300,     # `n_estimators` reused as max epochs
            early_stopping=True,
            n_iter_no_change=15,
            random_state=seed,
        )
    if backend == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=n_estimators or 400,
            max_depth=max_depth,
            learning_rate=learning_rate or 0.1,
            random_state=seed,
        )
    if backend == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=n_estimators or 300,
            max_depth=max_depth or 3,
            learning_rate=learning_rate or 0.1,
            random_state=seed,
        )
    raise ValueError(f"unknown backend {backend!r}; pick from {BACKENDS}")


def _accepts_sample_weight(estimator) -> bool:
    """True when the estimator's ``fit`` accepts a ``sample_weight`` argument."""
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def _balanced_oversample(X: np.ndarray, y: np.ndarray, seed: int):
    """Random oversampling of minority classes up to the majority count.

    Used to balance training for estimators (e.g. `MLPClassifier`) whose
    ``fit`` takes no ``sample_weight``.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max())
    parts = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        if len(idx) < target:
            extra = rng.choice(idx, size=target - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        parts.append(idx)
    order = np.concatenate(parts)
    rng.shuffle(order)
    return X[order], y[order]


class ClassicalClassifier:
    """Uniform wrapper around a classical classifier on the 225-dim features."""

    def __init__(self, backend: str, *, seed: int = 42, class_weighting: bool = True,
                 **overrides) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend
        self.seed = seed
        self.class_weighting = class_weighting
        # keep only the known tuning knobs
        self.overrides = {k: v for k, v in overrides.items()
                          if k in ("n_estimators", "max_depth", "learning_rate") and v is not None}
        self.estimator = _build_estimator(backend, seed, self.overrides)

    @classmethod
    def available_backends(cls) -> list[str]:
        """Backends whose libraries are importable in this environment."""
        ok = []
        for b in BACKENDS:
            try:
                _build_estimator(b, 0, {})
                ok.append(b)
            except Exception:
                pass
        return ok

    # ---- fit / predict ---------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        fit_kwargs = {}
        if self.class_weighting:
            if _accepts_sample_weight(self.estimator):
                from sklearn.utils.class_weight import compute_sample_weight

                fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y)
            else:
                # estimators like MLPClassifier take no sample_weight --
                # balance the training set by oversampling instead
                X, y = _balanced_oversample(X, y, self.seed)
        self.estimator.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def feature_importances(self) -> np.ndarray | None:
        """Per-feature importance (225,) when the backend exposes it, else None."""
        if hasattr(self.estimator, "feature_importances_"):
            return np.asarray(self.estimator.feature_importances_, dtype=np.float64)
        if hasattr(self.estimator, "coef_"):           # logistic regression
            return np.abs(np.asarray(self.estimator.coef_, dtype=np.float64)).ravel()
        return None                                    # e.g. rbf-SVM

    # ---- persistence -----------------------------------------------------

    def save(self, path, *, dims: dict, normalizer=None, metrics: dict | None = None,
             label_names=("human", "ai")) -> None:
        """Save the fitted model + everything needed to reload and run it."""
        import joblib

        payload = {
            "kind": "classical",
            "backend": self.backend,
            "estimator": self.estimator,
            "hparams": {"seed": self.seed, "class_weighting": self.class_weighting,
                        **self.overrides},
            "dims": dict(dims),
            "normalizer": normalizer.state_dict() if normalizer is not None else None,
            "metrics": metrics or {},
            "label_names": list(label_names),
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path) -> tuple["ClassicalClassifier", dict]:
        """Rebuild a fitted classifier from a checkpoint. Returns ``(clf, payload)``."""
        import joblib

        payload = joblib.load(path)
        clf = cls.__new__(cls)
        clf.backend = payload["backend"]
        hp = payload.get("hparams", {})
        clf.seed = hp.get("seed", 42)
        clf.class_weighting = hp.get("class_weighting", True)
        clf.overrides = {k: v for k, v in hp.items()
                         if k in ("n_estimators", "max_depth", "learning_rate")}
        clf.estimator = payload["estimator"]
        return clf, payload
