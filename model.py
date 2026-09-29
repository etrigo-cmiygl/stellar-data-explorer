"""
Classifying stellar type with scikit-learn.

The workflow is deliberately conventional: one stratified train/test split held
out before anything is fitted, all preprocessing inside Pipelines so that
scalers and encoders never see the test set, cross-validation on the training
set for model selection, and the test set scored exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.tree import DecisionTreeClassifier

from .analysis import COLOUR_ORDER, SPECTRAL_ORDER, TYPE_LABELS

#: Temperature, luminosity and radius span 1.3, 10 and 5.4 orders of magnitude,
#: so the linear model gets them logged. Absolute magnitude is already a
#: logarithmic quantity and is only centred and scaled.
LOG_COLUMNS = ["Temperature", "L", "R"]
LINEAR_COLUMNS = ["A_M"]
CATEGORICAL_COLUMNS = ["Spectral_Class", "Color"]
FEATURES = LOG_COLUMNS + LINEAR_COLUMNS + CATEGORICAL_COLUMNS

LABELS = {"Temperature": "Temperature", "L": "Luminosity", "R": "Radius",
          "A_M": "Absolute magnitude", "Spectral_Class": "Spectral class",
          "Color": "Colour"}

TEST_SIZE = 0.25
N_SPLITS = 5


def _log10(x):
    """Module-level (not a lambda) so the pipeline stays picklable."""
    return np.log10(np.clip(np.asarray(x, dtype=float), 1e-12, None))


def _preprocessor(kind: str) -> ColumnTransformer:
    """Preprocessing suited to a family of models.

    Linear models need everything on a comparable scale. Trees do not -- they
    split on thresholds, and no monotonic transform changes which side of a
    threshold a value falls on -- so they get the raw numbers, and their
    categoricals are ordinal-encoded in *physical* order (O to M, blue to red)
    so that one split can separate "hotter than F" from "cooler".
    """
    if kind == "linear":
        return ColumnTransformer([
            ("log_scaled", Pipeline([("log", FunctionTransformer(_log10)),
                                     ("scale", StandardScaler())]), LOG_COLUMNS),
            ("scaled", StandardScaler(), LINEAR_COLUMNS),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLUMNS),
        ])
    return ColumnTransformer([
        ("numeric", "passthrough", LOG_COLUMNS + LINEAR_COLUMNS),
        ("ordinal", OrdinalEncoder(categories=[SPECTRAL_ORDER, COLOUR_ORDER],
                                   handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL_COLUMNS),
    ])


def build_models(seed: int = 42) -> dict[str, Pipeline]:
    """Three models with genuinely different reasoning, not a leaderboard."""
    return {
        "Logistic Regression": Pipeline([
            ("prep", _preprocessor("linear")),
            ("clf", LogisticRegression(max_iter=5000, random_state=seed))]),
        "Decision Tree": Pipeline([
            ("prep", _preprocessor("tree")),
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, random_state=seed))]),
        "Random Forest": Pipeline([
            ("prep", _preprocessor("tree")),
            ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed))]),
    }


@dataclass
class Results:
    class_names: list[str]
    n_train: int
    n_test: int
    scores: pd.DataFrame
    best_model: str
    confusion: np.ndarray
    report: str
    contributions: pd.DataFrame
    permutation: pd.DataFrame
    grid: dict


def _feature_contributions(X_train, y_train, cv, seed: int) -> pd.DataFrame:
    """How useful is each measurement alone, and how much is it missed?

    Permutation importance is the usual tool, but it is blunted by redundancy:
    radius, luminosity and magnitude are locked together by the physics, so a
    model deprived of one simply reads it off the others and the importance
    comes out as zero. Training on one column at a time cuts through that.
    """
    def score(columns: list[str]) -> float:
        numeric = [c for c in columns if c not in CATEGORICAL_COLUMNS]
        categorical = [c for c in columns if c in CATEGORICAL_COLUMNS]
        blocks = []
        if numeric:
            blocks.append(("numeric", "passthrough", numeric))
        if categorical:
            order = {"Spectral_Class": SPECTRAL_ORDER, "Color": COLOUR_ORDER}
            blocks.append(("ordinal", OrdinalEncoder(
                categories=[order[c] for c in categorical],
                handle_unknown="use_encoded_value", unknown_value=-1), categorical))
        pipeline = Pipeline([
            ("prep", ColumnTransformer(blocks)),
            # n_jobs=1 on the forest: the parallelism belongs on cross_val_score
            # below. Nesting both oversubscribes the CPU and runs slower.
            ("clf", RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=seed))])
        return float(cross_val_score(pipeline, X_train[columns], y_train,
                                     cv=cv, n_jobs=-1).mean())

    full = score(FEATURES)
    rows = {LABELS[c]: {"alone": score([c]),
                        "lost_if_removed": full - score([f for f in FEATURES if f != c])}
            for c in FEATURES}
    out = pd.DataFrame(rows).T.sort_values("alone", ascending=False)
    out.attrs["full"] = full
    return out.round(4)


def run(df: pd.DataFrame, seed: int = 42) -> Results:
    """Fit, validate and evaluate. The test set is scored once, at the end."""
    class_names = [TYPE_LABELS[i] for i in sorted(TYPE_LABELS)]
    X = df[FEATURES].copy()
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype(str)
    y = df["Type"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    rows, fitted = [], {}
    for name, pipeline in build_models(seed).items():
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, n_jobs=-1)
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        fitted[name] = pipeline
        rows.append({
            "model": name,
            "cv_accuracy": cv_scores.mean(), "cv_sd": cv_scores.std(),
            "test_accuracy": accuracy_score(y_test, predicted),
            "precision": precision_score(y_test, predicted, average="macro", zero_division=0),
            "recall": recall_score(y_test, predicted, average="macro", zero_division=0),
            "f1": f1_score(y_test, predicted, average="macro", zero_division=0),
        })
    scores = pd.DataFrame(rows).set_index("model").round(4)

    # Highest cross-validated accuracy; ties broken towards the simpler model.
    best = scores["cv_accuracy"].max()
    best_model = next(m for m in build_models(seed)
                      if scores.loc[m, "cv_accuracy"] >= best - 1e-12)

    predicted = fitted[best_model].predict(X_test)
    perm = permutation_importance(fitted[best_model], X_test, y_test, n_repeats=20,
                                  random_state=seed, scoring="accuracy", n_jobs=-1)

    # A two-feature model, purely so its decision boundaries can be drawn on the
    # HR diagram. Its accuracy says how much of the problem lives in that plane.
    features_2d = df[["logT", "logL"]].to_numpy()
    Xa, Xb, ya, yb = train_test_split(features_2d, y, test_size=TEST_SIZE,
                                      stratify=y, random_state=seed)
    flat = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed).fit(Xa, ya)
    xx, yy = np.meshgrid(
        np.linspace(features_2d[:, 0].min() - 0.05, features_2d[:, 0].max() + 0.05, 240),
        np.linspace(features_2d[:, 1].min() - 0.6, features_2d[:, 1].max() + 0.6, 240))
    grid = {"xx": xx, "yy": yy,
            "zz": flat.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape),
            "accuracy": float(accuracy_score(yb, flat.predict(Xb)))}

    return Results(
        class_names=class_names,
        n_train=len(X_train), n_test=len(X_test),
        scores=scores,
        best_model=best_model,
        confusion=confusion_matrix(y_test, predicted, labels=range(len(class_names))),
        report=classification_report(y_test, predicted, labels=range(len(class_names)),
                                     target_names=class_names, zero_division=0),
        contributions=_feature_contributions(X_train, y_train, cv, seed),
        permutation=pd.DataFrame({"importance": perm.importances_mean.round(4)},
                                 index=[LABELS[c] for c in FEATURES]
                                 ).sort_values("importance", ascending=False),
        grid=grid,
    )
