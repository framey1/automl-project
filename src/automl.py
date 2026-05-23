from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import RANDOM_STATE


@dataclass
class AutoMLResult:
    model_name: str
    best_estimator: Pipeline
    best_params: Dict[str, Any]
    best_cv_score: float


def get_candidate_models() -> Dict[str, tuple[Pipeline, Dict[str, list]]]:
    return {
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            {
                "model__C": [0.1, 1.0],
                "model__solver": ["liblinear"],
            },
        ),
        "random_forest": (
            Pipeline([
                ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]),
            {
                "model__n_estimators": [50],
                "model__max_depth": [None, 5],
            },
        ),
        "svc": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, random_state=RANDOM_STATE)),
            ]),
            {
                "model__C": [1.0],
                "model__kernel": ["rbf", "linear"],
            },
        ),
        "gradient_boosting": (
            Pipeline([
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]),
            {
                "model__n_estimators": [50],
                "model__learning_rate": [0.1],
                "model__max_depth": [2],
            },
        ),
    }


def train_automl(X_train, y_train, scoring: str = "f1", cv: int = 3) -> AutoMLResult:
    candidates = get_candidate_models()
    results = []
    for model_name, (pipeline, param_grid) in candidates.items():
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        results.append(
            AutoMLResult(
                model_name=model_name,
                best_estimator=search.best_estimator_,
                best_params=search.best_params_,
                best_cv_score=float(search.best_score_),
            )
        )
    return max(results, key=lambda item: item.best_cv_score)
