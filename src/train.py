from __future__ import annotations

import json
import joblib

from src.automl import train_automl
from src.config import DATA_RAW, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COLUMN
from src.etl import extract, transform, load_processed
from src.evaluate import (
    calculate_metrics,
    save_class_distribution,
    save_confusion_matrix,
    save_feature_importance,
    save_metrics,
    save_roc_curve,
)


def log_monitoring(metrics: dict, params: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    monitoring_record = {"metrics": metrics, "params": params}
    (REPORTS_DIR / "monitoring_record.json").write_text(
        json.dumps(monitoring_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        import mlflow
        mlflow.set_experiment("automl_breast_cancer")
        with mlflow.start_run(run_name="custom_automl_pipeline"):
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            for key, value in params.items():
                mlflow.log_param(key, value)
            mlflow.sklearn.log_model(params["model_name"], "model")
    except Exception as exc:
        print(f"MLflow логгирование, пропущено: {exc}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = extract(DATA_RAW)
    X_train, X_test, y_train, y_test = transform(df)
    load_processed(X_train, X_test, y_train, y_test)

    result = train_automl(X_train, y_train)
    model = result.best_estimator

    metrics = calculate_metrics(model, X_test, y_test)
    params = {
        "model_name": result.model_name,
        "best_cv_f1": result.best_cv_score,
        **{k: str(v) for k, v in result.best_params.items()},
    }

    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    save_metrics(metrics)
    save_class_distribution(df, TARGET_COLUMN)
    save_confusion_matrix(model, X_test, y_test)
    save_roc_curve(model, X_test, y_test)
    save_feature_importance(model, X_train.columns)
    log_monitoring(metrics, params)

    print("Обучение завершено")
    print(json.dumps({"metrics": metrics, "params": params}, indent=2))


if __name__ == "__main__":
    main()
