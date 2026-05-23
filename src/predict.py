from __future__ import annotations

import joblib
import pandas as pd

from src.config import MODELS_DIR


def predict(input_csv: str) -> pd.DataFrame:
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    X = pd.read_csv(input_csv)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    result = X.copy()
    result["prediction"] = predictions
    result["probability_class_1"] = probabilities
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    predict(args.input_csv).to_csv(args.output, index=False)
    print(f"Прогнозы сохранены в {args.output}")
