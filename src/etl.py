from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.config import DATA_RAW, DATA_PROCESSED, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def extract(output_path: Path = DATA_RAW) -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df.rename(columns={"target": TARGET_COLUMN}, inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def transform(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if df.empty:
        raise ValueError("Dataset is empty")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing")
    if df.isna().sum().sum() > 0:
        df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def load_processed(X_train, X_test, y_train, y_test, output_dir: Path = DATA_PROCESSED) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)


def run_etl() -> pd.DataFrame:
    df = extract()
    X_train, X_test, y_train, y_test = transform(df)
    load_processed(X_train, X_test, y_train, y_test)
    return df


if __name__ == "__main__":
    run_etl()
    print(f"ETL завершен. Сохранено в {DATA_RAW}")
