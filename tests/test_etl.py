from src.etl import extract, transform
from src.config import TARGET_COLUMN


def test_extract_dataset_not_empty(tmp_path):
    df = extract(tmp_path / "data.csv")
    assert not df.empty
    assert TARGET_COLUMN in df.columns


def test_transform_returns_train_test_split(tmp_path):
    df = extract(tmp_path / "data.csv")
    X_train, X_test, y_train, y_test = transform(df)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
