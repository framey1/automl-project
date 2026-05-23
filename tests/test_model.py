from src.automl import train_automl
from src.etl import extract, transform


def test_model_can_predict(tmp_path):
    df = extract(tmp_path / "data.csv")
    X_train, X_test, y_train, _ = transform(df)
    result = train_automl(X_train.head(80), y_train.head(80), cv=2)
    predictions = result.best_estimator.predict(X_test.head(5))
    assert len(predictions) == 5
