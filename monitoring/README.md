# Monitoring

В проекте используется MLflow и локальный JSON-файл для мониторинга качества модели.

Что логируется:

- accuracy;
- precision;
- recall;
- f1;
- roc_auc;
- выбранная модель;
- лучшие гиперпараметры;
- best CV F1-score.

Команда запуска UI:

```bash
mlflow ui
```
