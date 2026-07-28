import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)


class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_classification(self, y_true, y_pred, class_names=None, model_name="model"):
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
        }
        self.results[model_name] = {"type": "classification", "metrics": metrics, "report": report}
        return metrics

    def evaluate_regression(self, y_true, y_pred, model_name="model"):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except Exception:
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE_%": mape * 100,
        }
        self.results[model_name] = {"type": "regression", "metrics": metrics}
        return metrics

    def compare_models(self):
        comparison = {}
        for name, res in self.results.items():
            comparison[name] = res["metrics"]
        return comparison

    def print_report(self, model_name=None):
        if model_name:
            targets = [model_name]
        else:
            targets = list(self.results.keys())

        for name in targets:
            res = self.results[name]
            print(f"\n{'='*60}")
            print(f"  Modelo: {name} ({res['type']})")
            print(f"{'='*60}")
            for metric, value in res["metrics"].items():
                print(f"  {metric:25s}: {value:.4f}")
            print(f"{'='*60}")
