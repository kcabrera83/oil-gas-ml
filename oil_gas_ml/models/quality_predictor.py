"""Predictor de calidad de crudo - modelo multi-output para rendimiento y valor."""

import numpy as np
import joblib
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class QualityPredictor:
    def __init__(self, base_model="gradient_boosting"):
        if base_model == "gradient_boosting":
            base = GradientBoostingRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42,
            )
        else:
            base = RandomForestRegressor(
                n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
            )

        self.model = MultiOutputRegressor(base, n_jobs=-1)
        self.base_model_name = base_model
        self.target_names = None
        self._trained = False

    def train(self, X_train, y_train, target_names=None):
        self.target_names = target_names or ["target_0", "target_1"]
        self.model.fit(X_train, y_train)
        self._trained = True
        return self

    def predict(self, X):
        if not self._trained:
            raise RuntimeError("Modelo no entrenado.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        results = {}
        for i, name in enumerate(self.target_names):
            results[name] = {
                "MAE": mean_absolute_error(y_test[:, i], y_pred[:, i]),
                "RMSE": np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])),
                "R2": r2_score(y_test[:, i], y_pred[:, i]),
            }
        return results

    def predict_quality_profile(self, X):
        predictions = self.predict(X)
        profiles = []
        for row in predictions:
            profile = {}
            for i, name in enumerate(self.target_names):
                profile[name] = round(float(row[i]), 2)
            profiles.append(profile)
        return profiles

    def save(self, path="outputs/models/quality_predictor.pkl"):
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "base_model_name": self.base_model_name,
            "target_names": self.target_names,
        }, filepath)
        return filepath

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls(base_model=data["base_model_name"])
        instance.model = data["model"]
        instance.target_names = data["target_names"]
        instance._trained = True
        return instance
