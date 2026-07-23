"""Modelo de regresión para valor de mercado y rendimiento de crudo."""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score


class CrudeRegressor:
    MODELS = {
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
        ),
        "svr": lambda: SVR(kernel="rbf", C=100, gamma="scale"),
        "mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            max_iter=500, random_state=42, early_stopping=True,
        ),
        "ridge": lambda: Ridge(alpha=1.0),
        "elastic_net": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
    }

    def __init__(self, model_name="gradient_boosting"):
        if model_name not in self.MODELS:
            raise ValueError(f"Modelo '{model_name}' no disponible. Opciones: {list(self.MODELS.keys())}")
        self.model_name = model_name
        self.model = self.MODELS[model_name]()
        self._trained = False
        self.target_name = None

    def train(self, X_train, y_train, target_name="target"):
        self.target_name = target_name
        self.model.fit(X_train, y_train)
        self._trained = True
        return self

    def predict(self, X):
        if not self._trained:
            raise RuntimeError("Modelo no entrenado.")
        return self.model.predict(X)

    def cross_validate(self, X, y, cv=5):
        scores_r2 = cross_val_score(self.model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        scores_neg_mae = cross_val_score(self.model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        return {
            "mean_r2": scores_r2.mean(),
            "std_r2": scores_r2.std(),
            "mean_mae": -scores_neg_mae.mean(),
        }

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = self.predict(X_test)
        return {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        }

    def get_feature_importance(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None

    def save(self, path="outputs/models/crude_regressor.pkl"):
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "model_name": self.model_name, "target_name": self.target_name}, filepath)
        return filepath

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls(model_name=data["model_name"])
        instance.model = data["model"]
        instance.target_name = data["target_name"]
        instance._trained = True
        return instance

    @staticmethod
    def train_all(X_train, y_train, target_name="target"):
        results = {}
        for name in CrudeRegressor.MODELS:
            print(f"  Entrenando {name}...")
            reg = CrudeRegressor(model_name=name)
            reg.train(X_train, y_train, target_name)
            results[name] = reg
        return results
