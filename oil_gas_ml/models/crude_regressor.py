import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score


class CrudeRegressor:

    def __init__(self, model_name="gradient_boosting", random_state=2024):
        self.random_state = random_state
        self.model_name = model_name
        self._trained = False
        self.target_name = None
        builders = {
            "random_forest": lambda rs: RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=rs, n_jobs=-1,
            ),
            "gradient_boosting": lambda rs: GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=rs,
            ),
            "extra_trees": lambda rs: ExtraTreesRegressor(
                n_estimators=200, max_depth=15, random_state=rs, n_jobs=-1,
            ),
            "svr": lambda rs: SVR(kernel="rbf", C=100, gamma="scale"),
            "mlp": lambda rs: MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                max_iter=500, random_state=rs, early_stopping=True,
            ),
            "ridge": lambda rs: Ridge(alpha=1.0),
            "elastic_net": lambda rs: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
        }
        if model_name not in builders:
            raise ValueError(f"'{model_name}' no existe")
        self.model = builders[model_name](random_state)

    def train(self, X_train, y_train, target_name="target"):
        self.target_name = target_name
        self.model.fit(X_train, y_train)
        self._trained = True
        return self

    def predict(self, X):
        if not self._trained:
            raise RuntimeError("Entrena el modelo primero")
        return self.model.predict(X)

    def cross_validate(self, X, y, cv=5):
        s1 = cross_val_score(self.model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        s2 = cross_val_score(self.model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        return {"mean_r2": s1.mean(), "std_r2": s1.std(), "mean_mae": -s2.mean()}

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = self.predict(X_test)
        return {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        }

    def save(self, path="outputs/models/crude_regressor.pkl"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "model_name": self.model_name, "target_name": self.target_name}, p)
        return p

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls(data["model_name"])
        instance.model = data["model"]
        instance.target_name = data["target_name"]
        instance._trained = True
        return instance
