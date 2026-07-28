import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


class CrudeClassifier:

    def __init__(self, model_name="random_forest", random_state=2024):
        self.random_state = random_state
        self.model_name = model_name
        self._trained = False
        self.class_names = None
        builders = {
            "random_forest": lambda rs: RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=rs, n_jobs=-1,
            ),
            "gradient_boosting": lambda rs: GradientBoostingClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=rs,
            ),
            "svm": lambda rs: SVC(
                kernel="rbf", C=10, gamma="scale", probability=True, random_state=rs,
            ),
            "knn": lambda rs: KNeighborsClassifier(
                n_neighbors=7, weights="distance", n_jobs=-1,
            ),
            "mlp": lambda rs: MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                max_iter=500, random_state=rs, early_stopping=True,
            ),
        }
        if model_name not in builders:
            raise ValueError(f"'{model_name}' no existe")
        self.model = builders[model_name](random_state)

    def train(self, X_train, y_train, class_names=None):
        self.class_names = class_names
        self.model.fit(X_train, y_train)
        self._trained = True
        return self

    def predict(self, X):
        if not self._trained:
            raise RuntimeError("Entrena el modelo primero")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self._trained:
            raise RuntimeError("Entrena el modelo primero")
        return self.model.predict_proba(X)

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        return {"mean_f1": scores.mean(), "std_f1": scores.std(), "scores": scores}

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)

    def save(self, path="outputs/models/crude_classifier.pkl"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "model_name": self.model_name, "class_names": self.class_names}, p)
        return p

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls(data["model_name"])
        instance.model = data["model"]
        instance.class_names = data["class_names"]
        instance._trained = True
        return instance
