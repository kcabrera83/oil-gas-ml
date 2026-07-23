"""Modelo de clasificación para tipo y calidad de crudo."""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


class CrudeClassifier:
    MODELS = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
        "svm": lambda: SVC(
            kernel="rbf", C=10, gamma="scale", probability=True, random_state=42,
        ),
        "knn": lambda: KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1,
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            max_iter=500, random_state=42, early_stopping=True,
        ),
    }

    def __init__(self, model_name="random_forest"):
        if model_name not in self.MODELS:
            raise ValueError(f"Modelo '{model_name}' no disponible. Opciones: {list(self.MODELS.keys())}")
        self.model_name = model_name
        self.model = self.MODELS[model_name]()
        self._trained = False
        self.class_names = None

    def train(self, X_train, y_train, class_names=None):
        self.class_names = class_names
        self.model.fit(X_train, y_train)
        self._trained = True
        return self

    def predict(self, X):
        if not self._trained:
            raise RuntimeError("Modelo no entrenado.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self._trained:
            raise RuntimeError("Modelo no entrenado.")
        return self.model.predict_proba(X)

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        return {"mean_f1": scores.mean(), "std_f1": scores.std(), "scores": scores}

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)

    def get_feature_importance(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None

    def save(self, path="outputs/models/crude_classifier.pkl"):
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "model_name": self.model_name, "class_names": self.class_names}, filepath)
        return filepath

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls(model_name=data["model_name"])
        instance.model = data["model"]
        instance.class_names = data["class_names"]
        instance._trained = True
        return instance

    @staticmethod
    def train_all(X_train, y_train, class_names=None):
        results = {}
        for name in CrudeClassifier.MODELS:
            print(f"  Entrenando {name}...")
            clf = CrudeClassifier(model_name=name)
            clf.train(X_train, y_train, class_names)
            results[name] = clf
        return results
