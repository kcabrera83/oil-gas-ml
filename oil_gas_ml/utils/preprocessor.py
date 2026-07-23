"""Preprocesamiento de datos de crudo petrolífero."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


NUMERIC_FEATURES = [
    "api_gravity", "viscosity_cp", "sulfur_content_pct", "water_content_pct",
    "asphaltene_content_pct", "total_acid_number", "pour_point_c", "flash_point_c",
    "density_kg_m3", "rvp_kpa", "salt_content_ptb", "metal_content_ppm",
    "nitrogen_content_pct", "carbon_residue_pct",
]

DERIVED_FEATURES = [
    "api_viscosity_ratio", "sulfur_api_ratio", "density_api_ratio",
    "pour_flash_diff", "heavy_component_index",
]


class CrudePreprocessor:
    def __init__(self, scaler_type="robust", test_size=0.2, random_state=42):
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
        self._fitted = False

    def _build_derived_features(self, df):
        df = df.copy()
        eps = 1e-6
        df["api_viscosity_ratio"] = df["api_gravity"] / (df["viscosity_cp"] + eps)
        df["sulfur_api_ratio"] = df["sulfur_content_pct"] / (df["api_gravity"] + eps)
        df["density_api_ratio"] = df["density_kg_m3"] / (df["api_gravity"] + eps)
        df["pour_flash_diff"] = df["flash_point_c"] - df["pour_point_c"]
        df["heavy_component_index"] = (
            df["asphaltene_content_pct"] + df["carbon_residue_pct"] + df["metal_content_ppm"] / 100
        )
        return df

    def fit(self, df):
        df = self._build_derived_features(df)
        all_features = NUMERIC_FEATURES + DERIVED_FEATURES

        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(df[all_features])

        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(self.imputer.transform(df[all_features]))

        for col in df.select_dtypes(include="object").columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].astype(str))

        self._fitted = True
        return self

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        df = self._build_derived_features(df)
        all_features = NUMERIC_FEATURES + DERIVED_FEATURES
        X = self.imputer.transform(df[all_features])
        X = self.scaler.transform(X)
        return pd.DataFrame(X, columns=all_features, index=df.index)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def prepare_classification(self, df, target_col="quality_class"):
        df = df.copy()
        X = self.fit_transform(df)
        y = self.label_encoders[target_col].fit_transform(df[target_col])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y,
        )
        return X_train, X_test, y_train, y_test, self.label_encoders[target_col]

    def prepare_regression(self, df, target_col="market_value_usd_bbl"):
        X = self.fit_transform(df)
        y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
        )
        return X_train, X_test, y_train, y_test

    def prepare_multi_target(self, df):
        X = self.fit_transform(df)
        targets = ["market_value_usd_bbl", "yield_recovery_pct"]
        y = df[targets].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
        )
        return X_train, X_test, y_train, y_test, targets

    def get_feature_names(self):
        return NUMERIC_FEATURES + DERIVED_FEATURES
