import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import seaborn as sns


COLORS = {
    "premium": "#2ecc71",
    "estandar": "#3498db",
    "inferior": "#e67e22",
    "deshidratado": "#e74c3c",
}

TYPE_COLORS = {
    "liviano": "#27ae60",
    "mediano": "#2980b9",
    "pesado": "#d35400",
    "extra_pesado": "#c0392b",
}


class CrudeVisualizer:
    def __init__(self, output_dir="outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    def plot_data_distribution(self, df):
        fig, axes = plt.subplots(3, 5, figsize=(22, 14))
        axes = axes.flatten()
        features = [c for c in df.columns if df[c].dtype in ("float64", "int64") and c != "yield_recovery_pct"]

        for i, col in enumerate(features[:15]):
            ax = axes[i]
            for ctype, color in TYPE_COLORS.items():
                subset = df[df["crude_type"] == ctype][col]
                if len(subset) > 0:
                    ax.hist(subset, bins=30, alpha=0.5, label=ctype, color=color, density=True)
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.legend(fontsize=7)

        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Distribución de Propiedades del Crudo por Tipo", fontsize=16, fontweight="bold", y=1.01)
        fig.tight_layout()
        return self._save(fig, "01_data_distribution")

    def plot_correlation_matrix(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            annot_kws={"size": 7}, linewidths=0.5,
        )
        ax.set_title("Matriz de Correlación - Propiedades del Crudo", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, "02_correlation_matrix")

    def plot_scatter_matrix(self, df, features=None, hue="quality_class"):
        if features is None:
            features = ["api_gravity", "sulfur_content_pct", "viscosity_cp", "density_kg_m3"]
        subset = df[features + [hue]].dropna()
        fig = sns.pairplot(subset, hue=hue, palette=COLORS, diag_kind="kde", plot_kws={"alpha": 0.5, "s": 15})
        fig.fig.suptitle("Matriz de Dispersión por Clase de Calidad", y=1.02, fontsize=14, fontweight="bold")
        return self._save(fig.fig, "03_scatter_matrix")

    def plot_feature_importance(self, importances, feature_names, title="Importancia de Características"):
        sorted_idx = np.argsort(importances)[-15:]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="#3498db", edgecolor="#2c3e50")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel("Importancia", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "04_feature_importance")

    def plot_model_comparison(self, results: dict, metric_filter=None):
        all_metrics = set()
        for m in results.values():
            all_metrics.update(m.keys())
        if metric_filter:
            all_metrics = all_metrics.intersection(metric_filter)
        metrics = sorted(all_metrics)
        models = list(results.keys())
        n_metrics = len(metrics)
        if n_metrics == 0:
            return None

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        bar_colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

        for i, metric in enumerate(metrics):
            values = []
            valid_models = []
            for m in models:
                if metric in results[m]:
                    values.append(results[m][metric])
                    valid_models.append(m)
            if not values:
                continue
            bars = axes[i].barh(valid_models, values, color=bar_colors[:len(valid_models)], edgecolor="#2c3e50")
            axes[i].set_title(metric, fontsize=11, fontweight="bold")
            axes[i].grid(axis="x", alpha=0.3)
            for bar, val in zip(bars, values):
                axes[i].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                             f"{val:.4f}", va="center", fontsize=9)

        fig.suptitle("Comparación de Modelos", fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        return self._save(fig, "05_model_comparison")

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title("Matriz de Confusión (conteo)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Real")
        axes[0].set_xlabel("Predicho")

        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="RdYlGn", ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
        axes[1].set_title("Matriz de Confusión (normalizada)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Real")
        axes[1].set_xlabel("Predicho")

        fig.tight_layout()
        return self._save(fig, "06_confusion_matrix")

    def plot_regression_results(self, y_true, y_pred, title="Predicción vs Real"):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        axes[0].scatter(y_true, y_pred, alpha=0.4, s=15, c="#3498db", edgecolors="#2c3e50", linewidth=0.3)
        lims = [min(y_true.min(), y_pred.min()) * 0.9, max(y_true.max(), y_pred.max()) * 1.1]
        axes[0].plot(lims, lims, "--", color="#e74c3c", linewidth=2, label="Perfecta predicción")
        axes[0].set_xlabel("Valor Real")
        axes[0].set_ylabel("Valor Predicho")
        axes[0].set_title("Dispersión", fontsize=11, fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.4, s=15, c="#e67e22", edgecolors="#2c3e50", linewidth=0.3)
        axes[1].axhline(y=0, color="#e74c3c", linestyle="--", linewidth=2)
        axes[1].set_xlabel("Valor Predicho")
        axes[1].set_ylabel("Residuo")
        axes[1].set_title("Residuos", fontsize=11, fontweight="bold")
        axes[1].grid(alpha=0.3)

        axes[2].hist(residuals, bins=40, color="#2ecc71", edgecolor="#2c3e50", alpha=0.7)
        axes[2].axvline(x=0, color="#e74c3c", linestyle="--", linewidth=2)
        axes[2].set_xlabel("Residuo")
        axes[2].set_ylabel("Frecuencia")
        axes[2].set_title("Distribución de Residuos", fontsize=11, fontweight="bold")
        axes[2].grid(alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return self._save(fig, "07_regression_results")

    def plot_crude_profile(self, row, sample_name="Muestra"):
        features = [
            ("API Gravity", row.get("api_gravity", 0), "°API", 0, 60),
            ("Viscosidad", row.get("viscosity_cp", 0), "cP", 0, None),
            ("Azufre", row.get("sulfur_content_pct", 0), "%", 0, 6),
            ("Asfaltenos", row.get("asphaltene_content_pct", 0), "%", 0, 15),
            ("TAN", row.get("total_acid_number", 0), "mg KOH/g", 0, 4),
            ("Punto de Fluidez", row.get("pour_point_c", 0), "°C", -40, 60),
            ("Punto de Inflamación", row.get("flash_point_c", 0), "°C", -20, 150),
            ("Densidad", row.get("density_kg_m3", 0), "kg/m³", 700, 1100),
            ("Contenido de Agua", row.get("water_content_pct", 0), "%", 0, 30),
            ("Sal", row.get("salt_content_ptb", 0), "PTB", 0, 50),
            ("Metales", row.get("metal_content_ppm", 0), "ppm", 0, 200),
            ("Nitrógeno", row.get("nitrogen_content_pct", 0), "%", 0, 0.5),
        ]

        n = len(features)
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()

        for i, (name, value, unit, vmin, vmax) in enumerate(features):
            ax = axes[i]
            vmax = vmax or value * 1.5 if value > 0 else 100
            ax.barh([0], [value], height=0.5, color="#3498db", edgecolor="#2c3e50")
            ax.set_xlim(vmin, vmax)
            ax.set_yticks([])
            ax.set_title(f"{name}\n{value:.2f} {unit}", fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

        quality = row.get("quality_class", "N/A")
        crude_type = row.get("crude_type", "N/A")
        fig.suptitle(f"Perfil del Crudo: {sample_name} | Tipo: {crude_type} | Calidad: {quality}",
                      fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return self._save(fig, "08_crude_profile")
