import numpy as np
import pandas as pd
from pathlib import Path


class CrudeDataGenerator:
    # rangos tipicos segun tipo de crudo (literatura SPE)
    CRUDE_TYPES = {
        "liviano": {"api_range": (35, 55), "visc_range": (1, 10), "sulf_range": (0.1, 0.5)},
        "mediano": {"api_range": (25, 35), "visc_range": (10, 100), "sulf_range": (0.5, 1.5)},
        "pesado": {"api_range": (10, 25), "visc_range": (100, 10000), "sulf_range": (1.5, 3.5)},
        "extra_pesado": {"api_range": (5, 10), "visc_range": (1000, 100000), "sulf_range": (3.0, 6.0)},
    }

    # thresholds para clasificar calidad
    QUALITY_CLASSES = {
        "premium": {"api_min": 35, "sulf_max": 0.5, "visc_max": 20},
        "estandar": {"api_min": 22, "sulf_max": 2.0, "visc_max": 500},
        "inferior": {"api_min": 10, "sulf_max": 4.0, "visc_max": 5000},
        "deshidratado": {"api_min": 5, "sulf_max": 6.0, "visc_max": 100000},
    }

    def __init__(self, seed=2024):
        self.rng = np.random.default_rng(seed)

    def _gen_param(self, low, high, n, log_scale=False):
        if log_scale:
            return np.exp(self.rng.uniform(np.log(low), np.log(high), n))
        return self.rng.uniform(low, high, n)

    def generate(self, n_samples=2000, include_quality=True):
        n = n_samples
        records = []

        for _ in range(n):
            crude_type = self.rng.choice(
                list(self.CRUDE_TYPES.keys()),
                p=[0.30, 0.35, 0.25, 0.10],
            )
            spec = self.CRUDE_TYPES[crude_type]

            api = self.rng.uniform(*spec["api_range"])
            viscosity = self._gen_param(*spec["visc_range"], 1, log_scale=True)[0]
            sulfur = self.rng.uniform(*spec["sulf_range"])

            density = 141.5 / (api + 131.5) * 1000

            water = self.rng.beta(2, 8) * 30
            asphaltene = self.rng.beta(2, 10) * 15 if crude_type in ("pesado", "extra_pesado") else self.rng.beta(1, 15) * 8
            tan = self.rng.gamma(2, 0.3) if crude_type in ("pesado", "extra_pesado") else self.rng.gamma(1.5, 0.15)
            pour_point = self.rng.normal(-10 if api > 30 else 20, 15)
            flash_point = self.rng.normal(30 if api > 35 else 70, 20)
            rvp = self._gen_param(15, 100 if api > 35 else 40, 1)
            salt = self._gen_param(1, 50, 1, log_scale=True)
            metals = self._gen_param(0.5, 200 if crude_type in ("pesado", "extra_pesado") else 30, 1, log_scale=True)
            nitrogen = self.rng.uniform(0.01, 0.3) if crude_type in ("pesado", "extra_pesado") else self.rng.uniform(0.005, 0.15)
            carbon_residue = self.rng.uniform(0.5, 12) if crude_type in ("pesado", "extra_pesado") else self.rng.uniform(0.1, 3)

            records.append({
                "api_gravity": round(api, 2),
                "viscosity_cp": round(viscosity, 2),
                "sulfur_content_pct": round(sulfur, 3),
                "water_content_pct": round(water, 2),
                "asphaltene_content_pct": round(asphaltene, 3),
                "total_acid_number": round(tan, 3),
                "pour_point_c": round(pour_point, 1),
                "flash_point_c": round(max(flash_point, -20), 1),
                "density_kg_m3": round(density, 2),
                "rvp_kpa": round(rvp[0], 2),
                "salt_content_ptb": round(salt[0], 2),
                "metal_content_ppm": round(metals[0], 2),
                "nitrogen_content_pct": round(nitrogen, 4),
                "carbon_residue_pct": round(carbon_residue, 3),
                "crude_type": crude_type,
            })

        df = pd.DataFrame(records)

        if include_quality:
            df["quality_class"] = df.apply(self._assign_quality, axis=1)
            df["yield_recovery_pct"] = self._compute_yield(df)
            df["market_value_usd_bbl"] = self._compute_market_value(df)

        return df

    def _assign_quality(self, row):
        api = row["api_gravity"]
        sulfur = row["sulfur_content_pct"]
        visc = row["viscosity_cp"]

        if api >= 35 and sulfur <= 0.5 and visc <= 20:
            return "premium"
        elif api >= 22 and sulfur <= 2.0 and visc <= 500:
            return "estandar"
        elif api >= 10 and sulfur <= 4.0 and visc <= 5000:
            return "inferior"
        else:
            return "deshidratado"

    def _compute_yield(self, df):
        base = 85 - 0.3 * df["sulfur_content_pct"] - 0.1 * df["asphaltene_content_pct"]
        noise = self.rng.normal(0, 2, len(df))
        return np.clip(base + noise, 20, 98).round(2)

    def _compute_market_value(self, df):
        base_price = df["api_gravity"] * 1.5
        sulfur_penalty = df["sulfur_content_pct"] * 8
        visc_penalty = np.log1p(df["viscosity_cp"]) * 2
        noise = self.rng.normal(0, 3, len(df))
        return np.clip(base_price - sulfur_penalty - visc_penalty + noise + 20, 5, 120).round(2)

    def save(self, df, path="data/crude_dataset.csv"):
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        return filepath


if __name__ == "__main__":
    gen = CrudeDataGenerator(seed=2024)
    df = gen.generate(n_samples=3000)
    path = gen.save(df)
    print(f"Listo: {len(df)} muestras en {path}")
