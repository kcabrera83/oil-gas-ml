import sys
import time
from pathlib import Path

import sys; sys.path.append(str(Path(__file__).resolve().parent))

from oil_gas_ml.data_generator import CrudeDataGenerator
from oil_gas_ml.utils.preprocessor import CrudePreprocessor
from oil_gas_ml.utils.visualizer import CrudeVisualizer
from oil_gas_ml.utils.metrics import ModelEvaluator
from oil_gas_ml.models.crude_classifier import CrudeClassifier
from oil_gas_ml.models.crude_regressor import CrudeRegressor
from oil_gas_ml.models.quality_predictor import QualityPredictor


def main():
    pass

    # 1. Generar dataset
    print("\n[1/6] Generando dataset sintético...")
    gen = CrudeDataGenerator(seed=2024)
    df = gen.generate(n_samples=3000)
    gen.save(df, path="data/crude_dataset.csv")
    print(f"      Dataset: {len(df)} muestras, {df.shape[1]} columnas")
    print(f"      Tipos: {df['crude_type'].value_counts().to_dict()}")
    print(f"      Calidad: {df['quality_class'].value_counts().to_dict()}")

    # 2. Visualización exploratoria
    print("\n[2/6] Generando visualizaciones exploratorias...")
    viz = CrudeVisualizer(output_dir="outputs/plots")
    viz.plot_data_distribution(df)
    viz.plot_correlation_matrix(df)
    pass

    # 3. Preprocesamiento
    print("\n[3/6] Preprocesando datos...")
    preprocessor = CrudePreprocessor(scaler_type="robust", test_size=0.2)

    # Clasificación
    X_cls_train, X_cls_test, y_cls_train, y_cls_test, le_quality = preprocessor.prepare_classification(
        df.copy(), target_col="quality_class"
    )
    class_names = le_quality.classes_
    print(f"      Clasificación: {X_cls_train.shape[0]} train, {X_cls_test.shape[0]} test")
    print(f"      Clases: {list(class_names)}")

    # Regresión - valor de mercado
    preprocessor_reg = CrudePreprocessor(scaler_type="robust", test_size=0.2)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = preprocessor_reg.prepare_regression(
        df.copy(), target_col="market_value_usd_bbl"
    )
    print(f"      Regresión (valor): {X_reg_train.shape[0]} train, {X_reg_test.shape[0]} test")

    # 4. Entrenamiento de clasificadores
    print("\n[4/6] Entrenando clasificadores de calidad...")
    evaluator = ModelEvaluator()
    cls_results = {}
    cls_models = CrudeClassifier.train_all(X_cls_train, y_cls_train, class_names=class_names)

    for name, clf in cls_models.items():
        metrics = evaluator.evaluate_classification(y_cls_test, clf.predict(X_cls_test), class_names, name)
        cls_results[name] = metrics
        print(f"      {name:25s} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}")

    # 5. Entrenamiento de regresores
    print("\n[5/6] Entrenando regresores de valor de mercado...")
    reg_results = {}
    reg_models = CrudeRegressor.train_all(X_reg_train, y_reg_train, target_name="market_value_usd_bbl")

    for name, reg in reg_models.items():
        metrics = evaluator.evaluate_regression(y_reg_test, reg.predict(X_reg_test), name)
        reg_results[name] = metrics
        print(f"      {name:25s} | R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f}")

    # 6. Predictor multi-output
    print("\n[6/6] Entrenando predictor multi-output (valor + rendimiento)...")
    preprocessor_multi = CrudePreprocessor(scaler_type="robust", test_size=0.2)
    X_multi_train, X_multi_test, y_multi_train, y_multi_test, targets = preprocessor_multi.prepare_multi_target(df.copy())
    qp = QualityPredictor(base_model="gradient_boosting")
    qp.train(X_multi_train, y_multi_train, target_names=targets)
    multi_results = qp.evaluate(X_multi_test, y_multi_test)
    for target_name, metrics in multi_results.items():
        print(f"      {target_name:30s} | R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f}")

    # Guardar mejores modelos
    print("\n[GUARDANDO] Mejores modelos...")
    best_cls_name = max(cls_results, key=lambda k: cls_results[k]["f1_weighted"])
    cls_models[best_cls_name].save("outputs/models/crude_classifier_best.pkl")
    print(f"      Mejor clasificador: {best_cls_name}")

    best_reg_name = max(reg_results, key=lambda k: reg_results[k]["R2"])
    reg_models[best_reg_name].save("outputs/models/crude_regressor_best.pkl")
    print(f"      Mejor regresor: {best_reg_name}")

    qp.save("outputs/models/quality_predictor.pkl")
    pass

    # Visualización de resultados
    print("\n[VISUALIZACIONES] Generando gráficos de resultados...")
    cls_comparison = {k: v["metrics"] for k, v in evaluator.results.items() if v["type"] == "classification"}
    reg_comparison = {k: v["metrics"] for k, v in evaluator.results.items() if v["type"] == "regression"}
    if cls_comparison:
        viz.plot_model_comparison(cls_comparison, metric_filter={"accuracy", "f1_weighted", "precision_weighted"})
    if reg_comparison:
        viz.plot_model_comparison(reg_comparison, metric_filter={"R2", "RMSE", "MAE"})

    best_clf = cls_models[best_cls_name]
    y_pred_cls = best_clf.predict(X_cls_test)
    viz.plot_confusion_matrix(y_cls_test, y_pred_cls, class_names)

    best_reg = reg_models[best_reg_name]
    y_pred_reg = best_reg.predict(X_reg_test)
    viz.plot_regression_results(y_reg_test, y_pred_reg, title=f"Predicción de Valor de Mercado ({best_reg_name})")

    importance = best_clf.get_feature_importance()
    if importance is not None:
        feature_names = preprocessor.get_feature_names()
        viz.plot_feature_importance(importance, feature_names, f"Importancia de Características ({best_cls_name})")

    # Perfil de ejemplo
    sample = df.iloc[0]
    viz.plot_crude_profile(sample, sample_name="Muestra #0")

    print("\n" + "=" * 70)
    pass
    print("\n  Archivos generados:")
    pass
    pass
    pass


if __name__ == "__main__":
    main()
