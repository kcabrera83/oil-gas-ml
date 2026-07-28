import sys
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

    gen = CrudeDataGenerator(seed=2024)
    df = gen.generate(n_samples=3000)
    viz = CrudeVisualizer(output_dir="outputs/plots")

    print("\n[1] ANÁLISIS EXPLORATORIO")
    print("-" * 40)
    print(f"\n  Estadísticas descriptivas:")
    print(df.describe().round(2).to_string())

    print(f"\n  Distribución por tipo:")
    for ctype, count in df["crude_type"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {ctype:15s}: {count:5d} ({pct:.1f}%)")

    print(f"\n  Distribución por calidad:")
    for qclass, count in df["quality_class"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {qclass:15s}: {count:5d} ({pct:.1f}%)")

    print("\n[2] ENTRENAMIENTO Y VALIDACIÓN CRUZADA")
    print("-" * 40)

    preprocessor = CrudePreprocessor(scaler_type="robust", test_size=0.2)
    X_cls_train, X_cls_test, y_cls_train, y_cls_test, le = preprocessor.prepare_classification(df.copy())
    class_names = le.classes_

    print("\n  Clasificación (validación cruzada 5-fold):")
    for name in CrudeClassifier.MODELS:
        clf = CrudeClassifier(model_name=name)
        cv_results = clf.cross_validate(X_cls_train, y_cls_train, cv=5)
        print(f"    {name:25s} | F1: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")

    preprocessor_reg = CrudePreprocessor(scaler_type="robust", test_size=0.2)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = preprocessor_reg.prepare_regression(df.copy())

    print("\n  Regresión (validación cruzada 5-fold):")
    for name in CrudeRegressor.MODELS:
        reg = CrudeRegressor(model_name=name)
        cv_results = reg.cross_validate(X_reg_train, y_reg_train, cv=5)
        print(f"    {name:25s} | R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")

    print("\n[3] EVALUACIÓN EN TEST SET")
    print("-" * 40)

    evaluator = ModelEvaluator()

    print("\n  Clasificadores:")
    best_f1 = 0
    best_cls_name = ""
    for name in CrudeClassifier.MODELS:
        clf = CrudeClassifier(model_name=name)
        clf.train(X_cls_train, y_cls_train, class_names)
        y_pred = clf.predict(X_cls_test)
        metrics = evaluator.evaluate_classification(y_cls_test, y_pred, class_names, name)
        print(f"    {name:25s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}")
        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]
            best_cls_name = name

    print("\n  Regresores:")
    best_r2 = -999
    best_reg_name = ""
    for name in CrudeRegressor.MODELS:
        reg = CrudeRegressor(model_name=name)
        reg.train(X_reg_train, y_reg_train)
        y_pred = reg.predict(X_reg_test)
        metrics = evaluator.evaluate_regression(y_reg_test, y_pred, name)
        print(f"    {name:25s} | R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f}")
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_reg_name = name

    print(f"\n  MEJORES MODELOS:")
    print(f"    Clasificador: {best_cls_name} (F1={best_f1:.4f})")
    print(f"    Regresor:     {best_reg_name} (R²={best_r2:.4f})")

    print("\n[4] ANÁLISIS DE IMPORTANCIA")
    print("-" * 40)
    best_clf = CrudeClassifier(model_name=best_cls_name)
    best_clf.train(X_cls_train, y_cls_train, class_names)
    importance = best_clf.get_feature_importance()
    feature_names = preprocessor.get_feature_names()

    if importance is not None:
        sorted_idx = importance.argsort()[::-1]
        print("\n  Top 10 características más importantes:")
        for rank, idx in enumerate(sorted_idx[:10], 1):
            print(f"    {rank:2d}. {feature_names[idx]:30s}: {importance[idx]:.4f}")

    print("\n[5] GUARDANDO RESULTADOS")
    print("-" * 40)

    best_clf.save("outputs/models/crude_classifier_best.pkl")
    best_reg = CrudeRegressor(model_name=best_reg_name)
    best_reg.train(X_reg_train, y_reg_train)
    best_reg.save("outputs/models/crude_regressor_best.pkl")

    preprocessor_multi = CrudePreprocessor(scaler_type="robust", test_size=0.2)
    X_m_train, X_m_test, y_m_train, y_m_test, targets = preprocessor_multi.prepare_multi_target(df.copy())
    qp = QualityPredictor()
    qp.train(X_m_train, y_m_train, target_names=targets)
    qp.save("outputs/models/quality_predictor.pkl")

    comparison = evaluator.compare_models()
    cls_comparison = {k: v for k, v in comparison.items() if evaluator.results[k]["type"] == "classification"}
    reg_comparison = {k: v for k, v in comparison.items() if evaluator.results[k]["type"] == "regression"}
    if cls_comparison:
        viz.plot_model_comparison(cls_comparison, metric_filter={"accuracy", "f1_weighted"})
    if reg_comparison:
        viz.plot_model_comparison(reg_comparison, metric_filter={"R2", "RMSE"})

    pass
    pass

    print("\n" + "=" * 70)
    pass


if __name__ == "__main__":
    main()
