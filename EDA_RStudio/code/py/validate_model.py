"""
Script de Validación de Modelos ML para IBEX35
==============================================

Soporta:
- Clasificadores (LightGBM, XGBoost, RandomForest, etc.)
- Regresores (LightGBM, XGBoost, RandomForest, etc.)

Uso:
    python validate_model.py --model modelo.joblib --data validation.csv --task classification
    python validate_model.py --model modelo.joblib --data validation.csv --task regression

Autor: TFM Master Data Science
Fecha: 2025
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    # Clasificación
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    # Regresión
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import warnings

warnings.filterwarnings("ignore")


# Features esperadas por el modelo
EXPECTED_FEATURES = [
    "sp500_return",
    "dax_return",
    "dax_momentum",
    "volatility_lag5",
    "returns_acceleration",
    "sp500_vol20",
    "vix_return",
    "volatility_5",
    "ftse100_momentum",
    "ftse100_return",
    "price_momentum",
    "roc_5",
    "spread_eu_us",
    "sp500_momentum",
    "oil_gold_ratio",
    "dax_return_lag1",
    "range_20",
    "risk_on_score",
    "high_max_20",
    "volume_sma20",
    "oil_vol20",
    "obv_norm",
    "ftse100_return_lag1",
    "range_5",
    "vix_zscore",
    "eurodollar_momentum",
    "gold_vol20",
    "returns_mean_5",
    "returns_lag5",
    "gold_return",
    "eurodollar_level",
    "vol_spread_target_sp500",
    "oil_return",
    "euribor_3m",
    "volatility_velocity",
    "spread_target_dax",
    "oil_momentum",
    "slowD",
    "vix_regime_Low",
    "vix_regime_Normal",
    "vix_regime_Elevated",
    "vix_regime_High",
]


def load_model(model_path):
    """Carga el modelo desde archivo joblib."""
    print(f"Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)
    model_type = type(model).__name__
    print(f"Modelo cargado: {model_type}")
    
    # Detectar si es clasificador o regresor
    is_classifier = hasattr(model, 'predict_proba') or 'Classifier' in model_type
    is_regressor = 'Regressor' in model_type or (hasattr(model, 'predict') and not is_classifier)
    
    print(f"Tipo detectado: {'Clasificador' if is_classifier else 'Regresor'}")
    
    return model, is_classifier


def load_validation_data(data_path):
    """Carga los datos de validación desde CSV o RDS."""
    print(f"Cargando datos desde: {data_path}")

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".rds"):
        import pyreadr
        result = pyreadr.read_r(data_path)
        df = list(result.values())[0]
    else:
        raise ValueError("Formato no soportado. Usa .csv o .rds")

    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def verify_features(df, expected_features):
    """Verifica que todas las features esperadas estén presentes."""
    print("\n=== VERIFICACIÓN DE FEATURES ===")

    available = set(df.columns)
    expected = set(expected_features)

    missing = expected - available
    extra = available - expected - {"date", "direction_next", "direction_next_5", 
                                     "direction_next_10", "direction_next_20",
                                     "returns_next", "returns_next_5", 
                                     "returns_next_10", "returns_next_20"}

    if missing:
        print(f"⚠️  Features FALTANTES ({len(missing)}): {missing}")
    else:
        print("✓ Todas las features esperadas están presentes")

    if extra:
        print(f"ℹ️  Features EXTRA (no usadas por modelo): {extra}")

    return len(missing) == 0


def prepare_data(df, target_col="direction_next"):
    """Prepara X e y para predicción."""
    print(f"\n=== PREPARACIÓN DE DATOS ===")
    print(f"Target: {target_col}")

    # Filtrar filas con target válido
    df_valid = df.dropna(subset=[target_col])
    print(f"Observaciones con target válido: {len(df_valid)} de {len(df)}")

    # Extraer features
    X = df_valid[EXPECTED_FEATURES].copy()
    y = df_valid[target_col].copy()

    # Verificar NAs en features
    na_counts = X.isna().sum()
    if na_counts.sum() > 0:
        print(f"⚠️  Features con NAs:")
        for col, count in na_counts[na_counts > 0].items():
            print(f"   - {col}: {count} NAs")
        
        print("Eliminando filas con NAs en features...")
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        print(f"Observaciones finales: {len(X)}")

    # Fechas para análisis temporal
    dates = df_valid.loc[X.index, "date"] if "date" in df_valid.columns else None

    return X, y, dates


def evaluate_classifier(model, X, y, dates=None):
    """Evalúa un modelo de clasificación."""
    print("\n=== EVALUACIÓN DEL MODELO (CLASIFICACIÓN) ===")

    # Predicciones
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X)
        y_pred_proba = y_pred  # Sin probabilidades

    y = y.astype(int)

    # Métricas
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    try:
        auc_roc = roc_auc_score(y, y_pred_proba)
    except ValueError:
        auc_roc = None

    print(f"\n--- Métricas de Clasificación ---")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if auc_roc:
        print(f"AUC-ROC:   {auc_roc:.4f}")

    # Baseline
    baseline = y.mean()
    print(f"\n--- Comparación con Baseline ---")
    print(f"Baseline (% clase 1): {baseline:.4f} ({baseline*100:.2f}%)")
    print(f"Mejora sobre random:  {(accuracy - 0.5)*100:.2f} pp")
    print(f"Mejora sobre baseline: {(accuracy - max(baseline, 1-baseline))*100:.2f} pp")

    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)
    print(f"\n--- Matriz de Confusión ---")
    print(f"                 Predicho")
    print(f"              Neg    Pos")
    print(f"Real Neg    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Real Pos    {cm[1,0]:5d}  {cm[1,1]:5d}")

    # Distribución
    print(f"\n--- Distribución de Predicciones ---")
    print(f"Predicciones positivas: {y_pred.sum()} ({y_pred.mean()*100:.1f}%)")
    print(f"Predicciones negativas: {(1-y_pred).sum()} ({(1-y_pred).mean()*100:.1f}%)")

    results = {
        "task": "classification",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "confusion_matrix": cm,
    }

    # Análisis temporal
    if dates is not None:
        results["predictions_df"] = _temporal_analysis(dates, y, y_pred, y_pred_proba)

    return results


def evaluate_regressor(model, X, y, dates=None):
    """Evalúa un modelo de regresión."""
    print("\n=== EVALUACIÓN DEL MODELO (REGRESIÓN) ===")

    # Predicciones
    y_pred = model.predict(X)

    # Métricas de regresión
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"\n--- Métricas de Regresión ---")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"R²:    {r2:.4f}")
    print(f"MSE:   {mse:.4f}")

    # Convertir a dirección para métricas de clasificación
    y_direction = (y > 0).astype(int)
    y_pred_direction = (y_pred > 0).astype(int)

    accuracy_dir = accuracy_score(y_direction, y_pred_direction)
    
    print(f"\n--- Métricas Direccionales (signo del retorno) ---")
    print(f"Accuracy direccional: {accuracy_dir:.4f} ({accuracy_dir*100:.2f}%)")
    
    # Baseline direccional
    baseline_dir = y_direction.mean()
    print(f"Baseline (% positivos): {baseline_dir:.4f} ({baseline_dir*100:.2f}%)")
    print(f"Mejora sobre random:  {(accuracy_dir - 0.5)*100:.2f} pp")

    # Matriz de confusión direccional
    cm = confusion_matrix(y_direction, y_pred_direction)
    print(f"\n--- Matriz de Confusión (Dirección) ---")
    print(f"                 Predicho")
    print(f"              Neg    Pos")
    print(f"Real Neg    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Real Pos    {cm[1,0]:5d}  {cm[1,1]:5d}")

    # Estadísticas de predicciones
    print(f"\n--- Estadísticas de Predicciones ---")
    print(f"Media real:      {y.mean():.4f}")
    print(f"Media predicha:  {y_pred.mean():.4f}")
    print(f"Std real:        {y.std():.4f}")
    print(f"Std predicha:    {y_pred.std():.4f}")
    print(f"Correlación:     {np.corrcoef(y, y_pred)[0,1]:.4f}")

    results = {
        "task": "regression",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mse": mse,
        "accuracy_directional": accuracy_dir,
        "confusion_matrix_directional": cm,
    }

    # Análisis temporal
    if dates is not None:
        results["predictions_df"] = _temporal_analysis_regression(dates, y, y_pred)

    return results


def _temporal_analysis(dates, y, y_pred, y_pred_proba):
    """Análisis temporal para clasificación."""
    print(f"\n--- Análisis Temporal ---")
    
    results_df = pd.DataFrame({
        "date": dates.values,
        "y_real": y.values,
        "y_pred": y_pred,
        "y_proba": y_pred_proba,
        "correct": (y.values == y_pred).astype(int)
    })
    
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df["week"] = results_df["date"].dt.isocalendar().week
    
    weekly = results_df.groupby("week").agg({
        "correct": ["sum", "count", "mean"]
    }).round(4)
    weekly.columns = ["aciertos", "total", "accuracy"]
    
    print("\nAccuracy por semana:")
    print(weekly.to_string())
    
    return results_df


def _temporal_analysis_regression(dates, y, y_pred):
    """Análisis temporal para regresión."""
    print(f"\n--- Análisis Temporal ---")
    
    y_direction = (y > 0).astype(int)
    y_pred_direction = (y_pred > 0).astype(int)
    
    results_df = pd.DataFrame({
        "date": dates.values,
        "y_real": y.values,
        "y_pred": y_pred,
        "direction_real": y_direction.values,
        "direction_pred": y_pred_direction,
        "correct": (y_direction.values == y_pred_direction).astype(int),
        "error": y.values - y_pred
    })
    
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df["week"] = results_df["date"].dt.isocalendar().week
    
    weekly = results_df.groupby("week").agg({
        "correct": ["sum", "count", "mean"],
        "error": ["mean", "std"]
    }).round(4)
    weekly.columns = ["aciertos_dir", "total", "accuracy_dir", "error_medio", "error_std"]
    
    print("\nMétricas por semana:")
    print(weekly.to_string())
    
    return results_df


def evaluate_model(model, X, y, dates=None, force_task=None):
    """
    Evalúa el modelo detectando automáticamente si es clasificador o regresor.
    
    Args:
        model: Modelo cargado (o tupla (model, is_classifier) de load_model)
        X: Features
        y: Target
        dates: Fechas (opcional)
        force_task: Forzar 'classification' o 'regression' (opcional)
    """
    # Si model es tupla de load_model
    if isinstance(model, tuple):
        model, is_classifier = model
    else:
        is_classifier = hasattr(model, 'predict_proba') or 'Classifier' in type(model).__name__
    
    # Permitir forzar el tipo de tarea
    if force_task == 'classification':
        is_classifier = True
    elif force_task == 'regression':
        is_classifier = False
    
    if is_classifier:
        return evaluate_classifier(model, X, y, dates)
    else:
        return evaluate_regressor(model, X, y, dates)


def main():
    parser = argparse.ArgumentParser(description="Validar modelo ML")
    parser.add_argument("--model", required=True, help="Ruta al modelo .joblib")
    parser.add_argument("--data", required=True, help="Ruta a datos de validación")
    parser.add_argument("--target", default="direction_next", help="Columna target")
    parser.add_argument("--task", default=None, choices=['classification', 'regression'],
                        help="Forzar tipo de tarea")
    parser.add_argument("--output", default=None, help="Guardar resultados en CSV")

    args = parser.parse_args()

    # Cargar
    model, is_classifier = load_model(args.model)
    df = load_validation_data(args.data)

    # Verificar
    verify_features(df, EXPECTED_FEATURES)

    # Preparar
    X, y, dates = prepare_data(df, args.target)

    if len(X) == 0:
        print("ERROR: No hay datos válidos para evaluar.")
        return

    # Evaluar
    results = evaluate_model(model, X, y, dates, force_task=args.task)

    # Guardar
    if args.output and "predictions_df" in results:
        results["predictions_df"].to_csv(args.output, index=False)
        print(f"\nResultados guardados en: {args.output}")

    print("\n=== VALIDACIÓN COMPLETADA ===")


if __name__ == "__main__":
    main()
