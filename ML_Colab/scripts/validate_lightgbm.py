#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 20:07:50 2025

@author: santi
"""
"""
Script de Validación del Modelo LightGBM para IBEX35
=====================================================

Este script carga un modelo LightGBM entrenado y lo valida con nuevos datos.

Uso:
    python validate_lightgbm.py --model modelo.joblib --data validation_scaled.csv

Autor: TFM Master Data Science
Fecha: 2025
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import warnings

warnings.filterwarnings("ignore")


# Features esperadas por el modelo (extraídas del archivo .txt)
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
    """Carga el modelo LightGBM desde archivo joblib."""
    print(f"Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)
    print(f"Modelo cargado: {type(model).__name__}")
    return model


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

    # Filtrar filas con target válido
    df_valid = df.dropna(subset=[target_col])
    print(f"Observaciones con target válido: {len(df_valid)} de {len(df)}")

    # Extraer features
    X = df_valid[EXPECTED_FEATURES].copy()
    y = df_valid[target_col].astype(int)

    # Verificar NAs en features
    na_counts = X.isna().sum()
    if na_counts.sum() > 0:
        print(f"⚠️  Features con NAs:")
        for col, count in na_counts[na_counts > 0].items():
            print(f"   - {col}: {count} NAs")
        
        # Opción: eliminar filas con NAs o imputar
        print("Eliminando filas con NAs en features...")
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        print(f"Observaciones finales: {len(X)}")

    # Fechas para análisis temporal
    dates = df_valid.loc[X.index, "date"] if "date" in df_valid.columns else None

    return X, y, dates


def evaluate_model(model, X, y, dates=None):
    """Genera predicciones y calcula métricas."""
    print("\n=== EVALUACIÓN DEL MODELO ===")

    # Predicciones
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Métricas básicas
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

    # Distribución de predicciones
    print(f"\n--- Distribución de Predicciones ---")
    print(f"Predicciones positivas: {y_pred.sum()} ({y_pred.mean()*100:.1f}%)")
    print(f"Predicciones negativas: {(1-y_pred).sum()} ({(1-y_pred).mean()*100:.1f}%)")

    # Análisis temporal si hay fechas
    if dates is not None:
        print(f"\n--- Análisis Temporal ---")
        results_df = pd.DataFrame({
            "date": dates.values,
            "y_real": y.values,
            "y_pred": y_pred,
            "y_proba": y_pred_proba,
            "correct": (y.values == y_pred).astype(int)
        })
        
        # Accuracy por semana
        results_df["date"] = pd.to_datetime(results_df["date"])
        results_df["week"] = results_df["date"].dt.isocalendar().week
        
        weekly = results_df.groupby("week").agg({
            "correct": ["sum", "count", "mean"]
        }).round(4)
        weekly.columns = ["aciertos", "total", "accuracy"]
        
        print("\nAccuracy por semana:")
        print(weekly.to_string())

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
            "confusion_matrix": cm,
            "predictions_df": results_df,
        }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser(description="Validar modelo LightGBM")
    parser.add_argument("--model", required=True, help="Ruta al modelo .joblib")
    parser.add_argument("--data", required=True, help="Ruta a datos de validación (.csv o .rds)")
    parser.add_argument("--target", default="direction_next", help="Columna target")
    parser.add_argument("--output", default=None, help="Guardar resultados en CSV")

    args = parser.parse_args()

    # Cargar modelo y datos
    model = load_model(args.model)
    df = load_validation_data(args.data)

    # Verificar features
    if not verify_features(df, EXPECTED_FEATURES):
        print("\n⚠️  ADVERTENCIA: Faltan features. Los resultados pueden ser incorrectos.")

    # Preparar datos
    X, y, dates = prepare_data(df, args.target)

    if len(X) == 0:
        print("ERROR: No hay datos válidos para evaluar.")
        return

    # Evaluar
    results = evaluate_model(model, X, y, dates)

    # Guardar resultados
    if args.output and "predictions_df" in results:
        results["predictions_df"].to_csv(args.output, index=False)
        print(f"\nResultados guardados en: {args.output}")

    print("\n=== VALIDACIÓN COMPLETADA ===")


if __name__ == "__main__":
    main()