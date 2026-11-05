#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funciones auxiliares para el proyecto IBEX35
@author: santi
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import os
import sys
import pickle
import json
from datetime import datetime
import warnings

# Data manipulation
import numpy as np
import pandas as pd

# File I/O
import pyreadr
import joblib

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Statistics
from scipy import stats
from scipy.stats import norm, chi2

# Machine Learning
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)

# Import config
from config import (
    ALL_TARGET_COLS,
    MODELS,
    PLOTS,
    CSV,
    RANDOM_STATE
)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_rds(file_path):
    """Carga archivo .rds desde Google Drive"""
    try:
        result = pyreadr.read_r(file_path)
        return result[None]
    except Exception as e:
        print(f"Error cargando {file_path}: {e}")
        return None

def prepare_data(df, target_col):
    """
    Prepara datos excluyendo TODOS los targets de las features

    Args:
        df: DataFrame completo
        target_col: Target específico a predecir

    Returns:
        X: Features (sin ningún target)
        y: Target específico
    """
# Excluir TODOS los targets + date
    exclude_cols = ALL_TARGET_COLS + ['date']

    # Features: todo excepto targets y date
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()

    # Target específico
    y = df[target_col].copy()

    # Verificación
    assert target_col not in X.columns, f" LEAK: {target_col} en features!"
    for other in ALL_TARGET_COLS:
        if other != target_col:
            assert other not in X.columns, f" LEAK: {other} en features!"

    print(f"✓ Features: {X.shape[1]} (targets excluidos: {ALL_TARGET_COLS})")

    return X, y

def create_sequences(X, y, lookback=20):
    """
    Crea secuencias para LSTM/GRU

    X: (n_samples, n_features)
    y: (n_samples,)

    Returns:
    X_seq: (n_sequences, lookback, n_features)
    y_seq: (n_sequences,)
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# NEW: Función para guardar modelos
def save_model(model, model_name, dataset_name, target_name, model_type='sklearn'):
    """
    Guarda modelo entrenado

    Args:
        model: Modelo entrenado
        model_name: Nombre del modelo (ej: 'XGBoost', 'GRU')
        dataset_name: Nombre del dataset (ej: 'financial_scaled')
        target_name: Nombre del target (ej: 'returns_next')
        model_type: 'sklearn', 'tensorflow', 'xgboost', 'lightgbm'
    """
    # Crear subdirectorio para este dataset/target
    save_dir = os.path.join(MODELS, dataset_name, target_name)
    os.makedirs(save_dir, exist_ok=True)

    # Nombre del archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace(" ", "_").replace("/", "_")
    safe_model_name = model_name.replace(" ", "_").replace("/", "_")
    basepath = os.path.join(save_dir, safe_model_name)


    try:
        if model_type == 'tensorflow':
            # TensorFlow/Keras: formato nativo .keras
            keras_path = basepath + ".keras"
            model.save(keras_path)
            print(f"  ✓ Modelo TensorFlow guardado: {keras_path}")
            return keras_path

        elif model_type == 'xgboost':
            # a) Mantener .joblib por compatibilidad
            joblib_path = basepath + ".joblib"
            joblib.dump(model, joblib_path)
            print(f"  ✓ Modelo XGBoost (.joblib) guardado: {joblib_path}")

            # b) Formato estable recomendado (Booster JSON)
            json_path = basepath + ".json"
            booster = model.get_booster()
            booster.save_model(json_path)
            print(f"  ✓ Modelo XGBoost (.json) guardado: {json_path}")

            # Devolvemos el formato “bueno” por defecto
            return json_path

        elif model_type == 'lightgbm':
            # a) Mantener .joblib por compatibilidad
            joblib_path = basepath + ".joblib"
            joblib.dump(model, joblib_path)
            print(f"  ✓ Modelo LightGBM (.joblib) guardado: {joblib_path}")

            # b) Formato nativo estable
            txt_path = basepath + ".txt"
            booster = getattr(model, "booster_", None)
            if booster is None:
                # por si el objeto es ya un Booster
                booster = model
            booster.save_model(txt_path)
            print(f"  ✓ Modelo LightGBM (.txt) guardado: {txt_path}")

            return txt_path

        else:
            # sklearn puro, ensembles, etc.
            joblib_path = basepath + ".joblib"
            joblib.dump(model, joblib_path)
            print(f"  ✓ Modelo sklearn guardado: {joblib_path}")
            return joblib_path

    except Exception as e:
        print(f"  ⚠ Error guardando modelo {model_name}: {e}")
        return None
print("✓ Funciones auxiliares definidas")


"""
Funciones para gestionar y visualizar el historial de entrenamiento de modelos deep learning
"""
def save_training_history(history, model_name, dataset_name, target_name,
                          save_path, save_format='pickle'):
    """
    Guarda el historial de entrenamiento de un modelo Keras.

    Parameters:
    -----------
    history : keras.callbacks.History
        Objeto History retornado por model.fit()
    model_name : str
        Nombre del modelo (e.g., 'GRU_medium')
    dataset_name : str
        Nombre del dataset (e.g., 'financial', 'sentiment')
    target_name : str
        Variable objetivo (e.g., 'return_next', 'direction_next')
    save_path : str
        Directorio donde guardar el archivo
    save_format : str, default='pickle'
        Formato: 'pickle', 'csv', o 'json'

    Returns:
    --------
    str : Ruta del archivo guardado
    """

    os.makedirs(save_path, exist_ok=True)

    filename = f"{dataset_name}_{target_name}_{model_name}_history"

    if save_format == 'pickle':
        filepath = os.path.join(save_path, f"{filename}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(history.history, f)

    elif save_format == 'csv':
        filepath = os.path.join(save_path, f"{filename}.csv")
        df = pd.DataFrame(history.history)
        df.index.name = 'epoch'
        df.to_csv(filepath)

    elif save_format == 'json':
        filepath = os.path.join(save_path, f"{filename}.json")
        with open(filepath, 'w') as f:
            json.dump(history.history, f, indent=2)
    else:
        raise ValueError(f"Formato no soportado: {save_format}")

    print(f"  ✓ Historial guardado: {filepath}")
    return filepath



def load_training_history(filepath, format='pickle'):
    """
    Carga un historial de entrenamiento guardado.

    Parameters:
    -----------
    filepath : str
        Ruta al archivo
    format : str, default='pickle'
        Formato del archivo: 'pickle', 'csv', o 'json'

    Returns:
    --------
    dict : Diccionario con el historial de entrenamiento
    """

    if format == 'pickle':
        with open(filepath, 'rb') as f:
            history_dict = pickle.load(f)
    elif format == 'csv':
        df = pd.read_csv(filepath, index_col='epoch')
        history_dict = df.to_dict('list')
    elif format == 'json':
        with open(filepath, 'r') as f:
            history_dict = json.load(f)
    else:
        raise ValueError(f"Formato no soportado: {format}")

    return history_dict


def get_best_epoch_info(history):
    """
    Extrae información sobre el mejor epoch basado en val_loss.

    Parameters:
    -----------
    history : keras.callbacks.History or dict
        Objeto History o diccionario con history.history

    Returns:
    --------
    dict : Información del mejor epoch
    """

    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history

    best_epoch = hist_dict['val_loss'].index(min(hist_dict['val_loss'])) + 1

    info = {
        'best_epoch': best_epoch,
        'best_val_loss': min(hist_dict['val_loss']),
        'train_loss_at_best': hist_dict['loss'][best_epoch - 1],
        'total_epochs': len(hist_dict['loss']),
        'final_val_loss': hist_dict['val_loss'][-1]
    }

    # Agregar métricas adicionales si existen
    for key in ['mae', 'mean_absolute_error', 'accuracy', 'acc']:
        if key in hist_dict:
            info[f'best_val_{key}'] = hist_dict[f'val_{key}'][best_epoch - 1]
            break

    return info





# FUNCIONES DE EVALUACIÓN
def evaluate_regression(y_true, y_pred, model_name="Model"):
    """
    Evalúa modelo de regresión

    Métricas:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - R²: Coefficient of determination
    - Direction Accuracy: % acierto en dirección
    """
    # Eliminar NAs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            'model': model_name,
            'RMSE': np.nan,
            'MAE': np.nan,
            'R2': np.nan,
            'Direction_Accuracy': np.nan,
            'n_samples': 0
        }

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Direccionalidad (crítico para trading)
    direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    return {
        'model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Direction_Accuracy': direction_accuracy,
        'n_samples': len(y_true)
    }

def evaluate_classification(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Evalúa modelo de clasificación

    Métricas:
    - Accuracy: % acierto total
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1: Harmonic mean of precision and recall
    - AUC: Area Under ROC Curve
    """
    # Eliminar NAs
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            'model': model_name,
            'Accuracy': np.nan,
            'Precision': np.nan,
            'Recall': np.nan,
            'F1': np.nan,
            'ROC_AUC': np.nan,
            'n_samples': 0
        }

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC si hay probabilidades
    auc = None
    if y_proba is not None:
        y_proba = y_proba[mask]
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = None

    return {
        'model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC_AUC': auc,
        'n_samples': len(y_true)
    }

print("✓ Funciones de evaluación definidas")

# FUNCIONES DE COMPARACIÓN ESTADÍSTICA

def mcnemar_test(y_true, pred1, pred2, model1_name='Model 1', model2_name='Model 2'):
    """
    Test de McNemar para comparar dos clasificadores

    H0: Los dos modelos tienen el mismo error rate
    H1: Los modelos tienen diferentes error rates

    Args:
        y_true: Valores reales
        pred1: Predicciones del modelo 1
        pred2: Predicciones del modelo 2
        model1_name: Nombre del modelo 1
        model2_name: Nombre del modelo 2

    Returns:
        dict con resultados del test
    """
    # Convertir a arrays
    y_true = np.array(y_true)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    # Calcular correctas/incorrectas
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)

    # Tabla de contingencia 2x2
    # n01: Modelo 1 correcto, Modelo 2 incorrecto
    # n10: Modelo 1 incorrecto, Modelo 2 correcto
    n01 = np.sum(correct1 & ~correct2)
    n10 = np.sum(~correct1 & correct2)
    n00 = np.sum(~correct1 & ~correct2)
    n11 = np.sum(correct1 & correct2)

    # Test estadístico de McNemar con corrección de continuidad
    if (n01 + n10) == 0:
        statistic = 0
        p_value = 1.0
    else:
        statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        p_value = 1 - chi2.cdf(statistic, df=1)

    # Calcular accuracies
    acc1 = np.mean(correct1)
    acc2 = np.mean(correct2)

    result = {
        'model1': model1_name,
        'model2': model2_name,
        'n_samples': len(y_true),
        'accuracy_model1': acc1,
        'accuracy_model2': acc2,
        'accuracy_diff': acc2 - acc1,
        'n_both_correct': n11,
        'n_both_wrong': n00,
        'n_only_model1_correct': n01,
        'n_only_model2_correct': n10,
        'mcnemar_statistic': statistic,
        'p_value': p_value,
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01
    }

    return result


def bootstrap_confidence_interval(y_true, pred1, pred2, metric='accuracy',
                                   n_bootstrap=1000, confidence=0.95, random_state=42):
    """
    Bootstrap para intervalo de confianza de la diferencia entre modelos

    Args:
        y_true: Valores reales
        pred1: Predicciones modelo 1
        pred2: Predicciones modelo 2
        metric: 'accuracy' o 'f1'
        n_bootstrap: Número de muestras bootstrap
        confidence: Nivel de confianza (0.95 = 95%)
        random_state: Semilla aleatoria

    Returns:
        dict con intervalos de confianza
    """
    np.random.seed(random_state)

    y_true = np.array(y_true)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    n = len(y_true)

    differences = []

    for _ in range(n_bootstrap):
        # Resample con reemplazo
        idx = np.random.choice(n, size=n, replace=True)
        y_boot = y_true[idx]
        p1_boot = pred1[idx]
        p2_boot = pred2[idx]

        # Calcular métrica
        if metric == 'accuracy':
            metric1 = np.mean(p1_boot == y_boot)
            metric2 = np.mean(p2_boot == y_boot)
        elif metric == 'f1':
            from sklearn.metrics import f1_score
            metric1 = f1_score(y_boot, p1_boot, average='binary', zero_division=0)
            metric2 = f1_score(y_boot, p2_boot, average='binary', zero_division=0)

        differences.append(metric2 - metric1)

    differences = np.array(differences)

    # Calcular percentiles
    alpha = 1 - confidence
    lower = np.percentile(differences, 100 * alpha/2)
    upper = np.percentile(differences, 100 * (1 - alpha/2))
    mean_diff = np.mean(differences)

    # ¿El intervalo incluye 0?
    significant = not (lower <= 0 <= upper)

    result = {
        'metric': metric,
        'mean_difference': mean_diff,
        'ci_lower': lower,
        'ci_upper': upper,
        'confidence': confidence,
        'significant': significant,
        'n_bootstrap': n_bootstrap
    }

    return result


def compare_models_statistical(y_true, pred1, pred2, model1_name, model2_name):
    """
    Comparación completa entre dos modelos
    """
    print(f"\n{'='*70}")
    print(f"COMPARACIÓN: {model1_name} vs {model2_name}")
    print(f"{'='*70}")

    # McNemar Test
    mcnemar_results = mcnemar_test(y_true, pred1, pred2, model1_name, model2_name)

    print(f"\n McNemar Test:")
    print(f"   Accuracy {model1_name}: {mcnemar_results['accuracy_model1']:.4f}")
    print(f"   Accuracy {model2_name}: {mcnemar_results['accuracy_model2']:.4f}")
    print(f"   Diferencia: {mcnemar_results['accuracy_diff']:.4f}")
    print(f"   Statistic: {mcnemar_results['mcnemar_statistic']:.4f}")
    print(f"   p-value: {mcnemar_results['p_value']:.4f}")

    if mcnemar_results['significant_001']:
        print(f"   Significativo al 0.1% (p < 0.001)")
    elif mcnemar_results['significant_005']:
        print(f"   Significativo al 5% (p < 0.05)")
    else:
        print(f"   NO significativo (p >= 0.05)")

    # Bootstrap CI
    boot_results = bootstrap_confidence_interval(y_true, pred1, pred2, metric='accuracy')

    print(f"\n Bootstrap (1000 iteraciones):")
    print(f"   Diferencia media: {boot_results['mean_difference']:.4f}")
    print(f"   IC 95%: [{boot_results['ci_lower']:.4f}, {boot_results['ci_upper']:.4f}]")

    if boot_results['significant']:
        print(f"   Intervalo NO incluye 0 → Diferencia significativa")
    else:
        print(f"   Intervalo incluye 0 → Diferencia NO significativa")

    return mcnemar_results, boot_results


print("✓ Funciones estadísticas para clasificación cargadas")

# FUNCIONES PARA COMPARACIÓN ESTADÍSTICA DE REGRESIÓN

def diebold_mariano_test(errors1, errors2, h=1):
    """
    Test Diebold-Mariano para comparar dos modelos de regresión

    H0: Los dos modelos tienen igual precisión de pronóstico
    H1: Los modelos tienen diferente precisión

    Args:
        errors1, errors2: arrays de errores (y_true - y_pred)
        h: horizonte (1 para returns_next, 5 para returns_next_5)

    Returns:
        dm_stat: estadístico DM
        p_value: p-valor bilateral
    """
    d = errors1**2 - errors2**2
    d_mean = np.mean(d)

    # Varianza con corrección Newey-West
    n = len(d)
    gamma = np.zeros(h)
    for j in range(h):
        if j < n:
            gamma[j] = np.mean((d[:(n-j)] - d_mean) * (d[j:] - d_mean))

    variance = (gamma[0] + 2 * np.sum(gamma[1:h])) / n

    if variance <= 0:
        return 0, 1.0

    dm_stat = d_mean / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


def bootstrap_rmse_comparison(y_true, pred1, pred2, n_iterations=10000):
    """
    Bootstrap para comparar RMSE entre dos modelos
    """
    n = len(y_true)
    diff_rmse = []

    for _ in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        rmse1 = np.sqrt(np.mean((y_true[idx] - pred1[idx])**2))
        rmse2 = np.sqrt(np.mean((y_true[idx] - pred2[idx])**2))
        diff_rmse.append(rmse1 - rmse2)

    diff_rmse = np.array(diff_rmse)
    ci_lower = np.percentile(diff_rmse, 2.5)
    ci_upper = np.percentile(diff_rmse, 97.5)
    p_value = 2 * min(np.mean(diff_rmse >= 0), np.mean(diff_rmse <= 0))

    return {
        'mean_diff': np.mean(diff_rmse),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'significant': 0 not in np.sign([ci_lower, ci_upper]) if ci_lower != ci_upper else False
    }


def compare_regression_models_statistical(y_test, pred_fin, pred_sent,
                                          model_name, h=1):
    """
    Compara modelos financial vs sentiment para regresión

    Returns:
        dict con resultados DM y Bootstrap
    """
    # Calcular errores
    errors_fin = y_test - pred_fin
    errors_sent = y_test - pred_sent

    # Test Diebold-Mariano
    dm_stat, dm_pval = diebold_mariano_test(errors_fin, errors_sent, h=h)

    # Bootstrap
    boot_results = bootstrap_rmse_comparison(y_test, pred_fin, pred_sent)

    # Calcular RMSE
    rmse_fin = np.sqrt(np.mean(errors_fin**2))
    rmse_sent = np.sqrt(np.mean(errors_sent**2))

    return {
        'model': model_name,
        'n_samples': len(y_test),
        'rmse_financial': rmse_fin,
        'rmse_sentiment': rmse_sent,
        'rmse_diff': rmse_fin - rmse_sent,
        'dm_statistic': dm_stat,
        'dm_pvalue': dm_pval,
        'dm_significant': dm_pval < 0.05,
        'boot_mean_diff': boot_results['mean_diff'],
        'boot_ci_lower': boot_results['ci_lower'],
        'boot_ci_upper': boot_results['ci_upper'],
        'boot_pvalue': boot_results['p_value'],
        'boot_significant': boot_results['significant']
    }

print("✓ Funciones de comparación estadística para REGRESIÓN definidas")

def run_regression_statistical_comparisons(reg_df, 
                                           targets=['returns_next', 'returns_next_5',
                                                    'returns_next_10', 'returns_next_20'],
                                           comparisons=None):
    """
    Ejecuta comparaciones estadísticas para REGRESIÓN

    Args:
        reg_df: DataFrame con columnas 'dataset', 'target', 'model', 'y_test', 'predictions'
        targets: Lista de targets a comparar
        comparisons: Lista de tuplas (dataset1_name, dataset2_name, label) a comparar
    """
    if comparisons is None:
        comparisons = [
            ('financial_scaled', 'sentiment_scaled', 'Financial_vs_Sentiment'),
            ('financial_unscaled', 'sentiment_unscaled', 'Financial_vs_Sentiment'),
            ('financial_scaled', 'financial_long_scaled', 'Horizons_scaled'),
            ('financial_unscaled', 'financial_long_unscaled', 'Horizons_unscaled')
        ]

    results_summary = []

    for target in targets:
        print(f"\n{'#'*70}")
        print(f"# REGRESIÓN - TARGET: {target}")
        print(f"{'#'*70}\n")

        # Determinar horizonte para DM test
        if 'next_5' in target:
            h = 5
        elif 'next_10' in target:
            h = 10
        elif 'next_20' in target:
            h = 20
        else:
            h = 1

        for dataset1, dataset2, comparison_label in comparisons:
            # Filtrar datos para cada dataset
            df1 = reg_df[(reg_df['dataset'] == dataset1) & (reg_df['target'] == target)]
            df2 = reg_df[(reg_df['dataset'] == dataset2) & (reg_df['target'] == target)]
            
            if df1.empty or df2.empty:
                print(f"⚠ Datos no disponibles para {dataset1} vs {dataset2}")
                continue

            # Obtener y_test (debe ser igual para ambos datasets)
            y_test1 = df1.iloc[0]['y_test']
            y_test2 = df2.iloc[0]['y_test']
            
            y1_arr = y_test1.values if hasattr(y_test1, 'values') else np.array(y_test1)
            y2_arr = y_test2.values if hasattr(y_test2, 'values') else np.array(y_test2)

            if len(y1_arr) != len(y2_arr):
                print(f" ERROR: {dataset1} vs {dataset2}: Test sets de diferente tamaño")
                continue

            # Filtrar NAs
            valid_mask = ~np.isnan(y1_arr) & ~np.isnan(y2_arr)
            if not np.any(valid_mask):
                print(f"⚠ No hay muestras válidas para {dataset1} vs {dataset2}")
                continue

            y_clean = y1_arr[valid_mask]
            scaling = 'scaled' if 'scaled' in dataset1 else 'unscaled'

            print(f"\n== {comparison_label} ({scaling}) - {np.sum(valid_mask)} muestras válidas ==")

            # Obtener modelos comunes
            models1 = set(df1['model'].unique())
            models2 = set(df2['model'].unique())
            common_models = models1 & models2

            for model_name in sorted(common_models):
                pred1_full = df1[df1['model'] == model_name].iloc[0]['predictions']
                pred2_full = df2[df2['model'] == model_name].iloc[0]['predictions']

                if len(pred1_full) != len(pred2_full):
                    print(f"⚠ {model_name}: Tamaños diferentes, saltando...")
                    continue

                # GRU: predicciones más cortas
                if len(pred1_full) < len(y1_arr):
                    print(f"⚠ {model_name}: Modelo secuencial (GRU), saltando...")
                    continue

                pred1_clean = pred1_full[valid_mask]
                pred2_clean = pred2_full[valid_mask]

                result = compare_regression_models_statistical(
                    y_clean, pred1_clean, pred2_clean,
                    model_name, h=h
                )

                result['comparison_type'] = comparison_label
                result['dataset1'] = dataset1
                result['dataset2'] = dataset2
                result['scaling'] = scaling
                result['target'] = target
                results_summary.append(result)

                print(f"\n→ {model_name}:")
                print(f"   RMSE: {dataset1.replace('_scaled','').replace('_unscaled','')}={result['rmse_financial']:.4f}, "
                      f"{dataset2.replace('_scaled','').replace('_unscaled','')}={result['rmse_sentiment']:.4f}")
                print(f"   DM p-value: {result['dm_pvalue']:.4f} "
                      f"{'✓ SIGNIFICATIVO' if result['dm_significant'] else ''}")
                print(f"   Bootstrap IC 95%: [{result['boot_ci_lower']:.4f}, "
                      f"{result['boot_ci_upper']:.4f}] "
                      f"{'✓ SIGNIFICATIVO' if result['boot_significant'] else ''}")

    df = pd.DataFrame(results_summary)

    if not df.empty:
        print("\nRESUMEN ESTADÍSTICO - REGRESIÓN")
        
        for comp_type in df['comparison_type'].unique():
            df_comp = df[df['comparison_type'] == comp_type]
            print(f"\n{comp_type}:")
            print(f"  Total comparaciones: {len(df_comp)}")
            print(f"  DM significativas (p<0.05): {df_comp['dm_significant'].sum()}")
            print(f"  Bootstrap significativas: {df_comp['boot_significant'].sum()}")

            mejoras = (df_comp['rmse_diff'] > 0).sum()
            print(f"  {df_comp.iloc[0]['dataset2'].split('_')[0]} mejor (RMSE menor): {mejoras} "
                  f"({mejoras/len(df_comp)*100:.1f}%)")

    return df

# Ejecutar
#regression_stats = run_regression_statistical_comparisons(reg_df)

#COMPARACIONES ESTADÍSTICAS: FINANCIAL vs SENTIMENT
def run_classification_statistical_comparisons(class_df,
                                               targets=['direction_next', 'direction_next_5',
                                                        'direction_next_10', 'direction_next_20'],
                                               comparisons=None):
    """
    Ejecuta comparaciones estadísticas para CLASIFICACIÓN

    Args:
        class_df: DataFrame con columnas 'dataset', 'target', 'model', 'y_test', 'predictions'
        targets: Lista de targets a comparar
        comparisons: Lista de tuplas (dataset1_name, dataset2_name, label) a comparar
    """
    if comparisons is None:
        comparisons = [
            ('financial_scaled', 'sentiment_scaled', 'Financial_vs_Sentiment'),
            ('financial_unscaled', 'sentiment_unscaled', 'Financial_vs_Sentiment'),
            ('financial_long_scaled', 'financial_scaled', 'Long_vs_Short_History'),
            ('financial_long_unscaled', 'financial_unscaled', 'Long_vs_Short_History'),
            ('financial_long_scaled', 'sentiment_scaled', 'Long_vs_Sentiment'),
            ('financial_long_unscaled', 'sentiment_unscaled', 'Long_vs_Sentiment')
        ]

    results_summary = []

    for target in targets:
        print(f"\n\n{'#'*70}")
        print(f"# CLASIFICACIÓN - TARGET: {target}")
        print(f"{'#'*70}")

        for dataset1, dataset2, comparison_label in comparisons:
            # Filtrar datos para cada dataset
            df1 = class_df[(class_df['dataset'] == dataset1) & (class_df['target'] == target)]
            df2 = class_df[(class_df['dataset'] == dataset2) & (class_df['target'] == target)]
            
            if df1.empty or df2.empty:
                print(f"\n⚠ Datos no disponibles para {dataset1} vs {dataset2}")
                continue

            # Obtener y_test
            y_test1 = df1.iloc[0]['y_test']
            y_test2 = df2.iloc[0]['y_test']

            y1_arr = y_test1.values if hasattr(y_test1, 'values') else np.array(y_test1)
            y2_arr = y_test2.values if hasattr(y_test2, 'values') else np.array(y_test2)

            if len(y1_arr) != len(y2_arr):
                print(f" ERROR: {dataset1} vs {dataset2}: Test sets de diferente tamaño")
                print(f"   {dataset1}: {len(y1_arr)} muestras, {dataset2}: {len(y2_arr)} muestras")
                continue

            # Identificar índices válidos (sin NAs)
            valid_mask = ~np.isnan(y1_arr) & ~np.isnan(y2_arr)

            if not np.any(valid_mask):
                print(f"⚠ No hay muestras válidas para {dataset1} vs {dataset2}")
                continue

            y1_clean = y1_arr[valid_mask]
            y2_clean = y2_arr[valid_mask]

            print(f"\n {comparison_label}: {np.sum(valid_mask)} muestras válidas / {len(y1_arr)} totales")

            if not np.array_equal(y1_clean, y2_clean):
                print(f"⚠ WARNING: Test sets diferentes entre {dataset1} y {dataset2}")
                continue

            print(f" Test sets idénticos")

            # Obtener modelos comunes
            models1 = set(df1['model'].unique())
            models2 = set(df2['model'].unique())
            common_models = models1 & models2

            scaling = 'scaled' if 'scaled' in dataset1 else 'unscaled'

            print(f"== {comparison_label} ({scaling}) - Comparando {len(common_models)} modelos ==")

            for model_name in sorted(common_models):
                pred1_full = df1[df1['model'] == model_name].iloc[0]['predictions']
                pred2_full = df2[df2['model'] == model_name].iloc[0]['predictions']

                if len(pred1_full) != len(pred2_full):
                    print(f"\n⚠ {model_name}: Tamaños diferentes (pred1={len(pred1_full)}, pred2={len(pred2_full)})")
                    continue

                if len(pred1_full) < len(y1_arr):
                    print(f"\n⚠ {model_name}: {len(pred1_full)} predicciones < {len(y1_arr)} y_test")
                    print(f"   Probablemente modelo secuencial (GRU)")
                    continue

                pred1_clean = pred1_full[valid_mask]
                pred2_clean = pred2_full[valid_mask]
                y_clean = y1_clean

                assert len(pred1_clean) == len(pred2_clean) == len(y_clean), \
                    f"Tamaños no coinciden: pred1={len(pred1_clean)}, pred2={len(pred2_clean)}, y={len(y_clean)}"

                mcnemar_res, boot_res = compare_models_statistical(
                    y_clean,
                    pred1_clean,
                    pred2_clean,
                    f"{model_name} ({dataset1})",
                    f"{model_name} ({dataset2})"
                )

                results_summary.append({
                    'comparison_type': comparison_label,
                    'dataset1': dataset1,
                    'dataset2': dataset2,
                    'scaling': scaling,
                    'target': target,
                    'model': model_name,
                    'n_samples': len(y_clean),
                    'n_nas_removed': len(y1_arr) - len(y_clean),
                    'acc_dataset1': mcnemar_res['accuracy_model1'],
                    'acc_dataset2': mcnemar_res['accuracy_model2'],
                    'diff': mcnemar_res['accuracy_diff'],
                    'mcnemar_pvalue': mcnemar_res['p_value'],
                    'mcnemar_significant': mcnemar_res['significant_005'],
                    'bootstrap_ci_lower': boot_res['ci_lower'],
                    'bootstrap_ci_upper': boot_res['ci_upper'],
                    'bootstrap_significant': boot_res['significant']
                })

    df = pd.DataFrame(results_summary)

    if not df.empty:
        print(f"\n{'='*70}")
        print("RESUMEN ESTADÍSTICO - CLASIFICACIÓN")
        print(f"{'='*70}\n")

        for comp_type in df['comparison_type'].unique():
            df_comp = df[df['comparison_type'] == comp_type]
            print(f"\n{comp_type}:")
            print(f"  Total comparaciones: {len(df_comp)}")
            print(f"  McNemar significativas (p<0.05): {df_comp['mcnemar_significant'].sum()}")
            print(f"  Bootstrap significativas: {df_comp['bootstrap_significant'].sum()}")

            mejoras = (df_comp['diff'] > 0).sum()
            print(f"  {df_comp.iloc[0]['dataset2'].split('_')[0]} mejor (accuracy mayor): {mejoras} "
                  f"({mejoras/len(df_comp)*100:.1f}%)")

    return df

# Ejecutar
#classification_stats = run_classification_statistical_comparisons(class_df)


def load_predictions_from_csv(base_path, datasets, targets, task='regression'):
    """
    Carga predicciones desde estructura de directorios CSV
    
    Args:
        base_path: Ruta base donde están los directorios (ej: '/content/drive/MyDrive/TFM/csv')
        datasets: Lista de datasets (ej: ['financial_scaled', 'sentiment_scaled'])
        targets: Lista de targets (ej: ['returns_next', 'returns_next_5'])
        task: 'regression' o 'classification'
    
    Returns:
        DataFrame con columnas: dataset, target, model, y_test, predictions
    """
    import os
    import pandas as pd
    
    data = []
    
    for dataset in datasets:
        for target in targets:
            dir_path = os.path.join(base_path, dataset, target)
            
            if not os.path.exists(dir_path):
                print(f"⚠ No existe: {dir_path}")
                continue
            
            # Listar archivos CSV en el directorio
            csv_files = [f for f in os.listdir(dir_path) if f.endswith('_predictions.parquet')]
            
            for csv_file in csv_files:
                # Extraer nombre del modelo
                model_name = csv_file.replace('_predictions.paquet', '')
                
                # Cargar CSV
                file_path = os.path.join(dir_path, csv_file)
                df = pd.read_parquet(file_path)
                
                # Verificar columnas
                if 'y_true' not in df.columns or 'y_pred' not in df.columns:
                    print(f"⚠ Columnas incorrectas en {file_path}")
                    continue
                
                # Guardar
                data.append({
                    'dataset': dataset,
                    'target': target,
                    'model': model_name,
                    'y_test': df['y_true'].values,
                    'predictions': df['y_pred'].values
                })
                
                print(f"✓ Cargado: {dataset}/{target}/{model_name} ({len(df)} muestras)")
    
    return pd.DataFrame(data)
'''
# USO:
#base_path = '/content/drive/MyDrive/TFM/csv'  # Ajusta esta ruta

# Para regresión
#datasets_reg = ['financial_scaled', 'sentiment_scaled', 
                'financial_unscaled', 'sentiment_unscaled',
                'financial_long_scaled', 'financial_long_unscaled']
#targets_reg = ['returns_next', 'returns_next_5', 'returns_next_10', 'returns_next_20']

#reg_predictions_df = load_predictions_from_csv(CSV, DATASETS_TO_PROCESS, TARGETS_REGRESSION, task='regression')
# Guardar resultados
#reg_predictions_df.to_csv(f"{CSV}/predictions_regression.csv", index=False)
'''
# -----------------------------------------------------------------------------
# 2. FUNCIONES DE COMPARACIÓN ENTRE HORIZONTES - CLASIFICACIÓN
# -----------------------------------------------------------------------------

def compare_classification_across_horizons(clf_predictions_df, classification_stats,
                                          horizons=['direction_next', 'direction_next_5',
                                                   'direction_next_10', 'direction_next_20']):
    """
    Compara modelos de clasificación entre horizontes temporales
    """
    results_summary = []

    for dataset_name in ['financial_scaled', 'sentiment_scaled',
                         'financial_unscaled', 'sentiment_unscaled',
                         'financial_long_scaled','financial_long_unscaled']:

        if dataset_name not in clf_predictions_df:
            continue

        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")

        # Identificar modelos comunes en todos los horizontes
        models_in_all_horizons = None

        for horizon in horizons:
            if horizon in clf_predictions_df[dataset_name]:
                models_in_horizon = set(clf_predictions_df[dataset_name][horizon].keys())
                if models_in_all_horizons is None:
                    models_in_all_horizons = models_in_horizon
                else:
                    models_in_all_horizons &= models_in_horizon

        if not models_in_all_horizons:
            print(f"⚠ No hay modelos comunes en todos los horizontes")
            continue

        print(f"✓ Modelos comunes: {sorted(models_in_all_horizons)}")

        for model_name in sorted(models_in_all_horizons):
            print(f"\n{'─'*70}")
            print(f"MODELO: {model_name}")
            print(f"{'─'*70}")

            # Recolectar datos para cada horizonte
            horizon_accuracies = {}
            horizon_predictions = {}
            horizon_y_test = {}

            for horizon in horizons:
                if horizon not in clf_predictions_df[dataset_name]:
                    continue
                if model_name not in clf_predictions_df[dataset_name][horizon]:
                    continue

                preds = clf_predictions_df[dataset_name][horizon][model_name]
                y_test = classification_stats[dataset_name][horizon]

                # Limpiar NAs
                y_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
                valid_mask = ~np.isnan(y_arr)

                y_clean = y_arr[valid_mask]
                pred_clean = preds[valid_mask] if len(preds) == len(y_arr) else None

                if pred_clean is None or len(pred_clean) == 0:
                    continue

                acc = np.mean(pred_clean == y_clean)
                horizon_accuracies[horizon] = acc
                horizon_predictions[horizon] = pred_clean
                horizon_y_test[horizon] = y_clean

            if len(horizon_accuracies) < 2:
                print(f"⚠ Menos de 2 horizontes disponibles para {model_name}")
                continue

            # ENCONTRAR EL TAMAÑO MÍNIMO
            min_size = min(len(horizon_y_test[h]) for h in horizon_y_test.keys())
            print(f"\n✓ Tamaño mínimo común: {min_size} observaciones")

            # ALINEAR TODOS LOS HORIZONTES (usar últimas n observaciones)
            for horizon in horizon_y_test.keys():
                horizon_y_test[horizon] = horizon_y_test[horizon][-min_size:]
                horizon_predictions[horizon] = horizon_predictions[horizon][-min_size:]
                horizon_accuracies[horizon] = np.mean(
                    horizon_predictions[horizon] == horizon_y_test[horizon]
                )

            # Mostrar accuracies
            print("\n📊 Accuracy por horizonte (últimas {} obs):".format(min_size))
            for horizon in horizons:
                if horizon in horizon_accuracies:
                    acc = horizon_accuracies[horizon]
                    days = horizon.split('_')[-1]
                    if days == 'next':
                        days = '1'
                    print(f"   {days:>2} días: {acc:.4f}")

            # Comparaciones pareadas
            horizon_list = sorted([h for h in horizons if h in horizon_accuracies])

            for i in range(len(horizon_list) - 1):
                h1 = horizon_list[i]
                h2 = horizon_list[i + 1]

                # Tests estadísticos
                mcnemar_res, boot_res = compare_models_statistical(
                    horizon_y_test[h1],
                    horizon_predictions[h1],
                    horizon_predictions[h2],
                    f"{h1}",
                    f"{h2}"
                )

                days1 = h1.split('_')[-1]
                days2 = h2.split('_')[-1]
                if days1 == 'next': days1 = '1'
                if days2 == 'next': days2 = '1'

                results_summary.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'horizon_1': h1,
                    'horizon_2': h2,
                    'days_1': int(days1),
                    'days_2': int(days2),
                    'n_samples': min_size,
                    'acc_horizon_1': mcnemar_res['accuracy_model1'],
                    'acc_horizon_2': mcnemar_res['accuracy_model2'],
                    'diff': mcnemar_res['accuracy_diff'],
                    'mcnemar_pvalue': mcnemar_res['p_value'],
                    'mcnemar_significant': mcnemar_res['significant_005'],
                    'bootstrap_ci_lower': boot_res['ci_lower'],
                    'bootstrap_ci_upper': boot_res['ci_upper'],
                    'bootstrap_significant': boot_res['significant']
                })

                print(f"\n  {h1} vs {h2}:")
                print(f"    Δ Accuracy: {mcnemar_res['accuracy_diff']:+.4f}")
                print(f"    McNemar p-value: {mcnemar_res['p_value']:.4f}")
                print(f"    Significativo: {'✓' if mcnemar_res['significant_005'] else '✗'}")

    return pd.DataFrame(results_summary)

print("Función compare_classification_across_horizons cargada correctamente")

def analyze_classification_horizon_effects(horizon_results_df):
    """
    Análisis cuantitativo del efecto del horizonte en clasificación
    """
    print("\n" + "="*70)
    print("ANÁLISIS: EFECTO DEL HORIZONTE TEMPORAL - CLASIFICACIÓN")
    print("="*70)

    if horizon_results_df.empty:
        print("No hay datos para analizar")
        return

    # 1. Tendencia general por dataset
    print("\n📈 TENDENCIA GENERAL:")

    for dataset in horizon_results_df['dataset'].unique():
        dataset_data = horizon_results_df[horizon_results_df['dataset'] == dataset]

        print(f"\n  {dataset}:")

        # Correlación días vs accuracy
        all_days = []
        all_accs = []

        for _, row in dataset_data.iterrows():
            all_days.extend([row['days_1'], row['days_2']])
            all_accs.extend([row['acc_horizon_1'], row['acc_horizon_2']])

        if len(all_days) > 2:
            corr = np.corrcoef(all_days, all_accs)[0, 1]

            if corr > 0.3:
                print(f"    Correlación: {corr:.3f} → Mejora con horizonte más largo")
            elif corr < -0.3:
                print(f"    Correlación: {corr:.3f} → Empeora con horizonte más largo")
            else:
                print(f"    Correlación: {corr:.3f} → Sin tendencia clara")

            # Accuracy promedio por horizonte
            horizon_means = {}
            for days in [1, 5, 10, 20]:
                accs = [row['acc_horizon_1'] for _, row in dataset_data.iterrows() if row['days_1'] == days]
                accs += [row['acc_horizon_2'] for _, row in dataset_data.iterrows() if row['days_2'] == days]
                if accs:
                    horizon_means[days] = np.mean(accs)

            if horizon_means:
                print(f"    Accuracy promedio por horizonte:")
                for days in sorted(horizon_means.keys()):
                    print(f"      {days:2d} días: {horizon_means[days]:.4f}")

    # 2. Diferencias significativas
    print(f"\n🔍 COMPARACIONES SIGNIFICATIVAS:")

    sig_comparisons = horizon_results_df[horizon_results_df['mcnemar_significant']]

    if len(sig_comparisons) > 0:
        print(f"  Total: {len(sig_comparisons)} / {len(horizon_results_df)} ({len(sig_comparisons)/len(horizon_results_df)*100:.1f}%)")

        for _, row in sig_comparisons.iterrows():
            direction = '📈' if row['diff'] > 0 else '📉'
            print(f"\n  {direction} {row['model']} ({row['dataset']})")
            print(f"     {row['days_1']}d → {row['days_2']}d: {row['diff']:+.4f}")
            print(f"     p-value: {row['mcnemar_pvalue']:.4f}")
    else:
        print("  ✗ NO se encontraron diferencias significativas")

    # 3. Estabilidad por modelo
    print(f"\n🎯 ESTABILIDAD POR MODELO:")

    stability_by_model = horizon_results_df.groupby('model').agg({
        'diff': ['mean', 'std'],
        'mcnemar_significant': 'sum'
    }).round(4)

    stability_by_model.columns = ['diff_mean', 'diff_std', 'n_significant']
    stability_by_model = stability_by_model.sort_values('diff_std')

    print("\n  Más estables (menor variación):")
    for model in stability_by_model.head(3).index:
        std = stability_by_model.loc[model, 'diff_std']
        mean = stability_by_model.loc[model, 'diff_mean']
        print(f"    {model}: std={std:.4f}, mean_diff={mean:+.4f}")

    print("\n  Más variables (mayor variación):")
    for model in stability_by_model.tail(3).index:
        std = stability_by_model.loc[model, 'diff_std']
        mean = stability_by_model.loc[model, 'diff_mean']
        n_sig = int(stability_by_model.loc[model, 'n_significant'])
        print(f"    {model}: std={std:.4f}, mean_diff={mean:+.4f}, sig={n_sig}")

# -----------------------------------------------------------------------------
# 3. FUNCIONES DE COMPARACIÓN ENTRE HORIZONTES - REGRESIÓN
# -----------------------------------------------------------------------------
def compare_regression_across_horizons(reg_predictions_df, regression_stats,
                                       horizons=['return_next', 'return_next_5',
                                                'return_next_10', 'return_next_20']):
    """
    Compara modelos de regresión entre horizontes temporales
    """
    results_summary = []
    for dataset_name in ['financial_scaled', 'sentiment_scaled',
                         'financial_unscaled', 'sentiment_unscaled',
                         'financial_long_scaled', 'financial_long_unscaled'
                         ]:
        if dataset_name not in reg_predictions_df:
            continue

        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")

        # Identificar modelos comunes
        models_in_all_horizons = None

        for horizon in horizons:
            if horizon in reg_predictions_df[dataset_name]:
                models_in_horizon = set(reg_predictions_df[dataset_name][horizon].keys())
                if models_in_all_horizons is None:
                    models_in_all_horizons = models_in_horizon
                else:
                    models_in_all_horizons &= models_in_horizon

        if not models_in_all_horizons:
            print(f"⚠ No hay modelos comunes en todos los horizontes")
            continue

        print(f"✓ Modelos comunes: {sorted(models_in_all_horizons)}")

        for model_name in sorted(models_in_all_horizons):
            print(f"\n{'─'*70}")
            print(f"MODELO: {model_name}")
            print(f"{'─'*70}")

            # Recolectar métricas para cada horizonte
            horizon_metrics = {}
            horizon_predictions = {}
            horizon_y_test = {}

            for horizon in horizons:
                if horizon not in reg_predictions_df[dataset_name]:
                    continue
                if model_name not in reg_predictions_df[dataset_name][horizon]:
                    continue

                preds = reg_predictions_df[dataset_name][horizon][model_name]
                y_test = regression_stats[dataset_name][horizon]

                # Limpiar NAs
                y_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
                valid_mask = ~np.isnan(y_arr)

                y_clean = y_arr[valid_mask]
                pred_clean = preds[valid_mask] if len(preds) == len(y_arr) else None

                if pred_clean is None or len(pred_clean) == 0:
                    continue

                # Calcular métricas
                mae = np.mean(np.abs(pred_clean - y_clean))
                rmse = np.sqrt(np.mean((pred_clean - y_clean)**2))

                horizon_metrics[horizon] = {'mae': mae, 'rmse': rmse}
                horizon_predictions[horizon] = pred_clean
                horizon_y_test[horizon] = y_clean

            if len(horizon_metrics) < 2:
                print(f"⚠ Menos de 2 horizontes disponibles")
                continue

            # ENCONTRAR EL TAMAÑO MÍNIMO
            min_size = min(len(horizon_y_test[h]) for h in horizon_y_test.keys())
            print(f"\n✓ Tamaño mínimo común: {min_size} observaciones")

            # ALINEAR TODOS LOS HORIZONTES (usar últimas n observaciones)
            for horizon in horizon_y_test.keys():
                horizon_y_test[horizon] = horizon_y_test[horizon][-min_size:]
                horizon_predictions[horizon] = horizon_predictions[horizon][-min_size:]
                
                # Recalcular métricas con datos alineados
                y_clean = horizon_y_test[horizon]
                pred_clean = horizon_predictions[horizon]
                mae = np.mean(np.abs(pred_clean - y_clean))
                rmse = np.sqrt(np.mean((pred_clean - y_clean)**2))
                horizon_metrics[horizon] = {'mae': mae, 'rmse': rmse}

            # Mostrar métricas
            print("\n📊 Métricas por horizonte (últimas {} obs):".format(min_size))
            for horizon in horizons:
                if horizon in horizon_metrics:
                    metrics = horizon_metrics[horizon]
                    days = horizon.split('_')[-1]
                    if days == 'next':
                        days = '1'
                    print(f"   {days:>2} días: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")

            # Comparaciones pareadas
            horizon_list = sorted([h for h in horizons if h in horizon_metrics])

            for i in range(len(horizon_list) - 1):
                h1 = horizon_list[i]
                h2 = horizon_list[i + 1]

                # Extraer horizonte numérico
                days1 = h1.split('_')[-1]
                if days1 == 'next': 
                    horizon_value = 1
                else:
                    horizon_value = int(days1)
                        
                # Calcular errores
                errors1 = horizon_y_test[h1] - horizon_predictions[h1]
                errors2 = horizon_y_test[h1] - horizon_predictions[h2]    
                # Tests estadísticos
                dm_result = diebold_mariano_test(
                    errors1,
                    errors2,
                    horizon_value
                )
                # dm_result es una tupla (statistic, p_value, significant)
                # dm_result es una tupla (statistic, p_value)
                dm_statistic, dm_pvalue = dm_result

                # Calcular significant manualmente
                dm_significant = dm_pvalue < 0.05
                
                mae_diff = horizon_metrics[h1]['mae'] - horizon_metrics[h2]['mae']
                boot_ci = bootstrap_confidence_interval(
                    horizon_y_test[h1],
                    horizon_predictions[h1],
                    horizon_predictions[h2]
                )
                # Extraer el horizonte numérico    
                days1 = h1.split('_')[-1]
                days2 = h2.split('_')[-1]
                if days1 == 'next': days1 = '1'
                if days2 == 'next': days2 = '1'


                # Luego en results_summary:
                results_summary.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'horizon_1': h1,
                    'horizon_2': h2,
                    'days_1': int(days1),
                    'days_2': int(days2),
                    'n_samples': min_size,
                    'mae_horizon_1': horizon_metrics[h1]['mae'],
                    'mae_horizon_2': horizon_metrics[h2]['mae'],
                    'rmse_horizon_1': horizon_metrics[h1]['rmse'],
                    'rmse_horizon_2': horizon_metrics[h2]['rmse'],
                    'mae_diff': mae_diff,
                    'dm_statistic': dm_statistic,
                    'dm_pvalue': dm_pvalue,
                    'dm_significant': dm_significant,
                    'bootstrap_ci_lower': boot_ci['ci_lower'],
                    'bootstrap_ci_upper': boot_ci['ci_upper'],
                    'bootstrap_significant': boot_ci['significant']
                    })

                print(f"\n  {h1} vs {h2}:")
                print(f"    Δ MAE: {mae_diff:.6f}")
                print(f"    DM statistic: {dm_statistic:.4f}")
                print(f"    p-value: {dm_pvalue:.4f}")
                print(f"    Significativo: {'✓' if dm_significant else '✗'}")
                
    return pd.DataFrame(results_summary)



def analyze_regression_horizon_effects(horizon_results_df):
    """
    Análisis cuantitativo del efecto del horizonte en regresión
    """
    if horizon_results_df.empty:
        print("No hay datos para analizar")
        return

    print(f"\n{'='*70}")
    print("ANÁLISIS: EFECTO DEL HORIZONTE TEMPORAL - REGRESIÓN")
    print(f"{'='*70}")

    # Por dataset
    for dataset in horizon_results_df['dataset'].unique():
        data = horizon_results_df[horizon_results_df['dataset'] == dataset]

        print(f"\nDATASET: {dataset}")
        print("─" * 70)

        # Diferencias significativas
        sig = data['dm_significant'].sum()
        total = len(data)

        print(f"\n  🔍 Comparaciones significativas (DM test):")
        print(f"    Total: {sig} / {total} ({sig/total*100:.1f}%)")

        # MAE promedio por horizonte
        print(f"\n  📊 MAE promedio por horizonte:")
        for days in [1, 5, 10, 20]:
            mae_list = []
            mae_list.extend(data[data['days_1'] == days]['mae_horizon_1'].values)
            mae_list.extend(data[data['days_2'] == days]['mae_horizon_2'].values)
            if mae_list:
                print(f"    {days:2d} días: {np.mean(mae_list):.6f}")

        # Tendencias por modelo
        print(f"\n  🎯 Tendencias por modelo:")
        for model in data['model'].unique():
            model_data = data[data['model'] == model].sort_values('days_1')

            if len(model_data) < 2:
                continue

            # Correlación días vs MAE
            mae_trend = np.corrcoef(model_data['days_1'], model_data['mae_horizon_1'])[0,1]

            print(f"\n    {model}:")
            print(f"      Correlación días vs MAE: {mae_trend:.3f}")

            if mae_trend > 0.5:
                print(f"      → Horizonte más largo = Menos predecible (mayor error)")
            elif mae_trend < -0.5:
                print(f"      → Horizonte más largo = Más predecible (menor error)")
            else:
                print(f"      → Sin tendencia clara")
                
            