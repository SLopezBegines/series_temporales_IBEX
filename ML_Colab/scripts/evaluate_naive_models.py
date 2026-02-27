"""
Módulo para evaluar modelos de clasificación: REALES vs INGENUOS

Autor: Santi - TFM IBEX35
Fecha: Noviembre 2024

Uso:
    from evaluate_naive_models import evaluate_model_vs_naive, evaluate_all_models
    
    # Para un solo modelo
    result = evaluate_model_vs_naive(y_test, y_pred, model_name="XGBoost")
    
    # Para múltiples modelos
    results_dict = {
        'XGBoost': (y_test, y_pred_xgb),
        'LightGBM': (y_test, y_pred_lgb)
    }
    df_results = evaluate_all_models(results_dict)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    balanced_accuracy_score, 
    cohen_kappa_score,
    precision_score, 
    recall_score, 
    f1_score
)
from scipy.stats import binomtest
import warnings
warnings.filterwarnings('ignore')


def evaluate_model_vs_naive(y_true, y_pred, model_name="Model", dataset="", target=""):
    """
    Evalúa si un modelo es realmente predictivo o simplemente ingenuo.
    
    Un modelo ingenuo es aquel que predice principalmente la clase mayoritaria,
    logrando alta accuracy pero sin capacidad real de predicción.
    
    Criterios de evaluación:
    ----------------------
    1. Cohen's Kappa < 0.1 (casi 0)
    2. Balanced Accuracy < 0.52 (cercano a 0.5)
    3. P-value binomial > 0.05 (no significativo para clase minoritaria)
    4. F1-Score de clase minoritaria < 0.1
    
    Si cumple ≥2 criterios → Clasificado como INGENUO
    
    Parameters
    ----------
    y_true : array-like
        Etiquetas verdaderas (valores 0 y 1)
    y_pred : array-like
        Predicciones del modelo (valores 0 y 1)
    model_name : str, optional
        Nombre del modelo (default: "Model")
    dataset : str, optional
        Nombre del dataset (default: "")
    target : str, optional
        Nombre de la variable target (default: "")
    
    Returns
    -------
    dict
        Diccionario con todas las métricas calculadas y el veredicto
        None si hay error en el procesamiento
    
    Examples
    --------
    >>> y_test = np.array([0, 1, 1, 0, 1, 1, 0])
    >>> y_pred = np.array([1, 1, 1, 1, 1, 0, 1])
    >>> result = evaluate_model_vs_naive(y_test, y_pred, model_name="XGBoost")
    >>> print(f"Veredicto: {result['Verdict']}")
    >>> print(f"Balanced Accuracy: {result['Balanced_Accuracy']:.4f}")
    """
    
    # Validación inicial
    if y_true is None or y_pred is None:
        return None
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ajustar tamaños si no coinciden
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Eliminar NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return None
    
    # Calcular prevalencia de cada clase
    class_0_count = np.sum(y_true == 0)
    class_1_count = np.sum(y_true == 1)
    total = len(y_true)
    prevalence_0 = class_0_count / total
    prevalence_1 = class_1_count / total
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Métricas básicas
    accuracy = (y_pred == y_true).mean()
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Métricas por clase
    precision_1 = precision_score(y_true, y_pred, zero_division=0)
    recall_1 = recall_score(y_true, y_pred, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, zero_division=0)
    
    precision_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # Sensibilidad y especificidad
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = 0
        specificity = 0
    
    # Test binomial para clase minoritaria
    minority_class = 0 if class_0_count < class_1_count else 1
    minority_mask = y_true == minority_class
    minority_correct = (y_pred[minority_mask] == y_true[minority_mask]).sum()
    minority_total = minority_mask.sum()
    
    p_value = binomtest(
        minority_correct, 
        n=minority_total, 
        p=0.0, 
        alternative='greater'
    ).pvalue if minority_total > 0 else 1.0
    
    # Criterios para clasificar como ingenuo
    is_naive_kappa = kappa < 0.1
    is_naive_balanced = balanced_acc < 0.52
    is_naive_pvalue = p_value > 0.05
    is_naive_f1_minority = (f1_0 < 0.1) if minority_class == 0 else (f1_1 < 0.1)
    
    # Clasificación final: es ingenuo si cumple 2 o más criterios
    naive_count = sum([is_naive_kappa, is_naive_balanced, 
                       is_naive_pvalue, is_naive_f1_minority])
    is_naive = naive_count >= 2
    
    results = {
        'Dataset': dataset,
        'Target': target,
        'Model': model_name,
        'N_samples': int(total),
        'Prevalence_Class_0': round(prevalence_0, 4),
        'Prevalence_Class_1': round(prevalence_1, 4),
        'Accuracy': round(accuracy, 4),
        'Balanced_Accuracy': round(balanced_acc, 4),
        'Cohen_Kappa': round(kappa, 4),
        'Sensitivity': round(sensitivity, 4),
        'Specificity': round(specificity, 4),
        'Precision_Class_0': round(precision_0, 4),
        'Recall_Class_0': round(recall_0, 4),
        'F1_Class_0': round(f1_0, 4),
        'Precision_Class_1': round(precision_1, 4),
        'Recall_Class_1': round(recall_1, 4),
        'F1_Class_1': round(f1_1, 4),
        'Minority_Class': int(minority_class),
        'Minority_Correct': int(minority_correct),
        'Minority_Total': int(minority_total),
        'P_Value_Binomial': round(p_value, 6),
        'Naive_Criteria_Count': int(naive_count),
        'Is_Naive': is_naive,
        'Verdict': 'INGENUO' if is_naive else 'REAL'
    }
    
    return results


def evaluate_all_models(results_dict, dataset="", target=""):
    """
    Evalúa múltiples modelos y devuelve un DataFrame con los resultados.
    
    Parameters
    ----------
    results_dict : dict
        Diccionario con estructura {model_name: (y_true, y_pred)}
    dataset : str, optional
        Nombre del dataset (default: "")
    target : str, optional
        Nombre de la variable target (default: "")
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con todas las métricas para todos los modelos,
        ordenado por Balanced_Accuracy descendente
    
    Examples
    --------
    >>> results = {
    ...     'XGBoost': (y_test, y_pred_xgb),
    ...     'RandomForest': (y_test, y_pred_rf),
    ...     'LightGBM': (y_test, y_pred_lgb)
    ... }
    >>> df = evaluate_all_models(results, dataset="IBEX35", target="direction_next")
    >>> print(df[['Model', 'Verdict', 'Balanced_Accuracy', 'Cohen_Kappa']])
    >>> 
    >>> # Filtrar solo modelos reales
    >>> real_models = df[df['Verdict'] == 'REAL']
    >>> print(f"Modelos reales: {len(real_models)}")
    """
    
    all_results = []
    
    for model_name, (y_true, y_pred) in results_dict.items():
        result = evaluate_model_vs_naive(y_true, y_pred, model_name, dataset, target)
        if result is not None:
            all_results.append(result)
    
    if len(all_results) == 0:
        print("ADVERTENCIA: No se pudo evaluar ningún modelo")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('Balanced_Accuracy', ascending=False)
    
    return df_results


def print_evaluation_summary(df_results):
    """
    Imprime un resumen de la evaluación de modelos.
    
    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame retornado por evaluate_all_models()
    
    Examples
    --------
    >>> df = evaluate_all_models(results_dict)
    >>> print_evaluation_summary(df)
    """
    
    if len(df_results) == 0:
        print("No hay resultados para mostrar")
        return
    
    print("=" * 80)
    print("RESUMEN DE EVALUACIÓN - MODELOS REALES VS INGENUOS")
    print("=" * 80)
    
    total = len(df_results)
    reales = (df_results['Verdict'] == 'REAL').sum()
    ingenuos = (df_results['Verdict'] == 'INGENUO').sum()
    
    print(f"\nTotal modelos evaluados: {total}")
    print(f"Modelos REALES: {reales} ({reales/total*100:.1f}%)")
    print(f"Modelos INGENUOS: {ingenuos} ({ingenuos/total*100:.1f}%)")
    
    # Resumen por modelo
    print("\n" + "-" * 80)
    print("RESUMEN POR TIPO DE MODELO")
    print("-" * 80)
    summary = df_results.groupby(['Model', 'Verdict']).size().unstack(fill_value=0)
    print(summary)
    
    # Top modelos reales
    print("\n" + "-" * 80)
    print("TOP 5 MODELOS REALES (ordenados por Balanced Accuracy)")
    print("-" * 80)
    
    real_models = df_results[df_results['Verdict'] == 'REAL']
    if len(real_models) > 0:
        top_real = real_models.head(5)[['Model', 'Target', 'Accuracy', 
                                        'Balanced_Accuracy', 'Cohen_Kappa', 
                                        'F1_Class_0', 'F1_Class_1']]
        print(top_real.to_string(index=False))
    else:
        print("No se encontraron modelos clasificados como REALES")
    
    print("\n" + "=" * 80)

'''
# Ejemplo de uso
if __name__ == "__main__":
    print("Módulo de evaluación de modelos: REALES vs INGENUOS")
    print("\nEjemplo de uso:")
    print("""
    from evaluate_naive_models import evaluate_model_vs_naive, evaluate_all_models
    
    # Evaluar un solo modelo
    result = evaluate_model_vs_naive(y_test, y_pred, model_name="XGBoost")
    print(f"Veredicto: {result['Verdict']}")
    
    # Evaluar múltiples modelos
    results_dict = {
        'XGBoost': (y_test, y_pred_xgb),
        'LightGBM': (y_test, y_pred_lgb),
        'RandomForest': (y_test, y_pred_rf)
    }
    
    df = evaluate_all_models(results_dict, dataset="IBEX35", target="direction_next")
    
    # Mostrar resumen
    print_evaluation_summary(df)
    
    # Filtrar solo modelos reales
    real_models = df[df['Verdict'] == 'REAL']
    print(f"\\nModelos reales: {len(real_models)}")
    
    # Guardar resultados
    df.to_csv('evaluation_results.csv', index=False)
    """)
'''