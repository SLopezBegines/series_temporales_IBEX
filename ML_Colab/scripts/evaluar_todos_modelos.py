"""
═══════════════════════════════════════════════════════════════════════════════
CHUNK DE CÓDIGO: EVALUACIÓN DE MODELOS REALES vs INGENUOS
═══════════════════════════════════════════════════════════════════════════════

Este código procesa tus archivos predictions_classification.csv y 
predictions_regression.csv para identificar modelos reales vs ingenuos.

Autor: Santi - TFM IBEX35
Uso: Simplemente ejecuta este script en el mismo directorio que tus archivos CSV
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import re
from evaluate_naive_models import evaluate_all_models, print_evaluation_summary
from config import CSV  # Importar ruta CSV desde config

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

# Archivos de entrada
FILE_CLASSIFICATION = 'predictions_classification.csv'
FILE_REGRESSION = 'predictions_regression.csv'

# Datasets a procesar
DATASETS_TO_PROCESS = [
    'financial_scaled',
    'financial_unscaled',
    'financial_long_scaled',
    'financial_long_unscaled',
    'sentiment_scaled',
    'sentiment_unscaled'
]

# Targets de clasificación
TARGETS_CLASSIFICATION = [
    'direction_next',
    'direction_next_5',
    'direction_next_10',
    'direction_next_20'
]

# Targets de regresión
TARGETS_REGRESSION = [
    'returns_next',
    'returns_next_5',
    'returns_next_10',
    'returns_next_20'
]

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def parse_numpy_array(s):
    """Parsear string de array numpy a numpy array"""
    if pd.isna(s):
        return None
    
    s = str(s).strip()
    s = re.sub(r'[\[\]]', '', s)
    
    try:
        values = []
        for val in s.split():
            if val.lower() == 'nan':
                continue
            try:
                values.append(float(val))
            except:
                continue
        return np.array(values) if len(values) > 0 else None
    except:
        return None


def regression_to_classification(y_true_reg, y_pred_reg):
    """
    Convierte predicciones de regresión a clasificación binaria
    1 = subida (return > 0), 0 = bajada (return <= 0)
    """
    if y_true_reg is None or y_pred_reg is None:
        return None, None
    
    # Ajustar tamaños
    min_len = min(len(y_true_reg), len(y_pred_reg))
    y_true_reg = y_true_reg[:min_len]
    y_pred_reg = y_pred_reg[:min_len]
    
    # Eliminar NaN
    mask = ~(np.isnan(y_true_reg) | np.isnan(y_pred_reg))
    y_true_reg = y_true_reg[mask]
    y_pred_reg = y_pred_reg[mask]
    
    if len(y_true_reg) == 0:
        return None, None
    
    # Convertir a clasificación
    y_true_class = (y_true_reg > 0).astype(int)
    y_pred_class = (y_pred_reg > 0).astype(int)
    
    return y_true_class, y_pred_class


def process_predictions(df_pred, targets, is_regression=False):
    """
    Procesa predicciones y evalúa modelos
    
    Parámetros:
    -----------
    df_pred : DataFrame con columnas [dataset, target, model, y_test, predictions]
    targets : lista de targets a procesar
    is_regression : bool, True si son predicciones de regresión
    
    Retorna:
    --------
    DataFrame con evaluación de todos los modelos
    """
    all_results = []
    
    for dataset_name in DATASETS_TO_PROCESS:
        for target_name in targets:
            
            # Filtrar datos
            subset = df_pred[
                (df_pred['dataset'] == dataset_name) & 
                (df_pred['target'] == target_name)
            ]
            
            if len(subset) == 0:
                continue
            
            print(f"\nProcesando: {dataset_name} - {target_name} ({len(subset)} modelos)")
            
            # Crear diccionario para evaluate_all_models
            models_dict = {}
            
            for idx, row in subset.iterrows():
                # Parsear arrays
                y_true = parse_numpy_array(row['y_test'])
                y_pred = parse_numpy_array(row['predictions'])
                
                # Si es regresión, convertir a clasificación
                if is_regression:
                    y_true, y_pred = regression_to_classification(y_true, y_pred)
                else:
                    # Para clasificación, ajustar tamaños si es necesario
                    if y_true is not None and y_pred is not None:
                        min_len = min(len(y_true), len(y_pred))
                        y_true = y_true[:min_len]
                        y_pred = y_pred[:min_len]
                        
                        # Eliminar NaN
                        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                        y_true = y_true[mask]
                        y_pred = y_pred[mask]
                
                if y_true is not None and y_pred is not None and len(y_true) > 0:
                    model_name = row['model']
                    models_dict[model_name] = (y_true, y_pred)
            
            # Evaluar modelos
            if len(models_dict) > 0:
                df_eval = evaluate_all_models(
                    models_dict,
                    dataset=dataset_name,
                    target=target_name
                )
                
                if len(df_eval) > 0:
                    all_results.append(df_eval)
                    
                    # Mostrar progreso
                    reales = (df_eval['Verdict'] == 'REAL').sum()
                    print(f"  ✓ Modelos REALES: {reales}/{len(df_eval)} ({reales/len(df_eval)*100:.1f}%)")
    
    # Consolidar resultados
    if len(all_results) > 0:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def print_detailed_summary(df_results, title):
    """Imprime resumen detallado de resultados"""
    
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    
    if len(df_results) == 0:
        print("No hay resultados para mostrar")
        return
    
    print(f"\nTotal modelos evaluados: {len(df_results)}")
    reales = (df_results['Verdict'] == 'REAL').sum()
    ingenuos = (df_results['Verdict'] == 'INGENUO').sum()
    print(f"Modelos REALES: {reales} ({reales/len(df_results)*100:.1f}%)")
    print(f"Modelos INGENUOS: {ingenuos} ({ingenuos/len(df_results)*100:.1f}%)")
    
    # Por modelo
    print("\n" + "-"*80)
    print("POR TIPO DE MODELO")
    print("-"*80)
    summary = df_results.groupby(['Model', 'Verdict']).size().unstack(fill_value=0)
    if 'REAL' in summary.columns:
        summary['Total'] = summary.sum(axis=1)
        summary['% Real'] = (summary['REAL'] / summary['Total'] * 100).round(1)
        summary = summary.sort_values('% Real', ascending=False)
    print(summary)
    
    # Por dataset
    print("\n" + "-"*80)
    print("POR DATASET")
    print("-"*80)
    summary_ds = df_results.groupby(['Dataset', 'Verdict']).size().unstack(fill_value=0)
    if 'REAL' in summary_ds.columns:
        summary_ds['Total'] = summary_ds.sum(axis=1)
        summary_ds['% Real'] = (summary_ds['REAL'] / summary_ds['Total'] * 100).round(1)
        summary_ds = summary_ds.sort_values('% Real', ascending=False)
    print(summary_ds)
    
    # Por target
    print("\n" + "-"*80)
    print("POR TARGET")
    print("-"*80)
    summary_tg = df_results.groupby(['Target', 'Verdict']).size().unstack(fill_value=0)
    if 'REAL' in summary_tg.columns:
        summary_tg['Total'] = summary_tg.sum(axis=1)
        summary_tg['% Real'] = (summary_tg['REAL'] / summary_tg['Total'] * 100).round(1)
        summary_tg = summary_tg.sort_values('% Real', ascending=False)
    print(summary_tg)
    
    # Top 10
    print("\n" + "-"*80)
    print("TOP 10 MODELOS REALES")
    print("-"*80)
    reales_df = df_results[df_results['Verdict'] == 'REAL']
    if len(reales_df) > 0:
        top10 = reales_df.head(10)[['Model', 'Dataset', 'Target', 
                                     'Balanced_Accuracy', 'Cohen_Kappa']]
        for idx, row in top10.iterrows():
            print(f"\n{idx+1}. {row['Model']} - {row['Dataset']} - {row['Target']}")
            print(f"   Balanced Accuracy: {row['Balanced_Accuracy']:.4f}")
            print(f"   Cohen's Kappa: {row['Cohen_Kappa']:.4f}")
    else:
        print("No se encontraron modelos REALES")

# ==============================================================================
# PROCESAMIENTO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("EVALUACIÓN DE MODELOS - TFM IBEX35")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 1. PROCESAR MODELOS DE CLASIFICACIÓN
    # -------------------------------------------------------------------------
    
    try:
        print("\n[1/2] Cargando predicciones de CLASIFICACIÓN...")
        df_class = pd.read_csv(CSV / FILE_CLASSIFICATION, quotechar='"')
        print(f"✓ Cargadas {len(df_class)} predicciones")
        
        print("\nEvaluando modelos de clasificación...")
        df_class_results = process_predictions(df_class, TARGETS_CLASSIFICATION, is_regression=False)
        
        if len(df_class_results) > 0:
            # Guardar resultados
            df_class_results.to_csv(CSV / 'evaluacion_clasificacion.csv', index=False, float_format='%.10f')
            df_class_results.to_parquet(CSV / 'evaluacion_clasificacion.parquet', index=False, engine='pyarrow')
            print(f"\n✓ Resultados guardados en: evaluacion_clasificacion.csv")
            
            # Mostrar resumen
            print_detailed_summary(df_class_results, "RESUMEN: MODELOS DE CLASIFICACIÓN")
        
    except FileNotFoundError:
        print(f"⚠ Archivo {FILE_CLASSIFICATION} no encontrado")
        df_class_results = pd.DataFrame()
    except Exception as e:
        print(f"⚠ Error procesando clasificación: {e}")
        df_class_results = pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # 2. PROCESAR MODELOS DE REGRESIÓN
    # -------------------------------------------------------------------------
    
    try:
        print("\n[2/2] Cargando predicciones de REGRESIÓN...")
        df_reg = pd.read_csv(CSV / FILE_REGRESSION, quotechar='"')
        print(f"✓ Cargadas {len(df_reg)} predicciones")
        
        print("\nEvaluando modelos de regresión (convertidos a clasificación)...")
        df_reg_results = process_predictions(df_reg, TARGETS_REGRESSION, is_regression=True)
        
        if len(df_reg_results) > 0:
            # Guardar resultados
            df_reg_results.to_csv(CSV / 'evaluacion_regresion.csv', index=False, float_format='%.10f')
            df_reg_results.to_parquet(CSV / 'evaluacion_regresion.parquet', index=False, engine='pyarrow')
            print(f"\n✓ Resultados guardados en: evaluacion_regresion.csv")
            
            # Mostrar resumen
            print_detailed_summary(df_reg_results, "RESUMEN: MODELOS DE REGRESIÓN")
        
    except FileNotFoundError:
        print(f"⚠ Archivo {FILE_REGRESSION} no encontrado")
        df_reg_results = pd.DataFrame()
    except Exception as e:
        print(f"⚠ Error procesando regresión: {e}")
        df_reg_results = pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # 3. COMPARACIÓN FINAL
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("COMPARACIÓN: CLASIFICACIÓN vs REGRESIÓN")
    print("="*80)
    
    if len(df_class_results) > 0 and len(df_reg_results) > 0:
        
        class_reales = (df_class_results['Verdict'] == 'REAL').sum()
        class_total = len(df_class_results)
        
        reg_reales = (df_reg_results['Verdict'] == 'REAL').sum()
        reg_total = len(df_reg_results)
        
        print(f"\nCLASIFICACIÓN:")
        print(f"  Total modelos: {class_total}")
        print(f"  Modelos REALES: {class_reales} ({class_reales/class_total*100:.1f}%)")
        
        print(f"\nREGRESIÓN:")
        print(f"  Total modelos: {reg_total}")
        print(f"  Modelos REALES: {reg_reales} ({reg_reales/reg_total*100:.1f}%)")
        
        print(f"\n{'='*80}")
        print("CONCLUSIÓN:")
        print("="*80)
        
        if reg_reales/reg_total > class_reales/class_total:
            diff = (reg_reales/reg_total - class_reales/class_total) * 100
            print(f"✓ Los modelos de REGRESIÓN tienen {diff:.1f}% MÁS capacidad predictiva")
            print(f"  que los modelos de CLASIFICACIÓN directa.")
        elif class_reales/class_total > reg_reales/reg_total:
            diff = (class_reales/class_total - reg_reales/reg_total) * 100
            print(f"✓ Los modelos de CLASIFICACIÓN tienen {diff:.1f}% MÁS capacidad predictiva")
            print(f"  que los modelos de REGRESIÓN convertida.")
        else:
            print("≈ Ambos enfoques tienen capacidad predictiva similar")
        
        # Mejor modelo global
        print(f"\n{'='*80}")
        print("MEJOR MODELO GLOBAL")
        print("="*80)
        
        df_all = pd.concat([df_class_results, df_reg_results], ignore_index=True)
        df_all_reales = df_all[df_all['Verdict'] == 'REAL'].sort_values('Balanced_Accuracy', ascending=False)
        
        if len(df_all_reales) > 0:
            best = df_all_reales.iloc[0]
            print(f"\nModelo: {best['Model']}")
            print(f"Dataset: {best['Dataset']}")
            print(f"Target: {best['Target']}")
            print(f"Balanced Accuracy: {best['Balanced_Accuracy']:.4f}")
            print(f"Cohen's Kappa: {best['Cohen_Kappa']:.4f}")
            print(f"F1 Clase 0: {best['F1_Class_0']:.4f}")
            print(f"F1 Clase 1: {best['F1_Class_1']:.4f}")
            
            # Guardar resultados consolidados
            df_all.to_csv(CSV / 'evaluacion_completa.csv', index=False, float_format='%.10f')
            df_all.to_parquet(CSV / 'evaluacion_completa.parquet', index=False, engine='pyarrow')
            print(f"\n✓ Resultados consolidados guardados en: evaluacion_completa.csv")
    
    print("\n" + "="*80)
    print("PROCESO COMPLETADO")
    print("="*80)
    print("\nArchivos generados:")
    print("  1. evaluacion_clasificacion.csv - Resultados de clasificación")
    print("  2. evaluacion_regresion.csv - Resultados de regresión")
    print("  3. evaluacion_completa.csv - Todos los resultados consolidados")
    print("\n" + "="*80)
