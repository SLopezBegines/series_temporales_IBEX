#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:59:49 2025

@author: santi
"""

"""
Analiza el log y genera lista de modelos faltantes
"""
import pandas as pd
import re
from config import CSV, MODELS
from pathlib import Path

def parse_log_completed_models(log_file):
    """Extrae modelos completados del log"""
    completed = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Buscar líneas que indican modelo completado
            if '✓' in line and 'completado' in line:
                # Extraer info del modelo
                match = re.search(r'(\w+)\s+completado', line)
                if match:
                    model = match.group(1)
                    # Buscar dataset y target en contexto previo
                    completed.append(model)
    
    return completed

def get_expected_models():
    """Genera lista completa de modelos esperados"""
    datasets = [
        'financial_scaled', 'financial_unscaled',
        'financial_long_scaled', 'financial_long_unscaled',
        'sentiment_scaled', 'sentiment_unscaled'
    ]
    
    targets_reg = ['returns_next', 'returns_next_5', 'returns_next_10', 'returns_next_20']
    targets_class = ['direction_next', 'direction_next_5', 'direction_next_10', 'direction_next_20']
    
    models_scaled = ['Linear', 'Ridge', 'Random Forest', 'XGBoost', 'LightGBM', 
                     'MLP', 'Ensemble', 'GRU', 'LSTM']
    models_unscaled = ['Random Forest', 'XGBoost', 'LightGBM', 'Ensemble', 'GRU', 'LSTM']
    models_class_scaled = ['Logistic', 'Random Forest', 'XGBoost', 'LightGBM', 
                           'MLP', 'Ensemble', 'GRU', 'LSTM']
    models_class_unscaled = ['Random Forest', 'XGBoost', 'LightGBM', 'Ensemble', 'GRU', 'LSTM']
    
    expected = []
    
    for dataset in datasets:
        is_scaled = 'scaled' in dataset and 'unscaled' not in dataset
        
        # Regresión
        for target in targets_reg:
            models = models_scaled if is_scaled else models_unscaled
            for model in models:
                expected.append({
                    'dataset': dataset,
                    'target': target,
                    'model': model,
                    'type': 'regression'
                })
        
        # Clasificación
        for target in targets_class:
            models = models_class_scaled if is_scaled else models_class_unscaled
            for model in models:
                expected.append({
                    'dataset': dataset,
                    'target': target,
                    'model': model,
                    'type': 'classification'
                })
    
    return pd.DataFrame(expected)

def check_model_exists(dataset, target, model):
    """Verifica si existe el modelo guardado"""
    model_dir = MODELS / dataset / target
    
    # Normalizar nombres de modelos
    model_file_map = {
        'Linear': 'Linear.joblib',
        'Ridge': 'Ridge.joblib',
        'Random Forest': 'Random_Forest.joblib',
        'XGBoost': 'XGBoost.joblib',
        'LightGBM': 'LightGBM.joblib',
        'MLP': 'MLP.joblib',
        'Logistic': 'Logistic.joblib',
        'Ensemble': 'Ensemble.joblib',
        'GRU': 'GRU_medium.keras',
        'LSTM': 'LSTM_medium.keras'
    }
    
    model_file = model_file_map.get(model)
    if not model_file:
        return False
    
    return (model_dir / model_file).exists()

def find_missing_models(log_file):
    """Identifica modelos faltantes"""
    print("Analizando modelos esperados vs completados...")
    
    # Obtener modelos esperados
    df_expected = get_expected_models()
    print(f"\nTotal modelos esperados: {len(df_expected)}")
    
    # Verificar cuáles existen
    df_expected['exists'] = df_expected.apply(
        lambda row: check_model_exists(row['dataset'], row['target'], row['model']),
        axis=1
    )
    
    # Modelos faltantes
    df_missing = df_expected[~df_expected['exists']].copy()
    
    print(f"Modelos completados: {df_expected['exists'].sum()}")
    print(f"Modelos faltantes: {len(df_missing)}")
    
    # Resumen por tipo
    print("\n" + "="*80)
    print("RESUMEN POR TIPO")
    print("="*80)
    summary = df_missing.groupby(['type', 'model']).size().unstack(fill_value=0)
    print(summary)
    
    # Resumen por dataset
    print("\n" + "="*80)
    print("RESUMEN POR DATASET")
    print("="*80)
    summary_ds = df_missing.groupby('dataset').size().sort_values(ascending=False)
    print(summary_ds)
    
    # Guardar lista de faltantes
    output_file = CSV / 'modelos_faltantes.csv'
    df_missing.to_csv(output_file, index=False)
    print(f"\n✓ Lista guardada en: {output_file}")
    
    return df_missing

if __name__ == "__main__":
    log_file = CSV / 'logs' / 'pipeline_20251130_175630.log'
    df_missing = find_missing_models(log_file)
    
    # Mostrar primeros faltantes
    print("\n" + "="*80)
    print("PRIMEROS 20 MODELOS FALTANTES")
    print("="*80)
    print(df_missing.head(20)[['dataset', 'target', 'model', 'type']])