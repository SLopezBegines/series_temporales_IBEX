#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelos LSTM para Series Temporales Financieras
Proyecto TFM IBEX35

Funciones especializadas para entrenar LSTM en predicción de mercados financieros
con soporte para regresión (returns) y clasificación (direction)

@author: santi
"""

# ============================================================================
# IMPORTS
# ============================================================================

import time
import numpy as np

# Deep Learning - TensorFlow/Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import config
from config import RANDOM_STATE, MODELS, PLOTS

# Import aux_functions
from aux_functions import (
    create_sequences,
    evaluate_regression,
    evaluate_classification,
    save_model,
    save_training_history,
    get_best_epoch_info
)
from modelos_ml import save_predictions
from visualization import plot_training_history
# ============================================================================
# ARQUITECTURAS LSTM
# ============================================================================

def get_lstm_architecture(input_shape, model_size='medium', task_type='regression'):
    """
    Devuelve arquitectura LSTM según tamaño y tipo de tarea
    
    Args:
        input_shape: (lookback, n_features)
        model_size: 'small', 'medium', 'large', 'xlarge'
        task_type: 'regression' o 'classification'
    
    Returns:
        model: Modelo compilado
        epochs_max: Número máximo de epochs
        batch_size: Tamaño de batch recomendado
    """
    
    lookback, n_features = input_shape
    
    # Arquitecturas según tamaño
    if model_size == 'small':
        # Simple LSTM - Rápido para pruebas
        model = Sequential([
            LSTM(32, activation='tanh', input_shape=input_shape),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid' if task_type == 'classification' else 'linear')
        ])
        epochs_max = 50
        batch_size = 16
        
    elif model_size == 'medium':
        # LSTM apilado - Balance velocidad/performance
        model = Sequential([
            LSTM(50, activation='tanh', return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, activation='tanh'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid' if task_type == 'classification' else 'linear')
        ])
        epochs_max = 100
        batch_size = 32
        
    elif model_size == 'large':
        # LSTM profundo - Mejor performance
        model = Sequential([
            LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='tanh'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid' if task_type == 'classification' else 'linear')
        ])
        epochs_max = 150
        batch_size = 32
        
    elif model_size == 'xlarge':
        # LSTM muy profundo - Máxima capacidad
        model = Sequential([
            LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='tanh'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid' if task_type == 'classification' else 'linear')
        ])
        epochs_max = 200
        batch_size = 64
        
    elif model_size == 'bidirectional':
        # Bidirectional LSTM - Contexto pasado y futuro
        model = Sequential([
            Bidirectional(LSTM(50, activation='tanh', return_sequences=True), 
                         input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(32, activation='tanh')),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid' if task_type == 'classification' else 'linear')
        ])
        epochs_max = 100
        batch_size = 32
    
    else:
        raise ValueError(f"model_size '{model_size}' no reconocido. Usa: small, medium, large, xlarge, bidirectional")
    
    # Compilar modelo
    if task_type == 'regression':
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    else:  # classification
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return model, epochs_max, batch_size

# ============================================================================
# ENTRENAMIENTO LSTM - REGRESIÓN
# ============================================================================

def train_lstm_regression(X_train, y_train, X_test, y_test,
                         dataset_name, target_name,
                         lookback=20, model_size='medium',
                         use_checkpoint=True, verbose=1):
    """
    Entrena LSTM para regresión (predicción de returns)
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de test
        dataset_name: Nombre del dataset
        target_name: Nombre del target
        lookback: Ventana temporal (días anteriores)
        model_size: 'small', 'medium', 'large', 'xlarge', 'bidirectional'
        use_checkpoint: Si guardar mejor modelo durante entrenamiento
        verbose: Nivel de verbosidad (0=silencioso, 1=progreso, 2=detallado)
    
    Returns:
        results: Lista con métricas [dict]
        models: Dict con modelo entrenado {'LSTM': model}
        history: Historial de entrenamiento
    """
    
    print(f"→ LSTM {model_size} (lookback={lookback})")
    print("  Creando secuencias temporales...")
    start_time = time.time()
    
    # Crear secuencias
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, lookback)
    
    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Test sequences: {X_test_seq.shape}")
    
    # Crear modelo
    input_shape = (lookback, X_train.shape[1])
    model, epochs_max, batch_size = get_lstm_architecture(
        input_shape, model_size, task_type='regression'
    )
    
    print(f"  Parámetros: {model.count_params():,}")
    print(f"  Ratio samples/params: {X_train_seq.shape[0] / model.count_params():.3f}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    # ModelCheckpoint (opcional)
    if use_checkpoint:
        checkpoint_path = MODELS / dataset_name / target_name / f"LSTM_{model_size}_checkpoint.keras"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        )
    
    # Entrenar
    print(f"  Entrenando (max {epochs_max} epochs)...")
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs_max,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=verbose
    )
    
    elapsed = time.time() - start_time
    print(f"  Epochs entrenados: {len(history.history['loss'])}")
    print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
    print(f"  Tiempo: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    
    # Predecir
    print("  Generando predicciones...")
    y_pred = model.predict(X_test_seq, verbose=0).flatten()
    
    # Evaluar
    result = evaluate_regression(y_test_seq, y_pred, model_name=f"LSTM_{model_size}")
    result['train_time'] = elapsed
    result['epochs'] = len(history.history['loss'])
    result['lookback'] = lookback
    result['model_size'] = model_size
    result['n_params'] = model.count_params()
    
    # Guardar modelo
    save_model(model, f'LSTM_{model_size}', dataset_name, target_name, 'tensorflow')
    # Guardar predicciones
    save_predictions(dataset_name, target_name, f'LSTM_{model_size}', y_test_seq, y_pred)
    
    # Guardar historial de entrenamiento
    save_training_history(
        history=history,
        model_name=f'LSTM_{model_size}',
        dataset_name=dataset_name,
        target_name=target_name,
        save_path=MODELS / 'history',
        save_format='pickle'
    )
    
    # Plot historial
    plot_training_history(
        history=history,
        model_name=f'LSTM_{model_size}',
        dataset_name=dataset_name,
        target_name=target_name,
        save_path=PLOTS,
        task_type='regression',
        show_plot=False
    )
    
    # Agregar info del mejor epoch
    best_info = get_best_epoch_info(history)
    result['best_epoch'] = best_info['best_epoch']
    result['best_val_loss'] = best_info['best_val_loss']
    
    return [result], {f'LSTM_{model_size}': model}, history.history

# ============================================================================
# ENTRENAMIENTO LSTM - CLASIFICACIÓN
# ============================================================================

def train_lstm_classification(X_train, y_train, X_test, y_test,
                              dataset_name, target_name,
                              lookback=20, model_size='medium',
                              use_checkpoint=True, verbose=1):
    """
    Entrena LSTM para clasificación (predicción de dirección)
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de test
        dataset_name: Nombre del dataset
        target_name: Nombre del target
        lookback: Ventana temporal (días anteriores)
        model_size: 'small', 'medium', 'large', 'xlarge', 'bidirectional'
        use_checkpoint: Si guardar mejor modelo durante entrenamiento
        verbose: Nivel de verbosidad
    
    Returns:
        results: Lista con métricas [dict]
        predictions: Dict con predicciones {model_name: pred}
        models: Dict con modelo entrenado {model_name: model}
        history: Historial de entrenamiento
    """
    
    print(f"→ LSTM {model_size} (lookback={lookback})")
    print("  Creando secuencias temporales...")
    start_time = time.time()
    
    # Crear secuencias
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, lookback)
    
    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Test sequences: {X_test_seq.shape}")
    
    # Crear modelo
    input_shape = (lookback, X_train.shape[1])
    model, epochs_max, batch_size = get_lstm_architecture(
        input_shape, model_size, task_type='classification'
    )
    
    print(f"  Parámetros: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    # ModelCheckpoint
    if use_checkpoint:
        checkpoint_path = MODELS / dataset_name / target_name / f"LSTM_{model_size}_checkpoint.keras"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        )
    
    # Entrenar
    print(f"  Entrenando (max {epochs_max} epochs)...")
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs_max,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=verbose
    )
    
    elapsed = time.time() - start_time
    print(f"  Epochs: {len(history.history['loss'])}")
    print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
    print(f"  Tiempo: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    
    # Predecir
    print("  Generando predicciones...")
    proba = model.predict(X_test_seq, verbose=0).flatten()
    pred = (proba > 0.5).astype(int)
    
    # Evaluar
    result = evaluate_classification(y_test_seq, pred, proba, model_name=f"LSTM_{model_size}")
    result['train_time'] = elapsed
    result['epochs'] = len(history.history['loss'])
    result['lookback'] = lookback
    result['model_size'] = model_size
    result['n_params'] = model.count_params()
    
    # Guardar modelo
    save_model(model, f'LSTM_{model_size}', dataset_name, target_name, 'tensorflow')
    
    # Guargar predicciones
    save_predictions(dataset_name, target_name, f'LSTM_{model_size}', y_test_seq, pred)
    
    # Guardar historial
    save_training_history(
        history=history,
        model_name=f'LSTM_{model_size}',
        dataset_name=dataset_name,
        target_name=target_name,
        save_path=MODELS / 'history',
        save_format='pickle'
    )
    
    # Plot historial
    plot_training_history(
        history=history,
        model_name=f'LSTM_{model_size}',
        dataset_name=dataset_name,
        target_name=target_name,
        save_path=PLOTS,
        task_type='classification',
        show_plot=False
    )
    
    # Info del mejor epoch
    best_info = get_best_epoch_info(history)
    result['best_epoch'] = best_info['best_epoch']
    result['best_val_loss'] = best_info['best_val_loss']
    
    return [result], {f'LSTM_{model_size}': pred}, {f'LSTM_{model_size}': model}, history.history

# ============================================================================
# ENTRENAMIENTO MÚLTIPLES CONFIGURACIONES
# ============================================================================

def train_lstm_suite(X_train, y_train, X_test, y_test,
                    dataset_name, target_name, task_type='regression',
                    model_sizes=['small', 'medium', 'large'],
                    lookbacks=[10, 20, 30],
                    verbose=1):
    """
    Entrena múltiples configuraciones de LSTM
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de test
        dataset_name: Nombre del dataset
        target_name: Nombre del target
        task_type: 'regression' o 'classification'
        model_sizes: Lista de tamaños a probar
        lookbacks: Lista de ventanas temporales a probar
        verbose: Nivel de verbosidad
    
    Returns:
        results: Lista con todos los resultados
        best_config: Mejor configuración encontrada
    """
    
    print(f"\n{'='*70}")
    print(f"LSTM SUITE - {task_type.upper()}")
    print(f"Dataset: {dataset_name} | Target: {target_name}")
    print(f"Configuraciones: {len(model_sizes)} tamaños × {len(lookbacks)} lookbacks = {len(model_sizes) * len(lookbacks)} modelos")
    print(f"{'='*70}")
    
    all_results = []
    
    for lookback in lookbacks:
        for model_size in model_sizes:
            print(f"\n{'-'*70}")
            print(f"Lookback: {lookback} | Size: {model_size}")
            print(f"{'-'*70}")
            
            try:
                if task_type == 'regression':
                    results, models, history = train_lstm_regression(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target_name,
                        lookback=lookback, model_size=model_size,
                        verbose=verbose
                    )
                else:  # classification
                    results, predictions, models, history = train_lstm_classification(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target_name,
                        lookback=lookback, model_size=model_size,
                        verbose=verbose
                    )
                
                all_results.extend(results)
                print(f"✓ Completado: {model_size} con lookback={lookback}")
                
            except Exception as e:
                print(f"❌ Error con {model_size} lookback={lookback}: {e}")
                continue
    
    # Encontrar mejor configuración
    if all_results:
        import pandas as pd
        results_df = pd.DataFrame(all_results)
        
        if task_type == 'regression':
            best_idx = results_df['RMSE'].idxmin()
            metric = 'RMSE'
        else:
            best_idx = results_df['Accuracy'].idxmax()
            metric = 'Accuracy'
        
        best_config = results_df.loc[best_idx]
        
        print(f"\n{'='*70}")
        print("MEJOR CONFIGURACIÓN")
        print(f"{'='*70}")
        print(f"Modelo: {best_config['model']}")
        print(f"Lookback: {best_config['lookback']}")
        print(f"Size: {best_config['model_size']}")
        print(f"{metric}: {best_config[metric]:.6f}")
        print(f"Epochs: {best_config['epochs']}")
        print(f"Params: {best_config['n_params']:,}")
        
        return all_results, best_config
    else:
        print("\n⚠ No se completó ningún modelo exitosamente")
        return [], None

# ============================================================================
# MENSAJE DE CONFIRMACIÓN
# ============================================================================

print("✓ Módulo lstm_models.py cargado")
print("  Funciones disponibles:")
print("    - train_lstm_regression()")
print("    - train_lstm_classification()")
print("    - train_lstm_suite()")
print("  Arquitecturas: small, medium, large, xlarge, bidirectional")
