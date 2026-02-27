#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funciones de entrenamiento de modelos ML/DL para proyecto IBEX35
@author: santi
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import time

# Machine Learning - Sklearn
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingClassifier, StackingRegressor 
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Machine Learning - Gradient Boosting
import xgboost as xgb
import lightgbm as lgb

# Deep Learning - TensorFlow/Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import config
from config import RANDOM_STATE, MODELS, PLOTS

# Import aux_functions
from aux_functions import (
    evaluate_regression,
    evaluate_classification,
    save_model,
    create_sequences,
    save_training_history,
    get_best_epoch_info
)
# Import visualization functions
from visualization import plot_training_history
# ============================================================================
# FUNCIONES AUXILIARES INTERNAS
# ============================================================================

def save_predictions(dataset_name, target_name, model_name, y_true, y_pred):
    """
    Guarda predicciones en formato CSV
    
    Args:
        dataset_name: Nombre del dataset
        target_name: Nombre del target
        model_name: Nombre del modelo
        y_true: Valores reales
        y_pred: Predicciones
    """
    import pandas as pd
    import numpy as np
    from config import CSV
    
    # Convertir a array si es necesario
    if hasattr(y_true, 'values'):  # Si es Series o DataFrame
        y_true = y_true.values
    elif isinstance(y_true, list):
        y_true = np.array(y_true)
    # Si ya es array, no hacer nada
    
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    elif isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'y_true': y_true.flatten() if y_true.ndim > 1 else y_true,
        'y_pred': y_pred.flatten() if y_pred.ndim > 1 else y_pred
    })
    
    # Guardar
    save_path = CSV / dataset_name / target_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    filepath = save_path / f"{model_name}_predictions.csv"
    df.to_csv(filepath, index=False, float_format='%.10f')
    filepath = save_path / f"{model_name}_predictions.parquet"
    df.to_parquet(CSV / filepath, index=False, engine='pyarrow')
    
    return filepath

# ============================================================================
# FUNCIONES DE ENTRENAMIENTO - REGRESIÓN
# ============================================================================
def train_linear_models(X_train, y_train, X_test, y_test,
                       dataset_name, target_name, use_scaled=True):
    """
    Entrena modelos lineales (solo con datos escalados)

    MODIFIED: Devuelve resultados Y modelos entrenados
    """
    if not use_scaled:
        return [], {}

    results = []
    models = {}

    print("→ Linear Regression")
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    save_predictions(dataset_name, target_name, "Linear", y_test, pred)
    results.append(evaluate_regression(y_test.values, pred, "Linear"))
    models['Linear'] = model
    # Guardar modelo
    save_model(model, 'Linear', dataset_name, target_name, 'sklearn')

    print("→ Ridge Regression")
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    save_predictions(dataset_name, target_name, "Ridge", y_test, pred)
    results.append(evaluate_regression(y_test.values, pred, "Ridge"))
    models['Ridge'] = model
    # Guardar modelo
    save_model(model, 'Ridge', dataset_name, target_name, 'sklearn')

    return results, models

def train_tree_models_regression(X_train, y_train, X_test, y_test,
                                 dataset_name, target_name):
    """
    Entrena modelos basados en árboles para regresión

    MODIFIED: Devuelve resultados Y modelos entrenados
    """
    results = []
    models = {}

    print("→ Random Forest")
    start_time = time.time()
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    save_predictions(dataset_name, target_name, "Random Forest", y_test, pred)
    elapsed = time.time() - start_time
    result = evaluate_regression(y_test.values, pred, "Random Forest")
    result['train_time'] = elapsed
    results.append(result)
    models['Random_Forest'] = model
    print(f"  Tiempo: {elapsed:.2f}s")
    save_model(model, 'Random_Forest', dataset_name, target_name, 'sklearn')

    print("→ XGBoost")
    start_time = time.time()
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    save_predictions(dataset_name, target_name, "XGBoost", y_test, pred)
    elapsed = time.time() - start_time
    result = evaluate_regression(y_test.values, pred, "XGBoost")
    result['train_time'] = elapsed
    results.append(result)
    models['XGBoost'] = model
    print(f"  Tiempo: {elapsed:.2f}s")
    save_model(model, 'XGBoost', dataset_name, target_name, 'xgboost')

    print("→ LightGBM")
    start_time = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    save_predictions(dataset_name, target_name, "LightGBM", y_test, pred)
    elapsed = time.time() - start_time
    result = evaluate_regression(y_test.values, pred, "LightGBM")
    result['train_time'] = elapsed
    results.append(result)
    models['LightGBM'] = model
    print(f"  Tiempo: {elapsed:.2f}s")
    save_model(model, 'LightGBM', dataset_name, target_name, 'lightgbm')

    return results, models

def train_mlp_regression(X_train, y_train, X_test, y_test,
                        dataset_name, target_name, use_scaled=True):
    """Entrena MLP para regresión"""
    if not use_scaled:
        return [], {}

    print("→ MLP (Multi-Layer Perceptron)")
    start_time = time.time()

    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=RANDOM_STATE,
        verbose=False
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    save_predictions(dataset_name, target_name, "MLP", y_test, pred)
    elapsed = time.time() - start_time

    result = evaluate_regression(y_test.values, pred, "MLP")
    result['train_time'] = elapsed
    print(f"  Tiempo: {elapsed:.2f}s")
    print(f"  Iterations: {model.n_iter_}")

    save_model(model, 'MLP', dataset_name, target_name, 'sklearn')

    return [result], {'MLP': model}

def train_gru_regression(X_train, y_train, X_test, y_test,
                        dataset_name, target_name,
                        lookback=20, model_size='medium'):
    """Entrena GRU para regresión"""
    print(f"→ GRU (lookback={lookback}, size={model_size})")
    print("  Creando secuencias...")
    start_time = time.time()

    # Crear secuencias
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, lookback)

    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Test sequences: {X_test_seq.shape}")

    # Arquitectura según tamaño
    if model_size == 'small':
        model = Sequential([
            GRU(32, activation='relu', input_shape=(lookback, X_train.shape[1])),
            Dropout(0.4),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
        epochs_max = 50
        batch_size = 16
    elif model_size == 'medium':
        model = Sequential([
            GRU(50, activation='relu', return_sequences=True,
                input_shape=(lookback, X_train.shape[1])),
            Dropout(0.3),
            GRU(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        epochs_max = 100
        batch_size = 32
    else:  # 'large'
        model = Sequential([
            GRU(64, activation='relu', return_sequences=True,
                input_shape=(lookback, X_train.shape[1])),
            Dropout(0.2),
            GRU(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            GRU(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        epochs_max = 150
        batch_size = 32

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print(f"  Parámetros: {model.count_params():,}")
    print(f"  Ratio samples/params: {X_train_seq.shape[0] / model.count_params():.3f}")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=0
        )
    ]

    # Entrenar
    print(f"  Entrenando (max {epochs_max} epochs)...")
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs_max,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

    elapsed = time.time() - start_time
    print(f"  Epochs entrenados: {len(history.history['loss'])}")
    print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
    print(f"  Tiempo: {elapsed:.2f}s")

    # Predecir
    pred = model.predict(X_test_seq, verbose=0).flatten()
    save_predictions(dataset_name, target_name, "GRU", y_test_seq, pred)

    result = evaluate_regression(y_test_seq, pred, f"GRU_{model_size}")
    result['train_time'] = elapsed
    result['epochs'] = len(history.history['loss'])

    save_model(model, f'GRU_{model_size}', dataset_name, target_name, 'tensorflow')
 # Guardar y plotear historial de entrenamiento
    save_training_history(
        history=history,
        model_name=f'GRU_{model_size}',
        dataset_name=dataset_name,
        target_name=target_name,
        save_path= MODELS / 'history',
        save_format='pickle'
    )

    plot_training_history(
        history=history,
        model_name=f'GRU_{model_size}',
        dataset_name=dataset_name,
        target_name=target_name,
        save_path=PLOTS,
        task_type='regression',
        show_plot=False
    )

    # Agregar info del mejor epoch al resultado
    best_info = get_best_epoch_info(history)
    result['best_epoch'] = best_info['best_epoch']
    result['best_val_loss'] = best_info['best_val_loss']

    return [result], {f'GRU_{model_size}': model}, history.history

def train_ensemble_regression(X_train, y_train, X_test, y_test,
                              dataset_name, target_name):
    """Entrena ensemble (stacking) para regresión"""
    print("→ Ensemble (Stacking)")
    start_time = time.time()

    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10,
                                    random_state=RANDOM_STATE, n_jobs=-1)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=6,
                                random_state=RANDOM_STATE, n_jobs=-1)),
        ('lgbm', lgb.LGBMRegressor(n_estimators=100, max_depth=6,
                                  random_state=RANDOM_STATE, verbose=-1, n_jobs=-1))
    ]

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    save_predictions(dataset_name, target_name, "Ensemble", y_test, pred)
    elapsed = time.time() - start_time

    result = evaluate_regression(y_test.values, pred, "Ensemble")
    result['train_time'] = elapsed
    print(f"  Tiempo: {elapsed:.2f}s")

    save_model(model, 'Ensemble', dataset_name, target_name, 'sklearn')

    return [result], {'Ensemble': model}

print("✓ Funciones de modelos de regresión definidas")

# MODELOS: CLASIFICACIÓN
def train_logistic_model(X_train, y_train, X_test, y_test,
                        dataset_name, target_name, use_scaled=True):
    """Entrena Logistic Regression"""
    if not use_scaled:
        return [], {}, {}

    print("→ Logistic Regression")
    start_time = time.time()

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    elapsed = time.time() - start_time
    result = evaluate_classification(y_test.values, pred, proba, model_name="Logistic")
    result['train_time'] = elapsed
    print(f"  Tiempo: {elapsed:.2f}s")
    #Guardar modelo
    save_model(model, 'Logistic', dataset_name, target_name, 'sklearn')

    predictions_dict = {'Logistic': pred}
    models_dict = {'Logistic': model}

    #Guardar estadísticas
    for model_name, preds in predictions_dict.items():
        save_predictions(dataset_name, target_name, model_name, y_test, preds)

    return [result], {'Logistic': preds}, {'Logistic': model}


def train_tree_models_classification(X_train, y_train, X_test, y_test,
                                     dataset_name, target_name):
    """Entrena modelos basados en árboles para clasificación"""
    results = []
    predictions = {}
    models = {}

    print("→ Random Forest")
    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.time() - start_time
    result = evaluate_classification(y_test.values, pred, proba, "Random Forest")
    result['train_time'] = elapsed
    results.append(result)
    predictions['RF'] = pred
    models['RF'] = model
    print(f"  Tiempo: {elapsed:.2f}s")
    save_model(model, 'Random_Forest', dataset_name, target_name, 'sklearn')

    # CORREGIDO: Guardar predicciones inmediatamente después de cada modelo
    save_predictions(dataset_name, target_name, 'Random_Forest', y_test, pred)

    print("→ XGBoost")
    start_time = time.time()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.time() - start_time
    result = evaluate_classification(y_test.values, pred, proba, "XGBoost")
    result['train_time'] = elapsed
    results.append(result)
    predictions['XGB'] = pred
    models['XGB'] = model
    print(f"  Tiempo: {elapsed:.2f}s")
    save_model(model, 'XGBoost', dataset_name, target_name, 'xgboost')

    # CORREGIDO: Guardar predicciones inmediatamente
    save_predictions(dataset_name, target_name, 'XGBoost', y_test, pred)

    print("→ LightGBM")
    start_time = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.time() - start_time
    result = evaluate_classification(y_test.values, pred, proba, "LightGBM")
    result['train_time'] = elapsed
    results.append(result)
    predictions['LGBM'] = pred
    models['LGBM'] = model
    print(f"  Tiempo: {elapsed:.2f}s")
    save_model(model, 'LightGBM', dataset_name, target_name, 'lightgbm')

    # Guardar predicciones
    save_predictions(dataset_name, target_name, 'LightGBM', y_test, pred)

    return results, predictions, models

def train_mlp_classification(X_train, y_train, X_test, y_test,
                             dataset_name, target_name, use_scaled=True):
    """Entrena MLP para clasificación"""
    if not use_scaled:
        return [], {}, {}

    print("→ MLP (Multi-Layer Perceptron)")
    start_time = time.time()

    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=RANDOM_STATE,
        verbose=False
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.time() - start_time

    result = evaluate_classification(y_test.values, pred, proba, "MLP")
    result['train_time'] = elapsed
    print(f"  Tiempo: {elapsed:.2f}s")
    print(f"  Iterations: {model.n_iter_}")

    save_model(model, 'MLP', dataset_name, target_name, 'sklearn')


    predictions_dict = {'MLP': pred}
    models_dict = {'MLP': model}

    #Guardar estadísticas
    for model_name, preds in predictions_dict.items():
        save_predictions(dataset_name, target_name, model_name, y_test, preds)



    return [result], {'MLP': preds}, {'MLP': model}

def train_gru_classification(X_train, y_train, X_test, y_test,
                             dataset_name, target_name,
                             lookback=20, model_size='medium'):

    """Entrena GRU para clasificación"""
    print(f"→ GRU (lookback={lookback}, size={model_size})")
    print("  Creando secuencias...")
    start_time = time.time()

    # Crear secuencias
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, lookback)

    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Test sequences: {X_test_seq.shape}")

    # Arquitectura
    if model_size == 'small':
        model = Sequential([
            GRU(32, activation='relu', input_shape=(lookback, X_train.shape[1])),
            Dropout(0.4),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        epochs_max = 50
        batch_size = 16
    elif model_size == 'medium':
        model = Sequential([
            GRU(50, activation='relu', return_sequences=True,
                input_shape=(lookback, X_train.shape[1])),
            Dropout(0.3),
            GRU(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        epochs_max = 100
        batch_size = 32
    else:  # 'large'
        model = Sequential([
            GRU(64, activation='relu', return_sequences=True,
                input_shape=(lookback, X_train.shape[1])),
            Dropout(0.2),
            GRU(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            GRU(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        epochs_max = 150
        batch_size = 32

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(f"  Parámetros: {model.count_params():,}")

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0)
    ]

    # Entrenar
    print(f"  Entrenando (max {epochs_max} epochs)...")
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs_max,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

    elapsed = time.time() - start_time
    print(f"  Epochs: {len(history.history['loss'])}")
    print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
    print(f"  Tiempo: {elapsed:.2f}s")

    # Predecir
    proba = model.predict(X_test_seq, verbose=0).flatten()
    pred = (proba > 0.5).astype(int)
    
    
    result = evaluate_classification(y_test_seq, pred, proba, f"GRU_{model_size}")
    result['train_time'] = elapsed
    result['epochs'] = len(history.history['loss'])

    save_model(model, f'GRU_{model_size}', dataset_name, target_name, 'tensorflow')

    predictions_dict = {f'GRU_{model_size}': pred}
    models_dict = {f'GRU_{model_size}': model}


    #Guardar estadísticas
    for model_name, preds in predictions_dict.items():
        save_predictions(dataset_name, target_name, model_name, y_test_seq, preds)

    # Guardar y plotear historial de entrenamiento
    save_training_history(
        history=history,
        model_name=f'GRU_{model_size}',
        dataset_name=dataset_name,
        target_name=target_name,
        save_path= MODELS / 'history',
        save_format='pickle'
    )

    plot_training_history(
        history=history,
        model_name=f'GRU_{model_size}',
        dataset_name=dataset_name,
        target_name=target_name,
        save_path=PLOTS,
        task_type='classification',
        show_plot=False
    )

    # Agregar info del mejor epoch al resultado
    best_info = get_best_epoch_info(history)
    result['best_epoch'] = best_info['best_epoch']
    result['best_val_loss'] = best_info['best_val_loss']

    return [result], {f'GRU_{model_size}': pred}, {f'GRU_{model_size}': model}, history.history

def train_ensemble_classification(X_train, y_train, X_test, y_test,
                                  dataset_name, target_name):
    """Entrena ensemble (stacking) para clasificación"""
    print("→ Ensemble (Stacking)")
    start_time = time.time()

    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                     random_state=RANDOM_STATE, n_jobs=-1)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6,
                                 random_state=RANDOM_STATE, eval_metric='logloss',
                                 n_jobs=-1)),
        ('lgbm', lgb.LGBMClassifier(n_estimators=100, max_depth=6,
                                   random_state=RANDOM_STATE, verbose=-1, n_jobs=-1))
    ]

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.time() - start_time

    result = evaluate_classification(y_test.values, pred, proba, "Ensemble")
    result['train_time'] = elapsed
    print(f"  Tiempo: {elapsed:.2f}s")

    save_model(model, 'Ensemble', dataset_name, target_name, 'sklearn')

    predictions_dict = {'Ensemble': pred}
    models_dict = {'Ensemble': model}

    #Guardar estadísticas
    for model_name, preds in predictions_dict.items():
        save_predictions(dataset_name, target_name, model_name, y_test, preds)

    return [result], {'Ensemble': preds}, {'Ensemble': model}

print("✓ Funciones de modelos de clasificación definidas")
