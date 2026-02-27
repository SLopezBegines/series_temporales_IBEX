#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline ML/DL - Versión 2
Ejecuta modelos según compatibilidad con datos scaled/unscaled

@author: santi
"""

# ============================================================================
# IMPORTS
# ============================================================================

import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Import config y funciones auxiliares
from config import config  # Importar la instancia, no el módulo
datasets_config = config.get_dataset_config() 

from aux_functions import load_rds, prepare_data

# Import funciones de entrenamiento
from modelos_ml import (
    train_linear_models,
    train_tree_models_regression,
    train_tree_models_classification,
    train_mlp_regression,
    train_mlp_classification,
    train_gru_regression,
    train_gru_classification,
    train_ensemble_regression,
    train_ensemble_classification,
    train_logistic_model
)

from lstm_models import (
    train_lstm_regression,
    train_lstm_classification,
    train_lstm_suite
)

from config import CSV

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

def setup_logging():
    """
    Configura el sistema de logging para consola y archivo.
    Retorna el logger configurado.
    """
    # Crear directorio de logs si no existe
    log_dir = CSV / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Nombre del archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'pipeline_{timestamp}.log'
    
    # Configurar formato
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Crear logger
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)
    
    # Evitar duplicar handlers si se llama múltiples veces
    if logger.handlers:
        logger.handlers.clear()
    
    # Handler para archivo (todo)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Handler para consola (INFO y superior)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Log file: {log_file}")
    
    return logger, log_file


# Inicializar logger global
logger, LOG_FILE = setup_logging()


# ============================================================================
# SISTEMA DE PROGRESO
# ============================================================================

class ProgressTracker:
    """
    Tracker de progreso para el pipeline.
    Muestra información en consola sobre el avance.
    """
    
    def __init__(self, config):
        """
        Args:
            config: Diccionario de configuración del pipeline
        """
        self.config = config
        self.total_datasets = len(config['datasets'])
        self.total_targets_reg = len(config['targets_regression'])
        self.total_targets_clf = len(config['targets_classification'])
        
        # Contadores
        self.current_dataset_idx = 0
        self.current_dataset_name = ""
        self.current_target = ""
        self.current_task = ""  # 'regression' o 'classification'
        self.current_model = ""
        self.models_completed = 0
        
        # Calcular total de modelos
        self.total_models = self._calculate_total_models()
        
        # Tiempo
        self.start_time = None
        
    def _calculate_total_models(self):
        """Calcula el número total de modelos a entrenar"""
        # Contar modelos por target según flags activos
        models_scaled = 0
        models_unscaled = 0
        
        # Regresión
        if self.config['train_linear']:
            models_scaled += 2  # Linear/Ridge cuenta como 1 llamada
        if self.config['train_trees']:
            models_scaled += 3
            models_unscaled += 3
        if self.config['train_mlp']:
            models_scaled += 1
        if self.config['train_ensemble']:
            models_scaled += 1
            models_unscaled += 1
        if self.config['train_gru']:
            models_scaled += 1
            models_unscaled += 1
        if self.config['train_lstm']:
            models_scaled += 1
            models_unscaled += 1
        
        # Contar datasets por tipo
        n_scaled = sum(1 for d in self.config['datasets'] if 'unscaled' not in d)
        n_unscaled = len(self.config['datasets']) - n_scaled
        
        # Total regresión
        total_reg = (n_scaled * self.total_targets_reg * models_scaled +
                     n_unscaled * self.total_targets_reg * models_unscaled)
        
        # Clasificación (similar pero con logit en lugar de linear)
        models_scaled_clf = 0
        models_unscaled_clf = 0
        
        if self.config['train_logit']:
            models_scaled_clf += 1
        if self.config['train_trees']:
            models_scaled_clf += 3
            models_unscaled_clf += 3
        if self.config['train_mlp']:
            models_scaled_clf += 1
        if self.config['train_ensemble']:
            models_scaled_clf += 1
            models_unscaled_clf += 1
        if self.config['train_gru']:
            models_scaled_clf += 1
            models_unscaled_clf += 1
        if self.config['train_lstm']:
            models_scaled_clf += 1
            models_unscaled_clf += 1
            
        total_clf = (n_scaled * self.total_targets_clf * models_scaled_clf +
                     n_unscaled * self.total_targets_clf * models_unscaled_clf)
        
        return total_reg + total_clf
    
    def start(self):
        """Inicia el tracking"""
        self.start_time = time.time()
        self._print_header()
    
    def _print_header(self):
        """Imprime cabecera inicial"""
        print("\n" + "=" * 70)
        print("PIPELINE ML/DL - IBEX35")
        print("=" * 70)
        print(f"Total datasets: {self.total_datasets}")
        print(f"Total modelos estimados: {self.total_models}")
        print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")
    
    def set_dataset(self, idx, name):
        """Actualiza el dataset actual"""
        self.current_dataset_idx = idx
        self.current_dataset_name = name
        print(f"\n{'─' * 70}")
        print(f"📁 Dataset {idx}/{self.total_datasets}: {name}")
        print(f"{'─' * 70}")
    
    def set_task(self, task, target):
        """Actualiza la tarea actual (regression/classification)"""
        self.current_task = task
        self.current_target = target
        task_emoji = "📈" if task == "regression" else "🎯"
        print(f"\n{task_emoji} {task.upper()}: {target}")
    
    def update_model(self, model_name, will_run=True):
        """Actualiza el modelo actual y muestra progreso"""
        self.current_model = model_name
        
        if will_run:
            self.models_completed += 1
            self._print_progress(model_name)
        else:
            print(f"   ⏭️  {model_name}: SKIP (requiere scaled)")
    
    def _print_progress(self, model_name):
        """Imprime la línea de progreso"""
        # Calcular porcentaje
        pct = (self.models_completed / self.total_models) * 100 if self.total_models > 0 else 0
        
        # Calcular ETA
        elapsed = time.time() - self.start_time if self.start_time else 0
        if self.models_completed > 0 and elapsed > 0:
            avg_time = elapsed / self.models_completed
            remaining = (self.total_models - self.models_completed) * avg_time
            eta_str = self._format_time(remaining)
        else:
            eta_str = "calculando..."
        
        # Barra de progreso
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Imprimir
        print(f"   ▶ {model_name}")
        print(f"   [{bar}] {pct:5.1f}% | Modelo {self.models_completed}/{self.total_models} | ETA: {eta_str}")
    
    def _format_time(self, seconds):
        """Formatea segundos a string legible"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def model_completed(self, model_name, success=True, metrics=None):
        """Marca un modelo como completado"""
        status = "✓" if success else "✗"
        extra = ""
        if metrics:
            if 'RMSE' in metrics:
                extra = f" (RMSE: {metrics['RMSE']:.6f})"
            elif 'Accuracy' in metrics:
                extra = f" (Acc: {metrics['Accuracy']:.4f})"
        
        print(f"   {status} {model_name} completado{extra}")
    
    def finish(self):
        """Finaliza el tracking"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETADO")
        print("=" * 70)
        print(f"Modelos entrenados: {self.models_completed}/{self.total_models}")
        print(f"Tiempo total: {self._format_time(elapsed)}")
        print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log: {LOG_FILE}")
        print("=" * 70 + "\n")


# ============================================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================================

DEFAULT_CONFIG = {
    # Datasets a procesar
    'datasets': [
        'financial_scaled',
        'financial_unscaled',
        'financial_long_scaled',
        'financial_long_unscaled',
        'sentiment_scaled',
        'sentiment_unscaled'
    ],
    
    # Targets
    'targets_regression': [
        'returns_next',
        'returns_next_5',
        'returns_next_10',
        'returns_next_20'
    ],
    'targets_classification': [
        'direction_next',
        'direction_next_5',
        'direction_next_10',
        'direction_next_20'
    ],
    
    # Flags de entrenamiento
    'train_linear': True,
    'train_logit': True,
    'train_trees': True,
    'train_mlp': True,
    'train_gru': True,
    'train_lstm': True,
    'train_ensemble': True,
    
    # Modo LSTM
    'lstm_mode': 'single',  # 'single' o 'suite'
    
    # Configuración LSTM
    'lstm_config': {
        'lookback': 20,
        'model_size': 'medium',
        'verbose': 0,
        'model_sizes': ['small', 'medium', 'large'],
        'lookbacks': [10, 20, 30]
    },
    
    # Configuración GRU
    'gru_config': {
        'lookback': 20,
        'model_size': 'medium',
        'verbose': 0
    }
}

# Modelos que REQUIEREN datos escalados (constante, no configurable)
MODELS_REQUIRE_SCALED = {'Linear', 'Ridge', 'Logistic', 'MLP'}

# Modelos que funcionan con cualquier tipo de datos (constante, no configurable)
MODELS_ANY_DATA = {'Random Forest', 'XGBoost', 'LightGBM', 'Ensemble', 'GRU', 'LSTM'}


def get_config(user_config=None):
    """
    Combina configuración del usuario con defaults.
    
    Args:
        user_config: Dict con configuración personalizada (opcional)
        
    Returns:
        Dict con configuración completa
    """
    if user_config is None:
        return DEFAULT_CONFIG.copy()
    
    # Copiar defaults
    config = DEFAULT_CONFIG.copy()
    
    # Actualizar con valores del usuario
    for key, value in user_config.items():
        if key in config:
            # Si es dict (lstm_config, gru_config), hacer merge
            if isinstance(value, dict) and isinstance(config[key], dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
        else:
            logger.warning(f"Clave de configuración desconocida: {key}")
    
    return config


def print_config_report(config):
    """
    Imprime un reporte formateado de la configuración del pipeline.
    
    Args:
        config: Diccionario de configuración (puede ser user_config o resultado de get_config)
    """
    # Si es config parcial del usuario, completar con defaults
    if 'datasets' not in config or 'train_linear' not in config:
        cfg = get_config(config)
    else:
        cfg = config
    
    # Contar datasets por tipo
    n_scaled = sum(1 for d in cfg['datasets'] if 'unscaled' not in d)
    n_unscaled = len(cfg['datasets']) - n_scaled
    
    # Calcular estimación de modelos
    models_reg_scaled = sum([
        2 if cfg['train_linear'] else 0,
        3 if cfg['train_trees'] else 0,
        1 if cfg['train_mlp'] else 0,
        1 if cfg['train_ensemble'] else 0,
        1 if cfg['train_gru'] else 0,
        1 if cfg['train_lstm'] else 0,
    ])
    models_reg_unscaled = sum([
        3 if cfg['train_trees'] else 0,
        1 if cfg['train_ensemble'] else 0,
        1 if cfg['train_gru'] else 0,
        1 if cfg['train_lstm'] else 0,
    ])
    models_clf_scaled = sum([
        1 if cfg['train_logit'] else 0,
        3 if cfg['train_trees'] else 0,
        1 if cfg['train_mlp'] else 0,
        1 if cfg['train_ensemble'] else 0,
        1 if cfg['train_gru'] else 0,
        1 if cfg['train_lstm'] else 0,
    ])
    models_clf_unscaled = sum([
        3 if cfg['train_trees'] else 0,
        1 if cfg['train_ensemble'] else 0,
        1 if cfg['train_gru'] else 0,
        1 if cfg['train_lstm'] else 0,
    ])
    
    n_targets_reg = len(cfg['targets_regression'])
    n_targets_clf = len(cfg['targets_classification'])
    
    total_reg = (n_scaled * n_targets_reg * models_reg_scaled + 
                 n_unscaled * n_targets_reg * models_reg_unscaled)
    total_clf = (n_scaled * n_targets_clf * models_clf_scaled + 
                 n_unscaled * n_targets_clf * models_clf_unscaled)
    total_models = total_reg + total_clf
    
    # Imprimir reporte
    print("\n" + "=" * 70)
    print("                    CONFIGURACIÓN DEL PIPELINE")
    print("=" * 70)
    
    # Datasets
    print("\n📁 DATASETS")
    print("─" * 40)
    print(f"   Total: {len(cfg['datasets'])} (scaled: {n_scaled}, unscaled: {n_unscaled})")
    for ds in cfg['datasets']:
        ds_type = "scaled" if 'unscaled' not in ds else "unscaled"
        print(f"   • {ds} [{ds_type}]")
    
    # Targets Regresión
    print(f"\n📈 TARGETS REGRESIÓN ({n_targets_reg})")
    print("─" * 40)
    if cfg['targets_regression']:
        for t in cfg['targets_regression']:
            print(f"   • {t}")
    else:
        print("   (ninguno)")
    
    # Targets Clasificación
    print(f"\n🎯 TARGETS CLASIFICACIÓN ({n_targets_clf})")
    print("─" * 40)
    if cfg['targets_classification']:
        for t in cfg['targets_classification']:
            print(f"   • {t}")
    else:
        print("   (ninguno)")
    
    # Modelos
    print("\n🤖 MODELOS A ENTRENAR")
    print("─" * 40)
    
    # Tabla de modelos
    print(f"   {'Modelo':<20} {'Activo':<10} {'Datos':<15}")
    print(f"   {'-'*20} {'-'*10} {'-'*15}")
    
    models_info = [
        ('Linear/Ridge', cfg['train_linear'], 'solo scaled'),
        ('Logistic', cfg['train_logit'], 'solo scaled'),
        ('MLP', cfg['train_mlp'], 'solo scaled'),
        ('Random Forest', cfg['train_trees'], 'todos'),
        ('XGBoost', cfg['train_trees'], 'todos'),
        ('LightGBM', cfg['train_trees'], 'todos'),
        ('Ensemble', cfg['train_ensemble'], 'todos'),
        ('GRU', cfg['train_gru'], 'todos'),
        ('LSTM', cfg['train_lstm'], 'todos'),
    ]
    
    for name, active, data_type in models_info:
        status = "✓ Sí" if active else "✗ No"
        print(f"   {name:<20} {status:<10} {data_type:<15}")
    
    # Configuración LSTM/GRU
    if cfg['train_lstm'] or cfg['train_gru']:
        print("\n⚙️  CONFIGURACIÓN DEEP LEARNING")
        print("─" * 40)
        
        if cfg['train_gru']:
            gru = cfg['gru_config']
            print(f"      GRU:")
            print(f"      lookback: {gru['lookback']}")
            print(f"      model_size: {gru['model_size']}")
        
        if cfg['train_lstm']:
            lstm = cfg['lstm_config']
            print(f"   LSTM (modo: {cfg['lstm_mode']}):")
            if cfg['lstm_mode'] == 'single':
                print(f"      lookback: {lstm['lookback']}")
                print(f"      model_size: {lstm['model_size']}")
            else:  # suite
                print(f"      lookbacks: {lstm.get('lookbacks', [10, 20, 30])}")
                print(f"      model_sizes: {lstm.get('model_sizes', ['small', 'medium', 'large'])}")
    
    # Resumen de ejecución
    print("\n📊 ESTIMACIÓN DE EJECUCIÓN")
    print("─" * 40)
    print(f"   Modelos regresión:      {total_reg:>6}")
    print(f"   Modelos clasificación:  {total_clf:>6}")
    print(f"   {'─'*30}")
    print(f"   TOTAL MODELOS:          {total_models:>6}")
    
    # Estimación de tiempo (muy aproximada)
    # Asumiendo ~2 min promedio por modelo tradicional, ~10 min por DL
    time_estimate_min = 0
    if cfg['train_linear']: time_estimate_min += n_scaled * n_targets_reg * 1
    if cfg['train_logit']: time_estimate_min += n_scaled * n_targets_clf * 1
    if cfg['train_trees']: time_estimate_min += len(cfg['datasets']) * (n_targets_reg + n_targets_clf) * 3
    if cfg['train_mlp']: time_estimate_min += n_scaled * (n_targets_reg + n_targets_clf) * 2
    if cfg['train_ensemble']: time_estimate_min += len(cfg['datasets']) * (n_targets_reg + n_targets_clf) * 5
    if cfg['train_gru']: time_estimate_min += len(cfg['datasets']) * (n_targets_reg + n_targets_clf) * 15
    if cfg['train_lstm']: 
        if cfg['lstm_mode'] == 'suite':
            n_lstm_configs = len(cfg['lstm_config'].get('lookbacks', [1])) * len(cfg['lstm_config'].get('model_sizes', [1]))
            time_estimate_min += len(cfg['datasets']) * (n_targets_reg + n_targets_clf) * 15 * n_lstm_configs
        else:
            time_estimate_min += len(cfg['datasets']) * (n_targets_reg + n_targets_clf) * 15
    
    if time_estimate_min < 60:
        time_str = f"{time_estimate_min:.0f} min"
    else:
        time_str = f"{time_estimate_min/60:.1f} h"
    
    print(f"\n   ⏱️  Tiempo estimado: ~{time_str} (aproximado)")
    
    print("\n" + "=" * 70)
    
    return cfg


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_dataset_type(dataset_name):
    """Determina si un dataset es scaled o unscaled"""
    if 'unscaled' in dataset_name:
        return 'unscaled'
    elif 'scaled' in dataset_name:
        return 'scaled'
    else:
        raise ValueError(f"No se puede determinar tipo de dataset: {dataset_name}")


def get_paired_dataset(dataset_name, datasets_config):
    """
    Dado un dataset, retorna su par (scaled <-> unscaled)
    
    Args:
        dataset_name: Nombre del dataset actual
        datasets_config: Configuración de datasets
        
    Returns:
        str o None: Nombre del dataset paired, o None si no existe
    """
    if 'unscaled' in dataset_name:
        paired_name = dataset_name.replace('_unscaled', '_scaled')
    else:
        paired_name = dataset_name.replace('_scaled', '_unscaled')
    
    return paired_name if paired_name in datasets_config else None


def format_model_name(base_model_name, dataset_name):
    """
    Formatea el nombre del modelo incluyendo el dataset
    
    Args:
        base_model_name: Nombre base del modelo (e.g., 'XGBoost')
        dataset_name: Nombre completo del dataset (e.g., 'financial_scaled')
        
    Returns:
        str: Nombre formateado (e.g., 'XGBoost_financial_scaled')
    """
    return f"{base_model_name}_{dataset_name}"


def add_dataset_info_to_results(results, dataset_name, target):
    """
    Añade información del dataset a los resultados
    
    Args:
        results: Lista de diccionarios con resultados
        dataset_name: Nombre del dataset
        target: Nombre del target
        
    Returns:
        Lista de resultados con campos añadidos
    """
    data_type = get_dataset_type(dataset_name)
    
    for r in results:
        r['dataset'] = dataset_name
        r['target'] = target
        r['data_type'] = data_type
        # Actualizar nombre del modelo para incluir dataset
        if 'model' in r:
            r['model_full'] = f"{r['model']}_{dataset_name}"
    
    return results


def log_model_execution(model_name, dataset_name, target, is_scaled, will_run):
    """
    Logging detallado de ejecución de modelos.
    Escribe a consola y archivo simultáneamente.
    
    Args:
        model_name: Nombre del modelo
        dataset_name: Nombre del dataset
        target: Variable objetivo
        is_scaled: Si el dataset está escalado
        will_run: Si el modelo se ejecutará
    """
    status = "EJECUTANDO" if will_run else "SALTANDO"
    reason = ""
    
    if not will_run:
        if model_name in MODELS_REQUIRE_SCALED and not is_scaled:
            reason = " (requiere scaled, dataset es unscaled)"
    
    message = f"{status}: {model_name} | {dataset_name} | {target}{reason}"
    
    if will_run:
        logger.info(message)
    else:
        logger.warning(message)


def log_section(title, level='info'):
    """Log de secciones con formato"""
    separator = "=" * 70
    if level == 'info':
        logger.info(separator)
        logger.info(title)
        logger.info(separator)
    else:
        logger.debug(separator)
        logger.debug(title)
        logger.debug(separator)


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_pipeline(user_config=None):
    """
    Ejecuta el pipeline completo.
    
    Args:
        user_config: Dict con configuración personalizada (opcional).
                     Las claves no especificadas usarán valores por defecto.
                     
    Claves de configuración disponibles:
        - datasets: Lista de datasets a procesar
        - targets_regression: Lista de targets de regresión
        - targets_classification: Lista de targets de clasificación
        - train_linear: bool - Entrenar Linear/Ridge (solo scaled)
        - train_logit: bool - Entrenar Logistic (solo scaled)
        - train_trees: bool - Entrenar RF/XGBoost/LightGBM
        - train_mlp: bool - Entrenar MLP (solo scaled)
        - train_gru: bool - Entrenar GRU
        - train_lstm: bool - Entrenar LSTM
        - train_ensemble: bool - Entrenar Ensemble
        - lstm_mode: 'single' o 'suite'
        - lstm_config: dict con lookback, model_size, verbose, etc.
        - gru_config: dict con lookback, model_size, verbose
    
    Returns:
        tuple: (all_reg_results, all_clf_results, execution_log)
    
    Ejemplo:
        # Ejecutar solo árboles en un dataset
        config = {
            'datasets': ['financial_scaled'],
            'train_linear': False,
            'train_mlp': False,
            'train_gru': False,
            'train_lstm': False
        }
        results = run_pipeline(config)
    """
    
    # Obtener configuración combinada (user + defaults)
    cfg = get_config(user_config)
    
    # Extraer variables para uso local (más legible)
    DATASETS = cfg['datasets']
    TARGETS_REG = cfg['targets_regression']
    TARGETS_CLF = cfg['targets_classification']
    TRAIN_LINEAR = cfg['train_linear']
    TRAIN_LOGIT = cfg['train_logit']
    TRAIN_TREES = cfg['train_trees']
    TRAIN_MLP = cfg['train_mlp']
    TRAIN_GRU = cfg['train_gru']
    TRAIN_LSTM = cfg['train_lstm']
    TRAIN_ENSEMBLE = cfg['train_ensemble']
    LSTM_MODE = cfg['lstm_mode']
    LSTM_CONFIG = cfg['lstm_config']
    GRU_CONFIG = cfg['gru_config']

    log_section(f"INICIANDO PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Resumen de configuración
    logger.info("CONFIGURACIÓN:")
    logger.info(f"   Datasets: {len(DATASETS)}")
    logger.info(f"   Targets Regresión: {len(TARGETS_REG)}")
    logger.info(f"   Targets Clasificación: {len(TARGETS_CLF)}")
    logger.info(f"   Modelos REQUIRE_SCALED: {MODELS_REQUIRE_SCALED}")
    logger.info(f"   Modelos ANY_DATA: {MODELS_ANY_DATA}")
    
    # Log de flags activos
    active_models = []
    if TRAIN_LINEAR: active_models.append('Linear/Ridge')
    if TRAIN_LOGIT: active_models.append('Logistic')
    if TRAIN_TREES: active_models.append('Trees')
    if TRAIN_MLP: active_models.append('MLP')
    if TRAIN_GRU: active_models.append('GRU')
    if TRAIN_LSTM: active_models.append('LSTM')
    if TRAIN_ENSEMBLE: active_models.append('Ensemble')
    logger.info(f"   Modelos activos: {', '.join(active_models)}")

    # Contar datasets por tipo
    n_scaled = sum(1 for d in DATASETS if 'unscaled' not in d)
    n_unscaled = len(DATASETS) - n_scaled

    logger.info(f"DATASETS: Scaled={n_scaled}, Unscaled={n_unscaled}")

    # Inicializar tracker de progreso
    progress = ProgressTracker(cfg)
    progress.start()

    # Obtener configuración de datasets
    datasets_config = config.get_dataset_config()

    # Listas para almacenar resultados
    all_reg_results = []
    all_clf_results = []

    # Contadores para logging
    execution_log = {
        'regression': {'executed': 0, 'skipped': 0},
        'classification': {'executed': 0, 'skipped': 0}
    }

    start_time = time.time()

    # ========================================================================
    # LOOP POR DATASETS
    # ========================================================================
    for idx, dataset_name in enumerate(DATASETS, 1):
        log_section(f"PROCESANDO: {dataset_name}")
        progress.set_dataset(idx, dataset_name)

        dataset_info = datasets_config[dataset_name]
        is_scaled = get_dataset_type(dataset_name) == 'scaled'

        # Cargar datos
        logger.info("Cargando datos...")
        train = load_rds(str(dataset_info['train']))
        test = load_rds(str(dataset_info['test']))
        logger.info(f"  Train: {train.shape}, Test: {test.shape}")
        logger.info(f"  Tipo de datos: {'SCALED' if is_scaled else 'UNSCALED'}")

        # ====================================================================
        # REGRESIÓN
        # ====================================================================
        for target in TARGETS_REG:
            logger.info(f"REGRESIÓN: {target} | Dataset: {dataset_name}")
            progress.set_task("regression", target)

            X_train, y_train = prepare_data(train, target)
            X_test, y_test = prepare_data(test, target)

            # ================================================================
            # MODELOS LINEALES (Linear, Ridge) - SOLO SCALED
            # ================================================================
            if TRAIN_LINEAR and is_scaled:
                progress.update_model('Linear/Ridge', will_run=True)
                log_model_execution('Linear/Ridge', dataset_name, target, is_scaled, True)
                try:
                    results, models = train_linear_models(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target, use_scaled=True
                    )
                    results = add_dataset_info_to_results(results, dataset_name, target)
                    all_reg_results.extend(results)
                    execution_log['regression']['executed'] += len(results)
                    logger.info(f"  {len(results)} modelos lineales completados")
                    progress.model_completed('Linear/Ridge', success=True)
                except Exception as e:
                    logger.error(f"  Error en modelos lineales: {e}")
                    progress.model_completed('Linear/Ridge', success=False)
            elif TRAIN_LINEAR and not is_scaled:
                progress.update_model('Linear/Ridge', will_run=False)
                log_model_execution('Linear/Ridge', dataset_name, target, is_scaled, False)
                execution_log['regression']['skipped'] += 2

            # ================================================================
            # MODELOS DE ÁRBOLES (RF, XGBoost, LightGBM) - TODOS LOS DATASETS
            # ================================================================
            if TRAIN_TREES:
                progress.update_model('Trees (RF/XGB/LGBM)', will_run=True)
                log_model_execution('Trees (RF/XGB/LGBM)', dataset_name, target, is_scaled, True)
                try:
                    results, models = train_tree_models_regression(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target
                    )
                    results = add_dataset_info_to_results(results, dataset_name, target)
                    all_reg_results.extend(results)
                    execution_log['regression']['executed'] += len(results)
                    logger.info(f"  {len(results)} modelos de árboles completados")
                    progress.model_completed('Trees', success=True)
                except Exception as e:
                    logger.error(f"  Error en modelos de árboles: {e}")
                    progress.model_completed('Trees', success=False)

            # ================================================================
            # MLP - SOLO SCALED
            # ================================================================
            if TRAIN_MLP and is_scaled:
                progress.update_model('MLP', will_run=True)
                log_model_execution('MLP', dataset_name, target, is_scaled, True)
                try:
                    results, models = train_mlp_regression(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target, use_scaled=True
                    )
                    results = add_dataset_info_to_results(results, dataset_name, target)
                    all_reg_results.extend(results)
                    execution_log['regression']['executed'] += len(results)
                    logger.info(f"  MLP completado")
                    progress.model_completed('MLP', success=True)
                except Exception as e:
                    logger.error(f"  Error en MLP: {e}")
                    progress.model_completed('MLP', success=False)
            elif TRAIN_MLP and not is_scaled:
                progress.update_model('MLP', will_run=False)
                log_model_execution('MLP', dataset_name, target, is_scaled, False)
                execution_log['regression']['skipped'] += 1

            # ================================================================
            # ENSEMBLE - TODOS LOS DATASETS
            # ================================================================
            if TRAIN_ENSEMBLE:
                progress.update_model('Ensemble', will_run=True)
                log_model_execution('Ensemble', dataset_name, target, is_scaled, True)
                try:
                    results, models = train_ensemble_regression(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target
                    )
                    results = add_dataset_info_to_results(results, dataset_name, target)
                    all_reg_results.extend(results)
                    execution_log['regression']['executed'] += len(results)
                    logger.info(f"  Ensemble completado")
                    progress.model_completed('Ensemble', success=True)
                except Exception as e:
                    logger.error(f"  Error en Ensemble: {e}")
                    progress.model_completed('Ensemble', success=False)

            # ================================================================
            # GRU - TODOS LOS DATASETS
            # ================================================================
            if TRAIN_GRU:
                progress.update_model('GRU', will_run=True)
                log_model_execution('GRU', dataset_name, target, is_scaled, True)
                try:
                    results, models, history = train_gru_regression(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target,
                        lookback=GRU_CONFIG['lookback'],
                        model_size=GRU_CONFIG['model_size']
                    )
                    results = add_dataset_info_to_results(results, dataset_name, target)
                    all_reg_results.extend(results)
                    execution_log['regression']['executed'] += len(results)
                    logger.info(f"  GRU completado")
                    progress.model_completed('GRU', success=True, metrics=results[0] if results else None)
                except Exception as e:
                    logger.error(f"  Error en GRU: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    progress.model_completed('GRU', success=False)

            # ================================================================
            # LSTM - TODOS LOS DATASETS
            # ================================================================
            if TRAIN_LSTM:
                progress.update_model('LSTM', will_run=True)
                log_model_execution('LSTM', dataset_name, target, is_scaled, True)
                try:
                    if LSTM_MODE == 'single':
                        results, models, history = train_lstm_regression(
                            X_train, y_train, X_test, y_test,
                            dataset_name, target,
                            lookback=LSTM_CONFIG['lookback'],
                            model_size=LSTM_CONFIG['model_size'],
                            verbose=LSTM_CONFIG['verbose']
                        )
                        results = add_dataset_info_to_results(results, dataset_name, target)
                        all_reg_results.extend(results)
                        execution_log['regression']['executed'] += len(results)
                        logger.info(f"  LSTM completado")
                        progress.model_completed('LSTM', success=True, metrics=results[0] if results else None)

                    elif LSTM_MODE == 'suite':
                        results, best_config = train_lstm_suite(
                            X_train, y_train, X_test, y_test,
                            dataset_name, target,
                            task_type='regression',
                            model_sizes=LSTM_CONFIG['model_sizes'],
                            lookbacks=LSTM_CONFIG['lookbacks'],
                            verbose=LSTM_CONFIG['verbose']
                        )
                        results = add_dataset_info_to_results(results, dataset_name, target)
                        all_reg_results.extend(results)
                        execution_log['regression']['executed'] += len(results)
                        logger.info(f"  LSTM suite completado ({len(results)} modelos)")
                        progress.model_completed('LSTM suite', success=True)

                except Exception as e:
                    logger.error(f"  Error en LSTM: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    progress.model_completed('LSTM', success=False)

        # ====================================================================
        # CLASIFICACIÓN
        # ====================================================================
        for target in TARGETS_CLF:
            logger.info(f"CLASIFICACIÓN: {target} | Dataset: {dataset_name}")
            progress.set_task("classification", target)

            X_train, y_train = prepare_data(train, target)
            X_test, y_test = prepare_data(test, target)

            # ================================================================
            # LOGISTIC REGRESSION - SOLO SCALED
            # ================================================================
            if TRAIN_LOGIT and is_scaled:
                progress.update_model('Logistic', will_run=True)
                log_model_execution('Logistic', dataset_name, target, is_scaled, True)
                try:
                    results, predictions, models = train_logistic_model(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target, use_scaled=True
                    )
                    results = add_dataset_info_to_results(results, dataset_name, target)
                    all_clf_results.extend(results)
                    execution_log['classification']['executed'] += len(results)
                    logger.info(f"  Regresión logística completada")
                    progress.model_completed('Logistic', success=True, metrics=results[0] if results else None)
                except Exception as e:
                    logger.error(f"  Error en regresión logística: {e}")
                    progress.model_completed('Logistic', success=False)
            elif TRAIN_LOGIT and not is_scaled:
                progress.update_model('Logistic', will_run=False)
                log_model_execution('Logistic', dataset_name, target, is_scaled, False)
                execution_log['classification']['skipped'] += 1

            # ================================================================
            # MODELOS DE ÁRBOLES - TODOS LOS DATASETS
            # ================================================================
            if TRAIN_TREES:
                progress.update_model('Trees (RF/XGB/LGBM)', will_run=True)
                log_model_execution('Trees (RF/XGB/LGBM)', dataset_name, target, is_scaled, True)
                try:
                    results, predictions, models = train_tree_models_classification(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target
                    )
                    results = add_dataset_info_to_results(results, dataset_name, target)
                    all_clf_results.extend(results)
                    execution_log['classification']['executed'] += len(results)
                    logger.info(f"  {len(results)} modelos de árboles completados")
                    progress.model_completed('Trees', success=True)
                except Exception as e:
                    logger.error(f"  Error en modelos de árboles: {e}")
                    progress.model_completed('Trees', success=False)

            # ================================================================
            # MLP - SOLO SCALED
            # ================================================================
            if TRAIN_MLP and is_scaled:
                progress.update_model('MLP', will_run=True)
                log_model_execution('MLP', dataset_name, target, is_scaled, True)
                try:
                    results, predictions, models = train_mlp_classification(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target, use_scaled=True
                    )
                    results = add_dataset_info_to_results(results, dataset_name, target)
                    all_clf_results.extend(results)
                    execution_log['classification']['executed'] += len(results)
                    logger.info(f"  MLP completado")
                    progress.model_completed('MLP', success=True, metrics=results[0] if results else None)
                except Exception as e:
                    logger.error(f"  Error en MLP: {e}")
                    progress.model_completed('MLP', success=False)
            elif TRAIN_MLP and not is_scaled:
                progress.update_model('MLP', will_run=False)
                log_model_execution('MLP', dataset_name, target, is_scaled, False)
                execution_log['classification']['skipped'] += 1

            # ================================================================
            # GRU - TODOS LOS DATASETS
            # ================================================================
            if TRAIN_GRU:
                progress.update_model('GRU', will_run=True)
                log_model_execution('GRU', dataset_name, target, is_scaled, True)
                try:
                    results, predictions, models, history = train_gru_classification(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target,
                        lookback=GRU_CONFIG['lookback'],
                        model_size=GRU_CONFIG['model_size']
                    )
                    results = add_dataset_info_to_results(results, dataset_name, target)
                    all_clf_results.extend(results)
                    execution_log['classification']['executed'] += len(results)
                    logger.info(f"  GRU completado")
                    progress.model_completed('GRU', success=True, metrics=results[0] if results else None)
                except Exception as e:
                    logger.error(f"  Error en GRU: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    progress.model_completed('GRU', success=False)

            # ================================================================
            # LSTM - TODOS LOS DATASETS
            # ================================================================
            if TRAIN_LSTM:
                progress.update_model('LSTM', will_run=True)
                log_model_execution('LSTM', dataset_name, target, is_scaled, True)
                try:
                    if LSTM_MODE == 'single':
                        results, predictions, models, history = train_lstm_classification(
                            X_train, y_train, X_test, y_test,
                            dataset_name, target,
                            lookback=LSTM_CONFIG['lookback'],
                            model_size=LSTM_CONFIG['model_size'],
                            verbose=LSTM_CONFIG['verbose']
                        )
                        results = add_dataset_info_to_results(results, dataset_name, target)
                        all_clf_results.extend(results)
                        execution_log['classification']['executed'] += len(results)
                        logger.info(f"  LSTM completado")
                        progress.model_completed('LSTM', success=True, metrics=results[0] if results else None)

                    elif LSTM_MODE == 'suite':
                        results, best_config = train_lstm_suite(
                            X_train, y_train, X_test, y_test,
                            dataset_name, target,
                            task_type='classification',
                            model_sizes=LSTM_CONFIG['model_sizes'],
                            lookbacks=LSTM_CONFIG['lookbacks'],
                            verbose=LSTM_CONFIG['verbose']
                        )
                        results = add_dataset_info_to_results(results, dataset_name, target)
                        all_clf_results.extend(results)
                        execution_log['classification']['executed'] += len(results)
                        logger.info(f"  LSTM suite completado ({len(results)} modelos)")
                        progress.model_completed('LSTM suite', success=True)

                except Exception as e:
                    logger.error(f"  Error en LSTM: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    progress.model_completed('LSTM', success=False)

            # ================================================================
            # ENSEMBLE - TODOS LOS DATASETS
            # ================================================================
            if TRAIN_ENSEMBLE:
                progress.update_model('Ensemble', will_run=True)
                log_model_execution('Ensemble', dataset_name, target, is_scaled, True)
                try:
                    results, predictions, models = train_ensemble_classification(
                        X_train, y_train, X_test, y_test,
                        dataset_name, target
                    )
                    results = add_dataset_info_to_results(results, dataset_name, target)
                    all_clf_results.extend(results)
                    execution_log['classification']['executed'] += len(results)
                    logger.info(f"  Ensemble completado")
                    progress.model_completed('Ensemble', success=True, metrics=results[0] if results else None)
                except Exception as e:
                    logger.error(f"  Error en Ensemble: {e}")
                    progress.model_completed('Ensemble', success=False)

    # ========================================================================
    # RESUMEN DE EJECUCIÓN
    # ========================================================================
    log_section("RESUMEN DE EJECUCIÓN")
    logger.info(f"REGRESIÓN: Ejecutados={execution_log['regression']['executed']}, Saltados={execution_log['regression']['skipped']}")
    logger.info(f"CLASIFICACIÓN: Ejecutados={execution_log['classification']['executed']}, Saltados={execution_log['classification']['skipped']}")
    total_exec = execution_log['regression']['executed'] + execution_log['classification']['executed']
    total_skip = execution_log['regression']['skipped'] + execution_log['classification']['skipped']
    logger.info(f"TOTAL: Ejecutados={total_exec}, Saltados={total_skip}")

    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================
    log_section("GUARDANDO RESULTADOS")

    # REGRESIÓN
    if all_reg_results:
        new_reg_df = pd.DataFrame(all_reg_results)
        reg_file_parquet = CSV / 'all_regression_results.parquet'
        reg_file_csv = CSV / 'all_regression_results.csv'

        # Usar model + dataset + target como clave única
        reg_key_cols = ['model', 'dataset', 'target']
        reg_key_cols = [c for c in reg_key_cols if c in new_reg_df.columns]

        if reg_file_parquet.exists():
            existing_reg = pd.read_parquet(reg_file_parquet)
            combined_reg = pd.concat([existing_reg, new_reg_df], ignore_index=True)
            combined_reg = combined_reg.drop_duplicates(subset=reg_key_cols, keep='last')

            n_added = len(new_reg_df)
            n_total = len(combined_reg)
            n_replaced = len(existing_reg) + n_added - n_total

            logger.info(f"Regresión: Nuevos={n_added}, Reemplazados={n_replaced}, Total={n_total}")
        else:
            combined_reg = new_reg_df
            logger.info(f"Regresión: {len(new_reg_df)} resultados guardados (archivo nuevo)")

        combined_reg.to_parquet(reg_file_parquet, index=False, engine='pyarrow')
        combined_reg.to_csv(reg_file_csv, index=False, float_format='%.10f')
        logger.info(f"  Archivos: {reg_file_parquet.name} + {reg_file_csv.name}")
    else:
        logger.warning("No hay resultados de regresión para guardar")

    # CLASIFICACIÓN
    if all_clf_results:
        new_clf_df = pd.DataFrame(all_clf_results)
        clf_file_parquet = CSV / 'all_classification_results.parquet'
        clf_file_csv = CSV / 'all_classification_results.csv'

        clf_key_cols = ['model', 'dataset', 'target']
        clf_key_cols = [c for c in clf_key_cols if c in new_clf_df.columns]

        if clf_file_parquet.exists():
            existing_clf = pd.read_parquet(clf_file_parquet)
            combined_clf = pd.concat([existing_clf, new_clf_df], ignore_index=True)
            combined_clf = combined_clf.drop_duplicates(subset=clf_key_cols, keep='last')

            n_added = len(new_clf_df)
            n_total = len(combined_clf)
            n_replaced = len(existing_clf) + n_added - n_total

            logger.info(f"Clasificación: Nuevos={n_added}, Reemplazados={n_replaced}, Total={n_total}")
        else:
            combined_clf = new_clf_df
            logger.info(f"Clasificación: {len(new_clf_df)} resultados guardados (archivo nuevo)")

        combined_clf.to_parquet(clf_file_parquet, index=False, engine='pyarrow')
        combined_clf.to_csv(clf_file_csv, index=False, float_format='%.10f')
        logger.info(f"  Archivos: {clf_file_parquet.name} + {clf_file_csv.name}")
    else:
        logger.warning("No hay resultados de clasificación para guardar")

    # ========================================================================
    # VALIDACIÓN FINAL
    # ========================================================================
    log_section("VALIDACIÓN DE COMPATIBILIDAD")

    # Verificar que modelos scaled-only NO están en datasets unscaled
    if all_reg_results:
        reg_df = pd.DataFrame(all_reg_results)
        for model in MODELS_REQUIRE_SCALED:
            invalid = reg_df[(reg_df['model'].str.contains(model, case=False, na=False)) &
                             (reg_df['data_type'] == 'unscaled')]
            if len(invalid) > 0:
                logger.warning(f"ALERTA: {model} encontrado en datos unscaled (regresión)")
            else:
                logger.info(f"{model}: Solo en datos scaled (correcto)")

    if all_clf_results:
        clf_df = pd.DataFrame(all_clf_results)
        for model in MODELS_REQUIRE_SCALED:
            invalid = clf_df[(clf_df['model'].str.contains(model, case=False, na=False)) &
                             (clf_df['data_type'] == 'unscaled')]
            if len(invalid) > 0:
                logger.warning(f"ALERTA: {model} encontrado en datos unscaled (clasificación)")
            else:
                logger.info(f"{model}: Solo en datos scaled (correcto)")

    # Verificar que modelos ANY_DATA están en AMBOS tipos de datasets
    logger.info("VERIFICACIÓN MODELS_ANY_DATA (deben estar en scaled Y unscaled):")
    if all_reg_results:
        reg_df = pd.DataFrame(all_reg_results)
        for model_base in ['Random Forest', 'XGBoost', 'LightGBM', 'Ensemble', 'GRU', 'LSTM']:
            model_results = reg_df[reg_df['model'].str.contains(model_base, case=False, na=False)]
            if len(model_results) > 0:
                n_scaled = (model_results['data_type'] == 'scaled').sum()
                n_unscaled = (model_results['data_type'] == 'unscaled').sum()
                status = "OK" if n_scaled > 0 and n_unscaled > 0 else "WARN"
                logger.info(f"  {status} {model_base}: scaled={n_scaled}, unscaled={n_unscaled}")

    elapsed = time.time() - start_time
    log_section(f"Pipeline completado en {elapsed / 60:.1f} min ({elapsed / 3600:.2f} h)")
    logger.info(f"Log guardado en: {LOG_FILE}")
    
    # Finalizar tracker de progreso
    progress.finish()

    return all_reg_results, all_clf_results, execution_log

'''
# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    # Ejemplo 1: Ejecutar con configuración por defecto
    # reg_results, clf_results, log = run_pipeline()
    
    # Ejemplo 2: Ejecutar con configuración personalizada
    # my_config = {
    #     'datasets': ['financial_scaled', 'financial_unscaled'],
    #     'targets_regression': ['returns_next'],
    #     'train_lstm': False,
    #     'train_gru': False
    # }
    # reg_results, clf_results, log = run_pipeline(my_config)
    
    # Ejecutar con defaults
    reg_results, clf_results, log = run_pipeline()
 
'''