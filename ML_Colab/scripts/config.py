"""
Configuración portable: detecta entorno y ajusta rutas automáticamente
"""

import os
import sys
from pathlib import Path

class Config:
    """Configuración del proyecto según entorno de ejecución"""
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.setup_paths()
        self._mount_drive_if_needed()
        
    def _detect_environment(self):
        """Detecta si está en Colab, Kaggle o local"""
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
            return 'colab'
        elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return 'kaggle'
        else:
            return 'local'
    
    def _mount_drive_if_needed(self):
        """Monta Google Drive si está en Colab"""
        if self.environment == 'colab':
            try:
                from google.colab import drive
                drive.mount('/content/drive', force_remount=False)
                print(f"✓ Google Drive montado")
            except Exception as e:
                print(f"Error montando Drive: {e}")
    
    def setup_paths(self):
        """Configura rutas según entorno"""
        if self.environment == 'colab':
            # Ajusta esta ruta según dónde guardes el proyecto en Drive
            self.PROJECT_ROOT = Path('/content/drive/MyDrive/TFM')
        elif self.environment == 'kaggle':
            self.PROJECT_ROOT = Path('/kaggle/working')
        else:  # local
            # Asume que config.py está en scripts/
            # Para local, ajusta según tu estructura
            self.PROJECT_ROOT = Path.home() / "Master" / "TFM" / "py_project"
            
        # Rutas principales
        self.DATA_RAW = self.PROJECT_ROOT / "data" / "raw"
        self.DATA_PROCESSED = self.PROJECT_ROOT / "data" / "processed"
        self.SCRIPTS = self.PROJECT_ROOT / "scripts"
        self.RESULTS = self.PROJECT_ROOT / "results"
        
        # Rutas específicas de resultados
        self.FIGURES = self.RESULTS / "figures"
        self.MODELS = self.RESULTS / "models"
        self.PLOTS = self.RESULTS / "plots"
        self.CSV = self.RESULTS / "csv"
        
        # Rutas de input data para ML
        self.INPUT_DATA = self.PROJECT_ROOT / "input_data"
        self.ML_FINANCIAL = self.INPUT_DATA / "ML_financial"
        self.ML_FINANCIAL_LONG = self.INPUT_DATA / "ML_financial_long"
        self.ML_SENTIMENT = self.INPUT_DATA / "ML_sentiment"
        
        # Crear directorios si no existen
        for path in [self.DATA_RAW, self.DATA_PROCESSED, 
                     self.FIGURES, self.MODELS, self.PLOTS, self.CSV,
                     self.INPUT_DATA, self.ML_FINANCIAL, 
                     self.ML_FINANCIAL_LONG, self.ML_SENTIMENT]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Añadir scripts al path para imports
        if str(self.SCRIPTS) not in sys.path:
            sys.path.insert(0, str(self.SCRIPTS))
    
    def get_dataset_config(self):
        """Devuelve configuración de datasets"""
        return {
            'financial_unscaled': {
                'train': self.ML_FINANCIAL / "train_unscaled.rds",
                'test': self.ML_FINANCIAL / "test_unscaled.rds",
                'scaled': False
            },
            'financial_scaled': {
                'train': self.ML_FINANCIAL / "train_scaled_zscore.rds",
                'test': self.ML_FINANCIAL / "test_scaled_zscore.rds",
                'scaled': True
            },
            'financial_long_unscaled': {
                'train': self.ML_FINANCIAL_LONG / "train_unscaled.rds",
                'test': self.ML_FINANCIAL_LONG / "test_unscaled.rds",
                'scaled': False
            },
            'financial_long_scaled': {
                'train': self.ML_FINANCIAL_LONG / "train_scaled_zscore.rds",
                'test': self.ML_FINANCIAL_LONG / "test_scaled_zscore.rds",
                'scaled': True
            },
            'sentiment_unscaled': {
                'train': self.ML_SENTIMENT / "train_unscaled.rds",
                'test': self.ML_SENTIMENT / "test_unscaled.rds",
                'scaled': False
            },
            'sentiment_scaled': {
                'train': self.ML_SENTIMENT / "train_scaled_zscore.rds",
                'test': self.ML_SENTIMENT / "test_scaled_zscore.rds",
                'scaled': True
            }
        }
    
    def info(self):
        """Imprime información del entorno"""
        print(f"Entorno: {self.environment.upper()}")
        print(f"Proyecto: {self.PROJECT_ROOT}")
        print(f"Data: {self.DATA_RAW}")
        print(f"Results: {self.RESULTS}")
        print(f"Models: {self.MODELS}")
        print(f"Plots: {self.PLOTS}")
        print(f"CSV: {self.CSV}")
    
    # Configuración de ML
    RANDOM_STATE = 42
    
    # Targets para ML
    ALL_TARGET_COLS = [
        'returns_next', 'returns_next_5', 'returns_next_10', 'returns_next_20',
        'direction_next', 'direction_next_5', 'direction_next_10', 'direction_next_20'
    ]
    
    REGRESSION_TARGETS = [
        'returns_next', 'returns_next_5', 'returns_next_10', 'returns_next_20'
    ]
    
    CLASSIFICATION_TARGETS = [
        'direction_next', 'direction_next_5', 'direction_next_10', 'direction_next_20'
    ]
    
    
    @staticmethod
    def setup_ml_environment():
        """Configura seeds y estilos para ML"""
        import numpy as np
        try:
            import tensorflow as tf
            tf.random.set_seed(Config.RANDOM_STATE)
        except ImportError:
            pass
        
        np.random.seed(Config.RANDOM_STATE)
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
        except ImportError:
            pass
        
        print("✓ Configuración ML establecida")

# Instancia global
config = Config()

# Atajos para importar - Rutas
PROJECT_ROOT = config.PROJECT_ROOT
DATA_RAW = config.DATA_RAW
DATA_PROCESSED = config.DATA_PROCESSED
RESULTS = config.RESULTS
FIGURES = config.FIGURES
MODELS = config.MODELS
PLOTS = config.PLOTS
CSV = config.CSV
INPUT_DATA = config.INPUT_DATA
ML_FINANCIAL = config.ML_FINANCIAL
ML_FINANCIAL_LONG = config.ML_FINANCIAL_LONG
ML_SENTIMENT = config.ML_SENTIMENT

# Atajos para importar - Configuración ML
RANDOM_STATE = config.RANDOM_STATE
ALL_TARGET_COLS = config.ALL_TARGET_COLS
REGRESSION_TARGETS = config.REGRESSION_TARGETS
CLASSIFICATION_TARGETS = config.CLASSIFICATION_TARGETS
