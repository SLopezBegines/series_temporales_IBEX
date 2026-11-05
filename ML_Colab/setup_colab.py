#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup Script para Google Colab
Proyecto TFM IBEX35

Este script configura el entorno de Colab para ejecutar el proyecto.
Alternativa al notebook para usuarios que prefieren scripts Python.

@author: santi
"""

import os
import sys
from pathlib import Path

print("="*70)
print("SETUP GOOGLE COLAB - PROYECTO TFM IBEX35")
print("="*70)

# ============================================================================
# 1. VERIFICAR ENTORNO
# ============================================================================

def check_environment():
    """Detecta si está en Colab"""
    try:
        import google.colab
        return 'colab'
    except:
        return 'local'

env = check_environment()
print(f"\n✓ Entorno detectado: {env.upper()}")

if env != 'colab':
    print("⚠ Este script está diseñado para Google Colab")
    print("  Para ejecución local, usar main_unified.py directamente")
    response = input("¿Continuar de todos modos? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)

# ============================================================================
# 2. VERIFICAR GPU
# ============================================================================

print("\n" + "="*70)
print("VERIFICANDO GPU")
print("="*70)

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU disponible: {len(gpus)} dispositivo(s)")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("⚠ No se detectó GPU")
        print("  Recomendación: Runtime → Change runtime type → GPU: T4")
except ImportError:
    print("⚠ TensorFlow no instalado aún")

# ============================================================================
# 3. MONTAR GOOGLE DRIVE
# ============================================================================

print("\n" + "="*70)
print("MONTANDO GOOGLE DRIVE")
print("="*70)

if env == 'colab':
    from google.colab import drive
    
    # Verificar si ya está montado
    if os.path.exists('/content/drive'):
        print("✓ Google Drive ya montado")
    else:
        print("Montando Google Drive...")
        drive.mount('/content/drive', force_remount=False)
        print("✓ Google Drive montado")

# ============================================================================
# 4. CONFIGURAR RUTAS
# ============================================================================

print("\n" + "="*70)
print("CONFIGURANDO RUTAS")
print("="*70)

# Solicitar ruta del proyecto o usar default
if env == 'colab':
    default_path = '/content/drive/MyDrive/TFM'
else:
    default_path = str(Path.home() / "Master" / "TFM" / "py_project")

print(f"\nRuta por defecto: {default_path}")
response = input("¿Usar esta ruta? (y/n): ")

if response.lower() == 'y':
    PROJECT_PATH = Path(default_path)
else:
    custom_path = input("Ingresa la ruta completa del proyecto: ")
    PROJECT_PATH = Path(custom_path)

# Rutas derivadas
SCRIPTS_PATH = PROJECT_PATH / 'scripts'
DATA_PATH = PROJECT_PATH / 'input_data'
RESULTS_PATH = PROJECT_PATH / 'results'

# Verificar que existen
print("\nVerificando estructura del proyecto:")
paths_ok = True

for name, path in [
    ('Proyecto', PROJECT_PATH),
    ('Scripts', SCRIPTS_PATH),
    ('Data', DATA_PATH),
    ('Results', RESULTS_PATH)
]:
    exists = path.exists()
    status = '✓' if exists else '✗'
    print(f"  {status} {name}: {path}")
    if not exists and name in ['Proyecto', 'Scripts', 'Data']:
        paths_ok = False

if not paths_ok:
    print("\n❌ Estructura del proyecto incompleta")
    print("Por favor, verifica las rutas y vuelve a ejecutar")
    sys.exit(1)

# Añadir al path
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

print(f"\n✓ Python path configurado")

# Cambiar directorio de trabajo
os.chdir(str(SCRIPTS_PATH))
print(f"✓ Directorio de trabajo: {os.getcwd()}")

# ============================================================================
# 5. INSTALAR DEPENDENCIAS
# ============================================================================

print("\n" + "="*70)
print("INSTALANDO DEPENDENCIAS")
print("="*70)

required_packages = {
    'pyreadr': 'pyreadr',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm'
}

import subprocess

for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"✓ {module} ya instalado")
    except ImportError:
        print(f"Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"✓ {module} instalado")

# ============================================================================
# 6. VERIFICAR ARCHIVOS DEL PROYECTO
# ============================================================================

print("\n" + "="*70)
print("VERIFICANDO ARCHIVOS DEL PROYECTO")
print("="*70)

required_files = [
    'config.py',
    'aux_functions.py',
    'modelos_ml.py',
    'lstm_models.py',
    'visualization.py'
]

# Buscar main script
main_files = ['main_unified.py', 'main.py']
main_script = None
for f in main_files:
    if (SCRIPTS_PATH / f).exists():
        main_script = f
        break

if main_script:
    required_files.append(main_script)
else:
    print("⚠ No se encontró main_unified.py ni main.py")

missing_files = []
for file in required_files:
    exists = (SCRIPTS_PATH / file).exists()
    status = '✓' if exists else '✗'
    print(f"  {status} {file}")
    if not exists:
        missing_files.append(file)

if missing_files:
    print(f"\n⚠ Archivos faltantes: {missing_files}")
    print("Por favor, sube estos archivos a scripts/ en Google Drive")
    sys.exit(1)
else:
    print("\n✓ Todos los archivos necesarios están presentes")

# ============================================================================
# 7. TEST DE IMPORTACIÓN
# ============================================================================

print("\n" + "="*70)
print("PROBANDO IMPORTS")
print("="*70)

try:
    # Imports estándar
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✓ Imports estándar: OK")
    
    # Config
    from config import config, MODELS, PLOTS, CSV
    print("✓ Config: OK")
    
    # Aux functions
    from aux_functions import load_rds, prepare_data
    print("✓ Aux functions: OK")
    
    # Modelos ML
    from modelos_ml import train_tree_models_regression
    print("✓ Modelos ML: OK")
    
    # LSTM
    from lstm_models import train_lstm_regression
    print("✓ LSTM models: OK")
    
    # Visualization
    from visualization import plot_statistical_results
    print("✓ Visualization: OK")
    
    print("\n✓ Todos los módulos se importaron correctamente")
    
except Exception as e:
    print(f"\n❌ Error en imports: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 8. CONFIGURAR ENTORNO ML
# ============================================================================

print("\n" + "="*70)
print("CONFIGURANDO ENTORNO ML")
print("="*70)

config.setup_ml_environment()
config.info()

# ============================================================================
# 9. TEST RÁPIDO OPCIONAL
# ============================================================================

print("\n" + "="*70)
print("TEST RÁPIDO (OPCIONAL)")
print("="*70)

response = input("\n¿Ejecutar test rápido de carga de datos? (y/n): ")

if response.lower() == 'y':
    print("\nCargando dataset de prueba...")
    try:
        datasets_config = config.get_dataset_config()
        dataset_name = 'financial_scaled'
        dataset_info = datasets_config[dataset_name]
        
        train = load_rds(str(dataset_info['train']))
        test = load_rds(str(dataset_info['test']))
        
        print(f"✓ Train: {train.shape}")
        print(f"✓ Test: {test.shape}")
        print(f"✓ Columnas: {list(train.columns[:5])}...")
        
        print("\n✓ Test de carga exitoso")
    except Exception as e:
        print(f"\n❌ Error en test: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 10. RESUMEN Y PRÓXIMOS PASOS
# ============================================================================

print("\n" + "="*70)
print("SETUP COMPLETADO")
print("="*70)

print("\n✓ Entorno configurado correctamente")
print(f"✓ Proyecto: {PROJECT_PATH}")
print(f"✓ Scripts: {SCRIPTS_PATH}")
print(f"✓ Main script: {main_script if main_script else 'NO ENCONTRADO'}")

print("\n" + "="*70)
print("PRÓXIMOS PASOS")
print("="*70)

if main_script:
    print(f"\n1. Ejecutar el pipeline completo:")
    print(f"   %run {main_script}")
    print("\n   O desde terminal:")
    print(f"   !python {main_script}")
    
    print("\n2. O ejecutar paso a paso:")
    print("   - Cargar datos con load_rds()")
    print("   - Preparar con prepare_data()")
    print("   - Entrenar modelos individualmente")
    
    print("\n3. Analizar resultados:")
    print(f"   - CSV: {CSV}")
    print(f"   - Modelos: {MODELS}")
    print(f"   - Plots: {PLOTS}")
else:
    print("\n⚠ No se encontró main_unified.py ni main.py")
    print("Por favor, sube uno de estos archivos para ejecutar el pipeline completo")

print("\n" + "="*70)
print("CONFIGURACIÓN DE ENTRENAMIENTO")
print("="*70)

print("""
Para ajustar qué modelos entrenar, edita el archivo main script:

# Datasets a procesar
DATASETS_TO_PROCESS = ['financial_scaled']

# Targets
TARGETS_REGRESSION = ['returns_next']
TARGETS_CLASSIFICATION = ['direction_next']

# Modelos a entrenar
TRAIN_LINEAR = True
TRAIN_TREES = True
TRAIN_MLP = True
TRAIN_GRU = False
TRAIN_LSTM = True
""")

print("\n✓ Setup finalizado exitosamente")
print("="*70)
