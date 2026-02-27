#!/bin/bash

# Script de configuración para proyecto Python
# Uso: bash setup_project.sh

PROJECT_NAME="project-ibex"

echo "Creando estructura de directorios..."

# Crear directorios
mkdir -p data/{raw,processed}
mkdir -p notebooks
mkdir -p scripts
mkdir -p results/{figures,models,plots,csv}
mkdir -p input_data/{ML_financial,ML_financial_long,ML_sentiment}

# Crear .gitignore
cat > .gitignore << 'EOF'
# Datos
data/raw/
data/processed/
input_data/

# Resultados
results/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
.venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# Sistema
.DS_Store
Thumbs.db
EOF

# Crear README.md
cat > README.md << 'EOF'
# Project IBEX35 - Machine Learning & Sentiment Analysis

Predicción de movimientos del IBEX35 mediante modelos de machine learning y análisis de sentimiento.

## Entorno de desarrollo

### Crear el entorno conda

```bash
conda create -n project-ibex python=3.10 -y
conda activate project-ibex
```

### Instalar dependencias básicas

```bash
conda install -c conda-forge \
    pandas numpy scipy scikit-learn \
    matplotlib seaborn plotly \
    jupyter jupyterlab spyder \
    xgboost lightgbm -y

pip install tensorflow transformers
```

### Exportar entorno (después de instalar todo)

```bash
conda env export > environment.yml
```

### Replicar entorno en otra máquina

```bash
conda env create -f environment.yml
conda activate project-ibex
```

## Estructura del proyecto

```
.
├── data/
│   ├── raw/              # Datos originales (no versionados)
│   └── processed/        # Datos procesados (no versionados)
├── input_data/           # Datasets para ML (no versionados)
│   ├── ML_financial/
│   ├── ML_financial_long/
│   └── ML_sentiment/
├── notebooks/            # Jupyter notebooks para exploración
│   └── colab_example.ipynb
├── scripts/              # Scripts de Python (.py)
│   ├── config.py         # Configuración portable (auto-detecta entorno)
│   ├── utils.py          # Funciones auxiliares
│   └── 01_example.py     # Ejemplo de uso
├── results/              # Resultados (no versionados)
│   ├── figures/
│   ├── models/
│   ├── plots/
│   └── csv/
├── environment.yml       # Dependencias del proyecto
├── .gitignore
└── README.md
```

## Uso portable (Local / Colab / Kaggle)

El proyecto detecta automáticamente dónde se ejecuta y ajusta las rutas.

### Uso local (Spyder/VS Code/Jupyter)

```python
from config import config
from utils import load_data, save_results

# Info del entorno
config.info()

# Cargar datos
df = load_data('mi_archivo.csv')

# Guardar resultados
save_results(df, 'resultados.csv', subdir='analisis')
```

### Uso en Google Colab

1. Sube la carpeta `project-ibex` a tu Google Drive
2. En Colab, importa config (monta Drive automáticamente):

```python
import sys
sys.path.append('/content/drive/MyDrive/project-ibex/scripts')

from config import config  # Monta Drive automáticamente
from utils import load_data, save_results

config.info()  # Verifica rutas
```

3. O ejecuta tu script completo:

```python
!python /content/drive/MyDrive/project-ibex/scripts/01_example.py
```

### Ajustar ruta de Drive (si es necesario)

Edita `scripts/config.py` línea 28:

```python
self.PROJECT_ROOT = Path('/content/drive/MyDrive/TU-CARPETA-AQUI')
```

## Flujo de trabajo

1. **Desarrollo local**: Prototipa en Spyder con datos pequeños
2. **Entrenamiento en Colab**: Sube código, ejecuta con GPU
3. **Resultados automáticos**: Guarda en `results/`, sincroniza con Drive

## Notas

- `config.py` detecta automáticamente Colab/Kaggle/Local
- `utils.py` gestiona rutas y guardado portablemente
- Todos los scripts usan las mismas funciones
- Versioniar solo código, no datos ni resultados
EOF

# Crear config.py para detección de entorno
cat > scripts/config.py << 'EOF'
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
            self.PROJECT_ROOT = Path('/content/drive/MyDrive/project-ibex')
        elif self.environment == 'kaggle':
            self.PROJECT_ROOT = Path('/kaggle/working')
        else:  # local
            # Asume que config.py está en scripts/
            self.PROJECT_ROOT = Path(__file__).parent.parent
        
        # Rutas principales
        self.DATA_RAW = self.PROJECT_ROOT / "data" / "raw"
        self.DATA_PROCESSED = self.PROJECT_ROOT / "data" / "processed"
        self.SCRIPTS = self.PROJECT_ROOT / "scripts"
        self.RESULTS = self.PROJECT_ROOT / "results"
        self.FIGURES = self.RESULTS / "figures"
        self.MODELS = self.RESULTS / "models"
        
        # Crear directorios si no existen
        for path in [self.DATA_RAW, self.DATA_PROCESSED, 
                     self.FIGURES, self.MODELS]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Añadir scripts al path para imports
        if str(self.SCRIPTS) not in sys.path:
            sys.path.insert(0, str(self.SCRIPTS))
    
    def info(self):
        """Imprime información del entorno"""
        print(f"Entorno: {self.environment.upper()}")
        print(f"Proyecto: {self.PROJECT_ROOT}")
        print(f"Data: {self.DATA_RAW}")
        print(f"Results: {self.RESULTS}")

# Instancia global
config = Config()

# Atajos para importar
PROJECT_ROOT = config.PROJECT_ROOT
DATA_RAW = config.DATA_RAW
DATA_PROCESSED = config.DATA_PROCESSED
RESULTS = config.RESULTS
FIGURES = config.FIGURES
MODELS = config.MODELS
EOF

# Crear utils.py con funciones auxiliares
cat > scripts/utils.py << 'EOF'
"""
Utilidades comunes para el proyecto IBEX35
"""

import pandas as pd
import numpy as np
from config import config, DATA_RAW, DATA_PROCESSED, RESULTS

def load_data(filename, processed=False):
    """Carga datos desde raw o processed"""
    path = DATA_PROCESSED if processed else DATA_RAW
    filepath = path / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"No existe: {filepath}")
    
    return pd.read_csv(filepath)

def save_results(df, filename, subdir=""):
    """Guarda resultados en results/"""
    save_path = RESULTS / subdir
    save_path.mkdir(exist_ok=True, parents=True)
    
    filepath = save_path / filename
    df.to_csv(filepath, index=False)
    print(f"✓ Guardado: {filepath}")
    
    return filepath

def save_model(model, filename, subdir=""):
    """Guarda modelo en results/models/"""
    import pickle
    from config import MODELS
    
    save_path = MODELS / subdir
    save_path.mkdir(exist_ok=True, parents=True)
    
    filepath = save_path / filename
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Modelo guardado: {filepath}")
    
    return filepath

def load_model(filename, subdir=""):
    """Carga modelo desde results/models/"""
    import pickle
    from config import MODELS
    
    filepath = MODELS / subdir / filename
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Modelo cargado: {filepath}")
    
    return model
EOF

# Crear ejemplo de uso
cat > scripts/01_example.py << 'EOF'
"""
Ejemplo de script portable para local y Colab
"""

from config import config
from utils import load_data, save_results
import pandas as pd

def main():
    # Info del entorno
    config.info()
    print("\n" + "="*50 + "\n")
    
    # Ejemplo: crear datos dummy
    print("Creando datos de ejemplo...")
    df = pd.DataFrame({
        'fecha': pd.date_range('2024-01-01', periods=100),
        'ibex': np.random.randn(100).cumsum() + 10000,
        'volumen': np.random.randint(1000, 5000, 100)
    })
    
    # Guardar
    save_results(df, 'ejemplo.csv', subdir='test')
    
    # Cargar (si existe)
    # df_loaded = load_data('ejemplo.csv', processed=True)
    
    print("\n✓ Script ejecutado correctamente")

if __name__ == "__main__":
    import numpy as np
    main()
EOF

# Crear notebook de ejemplo para Colab
cat > notebooks/colab_example.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Ejemplo de uso en Colab\n",
    "\n",
    "Este notebook detecta automáticamente que está en Colab y monta Drive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Instalar dependencias si es necesario\n",
    "# !pip install xgboost lightgbm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importar config (monta Drive automáticamente si es Colab)\n",
    "import sys\n",
    "sys.path.append('/content/drive/MyDrive/project-ibex/scripts')\n",
    "\n",
    "from config import config\n",
    "config.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ejecutar tu script\n",
    "from utils import load_data, save_results\n",
    "import pandas as pd\n",
    "\n",
    "# Tu código aquí..."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo ""
echo "✓ Estructura creada exitosamente"
echo ""
echo "Archivos creados:"
echo "  - scripts/config.py      (detección de entorno)"
echo "  - scripts/utils.py       (funciones auxiliares)"
echo "  - scripts/01_example.py  (ejemplo de uso)"
echo "  - notebooks/colab_example.ipynb"
echo ""
echo "Siguiente paso:"
echo "  1. conda activate project-ibex"
echo "  2. Instalar dependencias (ver README.md)"
echo "  3. conda env export > environment.yml"
echo ""
echo "Para Colab:"
echo "  - Sube la carpeta 'project-ibex' a tu Google Drive"
echo "  - Abre colab_example.ipynb desde Colab"
echo ""
