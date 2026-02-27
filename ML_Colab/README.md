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

pip install tensorflow transformers pyreader torch
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
