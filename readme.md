---
title: "Predicción de valores y tendencias de cierre del IBEX35 mediante *machine learning* y *webscraping*"
subtitle: "Trabajo de fin de Máster"
author: "Santiago López Begines, PhD"
toc-title: "Índice"
output: 
  pdf_document: 
    toc: true
    number_sections: true
    latex_engine: xelatex
    highlight: tango
---


# Descripción

Trabajo Fin de Máster en Data Science centrado en la predicción de movimientos direccionales del IBEX35 mediante machine learning y análisis de sentimiento de noticias financieras extraídas de GDELT.

# Estructura del Proyecto

```         
.
├── TFM_Santiago_Lopez_Begines.pdf      # Memoria principal del TFM
├── Anexos/                             # Documentación y reportes generados
├── EDA_RStudio/                        # Análisis exploratorio y preprocesamiento (R)
└── ML_Colab/                           # Pipeline de machine learning (Python)
```

------------------------------------------------------------------------

# 📄 Anexos/

Documentación completa del proyecto en formato HTML y PDF:

-   **Anexo 1**: Documentación técnica de variables financieras
-   **Anexos Fase 1-6**: Reportes completos generados desde archivos `.qmd`
    -   **Fase 1**: Exploración y preparación de datos financieros del IBEX35
    -   **Fase 2**: Feature engineering y análisis de correlaciones
    -   **Fase 3**: Descarga y preprocesamiento de datos GDELT
    -   **Fase 4**: Análisis de sentimiento e integración de variables
    -   **Fase 5**: Desarrollo y comparación de modelos ML/DL
    -   **Fase 6**: Validación final y conclusiones

------------------------------------------------------------------------

# 💻 EDA_RStudio/

Análisis exploratorio de datos, preprocesamiento y generación de features implementado en R.

## Estructura

```         
EDA_RStudio/
├── Fase1.qmd - Fase6.qmd    # Documentos Quarto con análisis y reportes
└── code/
    ├── R/                    # Scripts R organizados numéricamente
    ├── py/                   # Scripts Python auxiliares
    └── sh/                   # Scripts shell para procesamiento batch
```

## Archivos `.qmd` (Quarto Markdown)

Documentos principales que contienen el análisis completo y generan los reportes en `Anexos/`.
Cada `.qmd`: - Integra narrativa, código y visualizaciones - Realiza llamadas a funciones específicas mediante `source()` desde `code/R/` - Genera reportes HTML y PDF reproducibles

**Orden de ejecución**: Fase1.qmd → Fase2.qmd → ...
→ Fase6.qmd

## `code/R/`

Scripts organizados numéricamente según el flujo del pipeline:

### Configuración inicial (00-06)

-   `00.libraries.R`: Carga de paquetes R necesarios
-   `00.python_libraries.R`: Configuración de reticulate para integración Python-R
-   `01.general_functions.R`: Funciones auxiliares generales
-   `02.EDA_functions.R`: Funciones para análisis exploratorio
-   `03.transfer_py_to_r.R`: Transferencia de objetos entre Python y R
-   `04.financial_features.R`: Generación de indicadores técnicos y features financieras
-   `05.external_features.R`: Variables de mercados externos (S&P500, Euro Stoxx, commodities)
-   `06.cleaning_features.R`: Limpieza y tratamiento de valores atípicos

### Procesamiento GDELT (11-19)

-   `11.web_scraping_download_gdelt_parallel.R`: Descarga paralela de archivos GDELT (\>2000 ZIP)
-   `convert_zip_to_parquet.R`: Conversión de ZIP a formato Parquet
-   `12.filter_parquet_files.R`: Filtrado inicial de noticias relevantes al IBEX35
-   `13.filter_script.R`: Filtrado avanzado por palabras clave
-   `14.consolidate_filtered_batches.R`: Consolidación de batches filtrados
-   `15.sentiment_score.R`: Cálculo de scores de sentimiento (tone)
-   `16.analysis_sentiment_counts_ibex35.R`: Análisis de frecuencias de noticias
-   `17.analysis_sentiment_intensity_ibex35.R`: Análisis de intensidad de sentimiento
-   `18.Lags_sentimientos.R`: Generación de lags temporales de sentimiento
-   `19.cleaning_sentiment.R`: Limpieza final de variables de sentimiento

### Preparación para ML (20-25)

-   `20.feature_scaling.R`: Normalización de features para modelos ML
-   `21.forecasting_models.R`: Implementación de modelos de series temporales (ARIMA, Prophet)
-   `22.evaluate_naive_models.R`: Evaluación de modelos
-   `23.compare_predictions.R`: Comparación estadística de predicciones
-   `24.scaling_validation_data.R`: Escalado de datos de validación
-   `25.verify_consistency.R`: Verificación de consistencia de datos

## `code/py/`

Scripts Python auxiliares llamados desde R: - `ibex_downloader.py`: Descarga histórica de datos del IBEX35 (Yahoo Finance) - `stocks_downloader.py`: Descarga de datos de componentes individuales del IBEX35 - `stocks_list.py`: Gestión de lista de componentes del índice - `validate_model.py`: Validación de modelos entrenados

## `code/sh/`

Scripts shell para procesamiento eficiente: - `step1_zip_to_parquet.sh`: Conversión masiva de archivos GDELT - `02b_prefilter_fast.sh`: Pre-filtrado rápido de datos - `bootstrap_system_deps.sh`: Instalación de dependencias del sistema

------------------------------------------------------------------------

# 🤖 ML_Colab/

Pipeline de machine learning implementado en Python para ejecución en Google Colab (GPU).

## Estructura

```         
ML_Colab/
├── pipeline_ML_ibex35.ipynb    # Notebook principal con pipeline completo
├── README.md                   # Documentación específica del pipeline ML
├── environment.yml             # Especificación del entorno conda
├── setup_colab.py             # Configuración automática para Colab
├── setup_project.sh           # Script de configuración del proyecto
└── scripts/                   # Módulos Python del pipeline
```

## `scripts/`

Módulos organizados por funcionalidad:

### Configuración y utilidades

-   `config.py`: Parámetros globales y configuración del proyecto
-   `aux_functions.py`: Funciones auxiliares (carga de datos, métricas, etc.)

### Modelos implementados

-   `modelos_ml.py`: Modelos tradicionales (XGBoost, LightGBM, Random Forest, GRU, MLP)
-   `lstm_models.py`: Modelos de deep learning (LSTM,)
-   `evaluate_naive_models.py`: Evaluación de modelos

### Evaluación y validación

-   `evaluar_todos_modelos.py`: Pipeline de evaluación comparativa de todos los modelos
-   `validate_lightgbm.py`: Validación específica del modelo LightGBM
-   `validation_main.py`: Pipeline de validación en conjunto de test

### Análisis y visualización

-   `visualization.py`: Generación de gráficos y reportes visuales
-   `analize_log.py`: Análisis de logs de entrenamiento
-   `visualize_models_structure.py`: Visualización de arquitecturas de redes neuronales

### Ejecución

-   `main_pipeline.py`: Script principal para ejecutar el pipeline completo

------------------------------------------------------------------------

# Requisitos

## R (EDA_RStudio/)

-   R \>= 4.0
-   Paquetes principales:
    -   `tidyverse`: Manipulación y visualización de datos
    -   `quantmod`, `TTR`: Análisis financiero e indicadores técnicos
    -   `arrow`: Manejo de archivos Parquet
    -   `reticulate`: Integración Python-R
    -   `parallel`: Procesamiento paralelo
    -   `quarto`: Generación de reportes

## Python (ML_Colab/)

-   Python \>= 3.8
-   Entorno especificado en `ML_Colab/environment.yml`
-   Librerías principales:
    -   `pandas`, `numpy`: Manipulación de datos
    -   `scikit-learn`: Preprocesamiento y métricas
    -   `xgboost`, `lightgbm`: Modelos gradient boosting
    -   `tensorflow`, `keras`: Deep learning
    -   `statsmodels`: Modelos de series temporales
    -   `prophet`: Forecasting con modelos aditivos

------------------------------------------------------------------------

# Ejecución

## Análisis Exploratorio y Preprocesamiento (R)

1.  **Configurar entorno R**: Instalar paquetes listados en `00.libraries.R`
2.  **Ejecutar análisis por fases**: Renderizar documentos Quarto en orden secuencial

``` r
   quarto::quarto_render("EDA_RStudio/Fase1.qmd")
   quarto::quarto_render("EDA_RStudio/Fase2.qmd")
   # ... continuar con Fase3-6
```

3.  **Salidas**: Los reportes HTML/PDF se generan en `Anexos/`

Los archivos `.qmd` automáticamente ejecutan los scripts necesarios mediante `source()`.

## Pipeline de Machine Learning (Python)

### Opción 1: Google Colab (recomendado)

1.  Subir carpeta `ML_Colab/` a Google Drive
2.  Abrir `pipeline_ML_ibex35.ipynb` en Colab
3.  Ejecutar `setup_colab.py` para configurar entorno
4.  Ejecutar celdas secuencialmente

### Opción 2: Entorno local

``` bash
# Crear entorno conda
conda env create -f ML_Colab/environment.yml
conda activate tfm_ml

# Ejecutar pipeline completo
python ML_Colab/scripts/main_pipeline.py

# O ejecutar módulos específicos
python ML_Colab/scripts/evaluar_todos_modelos.py
```

------------------------------------------------------------------------

# Metodología

## Datos

-   **Financieros**: IBEX35 y componentes (2004-2024) via Yahoo Finance
-   **Sentimiento**: \>2000 archivos GDELT filtrados por relevancia al IBEX35
-   **Variables externas**: S&P500, Euro Stoxx 50, petróleo, oro, EUR/USD

## Features

-   Indicadores técnicos: RSI, MACD, Bollinger Bands, medias móviles
-   Variables de sentimiento: tone, conteos de noticias con lags
-   Variables de mercado: retornos, volatilidad, volumen
-   Total: \~50 features tras selección

## Modelos Implementados

-   **Tradicionales**: XGBoost, LightGBM, Random Forest
-   **Deep Learning**: LSTM, GRU, MLP
-   **Series temporales**: ARIMA, Prophet

## Evaluación

-   Métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Cohen's Kappa
-   Comparaciones estadísticas: McNemar, Diebold-Mariano
-   Intervalos de confianza: Bootstrap (1000 iteraciones)

------------------------------------------------------------------------

# Resultados Principales

-   **Mejor modelo**: LightGBM con 55-62% de precisión direccional
-   **Sentimiento**: Impacto marginal, mejoras no consistentemente significativas
-   **Variables clave**: RSI, medias móviles, retornos pasados
-   **Comparativa**: Modelos ML superan a ARIMA/Prophet y baselines
-   **Deep Learning**: Sin ventajas claras sobre métodos tradicionales

------------------------------------------------------------------------

# Notas Importantes

-   Los datos GDELT originales (\~150GB) y datasets procesados no se incluyen por tamaño
-   Los modelos entrenados (.joblib, .keras) están disponibles bajo petición
-   Reproducción completa requiere:
    -   Descargar datos GDELT (Fase 3)
    -   Ejecutar pipeline R completo (12-24 horas)
    -   Entrenar modelos en GPU (4-8 horas en Colab)


------------------------------------------------------------------------

# Licencia

Este proyecto académico se entrega como Trabajo Fin de Máster.
El código está disponible para fines educativos.
