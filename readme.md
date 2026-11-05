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

Trabajo Fin de Máster (diciembre 2025) del Máster en Big Data y Data Science aplicados a la Economía y a la Administración y Dirección de Empresas.

Desarrolla un sistema predictivo de los movimientos del IBEX35 mediante *machine learning*, integrando datos financieros (2004-2025) con análisis de sentimiento de noticias extraídas del proyecto GDELT (GKG 1.0).

Se implementaron **348 configuraciones experimentales** (clasificación y regresión) combinando:

-   **Modelos**: clásicos (ARIMA, ARIMAX, Prophet), lineales, basados en árboles, MLP y redes recurrentes (GRU, LSTM).
-   **Datasets**: escalados / sin escalar, serie larga (2004-2024) / serie corta (2014-2024), con / sin sentimiento.
-   **Horizontes**: 1, 5, 10 y 20 días.

> **Nota sobre el enfoque inicial**: el diseño original contemplaba hacer *web scraping* de prensa financiera (Cinco Días, Expansión, Reuters, CNBC, etc.). Las limitaciones técnicas y legales (RSS limitados, anti-scraping, suscripciones de pago, heterogeneidad de formatos y falta de histórico) llevaron a adoptar GDELT como fuente primaria de sentimiento. GDELT usa análisis léxico tradicional (no modelos de lenguaje contextuales) y no está especializado en mercados financieros, por lo que el sentimiento obtenido es básico.

# Estructura del Proyecto

```         
.
├── TFM_Santiago_Lopez_Begines.pdf      # Memoria principal del TFM (documento de referencia)
├── EDA_RStudio/                        # Análisis exploratorio y preprocesamiento (R)
└── ML_Colab/                           # Pipeline de machine learning (Python)
```

Los anexos a los que hace referencia la memoria (documentación técnica de variables y reportes de las fases) se generan a partir de los `.qmd` de `EDA_RStudio/`.

------------------------------------------------------------------------

# Fases del proyecto

La memoria estructura el trabajo en seis fases, cada una con su documento Quarto (`FaseN.qmd`):

| Fase | Contenido |
|------------------------|------------------------------------------------|
| **1** | Descarga y consolidación de datos: financieros (Yahoo Finance) y noticias GDELT |
| **2** | Análisis exploratorio de índices bursátiles (calidad de datos, retornos, volatilidad, normalidad, estacionalidad, correlaciones, causalidad de Granger) y de las noticias GDELT (scores de sentimiento, conteos, intensidad) |
| **3** | Ingeniería de características: variables técnicas y exógenas (97 variables), limpieza, reducción por correlación y Random Forest (44 variables finales), limpieza del sentimiento |
| **4** | Unión de datos financieros y de sentimiento, creación de targets de clasificación y normalización Z-score |
| **5** | Predicción: modelos clásicos (ARIMA, ARIMAX, Prophet) y modelos ML/DL; comparación estadística y evaluación frente a un modelo ingenuo (*naïve*) |
| **6** | Validación final con datos nuevos (17-oct-2025 a 30-nov-2025) |

------------------------------------------------------------------------

# 💻 EDA_RStudio/

Análisis exploratorio, preprocesamiento y generación de features, implementado en R (con Python integrado mediante `reticulate`).

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

Documentos principales que contienen el análisis completo. Cada `.qmd`:

-   Integra narrativa, código y visualizaciones.
-   Llama a funciones específicas mediante `source()` desde `code/R/`.
-   Genera reportes HTML y PDF reproducibles.

**Orden de ejecución**: Fase1.qmd → Fase2.qmd → ... → Fase6.qmd

Todo el código detallado de los modelos clásicos (ARIMA, ARIMAX, Prophet) de la Fase 5 se encuentra en `Fase5.qmd`.

## `code/R/`

Scripts organizados numéricamente según el flujo del pipeline:

### Configuración y features financieras (00-06)

-   `00.libraries.R`: Carga de paquetes R necesarios
-   `00.python_libraries.R`: Configuración de `reticulate` para integración Python-R
-   `01.general_functions.R`: Funciones auxiliares generales
-   `02.EDA_functions.R`: Funciones para análisis exploratorio
-   `03.transfer_py_to_r.R`: Transferencia de los datos descargados en Python a R (guarda en `.RDS`, `.xlsx` y `.parquet`)
-   `04.financial_features.R`: Indicadores técnicos derivados de OHLCV, con split temporal estricto (sin *look-ahead bias*)
-   `05.external_features.R`: Variables de índices y activos externos (S&P500, DAX, FTSE100, Euronext100, VIX, petróleo, oro, EUR/USD) y Euribor 3M
-   `06.cleaning_features.R`: Limpieza de variables (NAs, petróleo negativo en COVID-19), reducción por correlación/varianza (`reduce_by_correlation_variance()`) y ranking por Random Forest (`rank_features_by_importance()`)

### Procesamiento GDELT (11-19)

-   `11.web_scraping_download_gdelt_parallel.R`: Descarga paralela de los archivos diarios GDELT GKG 1.0 (`YYYYMMDD.gkg.csv.zip`)
-   `convert_zip_to_parquet.R`: Conversión de ZIP a formato Parquet (se ejecuta desde `code/sh/step1_zip_to_parquet.sh`)
-   `12.filter_parquet_files.R`: Filtrado de noticias relevantes para el IBEX35 con `duckdb`
-   `13.filter_script.R`: Script de filtrado (criterios por temas, organizaciones y fuentes)
-   `14.consolidate_filtered_batches.R`: Consolidación de los lotes filtrados en un único Parquet
-   `15.sentiment_score.R`: Procesado de los seis campos de TONE y agregación diaria/mensual/anual (`daily_sentiment`)
-   `16.analysis_sentiment_counts_ibex35.R`: Conteo de noticias positivas/negativas (cuatro categorías de sentimiento)
-   `17.analysis_sentiment_intensity_ibex35.R`: Categorización de valencia, intensidad (polarity) y volatilidad (SD del tone)
-   `18.Lags_sentimientos.R`: Lags y efectos retardados del sentimiento
-   `19.cleaning_sentiment.R`: Limpieza final del sentimiento (forward fill, mediana al inicio, correlación y varianza)

### Preparación y evaluación (20-25)

-   `20.feature_scaling.R`: Normalización Z-score (parámetros estimados solo en train y aplicados a test)
-   `21.forecasting_models.R`: Funciones de evaluación de pronósticos (RMSE, MAE, MAPE, *Direction Accuracy*) usadas con los modelos clásicos
-   `22.evaluate_naive_models.R`: Evaluación de modelos reales vs. ingenuos (`evaluate_all_models()`): métricas, test binomial y criterios *naïve*
-   `23.compare_predictions.R`: Comparación estadística de modelos: McNemar + bootstrap (clasificación) y Diebold-Mariano + bootstrap (regresión)
-   `24.scaling_validation_data.R`: Cálculo de features y escalado de los datos de validación con los parámetros del entrenamiento
-   `25.verify_consistency.R`: Verificación de que las features de validación coinciden con las del modelo

## `code/py/`

Scripts Python auxiliares llamados desde R:

-   `ibex_downloader.py`: Descarga histórica de datos del IBEX35 (Yahoo Finance)
-   `stocks_downloader.py`: Función `download_ibex()` para la descarga de índices con `yfinance`
-   `stocks_list.py`: Diccionario `stocks_to_download` con los tickers utilizados
-   `validate_model.py`: Carga, verificación de features y evaluación de modelos entrenados sobre los datos de validación

## `code/sh/`

Scripts shell para procesamiento eficiente:

-   `step1_zip_to_parquet.sh`: Conversión masiva de archivos GDELT ZIP a Parquet (ejecutado desde terminal por limitaciones de memoria en RStudio)
-   `02b_prefilter_fast.sh`: Pre-filtrado rápido de datos
-   `bootstrap_system_deps.sh`: Instalación de dependencias del sistema

------------------------------------------------------------------------

# 🤖 ML_Colab/

Pipeline de machine learning implementado en Python para ejecución en Google Colab (GPU).

## Estructura

```         
ML_Colab/
├── pipeline_ML_ibex35.ipynb    # Notebook principal con pipeline completo
├── README.md                   # Documentación específica del pipeline ML
├── environment.yml             # Especificación del entorno conda
├── setup_colab.py              # Configuración automática para Colab
├── setup_project.sh            # Script de configuración del proyecto
└── scripts/                    # Módulos Python del pipeline
```

## `scripts/`

### Configuración y utilidades

-   `config.py`: Parámetros globales y configuración del proyecto
-   `aux_functions.py`: Funciones auxiliares (carga de datos, métricas, etc.)

### Modelos implementados

-   `modelos_ml.py`: Modelos lineales, de árboles y MLP (Linear, Ridge, Logistic, Random Forest, XGBoost, LightGBM, Ensemble/Stacking, MLP)
-   `lstm_models.py`: Modelos recurrentes (GRU y LSTM; arquitecturas small, medium, large, etc.)
-   `evaluate_naive_models.py`: Evaluación de modelos reales vs. ingenuos

### Evaluación y validación

-   `evaluar_todos_modelos.py`: Pipeline de evaluación comparativa de todos los modelos
-   `validate_lightgbm.py`: Validación específica del modelo LightGBM
-   `validation_main.py`: Pipeline de validación con datos nuevos

### Análisis y visualización

-   `visualization.py`: Generación de gráficos y reportes visuales
-   `analize_log.py`: Análisis de logs de entrenamiento
-   `visualize_models_structure.py`: Visualización de arquitecturas de redes neuronales

### Ejecución

-   `main_pipeline.py`: Script principal para ejecutar el pipeline completo

------------------------------------------------------------------------

# Requisitos

## R (EDA_RStudio/)

-   R 4.5.2 y RStudio 2025.09.2
-   Paquetes principales:
    -   `tidyverse`, `lubridate`, `zoo`: Manipulación de datos y series temporales
    -   `duckdb`, `arrow`: Filtrado y manejo de archivos Parquet
    -   `reticulate`: Integración Python-R
    -   `caret`, `randomForest`: Reducción de variables y ranking por importancia
    -   `forecast`, `prophet`: Modelos clásicos (ARIMA/ARIMAX, Prophet) y test de Diebold-Mariano
    -   `fredr`: Descarga del Euribor 3M (requiere API key propia de FRED)
    -   `quarto`: Generación de reportes

## Python (ML_Colab/)

-   Python 3.13.5
-   Entorno especificado en `ML_Colab/environment.yml`
-   Librerías principales:
    -   `yfinance`: Descarga de datos financieros
    -   `pandas`, `numpy`: Manipulación de datos
    -   `scikit-learn`: Modelos lineales, Random Forest, MLP, stacking, preprocesamiento y métricas
    -   `xgboost`, `lightgbm`: Modelos gradient boosting
    -   `tensorflow`, `keras`: Redes recurrentes (GRU, LSTM)

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

Los archivos `.qmd` ejecutan automáticamente los scripts necesarios mediante `source()`.

### Descarga y procesamiento de GDELT

La conversión de ZIP a Parquet se ejecuta desde terminal (en RStudio se agotaba la memoria). Editar antes las variables `INPUT_DIR`, `OUTPUT_DIR`, `LOG_FILE` y `NUM_CORES` (2 cores para 16 GB de RAM):

``` bash
bash EDA_RStudio/code/sh/step1_zip_to_parquet.sh
```

| Paso | Script | Salida | Tamaño aprox. | Tiempo estimado |
|---------------|---------------|---------------|---------------|---------------|
| Descarga CSV.ZIP | `11.web_scraping_download_gdelt_parallel.R` | `gdelt_raw` | 144.9 GB (4283 archivos) | 3-4 h |
| Conversión a Parquet | `step1_zip_to_parquet.sh` + `convert_zip_to_parquet.R` | `gdelt_parquet` | 189.7 GB | 4-6 h |
| Filtrado | `12.filter_parquet_files.R` | `gdelt_filtered` | 65.1 MB (reducción del 99.6 %) | 1-2 h |
| Consolidación | `14.consolidate_filtered_batches.R` | `gdelt_filtered_consolidated` | 30.8 MB | 1-2 min |

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

-   **Financieros**: IBEX35 y 10 índices/activos externos, de enero de 2004 a octubre de 2025 (≈21.8 años), descargados con `yfinance`:
    -   Índices: S&P 500, DAX, SBF 120, Euronext 100, Nikkei 225, FTSE 100, VIX
    -   Divisa y materias primas: EUR/USD, oro (futuros), petróleo WTI (futuros)
    -   Euribor 3M (FRED, serie `IR3TIB01EZM156N`)
-   **Sentimiento**: GDELT 1.0 Global Knowledge Graph (GKG), del 2014-01-01 al 2025-10-16 (4283 archivos diarios, 144.9 GB). Tras el filtrado: 66 377 registros, 4257 días únicos y 136 745 artículos.
-   **Filtrado de noticias**: tres criterios booleanos (temas económicos/bursátiles en `THEMES`; empresas del IBEX35 en `ORGANIZATIONS`; fuentes españolas en `SOURCES`/`SOURCEURLS`). Una noticia es válida si cumple al menos 2 de los 3.
-   **Variables de sentimiento**: los seis componentes de TONE (tone, positive, negative, polarity, activity, self-reference) agregados a nivel diario.

## División temporal y datasets

-   **Train**: 2004-01-01 a 2024-10-15 · **Test**: 2024-10-16 a 2025-10-16 · **Validación final**: 2025-10-17 a 2025-11-30.
-   Normalización Z-score con parámetros calculados solo en train (sin *look-ahead bias*).
-   Tres conjuntos de datos, cada uno en versión escalada y sin escalar:
    -   `financial_long`: variables financieras, serie larga (2004-2024)
    -   `financial`: variables financieras, serie corta (2014-2024)
    -   `sentiment`: variables financieras + sentimiento GDELT (2014-2024)

## Features

-   Generadas: **97 variables** (técnicas, de calendario y exógenas).
-   Selección: limpieza (−5) → correlación \> 85 % (−30) → Random Forest con 90 % de importancia acumulada (−18) → **44 variables finales**.
-   Indicadores técnicos: retornos logarítmicos, volatilidad, velocidad y aceleración, RSI, ROC, estocástico, OBV, medias móviles, rangos.
-   Variables exógenas: retornos, momentum y volatilidad de S&P500, DAX, FTSE100 y Euronext100; VIX; petróleo; oro; EUR/USD; Euribor 3M.
-   Sentimiento: estadísticos diarios de TONE y sus lags (se elimina `mean_tone_score` por correlación con `median_tone_score`).

## Problemas y modelos

Dos problemas, cada uno a 1, 5, 10 y 20 días:

-   **Clasificación**: dirección del retorno (`direction_next`, 0 = baja, 1 = sube).
-   **Regresión**: retorno futuro (`returns_next`).

| Problema | Datos | Modelos | Nº |
|---------------|---------------|-----------------------------|---------------|
| Regresión | Scaled | Linear, Ridge, Random Forest, XGBoost, LightGBM, Ensemble, MLP, GRU, LSTM | 9 |
| Regresión | Unscaled | Random Forest, XGBoost, LightGBM, Ensemble, GRU, LSTM | 6 |
| Clasificación | Scaled | Logistic, Random Forest, XGBoost, LightGBM, Ensemble, MLP, GRU, LSTM | 8 |
| Clasificación | Unscaled | Random Forest, XGBoost, LightGBM, Ensemble, GRU, LSTM | 6 |

Total: 3 datasets × 4 targets × modelos = **348 configuraciones**. Además, se evalúan los modelos clásicos univariantes ARIMA, ARIMAX (con retornos de S&P500, DAX, VIX, petróleo y EUR/USD) y Prophet sobre el retorno.

## Evaluación

-   **Regresión**: RMSE, MAE, R², precisión direccional (signo del retorno).
-   **Clasificación**: Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC, Cohen's Kappa.
-   **Comparaciones estadísticas**: McNemar (clasificación), Diebold-Mariano (regresión) e intervalos de confianza por bootstrap.
-   **Evaluación frente a modelo ingenuo (*naïve*)**: test binomial frente a predecir siempre la clase mayoritaria, Cohen's Kappa \> 0.1, Balanced Accuracy \> 0.52 y F1 \> 0.1. Un modelo es "REAL" solo si supera significativamente al clasificador ingenuo.

------------------------------------------------------------------------

# Resultados Principales

-   **Análisis exploratorio**: tres periodos de volatilidad extrema (crisis 2008, deuda europea 2011-2012 y COVID-19). Correlaciones del IBEX35 con Euronext100 (0.87), DAX (0.82) y FTSE100 (0.79), y negativa con el VIX (−0.42). Causalidad de Granger (5 lags) significativa para S&P500, VIX, EUR/USD y DAX. Retornos no normales (curtosis ≈ 13.7).
-   **Precisión direccional**: entre el 53 % y el 62 %. Los modelos GRU y LSTM son los más consistentes (62.22 % y 61.97 %); en clasificación, Random Forest y Ensemble alcanzan 0.63 de accuracy.
-   **Modelos clásicos**: ARIMA(0,0,0), ARIMA con media y Prophet obtienen ≈ 59.84 % solo porque predicen siempre "sube" (en el test el 59.84 % de los días son de subida). Se clasifican como ingenuos.
-   **Sentimiento GDELT**: aporta mejoras marginales y generalmente no significativas. Solo 13 de 60 comparaciones de regresión (21.7 %, Diebold-Mariano) y 3 de 56 de clasificación (5.4 %, McNemar) mejoran al añadir sentimiento; la mejora más destacada es de ≈ 7 % en LightGBM (clasificación sin escalar, `direction_next`; regresión escalada, `returns_next_10`). En el ranking de Random Forest no aparece ninguna variable de sentimiento entre las 30 primeras.
-   **Horizonte y profundidad histórica**: sin diferencias consistentes y estadísticamente significativas entre horizontes (1, 5, 10, 20 días) ni entre serie corta y larga.
-   **Modelos con capacidad predictiva real**: tras los criterios *naïve* solo 10 de los modelos evaluados superan al clasificador ingenuo (p. ej., LightGBM clasificación `direction_next_5` con datos escalados; XGBoost y LightGBM en regresión).
-   **Validación final (17-oct-2025 a 30-nov-2025)**: los modelos seleccionados obtienen accuracy direccional de 36-46 % (LightGBM clasificación 46.15 %, LightGBM regresión 42.31 %, XGBoost regresión 36.36 %), por debajo del baseline de predecir siempre "sube". Indica sobreajuste y limitada generalización.
-   **Sesgo alcista**: todos los modelos tienden a predecir subidas, lo que limita su aplicabilidad práctica.

**Conclusión de la memoria**: la predicción direccional del IBEX35 mediante *machine learning* y análisis de sentimiento básico de GDELT presenta limitaciones fundamentales; los modelos apenas superan el azar y la mayoría no supera significativamente a un modelo ingenuo.

**Líneas futuras**: NLP más sofisticado (transformers, análisis por compañía), datos alternativos de mayor frecuencia, balanceo de clases, evaluación por regímenes de volatilidad y extensión a otros índices.

------------------------------------------------------------------------

# Notas Importantes

-   Los datos GDELT originales (≈ 145 GB en ZIP, ≈ 190 GB en Parquet) y los datasets procesados no se incluyen por tamaño.
-   Los modelos entrenados (`.joblib`, `.keras`) están disponibles bajo petición.
-   La descarga de Euribor requiere una API key propia de FRED (`fredr_set_key()`); no debe incluirse en el repositorio.
-   Parte del código de R y Python se programó con la ayuda de Claude Sonnet 4.5 (Anthropic).
-   Reproducción completa: descarga y procesado de GDELT (≈ 8-12 h, ver tabla en *Ejecución*) y entrenamiento de modelos de ML/DL en GPU (Colab).

------------------------------------------------------------------------

# Licencia

Este proyecto académico se entrega como Trabajo Fin de Máster.
El código está disponible para fines educativos.
