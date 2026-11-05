# ============================================================================
# FUNCIÓN: LIMPIEZA DE DATOS DE SENTIMIENTO
# ============================================================================

clean_sentiment_data <- function(train_df,
                                 test_df,
                                 na_threshold = 10,
                                 imputation_method = "median",
                                 remove_sd_vars = FALSE,
                                 verbose = TRUE) {
  #' Limpia datos de sentimiento GDELT
  #'
  #' @param train_df DataFrame de entrenamiento de sentimiento
  #' @param test_df DataFrame de test de sentimiento
  #' @param na_threshold Umbral de % NAs para eliminar variable (default: 10)
  #' @param imputation_method Método de imputación: "median", "zero", "forward" (default: "median")
  #' @param remove_sd_vars Eliminar todas las variables sd_* (default: FALSE)
  #' @param verbose Imprimir información del proceso (default: TRUE)
  #' @return Lista con train y test limpios

  library(dplyr)
  library(tidyr)

  if (verbose) log_cat("=== LIMPIEZA DE DATOS DE SENTIMIENTO ===\n\n")

  # --------------------------------------------------------------------------
  # 1. Análisis inicial de NAs
  # --------------------------------------------------------------------------

  na_train <- colSums(is.na(train_df))
  na_train_pct <- na_train / nrow(train_df) * 100
  na_train_sorted <- sort(na_train_pct[na_train_pct > 0], decreasing = TRUE)

  na_test <- colSums(is.na(test_df))
  na_test_pct <- na_test / nrow(test_df) * 100
  na_test_sorted <- sort(na_test_pct[na_test_pct > 0], decreasing = TRUE)

  if (verbose) {
    log_cat("=== NAs EN TRAIN ===\n")
    log_cat("Total observaciones:", nrow(train_df), "\n")
    if (length(na_train_sorted) > 0) {
      print(round(na_train_sorted, 2))
    } else {
      log_cat("Sin NAs\n")
    }

    log_cat("\n=== NAs EN TEST ===\n")
    log_cat("Total observaciones:", nrow(test_df), "\n")
    if (length(na_test_sorted) > 0) {
      print(round(na_test_sorted, 2))
    } else {
      log_cat("Sin NAs\n")
    }
    log_cat("\n")
  }

  # --------------------------------------------------------------------------
  # 2. Identificar variables a eliminar
  # --------------------------------------------------------------------------

  vars_to_remove <- c()

  # Variables con >threshold% NAs en test
  vars_high_na <- names(na_test_sorted[na_test_sorted > na_threshold])
  vars_to_remove <- c(vars_to_remove, vars_high_na)

  # Eliminar todas las sd_* si se especifica
  if (remove_sd_vars) {
    sd_vars <- grep("^sd_", names(train_df), value = TRUE)
    vars_to_remove <- c(vars_to_remove, sd_vars)
    if (verbose) {
      log_cat("Eliminando todas las variables de desviación estándar (sd_*)\n")
    }
  }

  vars_to_remove <- unique(vars_to_remove)

  if (verbose && length(vars_to_remove) > 0) {
    log_cat("=== VARIABLES A ELIMINAR ===\n")
    print(vars_to_remove)
    log_cat("Total:", length(vars_to_remove), "\n\n")
  }

  # --------------------------------------------------------------------------
  # 3. Eliminar variables
  # --------------------------------------------------------------------------

  train_step1 <- train_df %>%
    dplyr::select(-any_of(vars_to_remove))

  test_step1 <- test_df %>%
    dplyr::select(-any_of(vars_to_remove))

  if (verbose) {
    log_cat("Variables después de eliminación:", ncol(train_step1), "\n\n")
  }

  # --------------------------------------------------------------------------
  # 4. Imputar NAs restantes
  # --------------------------------------------------------------------------

  # Identificar columnas numéricas con NAs (excepto date)
  numeric_cols_train <- train_step1 %>%
    dplyr::select(where(is.numeric), -date) %>%
    names()

  numeric_cols_test <- test_step1 %>%
    dplyr::select(where(is.numeric), -date) %>%
    names()

  # Aplicar estrategia de imputación
  if (imputation_method == "median") {
    if (verbose) log_cat("Imputando NAs con mediana de train...\n")

    # Train: imputar con mediana
    for (col in numeric_cols_train) {
      if (sum(is.na(train_step1[[col]])) > 0) {
        median_value <- median(train_step1[[col]], na.rm = TRUE)
        train_step1[[col]] <- ifelse(is.na(train_step1[[col]]),
          median_value,
          train_step1[[col]]
        )
      }
    }

    # Test: imputar con mediana de train
    for (col in numeric_cols_test) {
      if (sum(is.na(test_step1[[col]])) > 0) {
        median_value <- median(train_df[[col]], na.rm = TRUE)
        test_step1[[col]] <- ifelse(is.na(test_step1[[col]]),
          median_value,
          test_step1[[col]]
        )
      }
    }
  } else if (imputation_method == "zero") {
    if (verbose) log_cat("Imputando NAs con cero...\n")

    train_step1 <- train_step1 %>%
      mutate(across(all_of(numeric_cols_train), ~ ifelse(is.na(.), 0, .)))

    test_step1 <- test_step1 %>%
      mutate(across(all_of(numeric_cols_test), ~ ifelse(is.na(.), 0, .)))
  } else if (imputation_method == "forward") {
    if (verbose) log_cat("Imputando NAs con forward fill...\n")

    train_step1 <- train_step1 %>%
      fill(all_of(numeric_cols_train), .direction = "down")

    test_step1 <- test_step1 %>%
      fill(all_of(numeric_cols_test), .direction = "down")

    # Si quedan NAs al inicio, usar mediana
    for (col in numeric_cols_train) {
      if (sum(is.na(train_step1[[col]])) > 0) {
        median_value <- median(train_step1[[col]], na.rm = TRUE)
        train_step1[[col]] <- ifelse(is.na(train_step1[[col]]),
          median_value,
          train_step1[[col]]
        )
      }
    }

    for (col in numeric_cols_test) {
      if (sum(is.na(test_step1[[col]])) > 0) {
        median_value <- median(train_df[[col]], na.rm = TRUE)
        test_step1[[col]] <- ifelse(is.na(test_step1[[col]]),
          median_value,
          test_step1[[col]]
        )
      }
    }
  }

  # --------------------------------------------------------------------------
  # 5. Verificación final
  # --------------------------------------------------------------------------

  na_train_final <- sum(is.na(train_step1))
  na_test_final <- sum(is.na(test_step1))

  if (verbose) {
    log_cat("\n=== VERIFICACIÓN FINAL ===\n")
    log_cat("Train:\n")
    log_cat("  - Observaciones:", nrow(train_step1), "\n")
    log_cat("  - Variables:", ncol(train_step1), "\n")
    log_cat("  - NAs totales:", na_train_final, "\n")

    log_cat("\nTest:\n")
    log_cat("  - Observaciones:", nrow(test_step1), "\n")
    log_cat("  - Variables:", ncol(test_step1), "\n")
    log_cat("  - NAs totales:", na_test_final, "\n\n")

    if (na_train_final > 0 || na_test_final > 0) {
      log_cat("⚠ ADVERTENCIA: Aún quedan NAs\n")
    } else {
      log_cat("✓ Limpieza completada: sin NAs\n")
    }
  }

  # --------------------------------------------------------------------------
  # 6. Retornar resultados
  # --------------------------------------------------------------------------

  return(list(
    train = train_step1,
    test = test_step1,
    vars_removed = vars_to_remove,
    n_vars_removed = length(vars_to_remove),
    n_vars_final = ncol(train_step1),
    imputation_method = imputation_method,
    na_before_train = sum(na_train),
    na_before_test = sum(na_test),
    na_after_train = na_train_final,
    na_after_test = na_test_final
  ))
}


# ============================================================================
# FUNCIÓN: ANÁLISIS DE CORRELACIÓN PARA SENTIMIENTO
# ============================================================================

analyze_sentiment_correlation <- function(sentiment_df,
                                          cor_cutoff = 0.90,
                                          save_plots = TRUE,
                                          output_dir = ".",
                                          verbose = TRUE) {
  #' Analiza correlaciones entre variables de sentimiento
  #'
  #' @param sentiment_df DataFrame de sentimiento limpio
  #' @param cor_cutoff Umbral de correlación (default: 0.90)
  #' @param save_plots Guardar gráficos (default: TRUE)
  #' @param output_dir Directorio de salida (default: ".")
  #' @param verbose Mostrar información (default: TRUE)
  #' @return Lista con análisis de correlación

  library(dplyr)
  library(corrplot)
  library(caret)

  if (verbose) log_cat("=== ANÁLISIS DE CORRELACIÓN EN SENTIMIENTO ===\n\n")

  # Preparar datos numéricos (sin date)
  numeric_data <- sentiment_df %>%
    dplyr::select(-date) %>%
    dplyr::select(where(is.numeric))

  if (verbose) {
    log_cat("Variables a analizar:", ncol(numeric_data), "\n")
    log_cat("Observaciones:", nrow(numeric_data), "\n\n")
  }

  # Calcular matriz de correlación
  cor_matrix <- cor(numeric_data, use = "complete.obs")


  # Guardar plot
  if (save_plots) {
    save_base_plot(
      plotname = "sentiment_correlation_matrix",
      plot_function = function() {
        corrplot(cor_matrix,
          method = "color",
          type = "upper",
          tl.cex = 0.7,
          tl.col = "black",
          title = sprintf("Matriz de Correlación (cutoff = %.2f)", cor_cutoff)
        )
      },
      show_plot = TRUE # No mostrar al guardar
    )
  }

  # Identificar variables con alta correlación
  if (verbose) log_cat("\nIdentificando correlaciones altas...\n")


  # Identificar variables altamente correlacionadas
  high_cor <- findCorrelation(cor_matrix,
    cutoff = cor_cutoff,
    names = TRUE,
    verbose = verbose
  )

  if (verbose) {
    log_cat(sprintf("\n=== VARIABLES CON CORRELACIÓN >%.2f ===\n", cor_cutoff))
    if (length(high_cor) > 0) {
      print(high_cor)
      log_cat("Total:", length(high_cor), "\n")
    } else {
      log_cat("Ninguna variable con correlación alta\n")
    }
  }

  # Encontrar pares específicos de alta correlación
  high_cor_pairs <- which(abs(cor_matrix) > cor_cutoff & cor_matrix != 1, arr.ind = TRUE)

  if (nrow(high_cor_pairs) > 0) {
    cor_pairs_df <- data.frame(
      var1 = rownames(cor_matrix)[high_cor_pairs[, 1]],
      var2 = colnames(cor_matrix)[high_cor_pairs[, 2]],
      correlation = cor_matrix[high_cor_pairs]
    ) %>%
      distinct() %>%
      arrange(desc(abs(correlation)))

    if (verbose && nrow(cor_pairs_df) > 0) {
      log_cat("\n=== PARES DE VARIABLES ALTAMENTE CORRELACIONADAS ===\n")
      print(cor_pairs_df)
    }
  } else {
    cor_pairs_df <- data.frame()
  }

  return(list(
    correlation_matrix = cor_matrix,
    high_cor_vars = high_cor,
    high_cor_pairs = cor_pairs_df,
    n_high_cor = length(high_cor)
  ))
}


'# ============================================================================
# Script de Uso
# ============================================================================

library(dplyr)
library(tidyr)
library(arrow)
library(corrplot)
library(caret)

# ----------------------------------------------------------------------------
# 1. Cargar datos
# ----------------------------------------------------------------------------

daily_sentiment <- read_parquet("output/analysis_ibex35/ibex35_daily_sentiment.parquet")

# Separar en train y test
daily_sentiment_train <- daily_sentiment %>%
  filter(date >= as.Date("2020-01-01") & date <= as.Date("2024-10-15"))

daily_sentiment_test <- daily_sentiment %>%
  filter(date > as.Date("2024-10-15"))

cat("Datos cargados:\n")
cat("Train:", nrow(daily_sentiment_train), "obs x", ncol(daily_sentiment_train), "vars\n")
cat("Test:", nrow(daily_sentiment_test), "obs x", ncol(daily_sentiment_test), "vars\n\n")

# ----------------------------------------------------------------------------
# 2. OPCIÓN A: Mantener variables sd_* imputando NAs
# ----------------------------------------------------------------------------

sentiment_cleaned <- clean_sentiment_data(
  train_df = daily_sentiment_train,
  test_df = daily_sentiment_test,
  na_threshold = 10,
  imputation_method = "median",  # Imputar sd_* con mediana
  remove_sd_vars = FALSE,        # Mantener sd_*
  verbose = TRUE
)

# Extraer resultados
daily_sentiment_train_clean <- sentiment_cleaned$train
daily_sentiment_test_clean <- sentiment_cleaned$test

# ----------------------------------------------------------------------------
# 3. OPCIÓN B: Eliminar todas las variables sd_* (más simple)
# ----------------------------------------------------------------------------

sentiment_cleaned_no_sd <- clean_sentiment_data(
  train_df = daily_sentiment_train,
  test_df = daily_sentiment_test,
  na_threshold = 10,
  imputation_method = "median",
  remove_sd_vars = TRUE,  # Eliminar todas las sd_*
  verbose = TRUE
)

# ----------------------------------------------------------------------------
# 4. Análisis de correlación
# ----------------------------------------------------------------------------

cor_analysis <- analyze_sentiment_correlation(
  sentiment_df = daily_sentiment_train_clean,
  cor_cutoff = 0.90,
  save_plots = TRUE,
  output_dir = "results/sentiment_analysis",
  verbose = TRUE
)

# Ver variables altamente correlacionadas
if(cor_analysis$n_high_cor > 0) {
  log_cat("\nVariables recomendadas para eliminar por alta correlación:\n")
  print(cor_analysis$high_cor_vars)
}

# ----------------------------------------------------------------------------
# 5. Eliminar variables altamente correlacionadas si las hay
# ----------------------------------------------------------------------------

if(cor_analysis$n_high_cor > 0) {
  daily_sentiment_train_final <- daily_sentiment_train_clean %>%
    dplyr::select(-all_of(cor_analysis$high_cor_vars))

  daily_sentiment_test_final <- daily_sentiment_test_clean %>%
    dplyr::select(-all_of(cor_analysis$high_cor_vars))

  log_cat("\n=== DATOS FINALES (sin correlaciones altas) ===\n")
  log_cat("Train:", nrow(daily_sentiment_train_final), "obs x",
      ncol(daily_sentiment_train_final), "vars\n")
  log_cat("Test:", nrow(daily_sentiment_test_final), "obs x",
      ncol(daily_sentiment_test_final), "vars\n")
} else {
  daily_sentiment_train_final <- daily_sentiment_train_clean
  daily_sentiment_test_final <- daily_sentiment_test_clean

  log_cat("\n✓ Sin correlaciones altas, usando datos limpios directamente\n")
}

# ----------------------------------------------------------------------------
# 6. Guardar resultados
# ----------------------------------------------------------------------------

saveRDS(daily_sentiment_train_final, "data/processed/sentiment_train_clean.rds")
saveRDS(daily_sentiment_test_final, "data/processed/sentiment_test_clean.rds")

# Guardar metadata
saveRDS(list(
  vars_removed = sentiment_cleaned$vars_removed,
  high_cor_vars = cor_analysis$high_cor_vars,
  imputation_method = sentiment_cleaned$imputation_method
), "data/processed/sentiment_cleaning_metadata.rds")

cat("\n=== ARCHIVOS GUARDADOS ===\n")
cat("- sentiment_train_clean.rds\n")
cat("- sentiment_test_clean.rds\n")
cat("- sentiment_cleaning_metadata.rds\n")
cat("\n✓ Limpieza completada\n")'


# ============================================================================
# IMPUTACIÓN DE NAs EN VARIABLES DE SENTIMENT
# ============================================================================

library(dplyr)
library(zoo)

impute_sentiment_nas <- function(train_df, test_df,
                                 method = "forward_fill",
                                 verbose = TRUE) {
  #' Imputa NAs en variables de sentiment después del join
  #'
  #' @param train_df DataFrame train con NAs en sentiment
  #' @param test_df DataFrame test con NAs en sentiment
  #' @param method Método: "forward_fill", "median", "mixed" (default: "forward_fill")
  #' @param verbose Imprimir información (default: TRUE)
  #' @return Lista con train y test imputados

  if (verbose) {
    log_cat("=== IMPUTACIÓN DE SENTIMENT NAs ===\n")
    log_cat("Método:", method, "\n\n")
  }

  # Identificar variables de sentiment
  sentiment_vars <- c(
    "mean_tone_score", "median_tone_score", "sd_tone_score",
    "min_tone_score", "max_tone_score",
    "mean_positive", "sd_positive",
    "mean_negative", "sd_negative",
    "mean_polarity", "sd_polarity",
    "mean_activity", "sd_activity",
    "mean_selfref", "sd_selfref",
    "total_records", "total_articles"
  )

  # Filtrar solo las que existen en los datos
  sentiment_vars <- sentiment_vars[sentiment_vars %in% names(train_df)]

  # Diagnóstico inicial
  if (verbose) {
    log_cat("Variables de sentiment encontradas:", length(sentiment_vars), "\n")
    log_cat("NAs en train:", sum(is.na(train_df[sentiment_vars])), "\n")
    log_cat("NAs en test:", sum(is.na(test_df[sentiment_vars])), "\n\n")
  }

  # --------------------------------------------------------------------------
  # MÉTODO 1: Forward Fill (mantener último valor conocido)
  # --------------------------------------------------------------------------

  if (method == "forward_fill") {
    train_imputed <- train_df %>%
      dplyr::arrange(date) %>%
      dplyr::mutate(across(
        all_of(sentiment_vars),
        ~ zoo::na.locf(., na.rm = FALSE, fromLast = FALSE)
      )) %>%
      # Backward fill por si quedan NAs al inicio
      dplyr::mutate(across(
        all_of(sentiment_vars),
        ~ zoo::na.locf(., na.rm = FALSE, fromLast = TRUE)
      ))

    test_imputed <- test_df %>%
      dplyr::arrange(date) %>%
      dplyr::mutate(across(
        all_of(sentiment_vars),
        ~ zoo::na.locf(., na.rm = FALSE, fromLast = FALSE)
      )) %>%
      # Si quedan NAs, usar mediana de train
      dplyr::mutate(across(
        all_of(sentiment_vars),
        ~ ifelse(is.na(.),
          median(train_imputed[[cur_column()]], na.rm = TRUE),
          .
        )
      ))
  }

  # --------------------------------------------------------------------------
  # MÉTODO 2: Mediana (valor central del periodo)
  # --------------------------------------------------------------------------

  else if (method == "median") {
    # Calcular medianas de train
    medians <- train_df %>%
      dplyr::summarise(across(
        all_of(sentiment_vars),
        ~ median(., na.rm = TRUE)
      )) %>%
      as.list()

    train_imputed <- train_df %>%
      dplyr::mutate(across(
        all_of(sentiment_vars),
        ~ ifelse(is.na(.), medians[[cur_column()]], .)
      ))

    test_imputed <- test_df %>%
      dplyr::mutate(across(
        all_of(sentiment_vars),
        ~ ifelse(is.na(.), medians[[cur_column()]], .)
      ))
  }

  # --------------------------------------------------------------------------
  # MÉTODO 3: Mixed (forward fill + valores neutrales para conteos)
  # --------------------------------------------------------------------------

  else if (method == "mixed") {
    # Variables de conteo: poner a 0 (no hubo artículos)
    count_vars <- c("total_records", "total_articles")
    count_vars <- count_vars[count_vars %in% sentiment_vars]

    # Variables de sentiment: forward fill
    sentiment_only <- setdiff(sentiment_vars, count_vars)

    # Train
    train_imputed <- train_df %>%
      dplyr::arrange(date) %>%
      # Forward fill para sentiment
      dplyr::mutate(across(
        all_of(sentiment_only),
        ~ zoo::na.locf(., na.rm = FALSE, fromLast = FALSE)
      )) %>%
      dplyr::mutate(across(
        all_of(sentiment_only),
        ~ zoo::na.locf(., na.rm = FALSE, fromLast = TRUE)
      )) %>%
      # Cero para conteos
      dplyr::mutate(across(
        all_of(count_vars),
        ~ replace_na(., 0)
      ))

    # Test
    test_imputed <- test_df %>%
      dplyr::arrange(date) %>%
      # Forward fill para sentiment
      dplyr::mutate(across(
        all_of(sentiment_only),
        ~ zoo::na.locf(., na.rm = FALSE, fromLast = FALSE)
      )) %>%
      # Si quedan NAs, usar mediana de train
      dplyr::mutate(across(
        all_of(sentiment_only),
        ~ ifelse(is.na(.),
          median(train_imputed[[cur_column()]], na.rm = TRUE),
          .
        )
      )) %>%
      # Cero para conteos
      dplyr::mutate(across(
        all_of(count_vars),
        ~ replace_na(., 0)
      ))
  }

  # --------------------------------------------------------------------------
  # Verificación final
  # --------------------------------------------------------------------------

  if (verbose) {
    log_cat("=== VERIFICACIÓN POST-IMPUTACIÓN ===\n")

    na_train_after <- sum(is.na(train_imputed[sentiment_vars]))
    na_test_after <- sum(is.na(test_imputed[sentiment_vars]))

    log_cat("NAs en sentiment (train):", na_train_after, "\n")
    log_cat("NAs en sentiment (test):", na_test_after, "\n")

    if (na_train_after > 0 || na_test_after > 0) {
      log_cat("\n⚠ ADVERTENCIA: Quedan NAs después de imputación\n")

      # Mostrar qué variables tienen NAs
      if (na_train_after > 0) {
        na_vars_train <- sentiment_vars[colSums(is.na(train_imputed[sentiment_vars])) > 0]
        log_cat("Train - Variables con NAs:", paste(na_vars_train, collapse = ", "), "\n")
      }
      if (na_test_after > 0) {
        na_vars_test <- sentiment_vars[colSums(is.na(test_imputed[sentiment_vars])) > 0]
        log_cat("Test - Variables con NAs:", paste(na_vars_test, collapse = ", "), "\n")
      }
    } else {
      log_cat("✓ Sin NAs en variables de sentiment\n")
    }
  }

  return(list(
    train = train_imputed,
    test = test_imputed,
    method_used = method,
    sentiment_vars = sentiment_vars
  ))
}

'# ============================================================================
# APLICAR IMPUTACIÓN
# ============================================================================

library(dplyr)
library(zoo)

# OPCIÓN 1: Forward Fill (RECOMENDADO para sentiment)
# Asume que el sentiment del último día persiste
sentiment_imputed <- impute_sentiment_nas(
  train_df = train_with_sentiment,
  test_df = test_with_sentiment,
  method = "forward_fill",
  verbose = TRUE
)

# OPCIÓN 2: Mediana (alternativa conservadora)
# sentiment_imputed <- impute_sentiment_nas(
#   train_df = train_with_sentiment,
#   test_df = test_with_sentiment,
#   method = "median",
#   verbose = TRUE
# )

# OPCIÓN 3: Mixed (forward fill + ceros en conteos)
# sentiment_imputed <- impute_sentiment_nas(
#   train_df = train_with_sentiment,
#   test_df = test_with_sentiment,
#   method = "mixed",
#   verbose = TRUE
# )

# Extraer datasets
train_final_clean <- sentiment_imputed$train
test_final_clean <- sentiment_imputed$test

# Verificación completa de NAs en todo el dataset
cat("\n=== VERIFICACIÓN COMPLETA ===\n")
cat("Train - NAs totales:", sum(is.na(train_final_clean)), "\n")
cat("Test - NAs totales:", sum(is.na(test_final_clean)), "\n")

# NAs solo en targets (esperado)
cat("\nNAs en targets (esperado):\n")
cat("returns_next (test):", sum(is.na(test_final_clean$returns_next)), "\n")
cat("returns_next_5 (test):", sum(is.na(test_final_clean$returns_next_5)), "\n")

# Guardar datasets finales
saveRDS(train_final_clean, "data/processed/train_final_with_sentiment.rds")
saveRDS(test_final_clean, "data/processed/test_final_with_sentiment.rds")'
