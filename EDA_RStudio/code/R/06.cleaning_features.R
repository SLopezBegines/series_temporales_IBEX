# FUNCIÓN: LIMPIEZA DE NAs Y VARIABLES PROBLEMÁTICAS


clean_na_and_redundant <- function(train_df, test_df,
                                   na_threshold = 10,
                                   vars_to_remove = NULL,
                                   verbose = TRUE,
                                   targets = c("returns_next", "returns_next_5", "returns_next_10", "returns_next_20")) {
  #' Limpia NAs y elimina variables problemáticas/redundantes
  #'
  #' @param train_df DataFrame de entrenamiento
  #' @param test_df DataFrame de test
  #' @param na_threshold Umbral de % NAs en test para eliminar variable (default: 10)
  #' @param vars_to_remove Vector de variables adicionales a eliminar (opcional)
  #' @param verbose Imprimir información del proceso (default: TRUE)
  #' @return Lista con train y test limpios

  library(dplyr)
  library(tidyr)

  if (verbose) log_cat("=== INICIANDO LIMPIEZA ===\n\n")


  # 1. Identificar variables con NAs estructurales en test


  na_test_pct <- colSums(is.na(test_df)) / nrow(test_df) * 100
  na_test_sorted <- sort(na_test_pct[na_test_pct > 0], decreasing = TRUE)

  if (verbose) {
    log_cat("=== NAs EN TEST (%) ===\n")
    log_cat(round(na_test_sorted, 2))
  }

  # Variables con >threshold% NAs
  vars_eliminar_na <- names(na_test_sorted[na_test_sorted > na_threshold])

  if (verbose) {
    log_cat(sprintf("\n=== VARIABLES A ELIMINAR (>%d%% NAs en test) ===\n", na_threshold))
    log_cat(vars_eliminar_na)
    log_cat("Total:", length(vars_eliminar_na), "\n\n")
  }


  # 2. Variables redundantes/problemáticas por defecto


  vars_eliminar_default <- c(
    "close", # Redundante con log_close
    "open", "high", "low", # Redundantes con rangos/close
    "close_smooth" # Redundante con close/log_close
  )

  # Combinar con variables adicionales del usuario
  vars_eliminar_total <- unique(c(
    vars_eliminar_na,
    vars_eliminar_default,
    vars_to_remove
  ))

  # Filtrar solo las que existen en los dataframes
  vars_eliminar_total <- vars_eliminar_total[vars_eliminar_total %in% names(train_df)]

  if (verbose) {
    log_cat("=== VARIABLES A ELIMINAR (TOTAL) ===\n")
    log_cat(vars_eliminar_total)
    log_cat("Total:", length(vars_eliminar_total), "\n\n")
  }


  # 3. Eliminar variables


  train_step1 <- train_df %>%
    dplyr::select(-any_of(vars_eliminar_total))

  test_step1 <- test_df %>%
    dplyr::select(-any_of(vars_eliminar_total))

  if (verbose) {
    log_cat("Variables después de eliminación:", ncol(train_step1), "\n\n")
  }


  # 4. Completar NAs restantes


  # Train: eliminar filas con NAs en targets
  train_clean <- train_step1 %>%
    drop_na(targets)

  # Test: forward fill + imputación con mediana de train
  test_clean <- test_step1 %>%
    # Forward fill para todas las variables excepto targets
    fill(-c(date, targets), .direction = "down")

  # Imputar NAs restantes con mediana de train
  numeric_cols <- test_clean %>%
    dplyr::select(where(is.numeric), -date, -targets) %>%
    names()

  for (col in numeric_cols) {
    if (sum(is.na(test_clean[[col]])) > 0) {
      median_value <- median(train_clean[[col]], na.rm = TRUE)
      test_clean[[col]] <- ifelse(is.na(test_clean[[col]]),
        median_value,
        test_clean[[col]]
      )
    }
  }


  # 5. Verificación final


  na_train_total <- sum(is.na(train_clean))
  na_train_predictors <- sum(is.na(train_clean %>%
    dplyr::select(-targets)))
  na_test_total <- sum(is.na(test_clean))
  na_test_predictors <- sum(is.na(test_clean %>%
    dplyr::select(-targets)))

  if (verbose) {
    log_cat("=== VERIFICACIÓN FINAL ===\n")
    log_cat("Train:\n")
    log_cat("  - Observaciones:", nrow(train_clean), "\n")
    log_cat("  - Variables:", ncol(train_clean), "\n")
    log_cat("  - NAs totales:", na_train_total, "\n")
    log_cat("  - NAs en predictores:", na_train_predictors, "\n\n")

    log_cat("Test:\n")
    log_cat("  - Observaciones:", nrow(test_clean), "\n")
    log_cat("  - Variables:", ncol(test_clean), "\n")
    log_cat("  - NAs totales:", na_test_total, "\n")
    log_cat("  - NAs en predictores:", na_test_predictors, "\n\n")

    if (na_test_predictors > 0) {
      log_cat("⚠ ADVERTENCIA: Test tiene NAs en predictores\n")
      na_cols <- test_clean %>%
        dplyr::select(-targets) %>%
        dplyr::select(where(~ sum(is.na(.)) > 0)) %>%
        names()
      log_cat("Columnas con NAs:", paste(na_cols, collapse = ", "), "\n")
    } else {
      log_cat("✓ Test OK: sin NAs en predictores\n")
    }
  }


  # 6. Retornar resultados


  return(list(
    train = train_clean,
    test = test_clean,
    vars_removed = vars_eliminar_total,
    n_vars_removed = length(vars_eliminar_total),
    n_vars_final = ncol(train_clean)
  ))
}


# APLICAR LIMPIEZA

'
library(dplyr)
library(tidyr)

# Ejecutar limpieza
cleaned_data <- clean_na_and_redundant(
  train_df = train_20,
  test_df = test_with_external,
  na_threshold = 10,
  vars_to_remove = NULL,  # Puedes añadir variables adicionales aquí
  verbose = TRUE
)

# Extraer resultados
train_step2 <- cleaned_data$train
test_step2 <- cleaned_data$test

# Información adicional
log_cat("\n=== RESUMEN ===\n")
log_cat("Variables eliminadas:", cleaned_data$n_vars_removed, "\n")
log_cat("Variables finales:", cleaned_data$n_vars_final, "\n")

# Guardar para siguiente paso
saveRDS(train_step2, "data/processed/train_step2_cleaned.rds")
saveRDS(test_step2, "data/processed/test_step2_cleaned.rds")
saveRDS(cleaned_data$vars_removed, "data/processed/vars_removed_step2.rds")'


# IMPUTACIÓN DE NAs DE PETRÓLEO (COVID-19 ABRIL-MAYO 2020)


library(dplyr)
library(zoo) # Para na.approx (interpolación)

impute_oil_covid_nas <- function(df, verbose = TRUE) {
  #' Imputa NAs en variables de petróleo durante COVID-19
  #'
  #' @param df DataFrame con variables de petróleo
  #' @param verbose Imprimir información (default: TRUE)
  #' @return DataFrame con NAs imputados

  if (verbose) {
    log_cat("=== IMPUTACIÓN DE NAs PETRÓLEO (COVID-19) ===\n\n")

    # Diagnóstico inicial
    oil_vars <- c(
      "oil_return", "oil_return_lag1", "oil_vol20",
      "oil_momentum", "risk_on_score"
    )
    oil_vars <- oil_vars[oil_vars %in% names(df)]

    for (var in oil_vars) {
      n_na <- sum(is.na(df[[var]]))
      if (n_na > 0) {
        log_cat(sprintf("%s: %d NAs\n", var, n_na))
      }
    }
    log_cat("\n")
  }

  df_imputed <- df %>%
    arrange(date) %>%
    mutate(
      # 1. oil_return: interpolación lineal
      oil_return = zoo::na.approx(oil_return, na.rm = FALSE),
      # Si quedan NAs al inicio/final: forward/backward fill
      oil_return = zoo::na.locf(oil_return, na.rm = FALSE, fromLast = FALSE),
      oil_return = zoo::na.locf(oil_return, na.rm = FALSE, fromLast = TRUE),

      # 2. oil_return_lag1: recalcular después de imputar oil_return
      oil_return_lag1 = dplyr::lag(oil_return, 1),

      # 3. oil_vol20: interpolación + fill (volatilidad es más suave)
      oil_vol20 = zoo::na.approx(oil_vol20, na.rm = FALSE, maxgap = 25),
      oil_vol20 = zoo::na.locf(oil_vol20, na.rm = FALSE, fromLast = FALSE),
      oil_vol20 = zoo::na.locf(oil_vol20, na.rm = FALSE, fromLast = TRUE),

      # 4. oil_momentum: si existe, interpolar
      oil_momentum = if ("oil_momentum" %in% names(.)) {
        zoo::na.approx(oil_momentum, na.rm = FALSE)
      } else {
        oil_momentum
      },
      oil_momentum = if ("oil_momentum" %in% names(.)) {
        zoo::na.locf(oil_momentum, na.rm = FALSE, fromLast = FALSE)
      } else {
        oil_momentum
      },

      # 5. risk_on_score: interpolación (es score compuesto)
      risk_on_score = zoo::na.approx(risk_on_score, na.rm = FALSE),
      risk_on_score = zoo::na.locf(risk_on_score, na.rm = FALSE, fromLast = FALSE),
      risk_on_score = zoo::na.locf(risk_on_score, na.rm = FALSE, fromLast = TRUE)
    )

  # Verificación post-imputación
  if (verbose) {
    log_cat("=== VERIFICACIÓN POST-IMPUTACIÓN ===\n")
    for (var in oil_vars) {
      n_na_after <- sum(is.na(df_imputed[[var]]))
      if (n_na_after > 0) {
        log_cat(sprintf("⚠ %s: %d NAs restantes\n", var, n_na_after))
      } else {
        log_cat(sprintf("✓ %s: sin NAs\n", var))
      }
    }
    log_cat("\n")
  }

  return(df_imputed)
}


'# Aplicar imputación
train_cleaned <- impute_oil_covid_nas(train_cleaned, verbose = TRUE)
test_cleaned <- impute_oil_covid_nas(test_cleaned, verbose = TRUE)

# Verificar que no quedan NAs en esas variables
log_cat("=== VERIFICACIÓN FINAL ===\n")
oil_vars <- c("oil_return", "oil_return_lag1", "oil_vol20",
              "oil_momentum", "risk_on_score")

for(var in oil_vars) {
  if(var %in% names(train_cleaned)) {
    n_na <- sum(is.na(train_cleaned[[var]]))
    log_cat(sprintf("%s: %d NAs\n", var, n_na))
  }
}

# Ver fechas específicas para confirmar
log_cat("\n=== VALORES IMPUTADOS (sample) ===\n")
train_cleaned %>%
  filter(date >= as.Date("2020-04-20") & date <= as.Date("2020-04-25")) %>%
  select(date, oil_return, oil_vol20, risk_on_score) %>%
  print()'


# FUNCIÓN: REDUCCIÓN POR CORRELACIÓN Y VARIANZA

reduce_by_correlation_variance <- function(train_df,
                                           test_df = NULL,
                                           cor_cutoff = 0.90,
                                           save_files = FALSE,
                                           output_dir = ".",
                                           verbose = TRUE,
                                           targets = c(
                                             "returns_next", "returns_next_5", "returns_next_10", "returns_next_20",
                                             "direction_next", "direction_next_5", "direction_next_10", "direction_next_20"
                                           )) {
  #' Reduce variables eliminando alta correlación y varianza cercana a cero
  #'
  #' @param train_df DataFrame de entrenamiento
  #' @param test_df DataFrame de test (opcional)
  #' @param cor_cutoff Umbral de correlación para eliminar (default: 0.90)
  #' @param save_plots Guardar plots de correlación (default: TRUE)
  #' @param save_files Guardar RDS y CSV (default: FALSE)
  #' @param output_dir Directorio para archivos de salida (default: ".")
  #' @param verbose Imprimir información del proceso (default: TRUE)
  #' @return Lista con dataframes reducidos y metadatos

  library(dplyr)
  library(caret)
  library(corrplot)

  if (verbose) log_cat("\n=== INICIANDO REDUCCIÓN POR CORRELACIÓN Y VARIANZA ===\n\n")

  # Crear directorio si no existe
  if (save_files) {
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  }

  exclude_cols <- c(
    "date", "weekday", "month", "quarter", targets
  )
  # Excluir categóricas también
  categ_vars <- train_df %>%
    dplyr::select(-any_of(exclude_cols)) %>%
    dplyr::select(where(~ is.character(.) || is.factor(.))) %>%
    names()

  exclude_cols <- c(exclude_cols, categ_vars)

  # 1. Preparar datos numéricos (sin date, categóricas y targets)
  train_numeric <- train_df %>%
    dplyr::select(-any_of(exclude_cols)) %>%
    dplyr::select(where(is.numeric))

  n_initial <- ncol(train_numeric)

  if (verbose) {
    log_cat("Variables totales: ", ncol(train_df), "\n")
    log_cat("Variables numéricas iniciales:", n_initial, "\n\n")
  }


  # 2. Análisis de correlación


  if (verbose) log_cat("Calculando matriz de correlación...\n")

  # Eliminar variables con varianza cero ANTES de correlación
  nzv_pre <- nearZeroVar(train_numeric)
  if (length(nzv_pre) > 0) {
    vars_nzv_pre <- colnames(train_numeric)[nzv_pre]
    if (verbose) {
      log_cat(sprintf("Eliminando %d variables con varianza ~0 antes de correlación\n", length(nzv_pre)))
    }
    train_numeric <- train_numeric[, -nzv_pre, drop = FALSE]
  } else {
    vars_nzv_pre <- character(0)
  }

  cor_matrix <- cor(train_numeric, use = "pairwise.complete.obs")

  # Verificar NAs y correlaciones perfectas
  if (any(is.na(cor_matrix))) {
    warning("Matriz de correlación contiene NAs. Verificar datos.")
    cor_matrix[is.na(cor_matrix)] <- 0
  }
  corrplot(cor_matrix,
    method = "color",
    type = "upper",
    tl.col = "black",
    tl.srt = 45,
    title = sprintf("Matriz de Correlación (umbral: %.2f)", cor_cutoff),
    mar = c(0, 0, 1, 0)
  )

  # Identificar variables con alta correlación
  if (verbose) log_cat("\nIdentificando correlaciones altas...\n")

  high_cor <- findCorrelation(cor_matrix,
    cutoff = cor_cutoff,
    names = FALSE, # Cambiar a FALSE para obtener índices
    verbose = verbose
  )

  # Convertir índices a nombres
  if (length(high_cor) > 0) {
    high_cor <- colnames(train_numeric)[high_cor]
  }

  if (verbose) {
    log_cat(sprintf("\n=== VARIABLES A ELIMINAR POR CORRELACIÓN (>%.2f) ===\n", cor_cutoff))
    if (length(high_cor) > 0) {
      print(high_cor)
    } else {
      log_cat("Ninguna\n")
    }
    log_cat("Total:", length(high_cor), "\n\n")
  }

  # Aplicar eliminación
  train_reduced_cor <- train_numeric %>%
    dplyr::select(-all_of(high_cor))


  # 3. Análisis de varianza


  if (verbose) log_cat("Analizando varianza...\n")

  nzv_stats <- nearZeroVar(train_reduced_cor, saveMetrics = TRUE)
  nzv_remove <- rownames(nzv_stats[nzv_stats$nzv == TRUE, ])

  if (verbose) {
    log_cat("\n=== VARIABLES A ELIMINAR POR VARIANZA CERCANA A CERO ===\n")
    if (length(nzv_remove) > 0) {
      print(nzv_remove)
    } else {
      log_cat("Ninguna variable con varianza problemática\n")
    }
    log_cat("Total:", length(nzv_remove), "\n\n")
  }

  # Aplicar eliminación
  train_reduced_final <- train_reduced_cor %>%
    dplyr::select(-all_of(nzv_remove))


  # 4. Resumen


  n_removed_pre <- length(vars_nzv_pre)
  n_removed_cor <- length(high_cor)
  n_removed_var <- length(nzv_remove)
  n_final <- ncol(train_reduced_final)

  # 5. Crear dataframes finales con date y targets


  vars_finales <- colnames(train_reduced_final)

  train_final <- train_df %>%
    dplyr::select(date, all_of(vars_finales), all_of(exclude_cols))

  # Aplicar mismas reducciones a test si existe
  test_final <- NULL
  if (!is.null(test_df)) {
    test_final <- test_df %>%
      dplyr::select(date, all_of(vars_finales), all_of(exclude_cols))

    if (verbose) {
      log_cat("Test reducido a las mismas variables\n")
    }
  }

  if (verbose) {
    log_cat("=== RESUMEN DE REDUCCIÓN ===\n")
    log_cat(sprintf("Variables iniciales:           %d\n", n_initial))
    log_cat(sprintf("Eliminadas (varianza pre):     %d\n", n_removed_pre))
    log_cat(sprintf("Eliminadas por correlación:    %d\n", n_removed_cor))
    log_cat(sprintf("Eliminadas por varianza:       %d\n", n_removed_var))
    log_cat(sprintf("Variables finales:             %d\n", n_final))
    log_cat(sprintf("Reducción total:     %.1f%%\n\n", (ncol(train_df) - ncol(train_final)) / ncol(train_df) * 100))
    log_cat(sprintf("Reducción total numéricas:     %.1f%%\n\n", (n_initial - n_final) / n_initial * 100))
  }
  # 6. Guardar archivos (opcional)


  if (save_files) {
    # RDS
    train_path <- file.path(output_dir, "RData/train_reduced_cor_var.rds")
    saveRDS(train_final, train_path)

    if (!is.null(test_final)) {
      test_path <- file.path(output_dir, "RData/test_reduced_cor_var.rds")
      saveRDS(test_final, test_path)
    }

    # CSV con lista de variables
    vars_path <- file.path(output_dir, "tables/variables_selected_cor_var.csv")
    write.csv(data.frame(variable = vars_finales),
      vars_path,
      row.names = FALSE
    )

    # CSV con variables eliminadas
    vars_removed_df <- data.frame(
      variable = c(high_cor, nzv_remove),
      reason = c(
        rep("high_correlation", length(high_cor)),
        rep("near_zero_variance", length(nzv_remove))
      )
    )
    removed_path <- file.path(output_dir, "tables/variables_removed_cor_var.csv")
    write.csv(vars_removed_df, removed_path, row.names = FALSE)

    if (verbose) {
      log_cat("\n=== ARCHIVOS GUARDADOS ===\n")
      log_cat("- train_reduced_cor_var.rds\n")
      if (!is.null(test_final)) log_cat("- test_reduced_cor_var.rds\n")
      log_cat("- variables_selected_cor_var.csv\n")
      log_cat("- variables_removed_cor_var.csv\n")
    }
  }


  # 7. Retornar resultados


  results <- list(
    train = train_final,
    test = test_final,
    variables_selected = vars_finales,
    variables_removed_cor = high_cor,
    variables_removed_var = nzv_remove,
    correlation_matrix = cor_matrix,
    n_initial = n_initial,
    n_removed_cor = n_removed_cor,
    n_removed_var = n_removed_var,
    n_final = n_final,
    reduction_pct = (n_initial - n_final) / n_initial * 100
  )

  class(results) <- c("reduced_features", "list")

  return(results)
}

# Método print para la clase
print.reduced_features <- function(x, ...) {
  log_cat("=== Feature Reduction Results ===\n")
  log_cat(sprintf("Initial variables:     %d\n", x$n_initial))
  log_cat(sprintf("Removed (correlation): %d\n", x$n_removed_cor))
  log_cat(sprintf("Removed (variance):    %d\n", x$n_removed_var))
  log_cat(sprintf("Final variables:       %d\n", x$n_final))
  log_cat(sprintf("Reduction:             %.1f%%\n", x$reduction_pct))
}


# APLICAR REDUCCIÓN POR CORRELACIÓN Y VARIANZA


'# Cargar datos limpios del paso anterior
# train_step2 <- readRDS("data/processed/train_step2_cleaned.rds")
# test_step2 <- readRDS("data/processed/test_step2_cleaned.rds")

# Ejecutar reducción
reduced_data <- reduce_by_correlation_variance(
  train_df = train_step2,
  test_df = test_step2,
  cor_cutoff = 0.90,
  save_plots = TRUE,
  save_files = TRUE,
  output_dir = "results/feature_reduction",
  verbose = TRUE
)

# Ver resumen
print(reduced_data)

# Extraer dataframes reducidos
train_for_rf <- reduced_data$train
test_for_rf <- reduced_data$test

# Ver variables seleccionadas
log_cat("\n=== VARIABLES SELECCIONADAS ===\n")
print(reduced_data$variables_selected)

# Dimensiones finales
log_cat("\n=== DIMENSIONES FINALES ===\n")
log_cat("Train:", nrow(train_for_rf), "obs x", ncol(train_for_rf), "vars\n")
log_cat("Test:", nrow(test_for_rf), "obs x", ncol(test_for_rf), "vars\n")

# Opcional: ver variables eliminadas por tipo
log_cat("\n=== VARIABLES ELIMINADAS ===\n")
log_cat("Por correlación alta:\n")
print(reduced_data$variables_removed_cor)
log_cat("\nPor varianza baja:\n")
print(reduced_data$variables_removed_var)'


# FUNCIÓN: RANDOM FOREST PARA RANKING DE IMPORTANCIA


rank_features_by_importance <- function(train_df,
                                        test_df = NULL,
                                        target_var = "returns_next",
                                        ntree = 500,
                                        mtry = NULL,
                                        nodesize = 5,
                                        top_n_vars = NULL,
                                        importance_threshold = NULL,
                                        save_plots = TRUE,
                                        save_files = TRUE,
                                        output_dir = output_path,
                                        plot_formats = c("tiff", "pdf"),
                                        plot_width = 16,
                                        plot_height = 16,
                                        seed = 123,
                                        verbose = TRUE,
                                        targets = c(
                                          "returns_next", "returns_next_5", "returns_next_10", "returns_next_20"
                                        )) {
  #' Entrena Random Forest y rankea variables por importancia
  #'
  #' @param train_df DataFrame de entrenamiento
  #' @param test_df DataFrame de test (opcional)
  #' @param target_var Variable target ("returns_next" o "returns_next_5")
  #' @param ntree Número de árboles (default: 500)
  #' @param mtry Número de variables por split (default: sqrt(n_vars))
  #' @param nodesize Tamaño mínimo de nodos terminales (default: 5)
  #' @param top_n_vars Número de variables a seleccionar (default: todas)
  #' @param importance_threshold Seleccionar vars hasta X% importancia acumulada
  #' @param save_plots Guardar gráficos (default: TRUE)
  #' @param save_files Guardar RDS y CSV (default: FALSE)
  #' @param output_dir Directorio para archivos de salida (default: ".")
  #' @param plot_formats Formatos de gráficos: "tiff", "pdf", "png" (default: c("png", "pdf"))
  #' @param plot_width Ancho de plots en cm (default: 16)
  #' @param plot_height Alto de plots en cm (default: 16)
  #' @param seed Semilla para reproducibilidad (default: 123)
  #' @param verbose Imprimir información del proceso (default: TRUE)
  #' @return Lista con resultados del RF e importancia

  library(dplyr)
  library(randomForest)
  library(ggplot2)

  if (verbose) log_cat("\n=== INICIANDO RANDOM FOREST PARA IMPORTANCIA ===\n\n")

  # Validar target
  if (!target_var %in% targets) {
    stop(paste0("target_var debe ser una de las siguientes: ", targets))
  }

  # Crear directorio si no existe
  if (save_plots || save_files) {
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  }


  # 1. Preparar datos


  other_target <- targets[targets != target_var]

  train_rf <- train_df %>%
    dplyr::select(-date, -all_of(other_target)) %>%
    drop_na()

  # Verificar y eliminar Inf/-Inf
  inf_check <- sapply(train_rf, function(x) any(is.infinite(x)))
  if (any(inf_check)) {
    vars_with_inf <- names(inf_check)[inf_check]
    if (verbose) {
      log_cat("⚠ Eliminando", length(vars_with_inf), "variables con valores infinitos:\n")
      print(vars_with_inf)
      log_cat("\n")
    }
    train_rf <- train_rf %>% dplyr::select(-all_of(vars_with_inf))
  }

  # Verificar target
  if (!is.numeric(train_rf[[target_var]])) {
    stop("La variable target debe ser numérica")
  }

  target_var_sd <- sd(train_rf[[target_var]], na.rm = TRUE)
  if (target_var_sd == 0 || is.na(target_var_sd)) {
    stop(sprintf("Variable target '%s' tiene varianza cero o NA", target_var))
  }

  n_obs <- nrow(train_rf)
  n_predictors <- ncol(train_rf) - 1

  if (verbose) {
    log_cat("=== DATOS PREPARADOS ===\n")
    log_cat("Observaciones:", n_obs, "\n")
    log_cat("Predictores:", n_predictors, "\n")
    log_cat("Target:", target_var, "\n")
    log_cat("Target SD:", round(target_var_sd, 6), "\n\n")
    log_cat("NAs en train:", sum(is.na(train_rf)), "\n")
    if (!is.null(test_df)) {
      log_cat(
        "NAs en test (predictores):",
        sum(is.na(test_df %>% dplyr::select(-c(date, all_of(targets))))), "\n"
      )
    }
    log_cat("\n")
  }


  # 2. Configurar parámetros RF


  if (is.null(mtry)) {
    mtry <- floor(sqrt(n_predictors))
  }

  if (verbose) {
    log_cat("=== PARÁMETROS RANDOM FOREST ===\n")
    log_cat("ntree:", ntree, "\n")
    log_cat("mtry:", mtry, "\n")
    log_cat("nodesize:", nodesize, "\n")
    log_cat("seed:", seed, "\n\n")
  }


  # 3. Entrenar Random Forest


  set.seed(seed)

  if (verbose) log_cat("Entrenando Random Forest...\n")

  start_time <- Sys.time()

  rf_formula <- as.formula(paste(target_var, "~ ."))

  rf_model <- randomForest(
    rf_formula,
    data = train_rf,
    ntree = ntree,
    importance = TRUE,
    mtry = mtry,
    nodesize = nodesize,
    keep.forest = TRUE
  )

  end_time <- Sys.time()
  time_elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

  if (verbose) {
    log_cat("\n=== RESULTADOS DEL MODELO ===\n")
    # log_cat(rf_model)
    log_cat("\nR-squared:", round(tail(rf_model$rsq, 1), 4), "\n")
    log_cat("MSE:", round(tail(rf_model$mse, 1), 6), "\n")
    log_cat("Tiempo de entrenamiento:", round(time_elapsed, 2), "segundos\n\n")
  }


  # 4. Extraer y ordenar importancia


  importance_df <- importance(rf_model) %>%
    as.data.frame() %>%
    tibble::rownames_to_column("variable") %>%
    arrange(desc(`%IncMSE`)) %>%
    mutate(
      cumsum_importance = cumsum(`%IncMSE`) / sum(`%IncMSE`) * 100,
      rank = row_number()
    )

  if (verbose) {
    log_cat("=== TOP 30 VARIABLES POR IMPORTANCIA ===\n")
    log_cat(
      head(importance_df[, c(
        "variable", "%IncMSE", "IncNodePurity",
        "cumsum_importance"
      )], 30),
      row.names = FALSE
    )
  }


  # 5. Análisis de importancia acumulada


  n_80 <- min(which(importance_df$cumsum_importance >= 80))
  n_90 <- min(which(importance_df$cumsum_importance >= 90))

  if (verbose) {
    log_cat("\n=== IMPORTANCIA ACUMULADA ===\n")
    log_cat("Variables para 80% importancia:", n_80, "\n")
    log_cat("Variables para 90% importancia:", n_90, "\n\n")
  }


  # 6. Visualizaciones usando save_plot


  if (save_plots) {
    # Plot 1: Importancia de variables (top 30)
    top_30 <- head(importance_df, 30)

    p1 <- ggplot(top_30, aes(x = reorder(variable, `%IncMSE`), y = `%IncMSE`)) +
      geom_col(fill = "steelblue", alpha = 0.8) +
      coord_flip() +
      labs(
        title = "Importancia de Variables (Random Forest)",
        subtitle = sprintf("Target: %s | n=%d | ntree=%d", target_var, n_obs, ntree),
        x = NULL,
        y = "% Incremento en MSE (mayor = más importante)"
      ) +
      theme_minimal(base_size = 12) +
      theme(panel.grid.major.y = element_blank())
    print(p1)
    save_plot(
      plotname = "rf_importance_top30",
      plot = p1,
      output_dir = output_dir,
      image_number = image_number,
      width = plot_width,
      height = plot_height * 0.8,
      formats = plot_formats,
      print_plot = FALSE,
      cleanup = FALSE
    )

    # Plot 2: Importancia acumulada
    p2 <- ggplot(importance_df, aes(x = rank, y = cumsum_importance)) +
      geom_line(size = 1.2, color = "steelblue") +
      geom_point(size = 2, color = "steelblue") +
      geom_hline(
        yintercept = c(80, 90), linetype = "dashed",
        color = c("red", "orange")
      ) +
      annotate("text",
        x = max(importance_df$rank) * 0.8, y = 82,
        label = "80%", color = "red", size = 4
      ) +
      annotate("text",
        x = max(importance_df$rank) * 0.8, y = 92,
        label = "90%", color = "orange", size = 4
      ) +
      labs(
        title = "Importancia Acumulada de Variables",
        subtitle = sprintf("Target: %s | n_80%%=%d | n_90%%=%d", target_var, n_80, n_90),
        x = "Número de variables",
        y = "% Importancia acumulada"
      ) +
      theme_minimal(base_size = 12)
    print(p2)
    save_plot(
      plotname = "rf_importance_cumulative",
      plot = p2,
      output_dir = output_dir,
      image_number = image_number,
      formats = plot_formats,
      print_plot = FALSE,
      cleanup = FALSE
    )

    if (verbose) {
      log_cat("Gráficos guardados en:", file.path(output_dir, "figures"), "\n\n")
    }
  }


  # 7. Selección de variables finales


  if (!is.null(importance_threshold)) {
    n_selected <- min(which(importance_df$cumsum_importance >= importance_threshold))
    selection_method <- sprintf("%d%% importancia acumulada", importance_threshold)
  } else if (!is.null(top_n_vars)) {
    n_selected <- min(top_n_vars, nrow(importance_df))
    selection_method <- sprintf("Top %d variables", n_selected)
  } else {
    n_selected <- nrow(importance_df)
    selection_method <- "Todas las variables"
  }

  vars_selected <- head(importance_df$variable, n_selected)

  if (verbose) {
    log_cat("=== SELECCIÓN FINAL ===\n")
    log_cat("Método:", selection_method, "\n")
    log_cat("Variables seleccionadas:", n_selected, "\n")
    log_cat(
      "Importancia acumulada:",
      round(importance_df$cumsum_importance[n_selected], 2), "%\n\n"
    )
  }


  # 8. Crear datasets finales


  train_final <- train_df %>%
    dplyr::select(date, all_of(vars_selected), targets)

  test_final <- NULL
  if (!is.null(test_df)) {
    test_final <- test_df %>%
      dplyr::select(date, all_of(vars_selected), targets)

    test_na <- sum(is.na(test_final %>%
      dplyr::select(-date, -targets)))

    if (verbose) {
      log_cat("=== DATASETS FINALES ===\n")
      log_cat("Train:", nrow(train_final), "obs x", ncol(train_final), "vars\n")
      log_cat("Test:", nrow(test_final), "obs x", ncol(test_final), "vars\n")

      if (test_na > 0) {
        log_cat("\n⚠ ADVERTENCIA: Test tiene", test_na, "NAs en predictores\n")
      } else {
        log_cat("\n✓ Test OK: sin NAs en predictores\n")
      }
    }
  } else {
    if (verbose) {
      log_cat("=== DATASET FINAL ===\n")
      log_cat("Train:", nrow(train_final), "obs x", ncol(train_final), "vars\n")
    }
  }


  # 9. Guardar archivos (opcional)


  if (save_files) {
    train_path <- file.path(output_dir, "/RData/train_final_ML.rds")
    saveRDS(train_final, train_path)

    if (!is.null(test_final)) {
      test_path <- file.path(output_dir, "/RData/test_final_ML.rds")
      saveRDS(test_final, test_path)
    }

    model_path <- file.path(output_dir, "/RData/rf_importance_model.rds")
    saveRDS(rf_model, model_path)

    importance_path <- file.path(output_dir, "/tables/variable_importance_ranking.csv")
    write.csv(importance_df, importance_path, row.names = FALSE)

    vars_path <- file.path(output_dir, "/tables/variables_finales_ML.csv")
    write.csv(data.frame(
      rank = 1:n_selected,
      variable = vars_selected,
      importance_pct = head(importance_df$`%IncMSE`, n_selected),
      cumsum_importance = head(importance_df$cumsum_importance, n_selected)
    ), vars_path, row.names = FALSE)

    if (verbose) {
      log_cat("\n=== ARCHIVOS GUARDADOS ===\n")
      log_cat("- train_final_ML.rds\n")
      if (!is.null(test_final)) log_cat("- test_final_ML.rds\n")
      log_cat("- rf_importance_model.rds\n")
      log_cat("- variable_importance_ranking.csv\n")
      log_cat("- variables_finales_ML.csv\n")
    }
  }


  # 10. Retornar resultados


  results <- list(
    train = train_final,
    test = test_final,
    rf_model = rf_model,
    importance = importance_df,
    variables_selected = vars_selected,
    n_vars_selected = n_selected,
    n_vars_80pct = n_80,
    n_vars_90pct = n_90,
    selection_method = selection_method,
    model_rsquared = tail(rf_model$rsq, 1),
    model_mse = tail(rf_model$mse, 1),
    training_time_secs = time_elapsed,
    parameters = list(
      target = target_var,
      ntree = ntree,
      mtry = mtry,
      nodesize = nodesize
    )
  )

  class(results) <- c("rf_importance_ranking", "list")

  if (verbose) log_cat("\n✓ Proceso completado.\n")

  return(results)
}

# Método print para la clase
print.rf_importance_ranking <- function(x, ...) {
  log_cat("=== Random Forest Importance Ranking ===\n")
  log_cat(sprintf("Target: %s\n", x$parameters$target))
  log_cat(sprintf("Model R²: %.4f\n", x$model_rsquared))
  log_cat(sprintf("Model MSE: %.6f\n", x$model_mse))
  log_cat(sprintf("Training time: %.2f seconds\n", x$training_time_secs))
  log_cat(sprintf(
    "\nVariables selected: %d (method: %s)\n",
    x$n_vars_selected, x$selection_method
  ))
  log_cat(sprintf("Variables for 80%% importance: %d\n", x$n_vars_80pct))
  log_cat(sprintf("Variables for 90%% importance: %d\n", x$n_vars_90pct))
}
