# ============================================================================
# PASO 2: VERIFICACIÓN DE CONSISTENCIA DE FEATURES
# ============================================================================

library(dplyr)

verify_features_consistency <- function(validation_df, verbose = TRUE) {
  #' Verifica que las features del set de validación coincidan con las del modelo
  #'
  #' @param validation_df DataFrame con features calculadas
  #' @param verbose Imprimir detalles
  #'
  #' @return Lista con resultados de verificación

  log_msg <- function(...) {
    if (verbose) cat(...)
  }

  log_msg("=== VERIFICACIÓN DE CONSISTENCIA DE FEATURES ===\n\n")

  # Features esperadas por el modelo LightGBM (extraídas del .txt)
  model_features <- c(
    "sp500_return", "dax_return", "dax_momentum", "volatility_lag5",
    "returns_acceleration", "sp500_vol20", "vix_return", "volatility_5",
    "ftse100_momentum", "ftse100_return", "price_momentum", "roc_5",
    "spread_eu_us", "sp500_momentum", "oil_gold_ratio", "dax_return_lag1",
    "range_20", "risk_on_score", "high_max_20", "volume_sma20", "oil_vol20",
    "obv_norm", "ftse100_return_lag1", "range_5", "vix_zscore",
    "eurodollar_momentum", "gold_vol20", "returns_mean_5", "returns_lag5",
    "gold_return", "eurodollar_level", "vol_spread_target_sp500", "oil_return",
    "euribor_3m", "volatility_velocity", "spread_target_dax", "oil_momentum",
    "slowD", "vix_regime_Low", "vix_regime_Normal", "vix_regime_Elevated",
    "vix_regime_High"
  )

  validation_cols <- names(validation_df)

  results <- list()

  # --------------------------------------------------------------------------
  # 1. Verificar columnas presentes/faltantes
  # --------------------------------------------------------------------------

  log_msg("--- 1. Verificación de columnas ---\n")

  missing_in_validation <- setdiff(model_features, validation_cols)
  extra_in_validation <- setdiff(validation_cols, c(
    model_features, "date",
    "returns_next", "returns_next_5",
    "returns_next_10", "returns_next_20",
    "direction_next", "direction_next_5",
    "direction_next_10", "direction_next_20"
  ))

  results$missing_features <- missing_in_validation
  results$extra_features <- extra_in_validation

  if (length(missing_in_validation) == 0) {
    log_msg("  ✓ Todas las features del modelo están presentes\n")
  } else {
    log_msg("  ✗ Features FALTANTES (", length(missing_in_validation), "):\n")
    for (f in missing_in_validation) {
      log_msg("    - ", f, "\n")
    }
  }

  if (length(extra_in_validation) > 0) {
    log_msg("  ⚠ Features EXTRA no usadas por el modelo (", length(extra_in_validation), "):\n")
    for (f in extra_in_validation) {
      log_msg("    - ", f, "\n")
    }
  }

  # --------------------------------------------------------------------------
  # 2. Verificar NAs en features predictoras
  # --------------------------------------------------------------------------

  log_msg("\n--- 2. Verificación de NAs en features predictoras ---\n")

  features_present <- intersect(model_features, validation_cols)
  na_counts <- sapply(validation_df[features_present], function(x) sum(is.na(x)))
  features_with_na <- na_counts[na_counts > 0]

  results$features_with_na <- features_with_na
  results$n_complete_rows <- sum(complete.cases(validation_df[features_present]))
  results$n_total_rows <- nrow(validation_df)

  if (length(features_with_na) == 0) {
    log_msg("  ✓ Sin NAs en features predictoras\n")
  } else {
    log_msg("  ✗ Features con NAs:\n")
    for (i in seq_along(features_with_na)) {
      log_msg(
        "    - ", names(features_with_na)[i], ": ",
        features_with_na[i], " NAs\n"
      )
    }
  }

  log_msg(
    "  Filas completas: ", results$n_complete_rows, " de ",
    results$n_total_rows, "\n"
  )

  # --------------------------------------------------------------------------
  # 3. Verificar rangos de valores (detectar anomalías)
  # --------------------------------------------------------------------------

  log_msg("\n--- 3. Verificación de rangos de valores ---\n")

  # Rangos esperados del modelo (extraídos de feature_infos en el .txt)
  expected_ranges <- list(
    sp500_return = c(-11.78, 8.22),
    dax_return = c(-10.87, 8.64),
    dax_momentum = c(-7.65, 4.18),
    volatility_lag5 = c(-1.29, 7.38),
    returns_acceleration = c(-10.00, 10.95),
    sp500_vol20 = c(-1.13, 8.65),
    vix_return = c(-4.23, 9.83),
    volatility_5 = c(-1.36, 11.35),
    ftse100_momentum = c(-8.27, 4.17),
    ftse100_return = c(-11.86, 8.91),
    price_momentum = c(-7.47, 5.06),
    roc_5 = c(-10.88, 4.92),
    spread_eu_us = c(-9.41, 9.87),
    sp500_momentum = c(-7.55, 5.99),
    oil_gold_ratio = c(-4.78, 3.01),
    dax_return_lag1 = c(-10.90, 8.66),
    range_20 = c(-1.41, 6.14),
    risk_on_score = c(-10.79, 8.85),
    high_max_20 = c(-2.54, 2.05),
    volume_sma20 = c(-1.89, 3.35),
    oil_vol20 = c(-1.21, 7.35),
    obv_norm = c(-2.92, 1.58),
    ftse100_return_lag1 = c(-11.91, 8.95),
    range_5 = c(-1.56, 8.53),
    vix_zscore = c(-1.55, 9.21),
    eurodollar_momentum = c(-4.23, 4.07),
    gold_vol20 = c(-1.84, 7.01),
    returns_mean_5 = c(-10.88, 4.92),
    returns_lag5 = c(-12.27, 6.65),
    gold_return = c(-5.65, 6.33),
    eurodollar_level = c(-2.25, 3.16),
    vol_spread_target_sp500 = c(-4.23, 4.96),
    oil_return = c(-9.99, 11.30),
    euribor_3m = c(-0.69, 2.46),
    volatility_velocity = c(-19.70, 17.72),
    spread_target_dax = c(-8.74, 5.73),
    oil_momentum = c(-17.51, 15.01),
    slowD = c(-1.94, 1.59),
    vix_regime_Low = c(-0.88, 1.13),
    vix_regime_Normal = c(-0.62, 1.60),
    vix_regime_Elevated = c(-0.55, 1.83),
    vix_regime_High = c(-0.23, 4.26)
  )

  out_of_range <- list()

  for (feat in names(expected_ranges)) {
    if (feat %in% validation_cols) {
      vals <- validation_df[[feat]]
      vals <- vals[!is.na(vals)]

      if (length(vals) > 0) {
        exp_min <- expected_ranges[[feat]][1]
        exp_max <- expected_ranges[[feat]][2]
        actual_min <- min(vals)
        actual_max <- max(vals)

        # Margen de tolerancia del 20%
        range_width <- exp_max - exp_min
        tolerance <- range_width * 0.2

        if (actual_min < (exp_min - tolerance) || actual_max > (exp_max + tolerance)) {
          out_of_range[[feat]] <- list(
            expected = expected_ranges[[feat]],
            actual = c(actual_min, actual_max)
          )
        }
      }
    }
  }

  results$out_of_range <- out_of_range

  if (length(out_of_range) == 0) {
    log_msg("  ✓ Todos los valores dentro de rangos esperados (±20% tolerancia)\n")
  } else {
    log_msg("  ⚠ Features con valores fuera de rango:\n")
    for (feat in names(out_of_range)) {
      log_msg("    - ", feat, ":\n")
      log_msg(
        "        Esperado: [", round(out_of_range[[feat]]$expected[1], 2),
        ", ", round(out_of_range[[feat]]$expected[2], 2), "]\n"
      )
      log_msg(
        "        Actual:   [", round(out_of_range[[feat]]$actual[1], 2),
        ", ", round(out_of_range[[feat]]$actual[2], 2), "]\n"
      )
    }
  }

  # --------------------------------------------------------------------------
  # 4. Resumen estadístico de features
  # --------------------------------------------------------------------------

  log_msg("\n--- 4. Resumen estadístico ---\n")

  stats_summary <- data.frame(
    feature = features_present,
    min = sapply(validation_df[features_present], min, na.rm = TRUE),
    mean = sapply(validation_df[features_present], mean, na.rm = TRUE),
    max = sapply(validation_df[features_present], max, na.rm = TRUE),
    sd = sapply(validation_df[features_present], sd, na.rm = TRUE),
    nas = sapply(validation_df[features_present], function(x) sum(is.na(x)))
  )
  rownames(stats_summary) <- NULL

  results$stats_summary <- stats_summary

  log_msg("  Features numéricas: ", length(features_present), "\n")
  log_msg("  (Ver results$stats_summary para detalles)\n")

  # --------------------------------------------------------------------------
  # 5. Verificación final
  # --------------------------------------------------------------------------

  log_msg("\n--- 5. RESULTADO FINAL ---\n")

  all_ok <- length(missing_in_validation) == 0 &&
    length(features_with_na) == 0 &&
    length(out_of_range) == 0

  results$ready_for_prediction <- all_ok

  if (all_ok) {
    log_msg("  ✓ DATOS LISTOS PARA PREDICCIÓN\n")
  } else {
    log_msg("  ✗ HAY PROBLEMAS QUE RESOLVER:\n")
    if (length(missing_in_validation) > 0) {
      log_msg("    - Features faltantes\n")
    }
    if (length(features_with_na) > 0) {
      log_msg("    - NAs en features predictoras\n")
    }
    if (length(out_of_range) > 0) {
      log_msg("    - Valores fuera de rango (posible drift)\n")
    }
  }

  log_msg("\n=== VERIFICACIÓN COMPLETADA ===\n")

  return(results)
}

# ============================================================================
# EJEMPLO DE USO
# ============================================================================
'
if (FALSE) {
  # Cargar datos de validación
  validation_df <- readRDS("validation_unscaled.rds")

  # Ejecutar verificación
  check_results <- verify_features_consistency(validation_df, verbose = TRUE)

  # Ver resumen estadístico
  print(check_results$stats_summary)

  # Ver si está listo para predicción
  print(check_results$ready_for_prediction)
}
'
