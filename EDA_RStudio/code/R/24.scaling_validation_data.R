# ============================================================================
# FUNCIÓN PARA CALCULAR FEATURES EN DATOS DE VALIDACIÓN
# ============================================================================
#
# Descripción: Calcula todas las features usadas en el modelo de predicción
#              IBEX35 para nuevos datos de validación. Incluye features
#              financieras, externas y opcionalmente aplica escalado z-score.
#
# Autor: Generado para TFM Master Data Science
# Fecha: 2025
#
# ============================================================================

library(TTR)
library(zoo)
library(lubridate)
library(dplyr)
library(tidyr)
library(caret)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

calculate_validation_features <- function(
  all_stocks_df,
  validation_start,
  validation_end,
  target_index = "ibex35",
  scaler_path = NULL,
  min_history_days = 252,
  verbose = TRUE
) {
  #' Calcula features para datos de validación

  #'

  #' @param all_stocks_df DataFrame en formato long con todos los índices.
  #'                      Columnas: date, index, open, high, low, close, volume
  #'                      Índices esperados: ibex35, s_p500, dax, ftse100,
  #'                                        volatility_index, oil, gold, euro_dollar
  #' @param validation_start Fecha inicio del período de validación (string o Date)
  #' @param validation_end Fecha fin del período de validación (string o Date)
  #' @param target_index Nombre del índice objetivo (default: "ibex35")
  #' @param scaler_path Ruta al archivo scaler_zscore.rds (opcional).
  #'                    Si se proporciona, aplica z-score scaling.
  #' @param min_history_days Días mínimos de historia para rolling windows (252)
  #' @param verbose Imprimir mensajes de progreso
  #'
  #' @return DataFrame con todas las features calculadas
  #'         Si se proporciona scaler, devuelve versión escalada

  log_msg <- function(...) {
    if (verbose) cat(...)
  }

  # --------------------------------------------------------------------------
  # VALIDACIÓN DE INPUTS
  # --------------------------------------------------------------------------

  required_cols <- c("date", "close", "index")
  if (!all(required_cols %in% names(all_stocks_df))) {
    stop("all_stocks_df debe contener: ", paste(required_cols, collapse = ", "))
  }

  validation_start <- as.Date(validation_start)
  validation_end <- as.Date(validation_end)

  # Convertir index a character para evitar problemas con factors
  all_stocks_df <- all_stocks_df %>%
    mutate(
      date = as.Date(date),
      index = as.character(index)
    ) %>%
    arrange(date)

  # Verificar que existe el target_index
  available_indices <- unique(all_stocks_df$index)
  if (!target_index %in% available_indices) {
    stop(
      "target_index '", target_index, "' no encontrado. ",
      "Índices disponibles: ", paste(available_indices, collapse = ", ")
    )
  }

  log_msg("=== CALCULANDO FEATURES DE VALIDACIÓN ===\n")
  log_msg("Período validación: ", validation_start, " a ", validation_end, "\n")
  log_msg("Índice objetivo: ", target_index, "\n")
  log_msg("Índices disponibles: ", paste(available_indices, collapse = ", "), "\n\n")

  # --------------------------------------------------------------------------
  # PREPARAR DATOS: HISTORIA + VALIDACIÓN
  # --------------------------------------------------------------------------

  # Calcular fecha de inicio para incluir historia
  history_start <- validation_start - min_history_days - 50 # margen extra

  # Filtrar rango necesario
  df_filtered <- all_stocks_df %>%
    filter(date >= history_start & date <= validation_end)

  log_msg("Rango de datos usado: ", min(df_filtered$date), " a ", max(df_filtered$date), "\n")

  # Extraer IBEX (target)
  ibex_df <- df_filtered %>%
    filter(index == target_index) %>%
    arrange(date)

  n_ibex_total <- nrow(ibex_df)
  n_ibex_validation <- sum(ibex_df$date >= validation_start)
  n_history <- n_ibex_total - n_ibex_validation

  log_msg("Observaciones IBEX total: ", n_ibex_total, "\n")
  log_msg("  - Historia (para rolling): ", n_history, "\n")
  log_msg("  - Validación: ", n_ibex_validation, "\n\n")

  # --------------------------------------------------------------------------
  # FEATURES FINANCIERAS (del IBEX)
  # --------------------------------------------------------------------------

  log_msg("--- Calculando features financieras ---\n")

  result <- ibex_df %>%
    mutate(
      # Retornos
      log_return_pct = c(NA, diff(log(close))) * 100,

      # Momentum y volatilidad
      price_momentum = (lag(close) - lag(close, 21)) / lag(close, 21) * 100,
      volatility_20 = rollapply(log_return_pct,
        width = 20, FUN = sd,
        fill = NA, align = "right"
      ),

      # Velocidades
      returns_velocity = log_return_pct - lag(log_return_pct),
      volatility_velocity = volatility_20 - lag(volatility_20),

      # Aceleraciones
      returns_acceleration = returns_velocity - lag(returns_velocity),

      # Targets (para validación)
      returns_next = lead(log_return_pct, 1),
      returns_next_5 = lead(log_return_pct, 5),
      returns_next_10 = lead(log_return_pct, 10),
      returns_next_20 = lead(log_return_pct, 20),

      # ROC
      roc_5 = ROC(close, n = 5) * 100,

      # Volumen
      obv = OBV(close, volume),
      volume_sma20 = SMA(volume, n = 20),

      # Lags
      returns_lag5 = lag(log_return_pct, 5),

      # Volatilidad lag
      volatility_lag5 = lag(volatility_20, 5),

      # Rolling stats - retornos (con lag para evitar look-ahead)
      returns_mean_5 = lag(rollmean(log_return_pct, k = 5, fill = NA, align = "right"), 1),

      # Rolling stats - volatilidad
      volatility_5 = lag(rollapply(log_return_pct,
        width = 5, FUN = sd,
        fill = NA, align = "right"
      ), 1),

      # Rolling stats - rango
      range_5 = lag(rollmean(high - low, k = 5, fill = NA, align = "right"), 1),
      range_20 = lag(rollmean(high - low, k = 20, fill = NA, align = "right"), 1),
      high_max_20 = lag(rollapply(high, width = 20, FUN = max, fill = NA, align = "right"), 1)
    )

  # Estocástico
  stoch_data <- stoch(
    HLC = as.matrix(ibex_df[, c("high", "low", "close")]),
    nFastK = 14, nFastD = 3, nSlowD = 3
  )
  result$slowD <- stoch_data[, "slowD"]

  # OBV normalizado (usando media/sd de los datos disponibles)
  obv_mean <- mean(result$obv, na.rm = TRUE)
  obv_sd <- sd(result$obv, na.rm = TRUE)
  result$obv_norm <- (result$obv - obv_mean) / obv_sd

  log_msg("  Features financieras calculadas\n")

  # --------------------------------------------------------------------------
  # FEATURES EXTERNAS (otros índices)
  # --------------------------------------------------------------------------

  log_msg("--- Calculando features externas ---\n")

  # Pivotar todos los índices externos a wide
  external_indices <- setdiff(available_indices, target_index)

  if (length(external_indices) > 0) {
    # Primero calcular vix_zscore sobre toda la historia disponible del VIX
    vix_zscore_df <- NULL
    if ("volatility_index" %in% external_indices) {
      vix_full <- all_stocks_df %>%
        filter(index == "volatility_index") %>%
        arrange(date) %>%
        mutate(
          vix_ma_252 = rollmean(close, k = 252, fill = NA, align = "right"),
          vix_sd_252 = rollapply(close, width = 252, FUN = sd, fill = NA, align = "right"),
          vix_zscore = (close - vix_ma_252) / vix_sd_252
        ) %>%
        dplyr::select(date, vix_zscore)

      vix_zscore_df <- vix_full
      log_msg("  VIX z-score calculado sobre historia completa\n")
    }

    df_ext_wide <- df_filtered %>%
      filter(index %in% external_indices) %>%
      dplyr::select(date, index, close) %>%
      pivot_wider(names_from = index, values_from = close) %>%
      arrange(date) %>%
      fill(-date, .direction = "down") # Forward fill

    # Merge con result
    result <- result %>%
      left_join(df_ext_wide, by = "date")

    # Añadir vix_zscore precalculado si existe
    if (!is.null(vix_zscore_df)) {
      result <- result %>%
        left_join(vix_zscore_df, by = "date")
    }

    # -- S&P 500 --
    if ("s_p500" %in% names(result)) {
      sp500_prices <- result$s_p500
      sp500_returns <- c(NA, diff(log(sp500_prices))) * 100

      result$sp500_return <- sp500_returns
      result$sp500_momentum <- (sp500_prices - lag(sp500_prices, 20)) /
        lag(sp500_prices, 20) * 100
      result$sp500_vol20 <- rollapply(sp500_returns,
        width = 20, FUN = sd,
        fill = NA, align = "right"
      )

      log_msg("  S&P 500 procesado\n")
    }

    # -- DAX --
    if ("dax" %in% names(result)) {
      dax_prices <- result$dax
      dax_returns <- c(NA, diff(log(dax_prices))) * 100

      result$dax_return <- dax_returns
      result$dax_return_lag1 <- lag(dax_returns, 1)
      result$dax_momentum <- (dax_prices - lag(dax_prices, 20)) /
        lag(dax_prices, 20) * 100

      log_msg("  DAX procesado\n")
    }

    # -- FTSE 100 --
    if ("ftse100" %in% names(result)) {
      ftse_prices <- result$ftse100
      ftse_returns <- c(NA, diff(log(ftse_prices))) * 100

      result$ftse100_return <- ftse_returns
      result$ftse100_return_lag1 <- lag(ftse_returns, 1)
      result$ftse100_momentum <- (ftse_prices - lag(ftse_prices, 20)) /
        lag(ftse_prices, 20) * 100

      log_msg("  FTSE 100 procesado\n")
    }

    # -- VIX --
    if ("volatility_index" %in% names(result)) {
      vix <- result$volatility_index
      result$vix_return <- c(NA, diff(log(vix))) * 100

      # vix_zscore ya está calculado sobre historia completa (arriba)

      # Régimen VIX (one-hot encoding)
      vix_regime <- cut(vix,
        breaks = c(0, 15, 20, 30, 100),
        labels = c("Low", "Normal", "Elevated", "High"),
        include.lowest = TRUE
      )
      result$vix_regime_Low <- as.numeric(vix_regime == "Low")
      result$vix_regime_Normal <- as.numeric(vix_regime == "Normal")
      result$vix_regime_Elevated <- as.numeric(vix_regime == "Elevated")
      result$vix_regime_High <- as.numeric(vix_regime == "High")

      log_msg("  VIX procesado\n")
    }

    # -- Oil --
    if ("oil" %in% names(result)) {
      oil_prices <- result$oil
      oil_returns <- c(NA, diff(log(oil_prices))) * 100

      result$oil_return <- oil_returns
      result$oil_momentum <- (oil_prices - lag(oil_prices, 20)) /
        lag(oil_prices, 20) * 100
      result$oil_vol20 <- rollapply(oil_returns,
        width = 20, FUN = sd,
        fill = NA, align = "right"
      )

      log_msg("  Oil procesado\n")
    }

    # -- Gold --
    if ("gold" %in% names(result)) {
      gold_prices <- result$gold
      gold_returns <- c(NA, diff(log(gold_prices))) * 100

      result$gold_return <- gold_returns
      result$gold_vol20 <- rollapply(gold_returns,
        width = 20, FUN = sd,
        fill = NA, align = "right"
      )

      log_msg("  Gold procesado\n")
    }

    # -- Oil/Gold ratio --
    if (all(c("oil", "gold") %in% names(result))) {
      result$oil_gold_ratio <- result$oil / result$gold
    }

    # -- Euro/Dollar --
    if ("euro_dollar" %in% names(result)) {
      eurodollar_prices <- result$euro_dollar
      eurodollar_returns <- c(NA, diff(log(eurodollar_prices))) * 100

      result$eurodollar_level <- eurodollar_prices
      result$eurodollar_momentum <- (eurodollar_prices - lag(eurodollar_prices, 20)) /
        lag(eurodollar_prices, 20) * 100

      log_msg("  Euro/Dollar procesado\n")
    }

    # -- Spreads --
    ibex_returns <- result$log_return_pct

    if ("sp500_return" %in% names(result)) {
      sp500_lag1 <- lag(result$sp500_return, 1)
      result$vol_spread_target_sp500 <- result$volatility_20 - result$sp500_vol20
    }

    if ("dax_return" %in% names(result)) {
      result$spread_target_dax <- ibex_returns - result$dax_return

      if ("sp500_return" %in% names(result)) {
        sp500_lag1 <- lag(result$sp500_return, 1)
        result$spread_eu_us <- result$dax_return - sp500_lag1
      }
    }

    # -- Risk Score --
    risk_components <- list()

    if ("sp500_return" %in% names(result)) {
      sp_mean <- mean(result$sp500_return, na.rm = TRUE)
      sp_sd <- sd(result$sp500_return, na.rm = TRUE)
      risk_components$equity <- (result$sp500_return - sp_mean) / sp_sd
    }

    if ("oil_return" %in% names(result)) {
      oil_mean <- mean(result$oil_return, na.rm = TRUE)
      oil_sd <- sd(result$oil_return, na.rm = TRUE)
      risk_components$oil <- (result$oil_return - oil_mean) / oil_sd
    }

    if ("gold_return" %in% names(result)) {
      gold_mean <- mean(result$gold_return, na.rm = TRUE)
      gold_sd <- sd(result$gold_return, na.rm = TRUE)
      risk_components$gold <- -(result$gold_return - gold_mean) / gold_sd
    }

    if (length(risk_components) >= 2) {
      result$risk_on_score <- Reduce("+", risk_components) / length(risk_components)
    }

    # Eliminar columnas raw de índices externos
    cols_to_remove <- c(
      "s_p500", "dax", "ftse100", "volatility_index",
      "oil", "gold", "euro_dollar"
    )
    result <- result %>% dplyr::select(-any_of(cols_to_remove))

    log_msg("  Features externas calculadas\n")
  } else {
    log_msg("  No hay índices externos disponibles\n")
  }

  # --------------------------------------------------------------------------
  # CREAR TARGETS DIRECCIONALES
  # --------------------------------------------------------------------------

  result <- result %>%
    mutate(
      direction_next = ifelse(returns_next > 0, 1, 0),
      direction_next_5 = ifelse(returns_next_5 > 0, 1, 0),
      direction_next_10 = ifelse(returns_next_10 > 0, 1, 0),
      direction_next_20 = ifelse(returns_next_20 > 0, 1, 0)
    )

  # --------------------------------------------------------------------------
  # FILTRAR SOLO PERÍODO DE VALIDACIÓN (eliminar historia usada para rolling)
  # --------------------------------------------------------------------------

  result <- result %>%
    filter(date >= validation_start & date <= validation_end)

  log_msg("\n--- Resumen ---\n")
  log_msg("Observaciones finales: ", nrow(result), "\n")
  log_msg("Features calculadas: ", ncol(result), "\n")

  # --------------------------------------------------------------------------
  # SELECCIONAR Y ORDENAR COLUMNAS COMO EN TRAIN
  # --------------------------------------------------------------------------

  # Columnas esperadas en el modelo (del colnames que proporcionaste)
  expected_cols <- c(
    "date",
    "sp500_return", "dax_return", "dax_momentum", "volatility_lag5",
    "returns_acceleration", "sp500_vol20", "vix_return", "volatility_5",
    "ftse100_momentum", "ftse100_return", "price_momentum", "roc_5",
    "spread_eu_us", "sp500_momentum", "oil_gold_ratio", "dax_return_lag1",
    "range_20", "risk_on_score", "high_max_20", "volume_sma20", "oil_vol20",
    "obv_norm", "ftse100_return_lag1", "range_5", "vix_zscore",
    "eurodollar_momentum", "gold_vol20", "returns_mean_5", "returns_lag5",
    "gold_return", "eurodollar_level", "vol_spread_target_sp500", "oil_return",
    "euribor_3m", "volatility_velocity", "spread_target_dax", "oil_momentum",
    "slowD", "returns_next", "returns_next_5", "returns_next_10",
    "returns_next_20", "vix_regime_Low", "vix_regime_Normal",
    "vix_regime_Elevated", "vix_regime_High", "direction_next",
    "direction_next_5", "direction_next_10", "direction_next_20"
  )

  # Identificar columnas faltantes
  missing_cols <- setdiff(expected_cols, names(result))
  if (length(missing_cols) > 0) {
    log_msg("\nADVERTENCIA: Columnas faltantes: ", paste(missing_cols, collapse = ", "), "\n")
    log_msg("  Estas columnas se llenarán con NA\n")

    for (col in missing_cols) {
      result[[col]] <- NA
    }
  }

  # Ordenar columnas
  available_cols <- intersect(expected_cols, names(result))
  result <- result %>% dplyr::select(all_of(available_cols))

  # --------------------------------------------------------------------------
  # APLICAR ESCALADO (si se proporciona scaler)
  # --------------------------------------------------------------------------

  if (!is.null(scaler_path)) {
    log_msg("\n--- Aplicando escalado z-score ---\n")

    if (!file.exists(scaler_path)) {
      warning("Archivo scaler no encontrado: ", scaler_path)
    } else {
      scaler <- readRDS(scaler_path)
      # Variables que el scaler espera
      scaler_vars <- names(scaler$mean)
      log_msg("  Variables en scaler: ", length(scaler_vars), "\n")


      # Variables a escalar (numéricas, excluyendo targets y date)
      exclude_vars <- c(
        "date",
        "returns_next", "returns_next_5", "returns_next_10", "returns_next_20",
        "direction_next", "direction_next_5", "direction_next_10", "direction_next_20",
        "vix_regime_Low", "vix_regime_Normal", "vix_regime_Elevated", "vix_regime_High"
      )

      numeric_vars <- result %>%
        dplyr::select(-any_of(exclude_vars)) %>%
        dplyr::select(where(is.numeric)) %>%
        names()

      # Solo escalar variables que están TANTO en result COMO en scaler
      vars_to_scale <- intersect(numeric_vars, scaler_vars)

      # Diagnóstico
      vars_missing_in_result <- setdiff(scaler_vars, numeric_vars)
      vars_missing_in_scaler <- setdiff(numeric_vars, scaler_vars)

      if (length(vars_missing_in_result) > 0) {
        log_msg(
          "  AVISO - Variables del scaler no encontradas en datos: ",
          paste(vars_missing_in_result, collapse = ", "), "\n"
        )
      }

      if (length(vars_missing_in_scaler) > 0) {
        log_msg(
          "  AVISO - Variables en datos no encontradas en scaler: ",
          paste(vars_missing_in_scaler, collapse = ", "), "\n"
        )
      }

      log_msg("  Variables a escalar: ", length(vars_to_scale), "\n")

      if (length(vars_to_scale) > 0) {
        # Aplicar escalado manualmente para evitar errores de predict()
        for (var in vars_to_scale) {
          result[[var]] <- (result[[var]] - scaler$mean[var]) / scaler$std[var]
        }
        log_msg("  Escalado aplicado correctamente\n")
      }
    }
  }

  # --------------------------------------------------------------------------
  # RESUMEN FINAL
  # --------------------------------------------------------------------------

  na_counts <- colSums(is.na(result))
  cols_with_na <- na_counts[na_counts > 0]

  if (length(cols_with_na) > 0) {
    log_msg("\nColumnas con NAs:\n")
    for (i in seq_along(cols_with_na)) {
      log_msg("  ", names(cols_with_na)[i], ": ", cols_with_na[i], " NAs\n")
    }
  }

  log_msg("\n=== COMPLETADO ===\n")

  return(result)
}

# ============================================================================
# EJEMPLO DE USO
# ============================================================================
'
if (FALSE) { # No ejecutar automáticamente

  # all_stocks_df debe tener formato long:
  # - Columnas: date, index, open, high, low, close, volume
  # - index: factor/character con valores como "ibex35", "s_p500", "dax", etc.

  # 1. Calcular features SIN escalado
  validation_unscaled <- calculate_validation_features(
    all_stocks_df = all_stocks_df,
    validation_start = "2025-10-17",
    validation_end = "2025-11-30",
    target_index = "ibex35",
    scaler_path = NULL,
    min_history_days = 252,
    verbose = TRUE
  )

  # 2. Calcular features CON escalado
  validation_scaled <- calculate_validation_features(
    all_stocks_df = all_stocks_df,
    validation_start = "2025-10-17",
    validation_end = "2025-11-30",
    target_index = "ibex35",
    scaler_path = "scaler_zscore.rds",
    min_history_days = 252,
    verbose = TRUE
  )

  # 3. Guardar
  saveRDS(validation_unscaled, "validation_unscaled.rds")
  saveRDS(validation_scaled, "validation_scaled.rds")

  # 4. Verificar columnas
  print(colnames(validation_scaled))
}'
