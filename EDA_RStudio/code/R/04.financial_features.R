# FUNCIÓN DE FEATURE ENGINEERING FINANCIERO - SIN LOOK-AHEAD BIAS
#
# Descripción: Calcula features financieras evitando look-ahead bias
#              mediante train/test split temporal
#
# Input:  dataframe con columnas: date, close, open, high, low, volume
# Output: list(train = df_train, test = df_test, stats = estadísticos_train)
#

library(TTR)
library(zoo)
library(lubridate)
library(dplyr)
library(recipes)
library(moments)

# FUNCIÓN AUXILIAR PARA YEO-JOHNSON

yeojohnson_transform <- function(x, lambda) {
  if (lambda != 0) {
    ((abs(x) + 1)^lambda - 1) / lambda * sign(x)
  } else {
    sign(x) * log(abs(x) + 1)
  }
}

# FUNCIÓN PRINCIPAL CON TRAIN/TEST SPLIT

#' Feature Engineering para Series Financieras con Train/Test Split
#'
#' @param df Dataframe con columnas: date, close, open, high, low, volume
#' @param train_start Fecha inicio período train
#' @param train_end Fecha fin período train
#' @param test_start Fecha inicio período test
#' @param test_end Fecha fin período test
#' @param apply_scaling Aplicar z-score y min-max rolling (sin look-ahead)
#' @param apply_yj Aplicar transformación Yeo-Johnson (estimada en train)
#' @return Lista: list(train = df, test = df, stats = list())
#'
calculate_financial_features_split <- function(df,
                                               train_start,
                                               train_end,
                                               test_start,
                                               test_end,
                                               apply_scaling = FALSE,
                                               apply_yj = FALSE) {
  # Validación
  required_cols <- c("date", "close", "open", "high", "low", "volume")
  if (!all(required_cols %in% names(df))) {
    stop("El dataframe debe contener: date, close, open, high, low, volume")
  }

  train_start <- as.Date(train_start)
  train_end <- as.Date(train_end)
  test_start <- as.Date(test_start)
  test_end <- as.Date(test_end)

  if (train_end >= test_start) {
    stop("train_end debe ser anterior a test_start")
  }

  log_cat("\n=== FEATURE ENGINEERING FINANCIERO SIN LOOK-AHEAD BIAS ===\n")
  log_cat(sprintf("Train: %s a %s\n", train_start, train_end))
  log_cat(sprintf("Test:  %s a %s\n\n", test_start, test_end))

  # Ordenar y filtrar
  df <- df %>%
    arrange(date) %>%
    filter(date >= train_start & date <= test_end)

  # Separar train y test
  df_train <- df %>% filter(date >= train_start & date <= train_end)
  df_test <- df %>% filter(date >= test_start & date <= test_end)

  log_cat(sprintf("Obs train: %d\n", nrow(df_train)))
  log_cat(sprintf("Obs test: %d\n\n", nrow(df_test)))

  # Inicializar
  train_stats <- list()

  # PROCESAR TRAIN

  log_cat("--- Procesando TRAIN ---\n")

  result_train <- df_train %>%
    mutate(
      # Fecha y temporales
      date = as.Date(date),
      weekday = wday(date),
      month = month(date),
      quarter = quarter(date),

      # Suavizado y retornos
      close_smooth = SMA(close, n = 5),
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
      volume_velocity = volume - lag(volume),

      # Aceleraciones
      returns_acceleration = returns_velocity - lag(returns_velocity),
      volatility_acceleration = volatility_velocity - lag(volatility_velocity),

      # Variables auxiliares
      returns_next = lead(log_return_pct, 1),
      returns_next_5 = lead(log_return_pct, 5),
      returns_next_10 = lead(log_return_pct, 10),
      returns_next_20 = lead(log_return_pct, 20),


      # Log
      log_close = log(close),

      # Medias móviles
      sma_20 = SMA(close, n = 20),
      ema_12 = EMA(close, n = 12),
      dist_sma20 = ((close - sma_20) / sma_20) * 100,

      # RSI
      rsi_14 = RSI(close, n = 14),

      # ROC
      roc_5 = ROC(close, n = 5) * 100,

      # Volumen
      obv = OBV(close, volume),
      volume_roc = ROC(volume, n = 5) * 100,
      volume_sma20 = SMA(volume, n = 20),
      volume_ratio = volume / volume_sma20,

      # Lags retornos
      returns_lag1 = lag(log_return_pct, 1),
      returns_lag5 = lag(log_return_pct, 5),
      returns_lag10 = lag(log_return_pct, 10),
      returns_lag20 = lag(log_return_pct, 20),

      # Lags volatilidad
      volatility_lag1 = lag(volatility_20, 1),
      volatility_lag5 = lag(volatility_20, 5),

      # Rolling stats - retornos
      returns_mean_5 = lag(rollmean(log_return_pct, k = 5, fill = NA, align = "right"), 1),
      returns_mean_20 = lag(rollmean(log_return_pct, k = 20, fill = NA, align = "right"), 1),

      # Rolling stats - volatilidad
      volatility_5 = lag(rollapply(log_return_pct,
        width = 5, FUN = sd,
        fill = NA, align = "right"
      ), 1),
      volatility_10 = lag(rollapply(log_return_pct,
        width = 10, FUN = sd,
        fill = NA, align = "right"
      ), 1),

      # Rolling stats - rango
      range_5 = lag(rollmean(high - low, k = 5, fill = NA, align = "right"), 1),
      range_20 = lag(rollmean(high - low, k = 20, fill = NA, align = "right"), 1),
      high_max_20 = lag(rollapply(high, width = 20, FUN = max, fill = NA, align = "right"), 1),
      low_min_20 = lag(rollapply(low, width = 20, FUN = min, fill = NA, align = "right"), 1),
      price_position_20 = lag((close - low_min_20) / (high_max_20 - low_min_20), 1)
    )

  # Estocástico (requiere cálculo aparte)
  stoch_train <- stoch(
    HLC = df_train[, c("high", "low", "close")],
    nFastK = 14, nFastD = 3, nSlowD = 3
  )
  result_train <- result_train %>%
    mutate(
      fastK = stoch_train[, "fastK"],
      fastD = stoch_train[, "fastD"],
      slowD = stoch_train[, "slowD"]
    )

  # OBV normalizado: usar estadísticos de train
  train_stats$obv_mean <- mean(result_train$obv, na.rm = TRUE)
  train_stats$obv_sd <- sd(result_train$obv, na.rm = TRUE)
  result_train <- result_train %>%
    mutate(obv_norm = (obv - train_stats$obv_mean) / train_stats$obv_sd)

  # Scaling (si se solicita)
  if (apply_scaling) {
    log_cat("  Aplicando scaling en train...\n")

    # Z-score rolling (252 días)
    result_train <- result_train %>%
      mutate(
        close_zscore_roll = (close - rollmean(close, k = 252, fill = NA, align = "right")) /
          rollapply(close, width = 252, FUN = sd, fill = NA, align = "right"),
        returns_zscore_roll = (log_return_pct - rollmean(log_return_pct, k = 252, fill = NA, align = "right")) /
          rollapply(log_return_pct, width = 252, FUN = sd, fill = NA, align = "right")
      )

    # Min-Max rolling (252 días)
    result_train <- result_train %>%
      mutate(
        close_minmax_roll = (close - rollapply(close, width = 252, FUN = min, fill = NA, align = "right")) /
          (rollapply(close, width = 252, FUN = max, fill = NA, align = "right") -
            rollapply(close, width = 252, FUN = min, fill = NA, align = "right")),
        returns_minmax_roll = (log_return_pct - rollapply(log_return_pct, width = 252, FUN = min, fill = NA, align = "right")) /
          (rollapply(log_return_pct, width = 252, FUN = max, fill = NA, align = "right") -
            rollapply(log_return_pct, width = 252, FUN = min, fill = NA, align = "right"))
      )

    # Guardar estadísticos de los últimos 252 días de train
    n_window <- min(252, nrow(result_train))

    train_stats$close_mean_252 <- mean(tail(result_train$close, n_window), na.rm = TRUE)
    train_stats$close_sd_252 <- sd(tail(result_train$close, n_window), na.rm = TRUE)
    train_stats$close_min_252 <- min(tail(result_train$close, n_window), na.rm = TRUE)
    train_stats$close_max_252 <- max(tail(result_train$close, n_window), na.rm = TRUE)

    train_stats$returns_mean_252 <- mean(tail(result_train$log_return_pct, n_window), na.rm = TRUE)
    train_stats$returns_sd_252 <- sd(tail(result_train$log_return_pct, n_window), na.rm = TRUE)
    train_stats$returns_min_252 <- min(tail(result_train$log_return_pct, n_window), na.rm = TRUE)
    train_stats$returns_max_252 <- max(tail(result_train$log_return_pct, n_window), na.rm = TRUE)
  }

  # Yeo-Johnson (si se solicita)
  if (apply_yj) {
    log_cat("  Aplicando Yeo-Johnson en train...\n")

    returns_clean <- na.omit(result_train$log_return_pct)
    yj_transform <- step_YeoJohnson(recipe(~., data = data.frame(x = returns_clean)), x)
    yj_prep <- prep(yj_transform, training = data.frame(x = returns_clean))
    lambda_yj <- yj_prep$steps[[1]]$lambdas

    train_stats$yj_lambda <- lambda_yj

    result_train <- result_train %>%
      mutate(return_yj = yeojohnson_transform(log_return_pct, lambda_yj))
  }

  # PROCESAR TEST

  log_cat("\n--- Procesando TEST ---\n")

  result_test <- df_test %>%
    mutate(
      # Fecha y temporales
      date = as.Date(date),
      weekday = wday(date),
      month = month(date),
      quarter = quarter(date),

      # Suavizado y retornos
      close_smooth = SMA(close, n = 5),
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
      volume_velocity = volume - lag(volume),

      # Aceleraciones
      returns_acceleration = returns_velocity - lag(returns_velocity),
      volatility_acceleration = volatility_velocity - lag(volatility_velocity),

      # Variables auxiliares
      returns_next = lead(log_return_pct, 1),
      returns_next_5 = lead(log_return_pct, 5),
      returns_next_10 = lead(log_return_pct, 10),
      returns_next_20 = lead(log_return_pct, 20),
      # Log
      log_close = log(close),

      # Medias móviles
      sma_20 = SMA(close, n = 20),
      ema_12 = EMA(close, n = 12),
      dist_sma20 = ((close - sma_20) / sma_20) * 100,

      # RSI
      rsi_14 = RSI(close, n = 14),

      # ROC
      roc_5 = ROC(close, n = 5) * 100,

      # Volumen
      obv = OBV(close, volume),
      volume_roc = ROC(volume, n = 5) * 100,
      volume_sma20 = SMA(volume, n = 20),
      volume_ratio = volume / volume_sma20,

      # Lags retornos
      returns_lag1 = lag(log_return_pct, 1),
      returns_lag5 = lag(log_return_pct, 5),
      returns_lag10 = lag(log_return_pct, 10),
      returns_lag20 = lag(log_return_pct, 20),

      # Lags volatilidad
      volatility_lag1 = lag(volatility_20, 1),
      volatility_lag5 = lag(volatility_20, 5),

      # Rolling stats - retornos
      returns_mean_5 = lag(rollmean(log_return_pct, k = 5, fill = NA, align = "right"), 1),
      returns_mean_20 = lag(rollmean(log_return_pct, k = 20, fill = NA, align = "right"), 1),

      # Rolling stats - volatilidad
      volatility_5 = lag(rollapply(log_return_pct,
        width = 5, FUN = sd,
        fill = NA, align = "right"
      ), 1),
      volatility_10 = lag(rollapply(log_return_pct,
        width = 10, FUN = sd,
        fill = NA, align = "right"
      ), 1),

      # Rolling stats - rango
      range_5 = lag(rollmean(high - low, k = 5, fill = NA, align = "right"), 1),
      range_20 = lag(rollmean(high - low, k = 20, fill = NA, align = "right"), 1),
      high_max_20 = lag(rollapply(high, width = 20, FUN = max, fill = NA, align = "right"), 1),
      low_min_20 = lag(rollapply(low, width = 20, FUN = min, fill = NA, align = "right"), 1),
      price_position_20 = lag((close - low_min_20) / (high_max_20 - low_min_20), 1)
    )

  # Estocástico
  stoch_test <- stoch(
    HLC = df_test[, c("high", "low", "close")],
    nFastK = 14, nFastD = 3, nSlowD = 3
  )
  result_test <- result_test %>%
    mutate(
      fastK = stoch_test[, "fastK"],
      fastD = stoch_test[, "fastD"],
      slowD = stoch_test[, "slowD"]
    )

  # OBV normalizado: usar estadísticos de TRAIN
  result_test <- result_test %>%
    mutate(obv_norm = (obv - train_stats$obv_mean) / train_stats$obv_sd)

  # Scaling (si se solicitó)
  if (apply_scaling) {
    log_cat("  Aplicando scaling en test (con stats de train)...\n")

    # Para rolling en test: concatenar últimas 252 obs de train con test
    close_concat <- c(tail(df_train$close, 252), df_test$close)
    returns_concat <- c(NA, diff(log(close_concat))) * 100
    # Ajustar para que coincida con test
    returns_test_for_rolling <- tail(returns_concat, nrow(df_test))

    # Z-score rolling
    close_ma_concat <- rollmean(close_concat, k = 252, fill = NA, align = "right")
    close_sd_concat <- rollapply(close_concat,
      width = 252, FUN = sd,
      fill = NA, align = "right"
    )
    close_zscore_concat <- (close_concat - close_ma_concat) / close_sd_concat
    close_zscore_test <- tail(close_zscore_concat, nrow(df_test))

    returns_ma_concat <- rollmean(returns_concat, k = 252, fill = NA, align = "right")
    returns_sd_concat <- rollapply(returns_concat,
      width = 252, FUN = sd,
      fill = NA, align = "right"
    )
    returns_zscore_concat <- (returns_concat - returns_ma_concat) / returns_sd_concat
    returns_zscore_test <- tail(returns_zscore_concat, nrow(df_test))

    # Rellenar NAs con estadísticos de train
    na_idx_close <- is.na(close_zscore_test)
    if (any(na_idx_close)) {
      close_zscore_test[na_idx_close] <-
        (df_test$close[na_idx_close] - train_stats$close_mean_252) /
          train_stats$close_sd_252
    }

    na_idx_returns <- is.na(returns_zscore_test)
    if (any(na_idx_returns)) {
      returns_zscore_test[na_idx_returns] <-
        (result_test$log_return_pct[na_idx_returns] - train_stats$returns_mean_252) /
          train_stats$returns_sd_252
    }

    result_test$close_zscore_roll <- close_zscore_test
    result_test$returns_zscore_roll <- returns_zscore_test

    # Min-Max rolling
    close_min_concat <- rollapply(close_concat,
      width = 252, FUN = min,
      fill = NA, align = "right"
    )
    close_max_concat <- rollapply(close_concat,
      width = 252, FUN = max,
      fill = NA, align = "right"
    )
    close_minmax_concat <- (close_concat - close_min_concat) /
      (close_max_concat - close_min_concat)
    close_minmax_test <- tail(close_minmax_concat, nrow(df_test))

    returns_min_concat <- rollapply(returns_concat,
      width = 252, FUN = min,
      fill = NA, align = "right"
    )
    returns_max_concat <- rollapply(returns_concat,
      width = 252, FUN = max,
      fill = NA, align = "right"
    )
    returns_minmax_concat <- (returns_concat - returns_min_concat) /
      (returns_max_concat - returns_min_concat)
    returns_minmax_test <- tail(returns_minmax_concat, nrow(df_test))

    # Rellenar NAs
    na_idx_close_mm <- is.na(close_minmax_test)
    if (any(na_idx_close_mm)) {
      close_minmax_test[na_idx_close_mm] <-
        (df_test$close[na_idx_close_mm] - train_stats$close_min_252) /
          (train_stats$close_max_252 - train_stats$close_min_252)
    }

    na_idx_returns_mm <- is.na(returns_minmax_test)
    if (any(na_idx_returns_mm)) {
      returns_minmax_test[na_idx_returns_mm] <-
        (result_test$log_return_pct[na_idx_returns_mm] - train_stats$returns_min_252) /
          (train_stats$returns_max_252 - train_stats$returns_min_252)
    }

    result_test$close_minmax_roll <- close_minmax_test
    result_test$returns_minmax_roll <- returns_minmax_test
  }

  # Yeo-Johnson (si se solicitó)
  if (apply_yj) {
    log_cat("  Aplicando Yeo-Johnson en test (con lambda de train)...\n")

    result_test <- result_test %>%
      mutate(return_yj = yeojohnson_transform(log_return_pct, train_stats$yj_lambda))
  }

  # RESUMEN

  n_features <- ncol(result_train) - length(required_cols)

  log_cat("RESUMEN - FEATURES FINANCIERAS SIN LOOK-AHEAD BIAS\n")
  log_cat(sprintf("Features creadas: %d\n", n_features))
  log_cat(sprintf("Train obs: %d\n", nrow(result_train)))
  log_cat(sprintf("Test obs: %d\n", nrow(result_test)))
  log_cat(sprintf("Stats guardados: %d\n", length(train_stats)))

  na_train <- colSums(is.na(result_train))
  na_test <- colSums(is.na(result_test))

  log_cat(sprintf("\nColumnas con NAs train: %d de %d\n", sum(na_train > 0), n_features))
  log_cat(sprintf("Columnas con NAs test: %d de %d\n", sum(na_test > 0), n_features))


  log_cat("✓ Sin look-ahead bias\n")
  log_cat("✓ Estadísticos de train aplicados a test\n\n")

  return(list(
    train = result_train,
    test = result_test,
    stats = train_stats
  ))
}

# EJEMPLO DE USO ####

# result <- calculate_financial_features_split(
#   df = ibex_raw,
#   train_start = "2010-01-01",
#   train_end = "2019-12-31",
#   test_start = "2020-01-01",
#   test_end = "2022-12-31",
#   apply_scaling = TRUE,
#   apply_yj = TRUE
# )
#
# train_features <- result$train
# test_features <- result$test
# train_statistics <- result$stats
#
# # Ver estadísticos guardados
# str(train_statistics)
# train_statistics$obv_mean
# train_statistics$close_mean_252
# train_statistics$yj_lambda
