# FUNCIÓN DE FEATURE ENGINEERING PARA ACTIVOS EXTERNOS - VERSIÓN CORREGIDA
library(TTR)
library(zoo)
library(dplyr)
library(tidyr)


get_default_config <- function() {
  list(
    equity_indices = c(
      "s_p500", "dax", "ftse100", "nikkei225",
      "euro_next100", "sbf120"
    ),
    volatility = c("volatility_index"),
    commodities = c("oil", "gold"),
    forex = c("euro_dollar")
  )
}

get_recommended_indices <- function(priority = "core") {
  core <- c("s_p500", "dax", "volatility_index", "oil", "euro_dollar")
  extended <- c(core, "ftse100", "gold", "euro_next100")
  all <- c(extended, "nikkei225", "sbf120")

  if (priority == "core") {
    return(core)
  } else if (priority == "extended") {
    return(extended)
  } else if (priority == "all") {
    return(all)
  } else {
    stop("priority debe ser 'core', 'extended' o 'all'")
  }
}

calculate_external_features_split <- function(all_stocks_df,
                                              indices,
                                              target_index,
                                              train_start,
                                              train_end,
                                              test_start,
                                              test_end,
                                              features_config,
                                              missing_method = "forward_fill",
                                              lag_international = TRUE) {
  train_start <- as.Date(train_start)
  train_end <- as.Date(train_end)
  test_start <- as.Date(test_start)
  test_end <- as.Date(test_end)

  log_cat("\n=== CALCULANDO FEATURES SIN LOOK-AHEAD BIAS ===\n")
  log_cat(sprintf("Train: %s a %s\n", train_start, train_end))
  log_cat(sprintf("Test:  %s a %s\n\n", test_start, test_end))

  # Pivotar solo el rango necesario (train + test, sin historia extra innecesaria)
  df_wide <- all_stocks_df %>%
    filter(date >= train_start & date <= test_end) %>%
    filter(index %in% c(target_index, indices)) %>%
    dplyr::select(date, index, close) %>%
    pivot_wider(names_from = index, values_from = close) %>%
    arrange(date)

  # Forward fill
  if (missing_method == "forward_fill") {
    df_wide <- df_wide %>% fill(-date, .direction = "down")
  }

  log_cat("Columnas disponibles:", paste(names(df_wide)[-1], collapse = ", "), "\n\n")

  # Separar train y test
  df_train <- df_wide %>% filter(date >= train_start & date <= train_end)
  df_test <- df_wide %>% filter(date >= test_start & date <= test_end)

  # Inicializar
  result_train <- data.frame(date = df_train$date)
  result_test <- data.frame(date = df_test$date)
  train_stats <- list()


  # EQUITY INDICES


  equity_indices <- intersect(indices, features_config$equity_indices)

  for (idx in equity_indices) {
    if (!idx %in% names(df_train)) {
      warning(sprintf("'%s' no encontrado", idx))
      next
    }

    log_cat(sprintf("Procesando: %s\n", idx))
    clean <- tolower(gsub("[^[:alnum:]]", "", idx))

    # === TRAIN ===
    prices_train <- df_train[[idx]]
    returns_train <- c(NA, diff(log(prices_train))) * 100

    result_train[[paste0(clean, "_return")]] <- returns_train
    result_train[[paste0(clean, "_return_lag1")]] <- lag(returns_train, 1)

    # Momentum 20
    result_train[[paste0(clean, "_momentum")]] <-
      (prices_train - lag(prices_train, 20)) / lag(prices_train, 20) * 100

    # Volatilidad rolling 20
    result_train[[paste0(clean, "_vol20")]] <-
      rollapply(returns_train, width = 20, FUN = sd, fill = NA, align = "right")

    # Guardar estadísticos
    train_stats[[paste0(clean, "_return_mean")]] <- mean(returns_train, na.rm = TRUE)
    train_stats[[paste0(clean, "_return_sd")]] <- sd(returns_train, na.rm = TRUE)

    # === TEST ===
    prices_test <- df_test[[idx]]
    returns_test <- c(NA, diff(log(prices_test))) * 100

    result_test[[paste0(clean, "_return")]] <- returns_test
    result_test[[paste0(clean, "_return_lag1")]] <- lag(returns_test, 1)

    # Para momentum y vol rolling: concatenar últimas 20 obs de train con test
    n_window <- 20
    prices_concat <- c(tail(prices_train, n_window), prices_test)

    # Momentum
    momentum_concat <- (prices_concat - lag(prices_concat, 20)) /
      lag(prices_concat, 20) * 100
    result_test[[paste0(clean, "_momentum")]] <-
      tail(momentum_concat, nrow(df_test))

    # Volatilidad: concatenar returns
    returns_concat <- c(tail(returns_train, n_window), returns_test)
    vol_concat <- rollapply(returns_concat,
      width = 20, FUN = sd,
      fill = NA, align = "right"
    )
    result_test[[paste0(clean, "_vol20")]] <-
      tail(vol_concat, nrow(df_test))
  }


  # SPREADS


  if (target_index %in% names(df_train)) {
    log_cat("\n--- Spreads ---\n")

    # Train
    target_ret_train <- c(NA, diff(log(df_train[[target_index]]))) * 100

    if ("s_p500" %in% equity_indices && "sp500_return_lag1" %in% names(result_train)) {
      result_train$spread_target_sp500 <-
        target_ret_train - result_train$sp500_return_lag1
    }

    if ("dax" %in% equity_indices && "dax_return" %in% names(result_train)) {
      result_train$spread_target_dax <-
        target_ret_train - result_train$dax_return
    }

    if (all(c("dax_return", "sp500_return_lag1") %in% names(result_train))) {
      result_train$spread_eu_us <-
        result_train$dax_return - result_train$sp500_return_lag1
    }

    # Test
    target_ret_test <- c(NA, diff(log(df_test[[target_index]]))) * 100

    if ("s_p500" %in% equity_indices && "sp500_return_lag1" %in% names(result_test)) {
      result_test$spread_target_sp500 <-
        target_ret_test - result_test$sp500_return_lag1
    }

    if ("dax" %in% equity_indices && "dax_return" %in% names(result_test)) {
      result_test$spread_target_dax <-
        target_ret_test - result_test$dax_return
    }

    if (all(c("dax_return", "sp500_return_lag1") %in% names(result_test))) {
      result_test$spread_eu_us <-
        result_test$dax_return - result_test$sp500_return_lag1
    }
  }


  # VIX


  if ("volatility_index" %in% indices && "volatility_index" %in% names(df_train)) {
    log_cat("--- VIX ---\n")

    # === TRAIN ===
    vix_train <- df_train$volatility_index

    result_train$vix_close <- vix_train
    result_train$vix_change <- vix_train - lag(vix_train)
    result_train$vix_return <- c(NA, diff(log(vix_train))) * 100
    result_train$vix_lag1 <- lag(vix_train, 1)

    # Z-score rolling 252
    vix_ma <- rollmean(vix_train, k = 252, fill = NA, align = "right")
    vix_sd <- rollapply(vix_train, width = 252, FUN = sd, fill = NA, align = "right")
    result_train$vix_zscore <- (vix_train - vix_ma) / vix_sd

    # Guardar estadísticos de train (últimos 252)
    train_stats$vix_mean_252 <- mean(tail(vix_train, min(252, length(vix_train))), na.rm = TRUE)
    train_stats$vix_sd_252 <- sd(tail(vix_train, min(252, length(vix_train))), na.rm = TRUE)

    result_train$vix_regime <- cut(vix_train,
      breaks = c(0, 15, 20, 30, 100),
      labels = c("Low", "Normal", "Elevated", "High"),
      include.lowest = TRUE
    )

    # === TEST ===
    vix_test <- df_test$volatility_index

    result_test$vix_close <- vix_test
    result_test$vix_change <- vix_test - lag(vix_test)
    result_test$vix_return <- c(NA, diff(log(vix_test))) * 100
    result_test$vix_lag1 <- lag(vix_test, 1)

    # Z-score: concatenar con train para rolling, o usar stats de train
    n_window <- 252
    vix_concat <- c(tail(vix_train, n_window), vix_test)

    vix_ma_concat <- rollmean(vix_concat, k = 252, fill = NA, align = "right")
    vix_sd_concat <- rollapply(vix_concat,
      width = 252, FUN = sd,
      fill = NA, align = "right"
    )
    vix_zscore_concat <- (vix_concat - vix_ma_concat) / vix_sd_concat

    vix_zscore_test <- tail(vix_zscore_concat, nrow(df_test))

    # Rellenar NAs con estadísticos de train
    na_idx <- is.na(vix_zscore_test)
    if (any(na_idx)) {
      vix_zscore_test[na_idx] <- (vix_test[na_idx] - train_stats$vix_mean_252) /
        train_stats$vix_sd_252
    }

    result_test$vix_zscore <- vix_zscore_test

    result_test$vix_regime <- cut(vix_test,
      breaks = c(0, 15, 20, 30, 100),
      labels = c("Low", "Normal", "Elevated", "High"),
      include.lowest = TRUE
    )
  }


  # COMMODITIES


  commodities <- intersect(indices, features_config$commodities)

  for (comm in commodities) {
    if (!comm %in% names(df_train)) next

    log_cat(sprintf("Procesando: %s\n", comm))
    clean <- tolower(gsub("[^[:alnum:]]", "", comm))

    # === TRAIN ===
    prices_train <- df_train[[comm]]
    returns_train <- c(NA, diff(log(prices_train))) * 100

    result_train[[paste0(clean, "_return")]] <- returns_train
    result_train[[paste0(clean, "_return_lag1")]] <- lag(returns_train, 1)
    result_train[[paste0(clean, "_momentum")]] <-
      (prices_train - lag(prices_train, 20)) / lag(prices_train, 20) * 100
    result_train[[paste0(clean, "_vol20")]] <-
      rollapply(returns_train, width = 20, FUN = sd, fill = NA, align = "right")

    train_stats[[paste0(clean, "_return_mean")]] <- mean(returns_train, na.rm = TRUE)
    train_stats[[paste0(clean, "_return_sd")]] <- sd(returns_train, na.rm = TRUE)

    # === TEST ===
    prices_test <- df_test[[comm]]
    returns_test <- c(NA, diff(log(prices_test))) * 100

    result_test[[paste0(clean, "_return")]] <- returns_test
    result_test[[paste0(clean, "_return_lag1")]] <- lag(returns_test, 1)

    # Momentum y vol: concatenar
    prices_concat <- c(tail(prices_train, 20), prices_test)
    momentum_concat <- (prices_concat - lag(prices_concat, 20)) /
      lag(prices_concat, 20) * 100
    result_test[[paste0(clean, "_momentum")]] <- tail(momentum_concat, nrow(df_test))

    returns_concat <- c(tail(returns_train, 20), returns_test)
    vol_concat <- rollapply(returns_concat,
      width = 20, FUN = sd,
      fill = NA, align = "right"
    )
    result_test[[paste0(clean, "_vol20")]] <- tail(vol_concat, nrow(df_test))
  }

  # Ratio Oil/Gold
  if (all(c("oil", "gold") %in% commodities) &&
    all(c("oil", "gold") %in% names(df_train))) {
    result_train$oil_gold_ratio <- df_train$oil / df_train$gold
    result_train$oil_gold_ratio_change <-
      result_train$oil_gold_ratio - lag(result_train$oil_gold_ratio)

    result_test$oil_gold_ratio <- df_test$oil / df_test$gold
    result_test$oil_gold_ratio_change <-
      result_test$oil_gold_ratio - lag(result_test$oil_gold_ratio)
  }


  # FOREX


  forex <- intersect(indices, features_config$forex)

  for (fx in forex) {
    if (!fx %in% names(df_train)) next

    log_cat(sprintf("Procesando: %s\n", fx))
    clean <- tolower(gsub("[^[:alnum:]]", "", fx))

    # === TRAIN ===
    prices_train <- df_train[[fx]]
    returns_train <- c(NA, diff(log(prices_train))) * 100

    result_train[[paste0(clean, "_return")]] <- returns_train
    result_train[[paste0(clean, "_level")]] <- prices_train

    ma20_train <- SMA(prices_train, n = 20)
    result_train[[paste0(clean, "_ma20")]] <- ma20_train
    result_train[[paste0(clean, "_deviation")]] <-
      (prices_train - ma20_train) / ma20_train * 100
    result_train[[paste0(clean, "_momentum")]] <-
      (prices_train - lag(prices_train, 20)) / lag(prices_train, 20) * 100
    result_train[[paste0(clean, "_return_lag1")]] <- lag(returns_train, 1)

    train_stats[[paste0(clean, "_return_mean")]] <- mean(returns_train, na.rm = TRUE)
    train_stats[[paste0(clean, "_return_sd")]] <- sd(returns_train, na.rm = TRUE)

    # === TEST ===
    prices_test <- df_test[[fx]]
    returns_test <- c(NA, diff(log(prices_test))) * 100

    result_test[[paste0(clean, "_return")]] <- returns_test
    result_test[[paste0(clean, "_level")]] <- prices_test
    result_test[[paste0(clean, "_return_lag1")]] <- lag(returns_test, 1)

    # MA20: concatenar
    prices_concat <- c(tail(prices_train, 20), prices_test)
    ma20_concat <- SMA(prices_concat, n = 20)
    ma20_test <- tail(ma20_concat, nrow(df_test))

    result_test[[paste0(clean, "_ma20")]] <- ma20_test
    result_test[[paste0(clean, "_deviation")]] <-
      (prices_test - ma20_test) / ma20_test * 100

    # Momentum
    momentum_concat <- (prices_concat - lag(prices_concat, 20)) /
      lag(prices_concat, 20) * 100
    result_test[[paste0(clean, "_momentum")]] <- tail(momentum_concat, nrow(df_test))
  }


  # CORRELACIONES


  log_cat("\n--- Correlaciones ---\n")

  if (target_index %in% names(df_train) && "s_p500" %in% names(df_train)) {
    # Train
    target_ret_train <- c(NA, diff(log(df_train[[target_index]]))) * 100
    sp500_ret_train <- c(NA, diff(log(df_train$s_p500))) * 100
    sp500_lag_train <- lag(sp500_ret_train, 1)

    corr_data_train <- data.frame(
      target = target_ret_train,
      sp500 = sp500_lag_train
    )

    result_train$corr_target_sp500_60 <- rollapply(
      corr_data_train,
      width = 60,
      FUN = function(x) {
        if (sum(!is.na(x[, 1])) < 30 || sum(!is.na(x[, 2])) < 30) {
          return(NA)
        }
        cor(x[, 1], x[, 2], use = "complete.obs")
      },
      fill = NA,
      align = "right",
      by.column = FALSE
    )

    # Test: concatenar últimas 60 obs
    target_ret_test <- c(NA, diff(log(df_test[[target_index]]))) * 100
    sp500_ret_test <- c(NA, diff(log(df_test$s_p500))) * 100
    sp500_lag_test <- lag(sp500_ret_test, 1)

    target_concat <- c(tail(target_ret_train, 60), target_ret_test)
    sp500_concat <- c(tail(sp500_lag_train, 60), sp500_lag_test)

    corr_data_concat <- data.frame(
      target = target_concat,
      sp500 = sp500_concat
    )

    corr_concat <- rollapply(
      corr_data_concat,
      width = 60,
      FUN = function(x) {
        if (sum(!is.na(x[, 1])) < 30 || sum(!is.na(x[, 2])) < 30) {
          return(NA)
        }
        cor(x[, 1], x[, 2], use = "complete.obs")
      },
      fill = NA,
      align = "right",
      by.column = FALSE
    )

    result_test$corr_target_sp500_60 <- tail(corr_concat, nrow(df_test))
  }

  if (target_index %in% names(df_train) && "dax" %in% names(df_train)) {
    # Train
    target_ret_train <- c(NA, diff(log(df_train[[target_index]]))) * 100
    dax_ret_train <- c(NA, diff(log(df_train$dax))) * 100

    corr_data_train <- data.frame(target = target_ret_train, dax = dax_ret_train)

    result_train$corr_target_dax_60 <- rollapply(
      corr_data_train,
      width = 60,
      FUN = function(x) {
        if (sum(!is.na(x[, 1])) < 30 || sum(!is.na(x[, 2])) < 30) {
          return(NA)
        }
        cor(x[, 1], x[, 2], use = "complete.obs")
      },
      fill = NA,
      align = "right",
      by.column = FALSE
    )

    # Test
    target_ret_test <- c(NA, diff(log(df_test[[target_index]]))) * 100
    dax_ret_test <- c(NA, diff(log(df_test$dax))) * 100

    target_concat <- c(tail(target_ret_train, 60), target_ret_test)
    dax_concat <- c(tail(dax_ret_train, 60), dax_ret_test)

    corr_data_concat <- data.frame(target = target_concat, dax = dax_concat)

    corr_concat <- rollapply(
      corr_data_concat,
      width = 60,
      FUN = function(x) {
        if (sum(!is.na(x[, 1])) < 30 || sum(!is.na(x[, 2])) < 30) {
          return(NA)
        }
        cor(x[, 1], x[, 2], use = "complete.obs")
      },
      fill = NA,
      align = "right",
      by.column = FALSE
    )

    result_test$corr_target_dax_60 <- tail(corr_concat, nrow(df_test))
  }


  # RISK SCORE


  log_cat("--- Risk Score ---\n")

  risk_train <- list()
  risk_test <- list()
  scaling <- list()

  if ("sp500_return" %in% names(result_train)) {
    ret_train <- result_train$sp500_return
    ret_test <- result_test$sp500_return

    scaling$sp500_mean <- mean(ret_train, na.rm = TRUE)
    scaling$sp500_sd <- sd(ret_train, na.rm = TRUE)

    risk_train$equity <- (ret_train - scaling$sp500_mean) / scaling$sp500_sd
    risk_test$equity <- (ret_test - scaling$sp500_mean) / scaling$sp500_sd
  }

  if ("vix_change" %in% names(result_train)) {
    chg_train <- result_train$vix_change
    chg_test <- result_test$vix_change

    scaling$vix_mean <- mean(chg_train, na.rm = TRUE)
    scaling$vix_sd <- sd(chg_train, na.rm = TRUE)

    risk_train$vix <- -(chg_train - scaling$vix_mean) / scaling$vix_sd
    risk_test$vix <- -(chg_test - scaling$vix_mean) / scaling$vix_sd
  }

  if ("oil_return" %in% names(result_train)) {
    ret_train <- result_train$oil_return
    ret_test <- result_test$oil_return

    scaling$oil_mean <- mean(ret_train, na.rm = TRUE)
    scaling$oil_sd <- sd(ret_train, na.rm = TRUE)

    risk_train$oil <- (ret_train - scaling$oil_mean) / scaling$oil_sd
    risk_test$oil <- (ret_test - scaling$oil_mean) / scaling$oil_sd
  }

  if ("gold_return" %in% names(result_train)) {
    ret_train <- result_train$gold_return
    ret_test <- result_test$gold_return

    scaling$gold_mean <- mean(ret_train, na.rm = TRUE)
    scaling$gold_sd <- sd(ret_train, na.rm = TRUE)

    risk_train$gold <- -(ret_train - scaling$gold_mean) / scaling$gold_sd
    risk_test$gold <- -(ret_test - scaling$gold_mean) / scaling$gold_sd
  }

  if (length(risk_train) >= 2) {
    result_train$risk_on_score <- Reduce("+", risk_train) / length(risk_train)
    result_test$risk_on_score <- Reduce("+", risk_test) / length(risk_test)
    train_stats$risk_scaling <- scaling
  }


  # VOL SPREAD


  if (target_index %in% names(df_train) && "sp500_vol20" %in% names(result_train)) {
    # Train
    target_ret_train <- c(NA, diff(log(df_train[[target_index]]))) * 100
    target_vol_train <- rollapply(target_ret_train,
      width = 20, FUN = sd,
      fill = NA, align = "right"
    )

    result_train$vol_spread_target_sp500 <-
      target_vol_train - result_train$sp500_vol20

    # Test
    target_ret_test <- c(NA, diff(log(df_test[[target_index]]))) * 100
    target_ret_concat <- c(tail(target_ret_train, 20), target_ret_test)
    target_vol_concat <- rollapply(target_ret_concat,
      width = 20, FUN = sd,
      fill = NA, align = "right"
    )
    target_vol_test <- tail(target_vol_concat, nrow(df_test))

    result_test$vol_spread_target_sp500 <-
      target_vol_test - result_test$sp500_vol20
  }


  # RESUMEN


  log_cat("\n== RESUMEN ==\n")
  log_cat(sprintf("Features: %d\n", ncol(result_train) - 1))
  log_cat(sprintf("Train obs: %d\n", nrow(result_train)))
  log_cat(sprintf("Test obs: %d\n", nrow(result_test)))
  log_cat(sprintf("Stats guardados: %d\n", length(train_stats)))
  log_cat("\n\n")

  return(list(
    train = result_train,
    test = result_test,
    stats = train_stats
  ))
}
