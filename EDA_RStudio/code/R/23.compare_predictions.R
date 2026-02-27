# Funciones para comparación estadística de modelos
# Clasificación: McNemar + Bootstrap
# Regresión: Diebold-Mariano + Bootstrap
# Usa funciones estándar: stats::mcnemar.test, forecast::dm.test, boot::boot
# Columnas compatibles con output de Python
# @author: santi


# DEPENDENCIAS ####
suppressPackageStartupMessages({
  library(tidyverse)
  library(boot)
  library(forecast) # dm.test
})


# UTILIDADES COMUNES ####


#' Parsea string de array numpy a vector numérico
#' @param x String con formato numpy array
#' @return Vector numérico
parse_numpy_array <- function(x) {
  x <- gsub("\\[|\\]", "", x)
  x <- gsub("\\s+", " ", x)
  x <- trimws(x)
  values <- strsplit(x, " ")[[1]]
  values <- values[values != ""]
  as.numeric(values)
}

#' Extrae tipo de scaling del nombre del dataset
#' @param dataset_name Nombre del dataset (ej: "financial_scaled")
#' @return String "scaled" o "unscaled"
extract_scaling <- function(dataset_name) {
  if (grepl("unscaled", dataset_name)) {
    "unscaled"
  } else if (grepl("scaled", dataset_name)) {
    "scaled"
  } else {
    "unknown"
  }
}


# CLASIFICACIÓN: McNEMAR + BOOTSTRAP ####


#' Test de McNemar para comparar dos clasificadores
#' Usa stats::mcnemar.test internamente
#' @param y_true Vector de valores reales (0/1)
#' @param pred1 Predicciones del modelo 1
#' @param pred2 Predicciones del modelo 2
#' @return Lista con resultados del test
mcnemar_test <- function(y_true, pred1, pred2) {
  y_true <- as.numeric(y_true)
  pred1 <- as.numeric(pred1)
  pred2 <- as.numeric(pred2)

  # Contar NAs antes de eliminar
  n_total <- length(y_true)
  valid_idx <- !is.na(y_true) & !is.na(pred1) & !is.na(pred2)
  n_nas_removed <- n_total - sum(valid_idx)

  y_true <- y_true[valid_idx]
  pred1 <- pred1[valid_idx]
  pred2 <- pred2[valid_idx]

  n_samples <- length(y_true)

  if (n_samples == 0) {
    return(list(p_value = NA, significant = NA, n_samples = 0, n_nas_removed = n_nas_removed))
  }

  # Calcular correctas/incorrectas
  correct1 <- (pred1 == y_true)
  correct2 <- (pred2 == y_true)

  # Construir tabla de contingencia para mcnemar.test
  # Filas: modelo 1 (correct/incorrect), Columnas: modelo 2 (correct/incorrect)
  contingency_table <- matrix(
    c(
      sum(correct1 & correct2), # ambos correctos
      sum(correct1 & !correct2), # solo modelo 1 correcto
      sum(!correct1 & correct2), # solo modelo 2 correcto
      sum(!correct1 & !correct2)
    ), # ambos incorrectos
    nrow = 2,
    byrow = TRUE
  )

  # McNemar test con corrección de continuidad
  test_result <- tryCatch(
    mcnemar.test(contingency_table, correct = TRUE),
    error = function(e) NULL
  )

  if (is.null(test_result)) {
    p_value <- NA
  } else {
    p_value <- test_result$p.value
  }

  # Accuracies
  acc1 <- mean(correct1)
  acc2 <- mean(correct2)

  list(
    n_samples = n_samples,
    n_nas_removed = n_nas_removed,
    acc1 = acc1,
    acc2 = acc2,
    diff = acc2 - acc1,
    p_value = p_value,
    significant = ifelse(is.na(p_value), NA, p_value < 0.05)
  )
}


#' Bootstrap CI para diferencia de accuracy usando boot::boot y boot::boot.ci
#' @param y_true Vector de valores reales
#' @param pred1 Predicciones modelo 1
#' @param pred2 Predicciones modelo 2
#' @param n_bootstrap Número de muestras bootstrap
#' @param confidence Nivel de confianza
#' @param seed Semilla aleatoria
#' @return Lista con CI y significancia
bootstrap_ci_classification <- function(y_true, pred1, pred2,
                                        n_bootstrap = 1000,
                                        confidence = 0.95,
                                        seed = 42) {
  set.seed(seed)

  valid_idx <- !is.na(y_true) & !is.na(pred1) & !is.na(pred2)
  y_true <- as.numeric(y_true[valid_idx])
  pred1 <- as.numeric(pred1[valid_idx])
  pred2 <- as.numeric(pred2[valid_idx])

  n <- length(y_true)

  if (n == 0) {
    return(list(ci_lower = NA, ci_upper = NA, significant = NA))
  }

  # Función para boot: diferencia de accuracy (modelo2 - modelo1)
  diff_stat <- function(data, indices) {
    d <- data[indices, ]
    mean(d$p2 == d$y) - mean(d$p1 == d$y)
  }

  data_df <- data.frame(y = y_true, p1 = pred1, p2 = pred2)

  # Bootstrap con boot::boot
  boot_result <- boot(data_df, diff_stat, R = n_bootstrap)

  # Calcular CI con boot.ci (usa método percentil por defecto, más robusto)
  ci_result <- tryCatch(
    boot.ci(boot_result, conf = confidence, type = "perc"),
    error = function(e) NULL
  )

  if (is.null(ci_result)) {
    # Fallback a percentiles manuales
    alpha <- 1 - confidence
    ci <- quantile(boot_result$t, probs = c(alpha / 2, 1 - alpha / 2), na.rm = TRUE)
    ci_lower <- unname(ci[1])
    ci_upper <- unname(ci[2])
  } else {
    ci_lower <- ci_result$percent[4]
    ci_upper <- ci_result$percent[5]
  }

  significant <- !(ci_lower <= 0 && ci_upper >= 0)

  list(
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    significant = significant
  )
}


#' Comparación de clasificadores con output compatible con Python
#' @param predictions_df DataFrame con predicciones
#' @param targets Vector de targets a analizar
#' @param comparisons Lista de comparaciones (dataset1, dataset2, comparison_type)
#' @param n_bootstrap Número de iteraciones bootstrap
#' @return DataFrame con columnas idénticas a Python
compare_classifiers <- function(predictions_df,
                                targets = c("direction_next", "direction_next_5", "direction_next_10", "direction_next_20"),
                                comparisons = NULL,
                                n_bootstrap = 1000) {
  if (is.null(comparisons)) {
    comparisons <- list(
      c("financial_scaled", "sentiment_scaled", "Financial_vs_Sentiment"),
      c("financial_unscaled", "sentiment_unscaled", "Financial_vs_Sentiment"),
      c("financial_scaled", "financial_long_scaled", "Financial_vs_Financial_long"),
      c("financial_unscaled", "financial_long_unscaled", "Financial_vs_Financial_long")
    )
  }

  results <- list()

  for (target in targets) {
    cat(sprintf(
      "\n%s\nTARGET: %s\n%s\n",
      strrep("=", 70), target, strrep("=", 70)
    ))

    for (comp in comparisons) {
      dataset1 <- comp[1]
      dataset2 <- comp[2]
      comparison_type <- comp[3]
      scaling <- extract_scaling(dataset1)
      scaled <- grepl("scaled", dataset1) & !grepl("unscaled", dataset1)

      cat(sprintf("\n%s vs %s (%s)\n", dataset1, dataset2, comparison_type))
      cat(strrep("-", 60), "\n")

      df1 <- predictions_df |> filter(dataset == dataset1, target == !!target)
      df2 <- predictions_df |> filter(dataset == dataset2, target == !!target)

      if (nrow(df1) == 0 || nrow(df2) == 0) {
        cat("  ⚠ Dataset no encontrado\n")
        next
      }

      common_models <- intersect(unique(df1$model), unique(df2$model))

      if (length(common_models) == 0) {
        cat("  ⚠ No hay modelos comunes\n")
        next
      }

      cat(sprintf("  ✓ Modelos comunes: %d\n", length(common_models)))

      for (model in common_models) {
        data1 <- df1 |> filter(model == !!model)
        data2 <- df2 |> filter(model == !!model)

        if (nrow(data1) == 0 || nrow(data2) == 0) next

        y1 <- data1$y_test[[1]]
        pred1 <- data1$predictions[[1]]
        y2 <- data2$y_test[[1]]
        pred2 <- data2$predictions[[1]]

        # Alinear tamaños
        min_len <- min(length(y1), length(y2), length(pred1), length(pred2))
        y1 <- tail(y1, min_len)
        pred1 <- tail(pred1, min_len)
        pred2 <- tail(pred2, min_len)

        # McNemar test
        mcn <- mcnemar_test(y1, pred1, pred2)

        if (is.na(mcn$n_samples) || mcn$n_samples == 0) next

        # Bootstrap CI
        boot_ci <- bootstrap_ci_classification(y1, pred1, pred2, n_bootstrap = n_bootstrap)

        # Resultado con columnas idénticas a Python
        results[[length(results) + 1]] <- tibble(
          comparison_type = comparison_type,
          dataset1 = dataset1,
          dataset2 = dataset2,
          scaling = scaling,
          target = target,
          model = model,
          n_samples = mcn$n_samples,
          n_nas_removed = mcn$n_nas_removed,
          acc_dataset1 = mcn$acc1,
          acc_dataset2 = mcn$acc2,
          diff = mcn$diff,
          mcnemar_pvalue = mcn$p_value,
          mcnemar_significant = mcn$significant,
          bootstrap_ci_lower = boot_ci$ci_lower,
          bootstrap_ci_upper = boot_ci$ci_upper,
          bootstrap_significant = boot_ci$significant
        )

        sig_symbol <- ifelse(mcn$significant, "✓", "✗")
        cat(sprintf(
          "    %s: Δacc=%.4f, McNemar p=%.4f %s\n",
          model, mcn$diff, mcn$p_value, sig_symbol
        ))
      }
    }
  }

  if (length(results) > 0) bind_rows(results) else tibble()
}


# REGRESIÓN: DIEBOLD-MARIANO + BOOTSTRAP ####


#' Test de Diebold-Mariano usando forecast::dm.test
#' @param errors1 Errores del modelo 1
#' @param errors2 Errores del modelo 2
#' @param horizon Horizonte de predicción
#' @return Lista con estadístico y p-value
diebold_mariano_test <- function(errors1, errors2, horizon = 1) {
  valid_idx <- !is.na(errors1) & !is.na(errors2)
  e1 <- errors1[valid_idx]
  e2 <- errors2[valid_idx]

  n <- length(e1)

  if (n < 10) {
    return(list(statistic = NA, p_value = NA, significant = NA))
  }

  # Usar forecast::dm.test
  # h = horizonte, power = 2 para MSE (squared errors)
  test_result <- tryCatch(
    dm.test(e1, e2, alternative = "two.sided", h = horizon, power = 2),
    error = function(e) NULL
  )

  if (is.null(test_result)) {
    return(list(statistic = NA, p_value = NA, significant = NA))
  }

  list(
    statistic = unname(test_result$statistic),
    p_value = test_result$p.value,
    significant = test_result$p.value < 0.05
  )
}


#' Bootstrap CI para diferencia de RMSE usando boot::boot y boot::boot.ci
#' @param y_true Vector de valores reales
#' @param pred1 Predicciones modelo 1
#' @param pred2 Predicciones modelo 2
#' @param n_bootstrap Número de muestras bootstrap
#' @param confidence Nivel de confianza
#' @param seed Semilla aleatoria
#' @return Lista con mean_diff, CI y p-value aproximado
bootstrap_ci_regression <- function(y_true, pred1, pred2,
                                    n_bootstrap = 1000,
                                    confidence = 0.95,
                                    seed = 42) {
  set.seed(seed)

  valid_idx <- !is.na(y_true) & !is.na(pred1) & !is.na(pred2)
  y_true <- y_true[valid_idx]
  pred1 <- pred1[valid_idx]
  pred2 <- pred2[valid_idx]

  n <- length(y_true)

  if (n == 0) {
    return(list(mean_diff = NA, ci_lower = NA, ci_upper = NA, p_value = NA, significant = NA))
  }

  # Función para boot: diferencia de RMSE (modelo1 - modelo2)
  diff_stat <- function(data, indices) {
    d <- data[indices, ]
    sqrt(mean((d$y - d$p1)^2)) - sqrt(mean((d$y - d$p2)^2))
  }

  data_df <- data.frame(y = y_true, p1 = pred1, p2 = pred2)

  # Bootstrap con boot::boot
  boot_result <- boot(data_df, diff_stat, R = n_bootstrap)

  mean_diff <- mean(boot_result$t, na.rm = TRUE)

  # Calcular CI con boot.ci
  ci_result <- tryCatch(
    boot.ci(boot_result, conf = confidence, type = "perc"),
    error = function(e) NULL
  )

  if (is.null(ci_result)) {
    alpha <- 1 - confidence
    ci <- quantile(boot_result$t, probs = c(alpha / 2, 1 - alpha / 2), na.rm = TRUE)
    ci_lower <- unname(ci[1])
    ci_upper <- unname(ci[2])
  } else {
    ci_lower <- ci_result$percent[4]
    ci_upper <- ci_result$percent[5]
  }

  # p-value aproximado (proporción de bootstrap samples que cruzan 0)
  if (mean_diff > 0) {
    p_value <- 2 * mean(boot_result$t <= 0, na.rm = TRUE)
  } else {
    p_value <- 2 * mean(boot_result$t >= 0, na.rm = TRUE)
  }
  p_value <- min(p_value, 1.0)

  significant <- !(ci_lower <= 0 && ci_upper >= 0)

  list(
    mean_diff = mean_diff,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    p_value = p_value,
    significant = significant
  )
}


#' Comparación de regresores con output compatible con Python
#' @param predictions_df DataFrame con predicciones
#' @param targets Vector de targets a analizar
#' @param comparisons Lista de comparaciones (dataset1, dataset2, comparison_type)
#' @param n_bootstrap Número de iteraciones bootstrap
#' @return DataFrame con columnas idénticas a Python
compare_regressors <- function(predictions_df,
                               targets = c("returns_next", "returns_next_5", "returns_next_10", "returns_next_20"),
                               comparisons = NULL,
                               n_bootstrap = 1000) {
  if (is.null(comparisons)) {
    comparisons <- list(
      c("financial_scaled", "sentiment_scaled", "Financial_vs_Sentiment"),
      c("financial_unscaled", "sentiment_unscaled", "Financial_vs_Sentiment"),
      c("financial_scaled", "financial_long_scaled", "Financial_vs_Financial_long"),
      c("financial_unscaled", "financial_long_unscaled", "Financial_vs_Financial_long")
    )
  }

  results <- list()

  for (target in targets) {
    cat(sprintf(
      "\n%s\nTARGET: %s\n%s\n",
      strrep("=", 70), target, strrep("=", 70)
    ))

    # Extraer horizonte del target
    horizon <- 1
    if (grepl("_\\d+$", target)) {
      horizon <- as.integer(sub(".*_(\\d+)$", "\\1", target))
    }

    for (comp in comparisons) {
      dataset1 <- comp[1]
      dataset2 <- comp[2]
      comparison_type <- comp[3]
      scaling <- extract_scaling(dataset1)
      scaled <- grepl("scaled", dataset1) & !grepl("unscaled", dataset1)

      cat(sprintf("\n%s vs %s (%s)\n", dataset1, dataset2, comparison_type))
      cat(strrep("-", 60), "\n")

      df1 <- predictions_df |> filter(dataset == dataset1, target == !!target)
      df2 <- predictions_df |> filter(dataset == dataset2, target == !!target)

      if (nrow(df1) == 0 || nrow(df2) == 0) {
        cat("  ⚠ Dataset no encontrado\n")
        next
      }

      common_models <- intersect(unique(df1$model), unique(df2$model))

      if (length(common_models) == 0) {
        cat("  ⚠ No hay modelos comunes\n")
        next
      }

      cat(sprintf("  ✓ Modelos comunes: %d\n", length(common_models)))

      for (model in common_models) {
        data1 <- df1 |> filter(model == !!model)
        data2 <- df2 |> filter(model == !!model)

        if (nrow(data1) == 0 || nrow(data2) == 0) next

        y1 <- data1$y_test[[1]]
        pred1 <- data1$predictions[[1]]
        y2 <- data2$y_test[[1]]
        pred2 <- data2$predictions[[1]]

        # Alinear tamaños
        min_len <- min(length(y1), length(y2), length(pred1), length(pred2))
        y1 <- tail(y1, min_len)
        pred1 <- tail(pred1, min_len)
        pred2 <- tail(pred2, min_len)

        # Eliminar NAs
        valid_idx <- !is.na(y1) & !is.na(pred1) & !is.na(pred2)
        y_clean <- y1[valid_idx]
        pred1_clean <- pred1[valid_idx]
        pred2_clean <- pred2[valid_idx]

        if (length(y_clean) < 10) {
          cat(sprintf("    %s: ⚠ Muy pocas observaciones\n", model))
          next
        }

        # Errores y métricas
        errors1 <- y_clean - pred1_clean
        errors2 <- y_clean - pred2_clean

        rmse1 <- sqrt(mean(errors1^2))
        rmse2 <- sqrt(mean(errors2^2))

        # Diebold-Mariano
        dm <- diebold_mariano_test(errors1, errors2, horizon = horizon)

        # Bootstrap
        boot_res <- bootstrap_ci_regression(y_clean, pred1_clean, pred2_clean,
          n_bootstrap = n_bootstrap
        )

        # Resultado con columnas idénticas a Python
        results[[length(results) + 1]] <- tibble(
          model = model,
          n_samples = length(y_clean),
          rmse_financial = rmse1,
          rmse_sentiment = rmse2,
          rmse_diff = rmse1 - rmse2,
          dm_statistic = dm$statistic,
          dm_pvalue = dm$p_value,
          dm_significant = dm$significant,
          boot_mean_diff = boot_res$mean_diff,
          boot_ci_lower = boot_res$ci_lower,
          boot_ci_upper = boot_res$ci_upper,
          boot_pvalue = boot_res$p_value,
          boot_significant = boot_res$significant,
          comparison_type = comparison_type,
          dataset1 = dataset1,
          dataset2 = dataset2,
          scaling = scaling,
          target = target
        )

        sig_symbol <- ifelse(dm$significant, "✓", "✗")
        cat(sprintf(
          "    %s: ΔRMSE=%.4f, DM p=%.4f %s\n",
          model, rmse1 - rmse2, dm$p_value, sig_symbol
        ))
      }
    }
  }

  if (length(results) > 0) bind_rows(results) else tibble()
}


# FUNCIÓN DE RESUMEN ESTADÍSTICA ####


#' Resumen de resultados
#' @param results_df DataFrame con resultados
#' @param test_type "classification" o "regression"
summarize_comparison_results <- function(results_df, test_type = "classification") {
  if (nrow(results_df) == 0) {
    cat("No hay resultados para resumir\n")
    return(invisible(NULL))
  }

  cat(sprintf("\n%s\n", strrep("=", 70)))
  cat("RESUMEN DE COMPARACIONES ESTADÍSTICAS\n")
  cat(sprintf("%s\n", strrep("=", 70)))

  for (comp in unique(results_df$comparison_type)) {
    comp_data <- results_df |> filter(comparison_type == comp)

    cat(sprintf(
      "\n%s\nCOMPARACIÓN: %s\n%s\n",
      strrep("-", 50), comp, strrep("-", 50)
    ))

    n_total <- nrow(comp_data)

    if (test_type == "classification") {
      n_sig <- sum(comp_data$mcnemar_significant, na.rm = TRUE)
      n_sig_boot <- sum(comp_data$bootstrap_significant, na.rm = TRUE)

      cat(sprintf("\n  Total: %d comparaciones\n", n_total))
      cat(sprintf(
        "  Significativas (McNemar): %d (%.1f%%)\n",
        n_sig, 100 * n_sig / n_total
      ))
      cat(sprintf(
        "  Significativas (Bootstrap): %d (%.1f%%)\n",
        n_sig_boot, 100 * n_sig_boot / n_total
      ))

      d1_better <- sum(comp_data$diff < 0, na.rm = TRUE)
      d2_better <- sum(comp_data$diff > 0, na.rm = TRUE)
      cat(sprintf("  Dataset1 mejor: %d | Dataset2 mejor: %d\n", d1_better, d2_better))
    } else {
      n_sig <- sum(comp_data$dm_significant, na.rm = TRUE)
      n_sig_boot <- sum(comp_data$boot_significant, na.rm = TRUE)

      cat(sprintf("\n  Total: %d comparaciones\n", n_total))
      cat(sprintf(
        "  Significativas (DM): %d (%.1f%%)\n",
        n_sig, 100 * n_sig / n_total
      ))
      cat(sprintf(
        "  Significativas (Bootstrap): %d (%.1f%%)\n",
        n_sig_boot, 100 * n_sig_boot / n_total
      ))

      d1_better <- sum(comp_data$rmse_diff > 0, na.rm = TRUE)
      d2_better <- sum(comp_data$rmse_diff < 0, na.rm = TRUE)
      cat(sprintf("  Dataset1 mejor RMSE: %d | Dataset2 mejor RMSE: %d\n", d1_better, d2_better))
    }
  }

  invisible(NULL)
}

# COMPARAZIÓN ENTRE HORIZONTES TEMPORALES ####

#' Extrae número de días del nombre del horizonte
#' @param horizon_name Nombre del horizonte (ej: "direction_next_5")
#' @return Número de días
extract_days <- function(horizon_name) {
  parts <- strsplit(horizon_name, "_")[[1]]
  last_part <- tail(parts, 1)
  if (last_part == "next") {
    return(1L)
  } else {
    return(as.integer(last_part))
  }
}


#' Compara clasificadores entre horizontes temporales
#' @param predictions_df DataFrame con predicciones (desde load_predictions_csv)
#' @param horizons Vector de horizontes a comparar
#' @param datasets Vector de datasets a analizar
#' @param n_bootstrap Número de iteraciones bootstrap
#' @return DataFrame con resultados de comparaciones pareadas entre horizontes
compare_classification_across_horizons <- function(
  predictions_df,
  horizons = c("direction_next", "direction_next_5", "direction_next_10", "direction_next_20"),
  datasets = c(
    "financial_scaled", "sentiment_scaled",
    "financial_unscaled", "sentiment_unscaled",
    "financial_long_scaled", "financial_long_unscaled"
  ),
  n_bootstrap = 1000
) {
  results <- list()

  for (dataset_name in datasets) {
    df_dataset <- predictions_df |> filter(dataset == dataset_name)

    if (nrow(df_dataset) == 0) next

    cat(sprintf(
      "\n%s\nDATASET: %s\n%s\n",
      strrep("=", 70), dataset_name, strrep("=", 70)
    ))

    # Encontrar modelos comunes en todos los horizontes
    models_by_horizon <- df_dataset |>
      filter(target %in% horizons) |>
      group_by(target) |>
      summarise(models = list(unique(model)), .groups = "drop")

    if (nrow(models_by_horizon) < 2) {
      cat("  ⚠ Menos de 2 horizontes disponibles\n")
      next
    }

    common_models <- Reduce(intersect, models_by_horizon$models)

    if (length(common_models) == 0) {
      cat("  ⚠ No hay modelos comunes en todos los horizontes\n")
      next
    }

    cat(sprintf("  ✓ Modelos comunes: %d\n", length(common_models)))

    for (model_name in sort(common_models)) {
      cat(sprintf(
        "\n%s\nMODELO: %s\n%s\n",
        strrep("-", 70), model_name, strrep("-", 70)
      ))

      # Recolectar datos por horizonte
      horizon_data <- list()

      for (horizon in horizons) {
        data_h <- df_dataset |>
          filter(target == horizon, model == model_name)

        if (nrow(data_h) == 0) next

        y_test <- data_h$y_test[[1]]
        preds <- data_h$predictions[[1]]

        # Limpiar NAs
        valid_idx <- !is.na(y_test) & !is.na(preds)

        horizon_data[[horizon]] <- list(
          y_test = y_test[valid_idx],
          predictions = preds[valid_idx]
        )
      }

      available_horizons <- names(horizon_data)
      if (length(available_horizons) < 2) {
        cat("  ⚠ Menos de 2 horizontes disponibles\n")
        next
      }

      # Alinear tamaños (usar mínimo común)
      min_size <- min(sapply(horizon_data, function(x) length(x$y_test)))
      cat(sprintf("\n  ✓ Tamaño mínimo común: %d observaciones\n", min_size))

      for (h in available_horizons) {
        n <- length(horizon_data[[h]]$y_test)
        horizon_data[[h]]$y_test <- tail(horizon_data[[h]]$y_test, min_size)
        horizon_data[[h]]$predictions <- tail(horizon_data[[h]]$predictions, min_size)
        horizon_data[[h]]$accuracy <- mean(horizon_data[[h]]$predictions == horizon_data[[h]]$y_test)
      }

      # Mostrar accuracies
      cat("\n Accuracy por horizonte:\n")
      for (h in horizons) {
        if (h %in% available_horizons) {
          days <- extract_days(h)
          cat(sprintf("     %2d días: %.4f\n", days, horizon_data[[h]]$accuracy))
        }
      }

      # Comparaciones pareadas consecutivas
      sorted_horizons <- intersect(horizons, available_horizons)

      for (i in seq_len(length(sorted_horizons) - 1)) {
        h1 <- sorted_horizons[i]
        h2 <- sorted_horizons[i + 1]

        days1 <- extract_days(h1)
        days2 <- extract_days(h2)

        y_test <- horizon_data[[h1]]$y_test
        pred1 <- horizon_data[[h1]]$predictions
        pred2 <- horizon_data[[h2]]$predictions

        # McNemar test
        mcn <- mcnemar_test(y_test, pred1, pred2)

        # Bootstrap CI
        boot_ci <- bootstrap_ci_classification(y_test, pred1, pred2,
          n_bootstrap = n_bootstrap
        )

        results[[length(results) + 1]] <- tibble(
          dataset = dataset_name,
          model = model_name,
          horizon_1 = h1,
          horizon_2 = h2,
          days_1 = days1,
          days_2 = days2,
          n_samples = min_size,
          acc_horizon_1 = horizon_data[[h1]]$accuracy,
          acc_horizon_2 = horizon_data[[h2]]$accuracy,
          diff = horizon_data[[h2]]$accuracy - horizon_data[[h1]]$accuracy,
          mcnemar_pvalue = mcn$p_value,
          mcnemar_significant = ifelse(is.na(mcn$significant), FALSE, mcn$significant),
          bootstrap_ci_lower = boot_ci$ci_lower,
          bootstrap_ci_upper = boot_ci$ci_upper,
          bootstrap_significant = ifelse(is.na(boot_ci$significant), FALSE, boot_ci$significant)
        )

        sig_symbol <- ifelse(mcn$significant %||% FALSE, "✓", "✗")
        cat(sprintf("\n  %s vs %s:\n", h1, h2))
        cat(sprintf("    Δ Accuracy: %+.4f\n", horizon_data[[h2]]$accuracy - horizon_data[[h1]]$accuracy))
        cat(sprintf("    McNemar p-value: %.4f %s\n", mcn$p_value, sig_symbol))
      }
    }
  }

  if (length(results) > 0) bind_rows(results) else tibble()
}


#' Compara regresores entre horizontes temporales
#' @param predictions_df DataFrame con predicciones
#' @param horizons Vector de horizontes a comparar
#' @param datasets Vector de datasets a analizar
#' @param n_bootstrap Número de iteraciones bootstrap
#' @return DataFrame con resultados de comparaciones pareadas
compare_regression_across_horizons <- function(
  predictions_df,
  horizons = c("returns_next", "returns_next_5", "returns_next_10", "returns_next_20"),
  datasets = c(
    "financial_scaled", "sentiment_scaled",
    "financial_unscaled", "sentiment_unscaled",
    "financial_long_scaled", "financial_long_unscaled"
  ),
  n_bootstrap = 1000
) {
  results <- list()

  for (dataset_name in datasets) {
    df_dataset <- predictions_df |> filter(dataset == dataset_name)

    if (nrow(df_dataset) == 0) next

    cat(sprintf(
      "\n%s\nDATASET: %s\n%s\n",
      strrep("=", 70), dataset_name, strrep("=", 70)
    ))

    # Encontrar modelos comunes
    models_by_horizon <- df_dataset |>
      filter(target %in% horizons) |>
      group_by(target) |>
      summarise(models = list(unique(model)), .groups = "drop")

    if (nrow(models_by_horizon) < 2) {
      cat("  ⚠ Menos de 2 horizontes disponibles\n")
      next
    }

    common_models <- Reduce(intersect, models_by_horizon$models)

    if (length(common_models) == 0) {
      cat("  ⚠ No hay modelos comunes\n")
      next
    }

    cat(sprintf("  ✓ Modelos comunes: %d\n", length(common_models)))

    for (model_name in sort(common_models)) {
      cat(sprintf(
        "\n%s\nMODELO: %s\n%s\n",
        strrep("-", 70), model_name, strrep("-", 70)
      ))

      horizon_data <- list()

      for (horizon in horizons) {
        data_h <- df_dataset |>
          filter(target == horizon, model == model_name)

        if (nrow(data_h) == 0) next

        y_test <- data_h$y_test[[1]]
        preds <- data_h$predictions[[1]]

        valid_idx <- !is.na(y_test) & !is.na(preds)

        y_clean <- y_test[valid_idx]
        pred_clean <- preds[valid_idx]

        horizon_data[[horizon]] <- list(
          y_test = y_clean,
          predictions = pred_clean,
          mae = mean(abs(y_clean - pred_clean)),
          rmse = sqrt(mean((y_clean - pred_clean)^2))
        )
      }

      available_horizons <- names(horizon_data)
      if (length(available_horizons) < 2) {
        cat("  ⚠ Menos de 2 horizontes disponibles\n")
        next
      }

      # Alinear tamaños
      min_size <- min(sapply(horizon_data, function(x) length(x$y_test)))
      cat(sprintf("\n  ✓ Tamaño mínimo común: %d observaciones\n", min_size))

      for (h in available_horizons) {
        horizon_data[[h]]$y_test <- tail(horizon_data[[h]]$y_test, min_size)
        horizon_data[[h]]$predictions <- tail(horizon_data[[h]]$predictions, min_size)

        y <- horizon_data[[h]]$y_test
        p <- horizon_data[[h]]$predictions
        horizon_data[[h]]$mae <- mean(abs(y - p))
        horizon_data[[h]]$rmse <- sqrt(mean((y - p)^2))
      }

      # Mostrar métricas
      cat("\n Métricas por horizonte:\n")
      for (h in horizons) {
        if (h %in% available_horizons) {
          days <- extract_days(h)
          cat(sprintf(
            "     %2d días: MAE=%.6f, RMSE=%.6f\n",
            days, horizon_data[[h]]$mae, horizon_data[[h]]$rmse
          ))
        }
      }

      # Comparaciones pareadas
      sorted_horizons <- intersect(horizons, available_horizons)

      for (i in seq_len(length(sorted_horizons) - 1)) {
        h1 <- sorted_horizons[i]
        h2 <- sorted_horizons[i + 1]

        days1 <- extract_days(h1)
        days2 <- extract_days(h2)

        y_test <- horizon_data[[h1]]$y_test
        pred1 <- horizon_data[[h1]]$predictions
        pred2 <- horizon_data[[h2]]$predictions

        errors1 <- y_test - pred1
        errors2 <- y_test - pred2

        # Diebold-Mariano test
        dm <- diebold_mariano_test(errors1, errors2, horizon = days1)

        # Bootstrap CI
        boot_ci <- bootstrap_ci_regression(y_test, pred1, pred2,
          n_bootstrap = n_bootstrap
        )

        mae_diff <- horizon_data[[h1]]$mae - horizon_data[[h2]]$mae

        results[[length(results) + 1]] <- tibble(
          dataset = dataset_name,
          model = model_name,
          horizon_1 = h1,
          horizon_2 = h2,
          days_1 = days1,
          days_2 = days2,
          n_samples = min_size,
          mae_horizon_1 = horizon_data[[h1]]$mae,
          mae_horizon_2 = horizon_data[[h2]]$mae,
          rmse_horizon_1 = horizon_data[[h1]]$rmse,
          rmse_horizon_2 = horizon_data[[h2]]$rmse,
          mae_diff = mae_diff,
          dm_statistic = dm$statistic,
          dm_pvalue = dm$p_value,
          dm_significant = ifelse(is.na(dm$significant), FALSE, dm$significant),
          bootstrap_ci_lower = boot_ci$ci_lower,
          bootstrap_ci_upper = boot_ci$ci_upper,
          bootstrap_significant = ifelse(is.na(boot_ci$significant), FALSE, boot_ci$significant)
        )

        sig_symbol <- ifelse(dm$significant %||% FALSE, "✓", "✗")
        cat(sprintf("\n  %s vs %s:\n", h1, h2))
        cat(sprintf("    Δ MAE: %.6f\n", mae_diff))
        cat(sprintf("    DM statistic: %.4f\n", dm$statistic))
        cat(sprintf("    p-value: %.4f %s\n", dm$p_value, sig_symbol))
      }
    }
  }

  if (length(results) > 0) bind_rows(results) else tibble()
}


# ANÁLISIS DE EFECTOS DE HORIZONTE  ####


#' Análisis cuantitativo del efecto del horizonte en clasificación
#' @param horizon_results_df DataFrame de compare_classification_across_horizons
analyze_classification_horizon_effects <- function(horizon_results_df) {
  if (nrow(horizon_results_df) == 0) {
    cat("No hay datos para analizar\n")
    return(invisible(NULL))
  }

  cat(sprintf("\n%s\n", strrep("=", 70)))
  cat("ANÁLISIS: EFECTO DEL HORIZONTE TEMPORAL - CLASIFICACIÓN\n")
  cat(sprintf("%s\n", strrep("=", 70)))

  # 1. Tendencia general por dataset
  cat("\n TENDENCIA GENERAL:\n")

  for (dataset in unique(horizon_results_df$dataset)) {
    data <- horizon_results_df |> filter(dataset == !!dataset)

    cat(sprintf("\n  %s:\n", dataset))

    # Correlación días vs accuracy
    all_days <- c(data$days_1, data$days_2)
    all_accs <- c(data$acc_horizon_1, data$acc_horizon_2)

    if (length(all_days) > 2) {
      corr <- cor(all_days, all_accs, use = "complete.obs")

      if (is.na(corr)) {
        cat("    Correlación: NA (varianza cero)\n")
      } else if (corr > 0.3) {
        cat(sprintf("    Correlación: %.3f → Mejora con horizonte más largo\n", corr))
      } else if (corr < -0.3) {
        cat(sprintf("    Correlación: %.3f → Empeora con horizonte más largo\n", corr))
      } else {
        cat(sprintf("    Correlación: %.3f → Sin tendencia clara\n", corr))
      }

      # Accuracy promedio por horizonte
      cat("    Accuracy promedio por horizonte:\n")
      for (days in c(1, 5, 10, 20)) {
        accs <- c(
          data$acc_horizon_1[data$days_1 == days],
          data$acc_horizon_2[data$days_2 == days]
        )
        if (length(accs) > 0) {
          cat(sprintf("      %2d días: %.4f\n", days, mean(accs, na.rm = TRUE)))
        }
      }
    }
  }

  # 2. Diferencias significativas
  cat(sprintf("\n COMPARACIONES SIGNIFICATIVAS:\n"))

  sig_comparisons <- horizon_results_df |> filter(mcnemar_significant == TRUE)

  if (nrow(sig_comparisons) > 0) {
    cat(sprintf(
      "  Total: %d / %d (%.1f%%)\n",
      nrow(sig_comparisons),
      nrow(horizon_results_df),
      100 * nrow(sig_comparisons) / nrow(horizon_results_df)
    ))

    for (i in seq_len(nrow(sig_comparisons))) {
      row <- sig_comparisons[i, ]
      direction <- ifelse(row$diff > 0, "Aumenta", "Disminuty")
      cat(sprintf("\n  %s %s (%s)\n", direction, row$model, row$dataset))
      cat(sprintf("     %dd → %dd: %+.4f\n", row$days_1, row$days_2, row$diff))
      cat(sprintf("     p-value: %.4f\n", row$mcnemar_pvalue))
    }
  } else {
    cat("  ✗ NO se encontraron diferencias significativas\n")
  }

  # 3. Estabilidad por modelo
  cat(sprintf("\n ESTABILIDAD POR MODELO:\n"))

  stability <- horizon_results_df |>
    group_by(model) |>
    summarise(
      diff_mean = mean(diff, na.rm = TRUE),
      diff_std = sd(diff, na.rm = TRUE),
      n_significant = sum(mcnemar_significant, na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(diff_std)

  cat("\n  Más estables (menor variación):\n")
  for (i in seq_len(min(3, nrow(stability)))) {
    row <- stability[i, ]
    cat(sprintf(
      "    %s: std=%.4f, mean_diff=%+.4f\n",
      row$model, row$diff_std, row$diff_mean
    ))
  }

  if (nrow(stability) > 3) {
    cat("\n  Más variables (mayor variación):\n")
    for (i in seq(max(1, nrow(stability) - 2), nrow(stability))) {
      row <- stability[i, ]
      cat(sprintf(
        "    %s: std=%.4f, mean_diff=%+.4f, sig=%d\n",
        row$model, row$diff_std, row$diff_mean, row$n_significant
      ))
    }
  }

  invisible(NULL)
}


#' Análisis cuantitativo del efecto del horizonte en regresión
#' @param horizon_results_df DataFrame de compare_regression_across_horizons
analyze_regression_horizon_effects <- function(horizon_results_df) {
  if (nrow(horizon_results_df) == 0) {
    cat("No hay datos para analizar\n")
    return(invisible(NULL))
  }

  cat(sprintf("\n%s\n", strrep("=", 70)))
  cat("ANÁLISIS: EFECTO DEL HORIZONTE TEMPORAL - REGRESIÓN\n")
  cat(sprintf("%s\n", strrep("=", 70)))

  for (dataset in unique(horizon_results_df$dataset)) {
    data <- horizon_results_df |> filter(dataset == !!dataset)

    cat(sprintf("\nDATASET: %s\n", dataset))
    cat(strrep("-", 70), "\n")

    # Diferencias significativas
    sig <- sum(data$dm_significant, na.rm = TRUE)
    total <- nrow(data)

    cat(sprintf("\n  🔍 Comparaciones significativas (DM test):\n"))
    cat(sprintf("    Total: %d / %d (%.1f%%)\n", sig, total, 100 * sig / total))

    # MAE promedio por horizonte
    cat(sprintf("\n MAE promedio por horizonte:\n"))
    for (days in c(1, 5, 10, 20)) {
      mae_list <- c(
        data$mae_horizon_1[data$days_1 == days],
        data$mae_horizon_2[data$days_2 == days]
      )
      if (length(mae_list) > 0) {
        cat(sprintf("    %2d días: %.6f\n", days, mean(mae_list, na.rm = TRUE)))
      }
    }

    # Tendencias por modelo
    cat(sprintf("\n  Tendencias por modelo:\n"))

    for (model in unique(data$model)) {
      model_data <- data |>
        filter(model == !!model) |>
        arrange(days_1)

      if (nrow(model_data) < 2) next

      mae_trend <- cor(model_data$days_1, model_data$mae_horizon_1, use = "complete.obs")

      cat(sprintf("\n    %s:\n", model))
      cat(sprintf("      Correlación días vs MAE: %.3f\n", mae_trend))

      if (is.na(mae_trend)) {
        cat("      → Datos insuficientes\n")
      } else if (mae_trend > 0.5) {
        cat("      → Horizonte más largo = Menos predecible (mayor error)\n")
      } else if (mae_trend < -0.5) {
        cat("      → Horizonte más largo = Más predecible (menor error)\n")
      } else {
        cat("      → Sin tendencia clara\n")
      }
    }
  }

  invisible(NULL)
}


# VISUALIZACIONES ####


#' Plot evolución de métricas por horizonte - Regresión
#' @param horizon_results_df DataFrame de compare_regression_across_horizons
#' @param metric "mae" o "rmse"
#' @param save_path Ruta para guardar (opcional)
plot_regression_horizon_evolution <- function(horizon_results_df,
                                              metric = "mae",
                                              save_path = NULL) {
  if (nrow(horizon_results_df) == 0) {
    warning("No hay datos para visualizar")
    return(invisible(NULL))
  }

  # Preparar datos en formato largo
  plot_data <- horizon_results_df |>
    dplyr::select(
      dataset, model, days_1, days_2,
      mae_horizon_1, mae_horizon_2, rmse_horizon_1, rmse_horizon_2
    ) |>
    pivot_longer(
      cols = c(mae_horizon_1, mae_horizon_2, rmse_horizon_1, rmse_horizon_2),
      names_to = c("metric_type", "horizon_num"),
      names_pattern = "(mae|rmse)_horizon_(1|2)",
      values_to = "value"
    ) |>
    mutate(
      days = ifelse(horizon_num == "1", days_1, days_2)
    ) |>
    filter(metric_type == metric) |>
    distinct(dataset, model, days, .keep_all = TRUE)

  p <- ggplot(plot_data, aes(x = days, y = value, color = model, group = model)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    facet_wrap(~dataset, scales = "free_y", ncol = 2) +
    scale_x_continuous(breaks = c(1, 5, 10, 20)) +
    labs(
      title = sprintf("Evolución de %s por Horizonte Temporal", toupper(metric)),
      x = "Horizonte (días)",
      y = toupper(metric),
      color = "Modelo"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold")
    )

  if (!is.null(save_path)) {
    save_plot("mae_horizon_evolution", p)
  }

  print(p)
  invisible(p)
}


#' Boxplot de métricas por horizonte - Regresión
#' @param horizon_results_df DataFrame de compare_regression_across_horizons
#' @param save_path Ruta para guardar (opcional)
plot_regression_horizon_boxplot <- function(horizon_results_df, save_path = NULL) {
  if (nrow(horizon_results_df) == 0) {
    warning("No hay datos para visualizar")
    return(invisible(NULL))
  }

  # Preparar datos
  plot_data <- bind_rows(
    horizon_results_df |>
      dplyr::select(dataset, model, days = days_1, mae = mae_horizon_1, rmse = rmse_horizon_1),
    horizon_results_df |>
      dplyr::select(dataset, model, days = days_2, mae = mae_horizon_2, rmse = rmse_horizon_2)
  ) |>
    distinct() |>
    mutate(days = factor(days, levels = c(1, 5, 10, 20)))

  p <- ggplot(plot_data, aes(x = days, y = mae, fill = days)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.2, alpha = 0.5, size = 1) +
    facet_wrap(~dataset, scales = "free_y", ncol = 2) +
    scale_fill_viridis_d(option = "plasma") +
    labs(
      title = "Distribución de MAE por Horizonte Temporal",
      x = "Horizonte (días)",
      y = "MAE"
    ) +
    theme_minimal() +
    theme(legend.position = "none")

  if (!is.null(save_path)) {
    save_plot("horizon_effects_regression", p)
  }

  print(p)
  invisible(p)
}


#' Plot evolución de accuracy por horizonte - Clasificación
#' @param horizon_results_df DataFrame de compare_classification_across_horizons
#' @param save_path Ruta para guardar (opcional)
plot_classification_horizon_evolution <- function(horizon_results_df, save_path = NULL) {
  if (nrow(horizon_results_df) == 0) {
    warning("No hay datos para visualizar")
    return(invisible(NULL))
  }

  # Preparar datos
  plot_data <- bind_rows(
    horizon_results_df |>
      dplyr::select(dataset, model, days = days_1, accuracy = acc_horizon_1),
    horizon_results_df |>
      dplyr::select(dataset, model, days = days_2, accuracy = acc_horizon_2)
  ) |>
    distinct()

  p <- ggplot(plot_data, aes(x = days, y = accuracy, color = model, group = model)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray50") +
    facet_wrap(~dataset, ncol = 2) +
    scale_x_continuous(breaks = c(1, 5, 10, 20)) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "Evolución de Accuracy por Horizonte Temporal",
      x = "Horizonte (días)",
      y = "Accuracy",
      color = "Modelo"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold")
    )

  if (!is.null(save_path)) {
    save_plot("temporal_horizon_evolution", p)
  }

  print(p)
  invisible(p)
}


#' Heatmap de significancia estadística - Clasificación
#' @param horizon_results_df DataFrame de compare_classification_across_horizons
#' @param save_path Ruta para guardar (opcional)
plot_classification_significance_heatmap <- function(horizon_results_df, save_path = NULL) {
  if (nrow(horizon_results_df) == 0) {
    warning("No hay datos para visualizar")
    return(invisible(NULL))
  }

  # Preparar datos
  plot_data <- horizon_results_df |>
    mutate(
      comparison = sprintf("%dd→%dd", days_1, days_2),
      significant = ifelse(mcnemar_significant, "Sí", "No"),
      label = sprintf("%.3f", mcnemar_pvalue)
    )

  p <- ggplot(plot_data, aes(x = comparison, y = model, fill = mcnemar_pvalue)) +
    geom_tile(color = "white") +
    geom_text(aes(label = label), size = 2.5) +
    facet_wrap(~dataset, ncol = 2) +
    scale_fill_gradient2(
      low = "darkgreen", mid = "white", high = "white",
      midpoint = 0.05,
      limits = c(0, 1),
      name = "p-value"
    ) +
    labs(
      title = "Test de McNemar: Comparaciones entre Horizontes",
      subtitle = "Verde = significativo (p < 0.05)",
      x = "Comparación de Horizontes",
      y = "Modelo"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.text = element_text(face = "bold")
    )

  if (!is.null(save_path)) {
    save_plot("heatmap_horizons_classification", p)
  }

  print(p)
  invisible(p)
}


#' Resumen ejecutivo comparando clasificación y regresión
#' @param classification_results DataFrame de compare_classification_across_horizons
#' @param regression_results DataFrame de compare_regression_across_horizons
#' @param save_path Ruta para guardar (opcional)
plot_executive_summary <- function(classification_results,
                                   regression_results,
                                   save_path = output_path,
                                   save_plots = FALSE) {
  # Panel 1: Proporción de comparaciones significativas
  sig_summary <- bind_rows(
    classification_results |>
      group_by(dataset) |>
      summarise(
        pct_significant = 100 * mean(mcnemar_significant, na.rm = TRUE),
        .groups = "drop"
      ) |>
      mutate(task = "Clasificación"),
    regression_results |>
      group_by(dataset) |>
      summarise(
        pct_significant = 100 * mean(dm_significant, na.rm = TRUE),
        .groups = "drop"
      ) |>
      mutate(task = "Regresión")
  )

  p1 <- ggplot(sig_summary, aes(x = dataset, y = pct_significant, fill = task)) +
    geom_col(position = "dodge") +
    geom_hline(yintercept = 5, linetype = "dashed", color = "red") +
    scale_fill_manual(values = c("Clasificación" = "#2E86AB", "Regresión" = "#A23B72")) +
    labs(
      title = "Comparaciones Significativas entre Horizontes",
      subtitle = "Línea roja = 5% esperado por azar",
      x = NULL,
      y = "% Significativas",
      fill = "Tarea"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  # Panel 2: Tendencia promedio por horizonte
  trend_class <- bind_rows(
    classification_results |> dplyr::select(days = days_1, value = acc_horizon_1),
    classification_results |> dplyr::select(days = days_2, value = acc_horizon_2)
  ) |>
    group_by(days) |>
    summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop") |>
    mutate(task = "Clasificación (Acc)")

  trend_reg <- bind_rows(
    regression_results |> dplyr::select(days = days_1, value = mae_horizon_1),
    regression_results |> dplyr::select(days = days_2, value = mae_horizon_2)
  ) |>
    group_by(days) |>
    summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop") |>
    mutate(task = "Regresión (MAE)")

  p2 <- ggplot() +
    geom_line(
      data = trend_class, aes(x = days, y = mean_value),
      color = "#2E86AB", linewidth = 1.5
    ) +
    geom_point(
      data = trend_class, aes(x = days, y = mean_value),
      color = "#2E86AB", size = 3
    ) +
    scale_x_continuous(breaks = c(1, 5, 10, 20)) +
    labs(
      title = "Tendencia de Accuracy \npor Horizonte",
      x = "Horizonte (días)",
      y = "Accuracy Promedio"
    ) +
    theme_minimal()

  p3 <- ggplot() +
    geom_line(
      data = trend_reg, aes(x = days, y = mean_value),
      color = "#A23B72", linewidth = 1.5
    ) +
    geom_point(
      data = trend_reg, aes(x = days, y = mean_value),
      color = "#A23B72", size = 3
    ) +
    scale_x_continuous(breaks = c(1, 5, 10, 20)) +
    labs(
      title = "Tendencia de MAE \npor Horizonte",
      x = "Horizonte (días)",
      y = "MAE Promedio"
    ) +
    theme_minimal()

  # Combinar con patchwork si está disponible
  if (requireNamespace("patchwork", quietly = TRUE)) {
    library(patchwork)
    p_combined <- p1 / (p2 | p3) +
      plot_annotation(
        title = "Resumen Ejecutivo: Efecto del Horizonte Temporal",
        theme = theme(plot.title = element_text(face = "bold", size = 12))
      )

    if (save_plots) {
      save_plot("horizon_effects_executive_summary", p_combined)
    }

    print(p_combined)
    return(invisible(p_combined))
  } else {
    # Sin patchwork, mostrar plots individuales
    print(p1)
    print(p2)
    print(p3)
    return(invisible(list(p1 = p1, p2 = p2, p3 = p3)))
  }
}


# DETERMINAR HORIZONTE ÓPTIMO ####


#' Determina el horizonte óptimo por modelo-dataset
#' @param classification_results DataFrame de compare_classification_across_horizons
#' @param regression_results DataFrame de compare_regression_across_horizons
#' @return Lista con DataFrames de horizontes óptimos
determine_optimal_horizons <- function(classification_results = NULL,
                                       regression_results = NULL) {
  results <- list()

  # Clasificación: horizonte con mayor accuracy
  if (!is.null(classification_results) && nrow(classification_results) > 0) {
    # Extraer todas las combinaciones modelo-dataset-horizonte con su accuracy
    class_metrics <- bind_rows(
      classification_results |>
        dplyr::select(dataset, model, days = days_1, accuracy = acc_horizon_1),
      classification_results |>
        dplyr::select(dataset, model, days = days_2, accuracy = acc_horizon_2)
    ) |>
      distinct() |>
      group_by(dataset, model) |>
      slice_max(accuracy, n = 1, with_ties = FALSE) |>
      ungroup() |>
      rename(optimal_horizon = days, best_accuracy = accuracy) |>
      arrange(dataset, model)

    results$classification <- class_metrics

    # Resumen general
    cat("HORIZONTES ÓPTIMOS - CLASIFICACIÓN\n")
    cat(strrep("=", 70), "\n")

    # Por dataset
    for (ds in unique(class_metrics$dataset)) {
      cat(sprintf("\n%s:\n", ds))
      ds_data <- class_metrics |> filter(dataset == ds)
      for (i in seq_len(nrow(ds_data))) {
        cat(sprintf(
          "  %s: %d días (acc=%.4f)\n",
          ds_data$model[i], ds_data$optimal_horizon[i], ds_data$best_accuracy[i]
        ))
      }
    }

    # Horizonte más común
    horizon_counts <- table(class_metrics$optimal_horizon)
    most_common <- as.integer(names(which.max(horizon_counts)))
    cat(sprintf(
      "\n Horizonte más frecuente: %d días (%d/%d modelos)\n",
      most_common, max(horizon_counts), nrow(class_metrics)
    ))
  }

  # Regresión: horizonte con menor MAE
  if (!is.null(regression_results) && nrow(regression_results) > 0) {
    reg_metrics <- bind_rows(
      regression_results |>
        dplyr::select(dataset, model, days = days_1, mae = mae_horizon_1),
      regression_results |>
        dplyr::select(dataset, model, days = days_2, mae = mae_horizon_2)
    ) |>
      distinct() |>
      group_by(dataset, model) |>
      slice_min(mae, n = 1, with_ties = FALSE) |>
      ungroup() |>
      rename(optimal_horizon = days, best_mae = mae) |>
      arrange(dataset, model)

    results$regression <- reg_metrics

    cat("HORIZONTES ÓPTIMOS - REGRESIÓN\n")
    cat(strrep("=", 70), "\n")

    for (ds in unique(reg_metrics$dataset)) {
      cat(sprintf("\n%s:\n", ds))
      ds_data <- reg_metrics |> filter(dataset == ds)
      for (i in seq_len(nrow(ds_data))) {
        cat(sprintf(
          "  %s: %d días (MAE=%.6f)\n",
          ds_data$model[i], ds_data$optimal_horizon[i], ds_data$best_mae[i]
        ))
      }
    }

    horizon_counts <- table(reg_metrics$optimal_horizon)
    most_common <- as.integer(names(which.max(horizon_counts)))
    cat(sprintf(
      "\n Horizonte más frecuente: %d días (%d/%d modelos)\n",
      most_common, max(horizon_counts), nrow(reg_metrics)
    ))
  }

  invisible(results)
}
