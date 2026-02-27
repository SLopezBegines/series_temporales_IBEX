# PREPARAR DATASETS ESTANDARIZADOS Y NO ESTANDARIZADOS


library(dplyr)
library(caret)

create_scaled_versions <- function(train_df, test_df,
                                   scale_method = "center_scale",
                                   save_scaler = TRUE,
                                   output_dir = "output/RData") {
  #' Crea versiones estandarizadas de los datasets
  #'
  #' @param train_df DataFrame train
  #' @param test_df DataFrame test
  #' @param scale_method "center_scale", "minmax", o "robust"
  #' @param save_scaler Guardar objeto scaler (default: TRUE)
  #' @param output_dir Directorio de salida
  #' @return Lista con datasets escalados y sin escalar

  log_cat("=== CREANDO VERSIONES ESTANDARIZADAS ===\n\n")

  # Variables a NO estandarizar
  exclude_vars <- c(
    "date",
    "returns_next", "returns_next_5", "returns_next_10", "returns_next_20",
    "direction_next", "direction_next_5", "direction_next_10", "direction_next_20",
    "weekday", "month", "quarter"
  ) # Categóricas si existen y targets

  # Variables numéricas para estandarizar
  numeric_vars <- train_df %>%
    dplyr::select(-any_of(exclude_vars)) %>%
    dplyr::select(where(is.numeric)) %>%
    names()

  log_cat("Variables a estandarizar:", length(numeric_vars), "\n")


  # Método de estandarización


  if (scale_method == "center_scale") {
    # Z-score: (x - mean) / sd
    preProc <- preProcess(train_df[numeric_vars],
      method = c("center", "scale")
    )
    method_name <- "zscore"
  } else if (scale_method == "minmax") {
    # Min-Max: (x - min) / (max - min) → [0, 1]
    preProc <- preProcess(train_df[numeric_vars],
      method = c("range")
    )
    method_name <- "minmax"
  } else if (scale_method == "robust") {
    # Robust: (x - median) / IQR (menos sensible a outliers)
    # Implementación manual
    medians <- sapply(train_df[numeric_vars], median, na.rm = TRUE)
    iqrs <- sapply(train_df[numeric_vars], IQR, na.rm = TRUE)

    train_scaled <- train_df
    test_scaled <- test_df

    for (var in numeric_vars) {
      train_scaled[[var]] <- (train_df[[var]] - medians[var]) / iqrs[var]
      test_scaled[[var]] <- (test_df[[var]] - medians[var]) / iqrs[var]
    }

    method_name <- "robust"

    preProc <- list(medians = medians, iqrs = iqrs, method = "robust")
  }

  # Aplicar transformación (para center_scale y minmax)
  if (scale_method %in% c("center_scale", "minmax")) {
    train_scaled <- train_df
    test_scaled <- test_df

    train_scaled[numeric_vars] <- predict(preProc, train_df[numeric_vars])
    test_scaled[numeric_vars] <- predict(preProc, test_df[numeric_vars])
  }


  # Verificación


  log_cat("\n=== VERIFICACIÓN ===\n")
  log_cat("Método:", method_name, "\n\n")

  # Sample de una variable antes/después
  sample_var <- numeric_vars[1]
  log_cat("Ejemplo - Variable:", sample_var, "\n")
  log_cat("Original (train):\n")
  print(summary(train_df[[sample_var]]))
  log_cat("\nEscalada (train):\n")
  print(summary(train_scaled[[sample_var]]))


  # Guardar


  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  # Datasets sin escalar (ya guardados, pero por consistencia)
  saveRDS(train_df, file.path(output_dir, "train_unscaled.rds"))
  saveRDS(test_df, file.path(output_dir, "test_unscaled.rds"))

  # Datasets escalados
  saveRDS(
    train_scaled,
    file.path(output_dir, paste0("train_scaled_", method_name, ".rds"))
  )
  saveRDS(
    test_scaled,
    file.path(output_dir, paste0("test_scaled_", method_name, ".rds"))
  )

  # Guardar scaler para uso futuro
  if (save_scaler) {
    saveRDS(
      preProc,
      file.path(output_dir, paste0("scaler_", method_name, ".rds"))
    )
  }

  log_cat("\n=== ARCHIVOS GUARDADOS ===\n")
  log_cat("- train_unscaled.rds\n")
  log_cat("- test_unscaled.rds\n")
  log_cat("- train_scaled_", method_name, ".rds\n", sep = "")
  log_cat("- test_scaled_", method_name, ".rds\n", sep = "")
  if (save_scaler) log_cat("- scaler_", method_name, ".rds\n", sep = "")

  return(list(
    train_unscaled = train_df,
    test_unscaled = test_df,
    train_scaled = train_scaled,
    test_scaled = test_scaled,
    scaler = preProc,
    method = method_name,
    vars_scaled = numeric_vars
  ))
}

# Ejemplo de uso:
'# Para datasets SOLO financieros
scaled_financial <- create_scaled_versions(
  train_df = train_final,
  test_df = test_final,
  scale_method = "center_scale",  # Z-score estándar
  save_scaler = TRUE,
  output_dir = "output/RData/financial"
)

# Para datasets CON sentiment
scaled_sentiment <- create_scaled_versions(
  train_df = train_final_with_sentiment,
  test_df = test_final_with_sentiment,
  scale_method = "center_scale",
  save_scaler = TRUE,
  output_dir = "output/RData/with_sentiment"
)

# En Python, cuando hagas ML:

# Para TREE-BASED models (Random Forest, XGBoost)
train = pd.read_pickle("train_unscaled.rds")  # Usar sin escalar
test = pd.read_pickle("test_unscaled.rds")

# Para LINEAR models (Logistic, SVM, Neural Networks)
train = pd.read_pickle("train_scaled_zscore.rds")  # Usar escalados
test = pd.read_pickle("test_scaled_zscore.rds")'
