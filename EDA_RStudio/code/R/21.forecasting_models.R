# FUNCIONES DE EVALUACIÓN ####

evaluate_forecast <- function(actual, predicted, model_name = "Model") {
  # Eliminar NAs
  mask <- !is.na(actual) & !is.na(predicted)
  actual <- actual[mask]
  predicted <- predicted[mask]

  if (length(actual) == 0) {
    return(data.frame(
      model = model_name,
      RMSE = NA,
      MAE = NA,
      MAPE = NA,
      Direction_Accuracy = NA,
      n_samples = 0
    ))
  }

  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  mape <- mean(abs((actual - predicted) / (actual + 1e-10))) * 100 # +epsilon para evitar div/0

  # Direction accuracy (crítico para trading)
  direction_accuracy <- mean(sign(actual) == sign(predicted)) * 100

  data.frame(
    model = model_name,
    RMSE = rmse,
    MAE = mae,
    MAPE = mape,
    Direction_Accuracy = direction_accuracy,
    n_samples = length(actual)
  )
}

#' Convertir predicciones de regresión a clasificación binaria
#'
#' @param y_true_regression Vector numérico con valores reales de regresión
#' @param y_pred_regression Vector numérico con predicciones de regresión
#'
#' @return Lista con dos elementos: y_true_class y y_pred_class
#'         NULL si hay error o datos vacíos
#'
#' @details
#' Convierte predicciones continuas (returns) a clasificación binaria:
#' - 1 = Predicción de subida (return > 0)
#' - 0 = Predicción de bajada (return <= 0)
#'
#' @examples
#' y_true <- c(0.5, -0.3, 0.2, -0.1, 0.8)
#' y_pred <- c(0.4, -0.2, 0.1, 0.3, 0.6)
#' result <- regression_to_classification(y_true, y_pred)
#' y_true_class <- result$y_true_class
#' y_pred_class <- result$y_pred_class
#'
#' @export
regression_to_classification <- function(y_true_regression, y_pred_regression) {
  # Validar inputs
  if (is.null(y_true_regression) || is.null(y_pred_regression)) {
    return(NULL)
  }

  # Convertir a vectores numéricos si no lo son
  y_true_regression <- as.numeric(y_true_regression)
  y_pred_regression <- as.numeric(y_pred_regression)

  # Ajustar tamaños al mínimo común
  min_len <- min(length(y_true_regression), length(y_pred_regression))
  y_true_regression <- y_true_regression[1:min_len]
  y_pred_regression <- y_pred_regression[1:min_len]

  # Eliminar NaN/NA
  mask <- !is.na(y_true_regression) & !is.na(y_pred_regression)
  y_true_regression <- y_true_regression[mask]
  y_pred_regression <- y_pred_regression[mask]

  # Validar que quedan datos
  if (length(y_true_regression) == 0) {
    return(NULL)
  }

  # Convertir a clasificación binaria (1 si > 0, 0 si <= 0)
  y_true_class <- as.integer(y_true_regression > 0)
  y_pred_class <- as.integer(y_pred_regression > 0)

  # Retornar como lista
  return(list(
    y_true_class = y_true_class,
    y_pred_class = y_pred_class
  ))
}
