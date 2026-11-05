# Script R para evaluar modelos reales vs ingenuos
# Santi - TFM IBEX35

library(caret)
library(dplyr)

evaluate_model_vs_naive <- function(y_true, y_pred, model_name = "Model",
                                    dataset = "", target = "") {
  # Convertir a factor si no lo son
  y_true <- as.factor(y_true)
  y_pred <- as.factor(y_pred)

  # Asegurar mismos niveles
  levels(y_pred) <- levels(y_true)

  # Calcular prevalencia
  class_counts <- table(y_true)
  total <- length(y_true)
  prevalence_0 <- class_counts[1] / total
  prevalence_1 <- class_counts[2] / total

  # Matriz de confusión
  cm <- confusionMatrix(y_pred, y_true)

  # Métricas
  accuracy <- cm$overall["Accuracy"]
  balanced_acc <- cm$byClass["Balanced Accuracy"]
  kappa <- cm$overall["Kappa"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  f1_score <- cm$byClass["F1"]

  # Métricas por clase
  precision_0 <- cm$byClass["Neg Pred Value"]
  recall_0 <- cm$byClass["Specificity"] # Recall de clase 0 = Especificidad
  f1_0 <- ifelse(is.na(precision_0) | is.na(recall_0), 0,
    2 * precision_0 * recall_0 / (precision_0 + recall_0)
  )

  precision_1 <- cm$byClass["Pos Pred Value"]
  recall_1 <- cm$byClass["Sensitivity"]
  f1_1 <- ifelse(is.na(precision_1) | is.na(recall_1), 0,
    2 * precision_1 * recall_1 / (precision_1 + recall_1)
  )

  # Test binomial para clase minoritaria
  minority_class <- ifelse(class_counts[1] < class_counts[2], 0, 1)
  minority_mask <- as.numeric(as.character(y_true)) == minority_class
  minority_correct <- sum(as.numeric(as.character(y_pred))[minority_mask] ==
    as.numeric(as.character(y_true))[minority_mask])
  minority_total <- sum(minority_mask)
  # La probabilidad de acertar por azar es la prevalencia de esa clase
  minority_prevalence <- ifelse(minority_class == 0, prevalence_0, prevalence_1)
  # Evitar error si minority_total es 0
  p_value <- ifelse(minority_total > 0,
    binom.test(minority_correct, minority_total,
      p = minority_prevalence, alternative = "greater"
    )$p.value,
    1.0
  )

  # Criterios para clasificar como ingenuo
  is_naive_kappa <- kappa < 0.1
  is_naive_balanced <- balanced_acc < 0.52
  is_naive_pvalue <- p_value > 0.05
  is_naive_f1_minority <- ifelse(minority_class == 0, f1_0 < 0.1, f1_1 < 0.1)

  # Clasificación final: es ingenuo si cumple 2 o más criterios
  naive_count <- sum(c(
    is_naive_kappa, is_naive_balanced,
    is_naive_pvalue, is_naive_f1_minority
  ))
  is_naive <- naive_count >= 2

  data.frame(
    Dataset = dataset,
    Target = target,
    Model = model_name,
    N_samples = total,
    Prevalence_Class_0 = round(prevalence_0, 4),
    Prevalence_Class_1 = round(prevalence_1, 4),
    Accuracy = round(accuracy, 4),
    Balanced_Accuracy = round(balanced_acc, 4),
    Cohen_Kappa = round(kappa, 4),
    Sensitivity = round(sensitivity, 4),
    Specificity = round(specificity, 4),
    F1_Score = round(f1_score, 4),
    Precision_Class_0 = round(precision_0, 4),
    Recall_Class_0 = round(recall_0, 4),
    F1_Class_0 = round(f1_0, 4),
    Precision_Class_1 = round(precision_1, 4),
    Recall_Class_1 = round(recall_1, 4),
    F1_Class_1 = round(f1_1, 4),
    Minority_Correct = minority_correct,
    Minority_Total = minority_total,
    P_Value_Binomial = round(p_value, 6),
    Naive_Criteria_Count = naive_count,
    Is_Naive = is_naive,
    Verdict = ifelse(is_naive, "INGENUO", "REAL"),
    stringsAsFactors = FALSE
  )
}

evaluate_all_models <- function(results_list) {
  # results_list: lista con elementos list(name="ModelX", dataset="", target="",
  #                                        y_true=..., y_pred=...)

  all_results <- lapply(results_list, function(x) {
    evaluate_model_vs_naive(x$y_true, x$y_pred, x$name, x$dataset, x$target)
  })

  df_results <- bind_rows(all_results)
  df_results <- df_results %>% arrange(desc(Balanced_Accuracy))

  return(df_results)
}

# Ejemplo de uso:
# results_list <- list(
#   list(name = "XGBoost", dataset = "financial", target = "direction_next",
#        y_true = y_test, y_pred = pred_xgb),
#   list(name = "RandomForest", dataset = "financial", target = "direction_next",
#        y_true = y_test, y_pred = pred_rf)
# )
# df <- evaluate_all_models(results_list)
#
# # Filtrar modelos REALES
# reales <- df %>% filter(Verdict == "REAL")
#
# # Filtrar modelos INGENUOS
# ingenuos <- df %>% filter(Verdict == "INGENUO")
