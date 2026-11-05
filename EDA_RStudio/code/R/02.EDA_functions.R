# Función para detectar gaps en días de trading
detect_trading_gaps <- function(df, date_col = "date", index_col = "index") {
  df |>
    dplyr::arrange(!!sym(date_col)) |>
    dplyr::group_by(!!sym(index_col)) |>
    dplyr::mutate(
      dias_gap = as.numeric(difftime(!!sym(date_col), lag(!!sym(date_col)), units = "days"))
    ) |>
    dplyr::filter(dias_gap > 7) |> # Gaps > 1 semana (considerar festivos/fines de semana)
    dplyr::select(!!sym(index_col), !!sym(date_col), dias_gap) |>
    dplyr::arrange(desc(dias_gap))
}


# VISUALIZACIONES - VALIDACIÓN TEMPORAL DATOS FINANCIEROS


library(tidyverse)
library(lubridate)
library(scales)
library(patchwork)

# Cargar resultados de validación (ajusta según tu estructura)
# validation_results <- readRDS("output/RData/temporal_validation_results.rds")


# 1. TIMELINE DE COBERTURA TEMPORAL POR ÍNDICE


plot_temporal_coverage_indices <- function(temporal_range_df) {
  p <- temporal_range_df %>%
    mutate(index = fct_reorder(index, fecha_inicio)) %>%
    ggplot(aes(y = index)) +
    geom_segment(
      aes(x = fecha_inicio, xend = fecha_fin, yend = index),
      size = 2,
      color = "#2C3E50"
    ) +
    geom_point(aes(x = fecha_inicio), color = "#27AE60", size = 2) +
    geom_point(aes(x = fecha_fin), color = "#E74C3C", size = 2) +
    geom_text(
      aes(x = fecha_inicio, label = format(fecha_inicio, "%Y")),
      hjust = 1.2, size = 2, color = "#27AE60", fontface = "bold"
    ) +
    geom_text(
      aes(x = fecha_fin, label = format(fecha_fin, "%Y")),
      hjust = -0.2, size = 2, color = "#E74C3C", fontface = "bold"
    ) +
    geom_text(
      aes(
        x = fecha_inicio + (fecha_fin - fecha_inicio) / 2,
        label = paste0(años_datos, " años")
      ),
      vjust = -0.8, size = 2.5, color = "#34495E"
    ) +
    labs(
      title = "Cobertura Temporal por Índice Bursátil",
      subtitle = "Verde: Inicio | Rojo: Fin | Barra: Período disponible",
      x = "Fecha",
      y = NULL,
      caption = "Fuente: Yahoo Finance"
    ) +
    scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
    theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank()
    )

  return(p)
}


# 2. MAPA DE CALOR: COMPLETITUD DE DATOS POR ÍNDICE Y AÑO


plot_completeness_heatmap <- function(stocks_df) {
  # Preparar datos: observaciones por índice y año
  completeness_data <- stocks_df %>%
    mutate(year = year(date)) %>%
    group_by(index, year) %>%
    summarise(
      n_obs = n(),
      .groups = "drop"
    ) %>%
    group_by(index) %>%
    mutate(
      completeness_pct = (n_obs / 252) * 100, # 252 días trading aprox
      completeness_pct = pmin(completeness_pct, 100) # Cap at 100%
    )

  p <- ggplot(completeness_data, aes(x = year, y = index, fill = completeness_pct)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(
      aes(label = round(completeness_pct, 0)),
      size = 2.5,
      color = ifelse(completeness_data$completeness_pct > 70, "white", "black")
    ) +
    scale_fill_gradient2(
      low = "#E74C3C",
      mid = "#F39C12",
      high = "#27AE60",
      midpoint = 70,
      limits = c(0, 100),
      name = "Completitud (%)",
      breaks = seq(0, 100, 25)
    ) +
    scale_x_continuous(breaks = seq(
      min(completeness_data$year),
      max(completeness_data$year), 2
    )) +
    labs(
      title = "Mapa de Calor: Completitud de Datos por Año",
      subtitle = "Verde: >70% días trading | Naranja: 40-70% | Rojo: <40%",
      x = "Año",
      y = NULL,
      caption = "Base: 252 días de trading/año"
    ) +
    theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank(),
      legend.position = "right"
    )

  return(p)
}


# 3. DISTRIBUCIÓN DE AÑOS DE DATOS POR EMPRESA


plot_companies_distribution <- function(temporal_range) {
  p1 <- ggplot(temporal_range, aes(x = años_datos)) +
    geom_histogram(bins = 20, fill = "#3498DB", color = "white", alpha = 0.8) +
    geom_vline(
      xintercept = median(temporal_range$años_datos),
      linetype = "dashed",
      color = "#E74C3C",
      size = 1
    ) +
    annotate(
      "text",
      x = median(temporal_range$años_datos),
      y = Inf,
      label = sprintf("Mediana: %.1f años", median(temporal_range$años_datos)),
      vjust = 1.5,
      color = "#E74C3C",
      fontface = "bold"
    ) +
    labs(
      title = "Distribución de Años de Datos Disponibles",
      subtitle = "Empresas IBEX35",
      x = "Años de datos históricos",
      y = "Número de empresas"
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  # Box plot por sector
  # Si existe la columna 'sector', añade el boxplot
  if ("sector" %in% names(temporal_range)) {
    p2 <- ggplot(temporal_range, aes(x = años_datos, y = fct_reorder(sector, años_datos))) +
      geom_boxplot(fill = "#9B59B6", alpha = 0.6, outlier.color = "#E74C3C") +
      geom_point(alpha = 0.4, position = position_jitter(height = 0.2), color = "#34495E") +
      labs(
        title = "Años de Datos por Sector",
        x = "Años de datos históricos",
        y = NULL
      ) +
      theme_minimal(base_size = 10) +
      theme(plot.title = element_text(face = "bold"))

    return(p1 / p2) # combina los dos gráficos con patchwork
  } else {
    return(p1) # solo devuelve el histograma
  }
}

# 4. DETECCIÓN DE GAPS: VISUALIZACIÓN TEMPORAL


plot_gaps_timeline <- function(gaps_df, stocks_df, top_n = 10) {
  if (nrow(gaps_df) == 0) {
    return(
      ggplot() +
        annotate("text",
          x = 0.5, y = 0.5,
          label = "No se detectaron gaps significativos (>7 días)",
          size = 6, color = "#27AE60"
        ) +
        theme_void()
    )
  }

  # Top gaps
  top_gaps <- gaps_df %>%
    head(top_n) %>%
    mutate(label = paste0(index, "\n", dias_gap, " días"))

  # Obtener rango completo para cada índice
  date_ranges <- stocks_df %>%
    group_by(index) %>%
    summarise(
      min_date = min(date),
      max_date = max(date)
    )

  top_gaps <- top_gaps %>%
    left_join(date_ranges, by = "index")

  p <- ggplot(top_gaps) +
    geom_segment(
      aes(
        x = min_date, xend = max_date, y = fct_reorder(index, -dias_gap),
        yend = fct_reorder(index, -dias_gap)
      ),
      color = "gray80",
      size = 2
    ) +
    geom_point(
      aes(x = date, y = fct_reorder(index, -dias_gap), size = dias_gap),
      color = "#E74C3C",
      alpha = 0.7
    ) +
    geom_text(
      aes(x = date, y = fct_reorder(index, -dias_gap), label = paste(dias_gap, "d")),
      vjust = -1,
      size = 2,
      fontface = "bold"
    ) +
    scale_size_continuous(range = c(3, 10), name = "Días de gap") +
    scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
    labs(
      title = sprintf("Top %d Gaps Detectados en Series Temporales", top_n),
      subtitle = "Períodos sin datos >7 días | Tamaño = magnitud del gap",
      x = "Fecha del gap",
      y = NULL,
      caption = "Línea gris: período total disponible"
    ) +
    theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )

  return(p)
}


# 5. ANOMALÍAS: DASHBOARD DE CALIDAD DE DATOS


plot_anomalies_dashboard <- function(anomalies_price, anomalies_logic, na_analysis) {
  # Plot 1: Precios negativos/cero por índice
  p1 <- if (nrow(anomalies_price) > 0) {
    anomalies_price %>%
      count(index) %>%
      ggplot(aes(x = fct_reorder(index, n), y = n)) +
      geom_col(fill = "#E74C3C", alpha = 0.8) +
      geom_text(aes(label = n), hjust = -0.2, size = 2) +
      coord_flip() +
      labs(
        title = "Precios Anómalos (≤0)",
        x = NULL,
        y = "N° de observaciones"
      ) +
      theme_minimal(base_size = 10)
  } else {
    ggplot() +
      annotate("text",
        x = 0.5, y = 0.5, label = "✓ Sin precios anómalos",
        size = 4, color = "#27AE60"
      ) +
      theme_void()
  }

  # Plot 2: Inconsistencias lógicas
  p2 <- if (nrow(anomalies_logic) > 0) {
    anomalies_logic %>%
      count(index) %>%
      ggplot(aes(x = fct_reorder(index, n), y = n)) +
      geom_col(fill = "#F39C12", alpha = 0.8) +
      geom_text(aes(label = n), hjust = -0.2, size = 2) +
      coord_flip() +
      labs(
        title = "Inconsistencias Lógicas",
        subtitle = "(high<low, close fuera rango)",
        x = NULL,
        y = "N° de observaciones"
      ) +
      theme_minimal(base_size = 10)
  } else {
    ggplot() +
      annotate("text",
        x = 0.5, y = 0.5, label = "✓ Sin inconsistencias",
        size = 4, color = "#27AE60"
      ) +
      theme_void()
  }
  '
  # Plot 3: Valores faltantes (NAs)
  p3 <- if (nrow(na_analysis) > 0) {
    na_analysis %>%
      ggplot(aes(x = fct_reorder(index, pct_na_close), y = pct_na_close)) +
      geom_col(fill = "#9B59B6", alpha = 0.8) +
      geom_text(aes(label = sprintf("%.1f%%", pct_na_close)),
        hjust = -0.2, size = 2
      ) +
      coord_flip() +
      labs(
        title = "Valores Faltantes (NA)",
        subtitle = "% de observaciones con close=NA",
        x = NULL,
        y = "Porcentaje"
      ) +
      theme_minimal(base_size = 10)
  } else {
    ggplot() +
      annotate("text",
        x = 0.5, y = 0.5, label = "✓ Sin valores faltantes",
        size = 4, color = "#27AE60"
      ) +
      theme_void()
  }
'
  # Plot 4: Resumen general
  total_obs <- sum(na_analysis$n_total, na.rm = TRUE)
  if (total_obs == 0) total_obs <- nrow(anomalies_price) + nrow(anomalies_logic)

  quality_summary <- tibble(
    categoria = c("Precios anómalos", "Inconsistencias", "NAs en close"),
    n_casos = c(
      nrow(anomalies_price),
      nrow(anomalies_logic),
      sum(na_analysis$na_close, na.rm = TRUE)
    )
  ) %>%
    mutate(pct = n_casos / max(total_obs, 1) * 100)
  '

  p4 <- ggplot(quality_summary, aes(x = "", y = n_casos, fill = categoria)) +
    geom_col(width = 1, color = "white", size = 1) +
    coord_polar("y", start = 0) +
    scale_fill_manual(values = c("#E74C3C", "#F39C12", "#9B59B6")) +
    labs(title = "Distribución de Problemas", fill = NULL) +
    theme_void(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      legend.position = "bottom"
    )

  # Combinar plots
  layout <- "
  AABB
  CCDD
  "
'
  # Combinar plots
  layout <- "AABB"


  # combined <- p1 + p2 + p3 + p4 +
  combined <- p1 + p2 +
    plot_layout(design = layout) +
    plot_annotation(
      title = "Dashboard de Calidad de Datos",
      subtitle = "Detección de anomalías y problemas en series temporales",
      theme = theme(
        plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 10)
      )
    )

  return(combined)
}


# 6. FUNCIÓN MAESTRA: GENERAR TODAS LAS VISUALIZACIONES


generate_validation_plots <- function(validation_results, all_stocks_df,
                                      output_path = "output") {
  # Crear directorio si no existe
  if (!dir.exists(output_path)) {
    dir.create(output_path, recursive = TRUE)
  }

  cat("Generando visualizaciones de validación temporal...\n\n")

  # Plot 1: Timeline de cobertura
  cat("1. Timeline de cobertura temporal...\n")
  p1 <- plot_temporal_coverage_indices(validation_results$temporal_range)
  print(p1)
  save_plot(
    "temporal_coverage_timeline",
    p1
  )


  # Plot 2: Mapa de calor completitud
  cat("2. Mapa de calor de completitud...\n")
  p2 <- plot_completeness_heatmap(all_stocks_df)
  print(p2)
  save_plot(
    "completeness_heatmap",
    p2
  )


  # Plot 3: Distribución empresas
  cat("3. Distribución de datos ...\n")
  p3 <- plot_companies_distribution(validation_results$temporal_range)
  print(p3)
  save_plot(
    "distribution",
    p3
  )

  # Plot 4: Gaps timeline
  cat("4. Timeline de gaps detectados...\n")
  p4 <- plot_gaps_timeline(validation_results$gaps, all_stocks_df, top_n = 10)
  print(p4)
  save_plot(
    "gaps_timeline",
    p4
  )
  # Plot 5: Dashboard anomalías
  cat("5. Dashboard de anomalías...\n")
  p5 <- plot_anomalies_dashboard(
    validation_results$anomalies_price,
    validation_results$anomalies_logic,
    validation_results$na_analysis
  )
  print(p5)
  save_plot(
    "anomalies_dashboard",
    p5
  )

  cat("\n✓ Visualizaciones completadas y guardadas en:", output_path, "\n")

  # Retornar plots en lista
  return(list(
    temporal_coverage = p1,
    completeness_heatmap = p2,
    distribution = p3,
    gaps_timeline = p4,
    anomalies_dashboard = p5
  ))
}


# EJECUCIÓN


# Ejemplo de uso:
# plots <- generate_validation_plots(validation_results, all_stocks_df)
#
# # Ver plot individual
# print(plots$temporal_coverage)
