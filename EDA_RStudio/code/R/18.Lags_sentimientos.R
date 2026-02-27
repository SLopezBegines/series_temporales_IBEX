# CÁLCULO DE LAGS Y EFECTOS RETARDADOS DEL SENTIMIENTO ####

log_cat("\n=== ANÁLISIS DE LAGS - EFECTOS RETARDADOS ===\n")

library(data.table)
library(ggplot2)
library(patchwork)
library(forecast)


# 1. PREPARAR DATOS CON LAGS ####

log_cat("Calculando lags del sentimiento...\n")

# Asegurar ordenamiento temporal
daily_sentiment <- daily_sentiment[order(date)]

# Calcular lags (1, 2, 3, 5, 7 días)
lags_to_compute <- c(1, 2, 3, 5, 7, 14, 21, 30)

sentiment_with_lags <- daily_sentiment %>%
  mutate(
    # === LAGS DE TONE SCORE ===
    sentiment_lag1 = lag(mean_tone_score, 1),
    sentiment_lag2 = lag(mean_tone_score, 2),
    sentiment_lag3 = lag(mean_tone_score, 3),
    sentiment_lag5 = lag(mean_tone_score, 5),
    sentiment_lag7 = lag(mean_tone_score, 7),
    sentiment_lag14 = lag(mean_tone_score, 14),
    sentiment_lag21 = lag(mean_tone_score, 21),
    sentiment_lag30 = lag(mean_tone_score, 30),

    # === LAGS DE VOLATILIDAD ===
    volatility_lag1 = lag(sd_tone_score, 1),
    volatility_lag3 = lag(sd_tone_score, 3),
    volatility_lag7 = lag(sd_tone_score, 7),

    # === LAGS DE VOLUMEN ===
    volume_lag1 = lag(total_articles, 1),
    volume_lag3 = lag(total_articles, 3),
    volume_lag7 = lag(total_articles, 7),

    # === LAGS DE POLARITY ===
    polarity_lag1 = lag(mean_polarity, 1),
    polarity_lag3 = lag(mean_polarity, 3),
    polarity_lag7 = lag(mean_polarity, 7),

    # === MOVING AVERAGES (capturan tendencia) ===
    sentiment_ma3 = frollmean(mean_tone_score, 3, align = "right"),
    sentiment_ma7 = frollmean(mean_tone_score, 7, align = "right"),
    sentiment_ma14 = frollmean(mean_tone_score, 14, align = "right"),
    sentiment_ma30 = frollmean(mean_tone_score, 30, align = "right"),
    volatility_ma7 = frollmean(sd_tone_score, 7, align = "right"),
    volume_ma7 = frollmean(total_articles, 7, align = "right"),

    # === CAMBIOS (momentum/aceleración) ===
    sentiment_change = mean_tone_score - sentiment_lag1,
    sentiment_change_lag1 = lag(sentiment_change, 1),
    sentiment_accel = sentiment_change - sentiment_change_lag1,

    # === DIFERENCIAS CON MAs (desviación de tendencia) ===
    sentiment_dev_ma7 = mean_tone_score - sentiment_ma7,
    sentiment_dev_ma30 = mean_tone_score - sentiment_ma30,

    # === VARIABLES INDICADORAS ===
    extreme_sentiment = abs(mean_tone_score) > 2,
    extreme_sentiment_lag1 = lag(extreme_sentiment, 1),
    high_volatility = sd_tone_score > quantile(sd_tone_score, 0.75, na.rm = TRUE),
    high_volatility_lag1 = lag(high_volatility, 1)
  ) %>%
  as.data.table()

log_cat(sprintf(
  "Variables creadas: %d lags y transformaciones\n",
  ncol(sentiment_with_lags) - ncol(daily_sentiment)
))


# 2. AUTOCORRELACIÓN (ACF) ####

log_cat("\n=== ANÁLISIS DE AUTOCORRELACIÓN ===\n")

# Calcular ACF para tone score
acf_sentiment <- acf(sentiment_with_lags$mean_tone_score,
  lag.max = 30,
  na.action = na.pass,
  plot = FALSE
)

acf_data <- data.table(
  lag = 0:30,
  acf = as.numeric(acf_sentiment$acf)
)

log_cat("Autocorrelación del sentimiento:\n")
print(acf_data[lag %in% c(1, 2, 3, 5, 7, 14, 21, 30)])

# Calcular PACF (autocorrelación parcial)
pacf_sentiment <- pacf(sentiment_with_lags$mean_tone_score,
  lag.max = 30,
  na.action = na.pass,
  plot = FALSE
)

pacf_data <- data.table(
  lag = 1:30,
  pacf = as.numeric(pacf_sentiment$acf)
)


# 3. CORRELACIÓN ENTRE VARIABLE Y SUS LAGS ####

log_cat("\n=== CORRELACIÓN SENTIMIENTO vs LAGS ===\n")

lag_correlations <- sentiment_with_lags[, .(
  lag1 = cor(mean_tone_score, sentiment_lag1, use = "complete.obs"),
  lag2 = cor(mean_tone_score, sentiment_lag2, use = "complete.obs"),
  lag3 = cor(mean_tone_score, sentiment_lag3, use = "complete.obs"),
  lag5 = cor(mean_tone_score, sentiment_lag5, use = "complete.obs"),
  lag7 = cor(mean_tone_score, sentiment_lag7, use = "complete.obs"),
  lag14 = cor(mean_tone_score, sentiment_lag14, use = "complete.obs"),
  lag21 = cor(mean_tone_score, sentiment_lag21, use = "complete.obs"),
  lag30 = cor(mean_tone_score, sentiment_lag30, use = "complete.obs")
)]

lag_corr_long <- melt(lag_correlations,
  measure.vars = names(lag_correlations),
  variable.name = "Lag",
  value.name = "Correlation"
)
lag_corr_long[, Lag := as.integer(gsub("lag", "", Lag))]

log_cat("Correlaciones:\n")
print(lag_corr_long)


# 4. VISUALIZACIONES ####

log_cat("\n=== GENERANDO VISUALIZACIONES DE LAGS ===\n")

# 4.1 ACF Plot
log_cat("Gráfico 1: ACF (Autocorrelación)\n")

# Banda de confianza (95%)
conf_level <- qnorm(0.975) / sqrt(nrow(sentiment_with_lags))

p_acf <- ggplot(acf_data, aes(x = lag, y = acf)) +
  geom_hline(yintercept = 0, color = "black") +
  geom_hline(
    yintercept = c(-conf_level, conf_level),
    linetype = "dashed", color = "blue"
  ) +
  geom_segment(aes(xend = lag, yend = 0), color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2) +
  labs(
    title = "Autocorrelación del Sentimiento (ACF)",
    subtitle = "Líneas azules = banda de confianza 95%",
    x = "Lag (días)",
    y = "Autocorrelación"
  ) +
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10, hjust = 0.5),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    panel.grid.minor = element_blank()
  )

print(p_acf)
save_plot("ibex35_sentiment_acf", p_acf)

# 4.2 PACF Plot
log_cat("Gráfico 2: PACF (Autocorrelación Parcial)\n")

p_pacf <- ggplot(pacf_data, aes(x = lag, y = pacf)) +
  geom_hline(yintercept = 0, color = "black") +
  geom_hline(
    yintercept = c(-conf_level, conf_level),
    linetype = "dashed", color = "blue"
  ) +
  geom_segment(aes(xend = lag, yend = 0), color = "darkred", linewidth = 1) +
  geom_point(color = "darkred", size = 2) +
  labs(
    title = "Autocorrelación Parcial del Sentimiento (PACF)",
    subtitle = "Líneas azules = banda de confianza 95%",
    x = "Lag (días)",
    y = "Autocorrelación Parcial"
  ) +
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10, hjust = 0.5),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    panel.grid.minor = element_blank()
  )

print(p_pacf)
save_plot("ibex35_sentiment_pacf", p_pacf)

# 4.3 Correlación con lags
log_cat("Gráfico 3: Correlación sentimiento vs lags\n")

p_lag_corr <- ggplot(lag_corr_long, aes(x = Lag, y = Correlation)) +
  geom_hline(yintercept = 0, color = "gray50") +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_point(color = "steelblue", size = 3) +
  geom_text(aes(label = sprintf("%.3f", Correlation)),
    vjust = -1, size = 3
  ) +
  scale_x_continuous(breaks = c(1, 2, 3, 5, 7, 14, 21, 30)) +
  ylim(-0.2, 1) +
  labs(
    title = "Persistencia del Sentimiento",
    subtitle = "Correlación entre sentimiento actual y retardado",
    x = "Días de retraso (lag)",
    y = "Correlación con sentimiento actual"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10, hjust = 0.5),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    panel.grid.minor = element_blank()
  )

print(p_lag_corr)
save_plot("ibex35_sentiment_lag_correlation", p_lag_corr)

# 4.4 Comparación visual: actual vs lags
log_cat("Gráfico 4: Serie temporal - Actual vs Lags\n")

# Crear subset para visualización (últimos 180 días)
recent_data <- sentiment_with_lags[date >= max(date) - 180]

p_lags_series <- ggplot(recent_data, aes(x = date)) +
  geom_line(aes(y = mean_tone_score, color = "Actual"),
    alpha = 0.8, linewidth = 1
  ) +
  geom_line(aes(y = sentiment_lag1, color = "Lag 1"),
    alpha = 0.5, linewidth = 0.8
  ) +
  geom_line(aes(y = sentiment_lag7, color = "Lag 7"),
    alpha = 0.5, linewidth = 0.8
  ) +
  geom_line(aes(y = sentiment_ma7, color = "MA 7"),
    alpha = 0.8, linewidth = 1, linetype = "dashed"
  ) +
  scale_color_manual(values = c(
    "Actual" = "black",
    "Lag 1" = "steelblue",
    "Lag 7" = "darkgreen",
    "MA 7" = "red"
  )) +
  labs(
    title = "Sentimiento: Actual vs Lags y Media Móvil",
    subtitle = "Últimos 6 meses",
    x = "Fecha",
    y = "Tone Score",
    color = "Variable"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10, hjust = 0.5),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    legend.position = "top"
  )

print(p_lags_series)
save_plot("ibex35_sentiment_lags_timeseries", p_lags_series)

# 4.5 Scatter plots: Actual vs Lags
log_cat("Gráfico 5: Scatter plots - Actual vs Lags\n")

p_scatter_lag1 <- ggplot(
  sentiment_with_lags,
  aes(x = sentiment_lag1, y = mean_tone_score)
) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Lag 1 día",
    subtitle = sprintf("r = %.3f", lag_corr_long[Lag == 1, Correlation]),
    x = "Sentimiento (t-1)",
    y = "Sentimiento (t)"
  ) +
  theme_minimal(base_size = 10)

p_scatter_lag3 <- ggplot(
  sentiment_with_lags,
  aes(x = sentiment_lag3, y = mean_tone_score)
) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Lag 3 días",
    subtitle = sprintf("r = %.3f", lag_corr_long[Lag == 3, Correlation]),
    x = "Sentimiento (t-3)",
    y = "Sentimiento (t)"
  ) +
  theme_minimal(base_size = 10)

p_scatter_lag7 <- ggplot(
  sentiment_with_lags,
  aes(x = sentiment_lag7, y = mean_tone_score)
) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Lag 7 días",
    subtitle = sprintf("r = %.3f", lag_corr_long[Lag == 7, Correlation]),
    x = "Sentimiento (t-7)",
    y = "Sentimiento (t)"
  ) +
  theme_minimal(base_size = 10)

p_scatter_lag30 <- ggplot(
  sentiment_with_lags,
  aes(x = sentiment_lag30, y = mean_tone_score)
) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Lag 30 días",
    subtitle = sprintf("r = %.3f", lag_corr_long[Lag == 30, Correlation]),
    x = "Sentimiento (t-30)",
    y = "Sentimiento (t)"
  ) +
  theme_minimal(base_size = 10)

p_scatter_combined <- (p_scatter_lag1 | p_scatter_lag3) /
  (p_scatter_lag7 | p_scatter_lag30) +
  plot_annotation(
    title = "Relación entre Sentimiento Actual y Retardado",
    subtitle = "Línea roja = regresión lineal | Línea punteada = identidad (sin cambio)",
    theme = theme(
      plot.title = element_text(size = 10, hjust = 0.5),
      plot.subtitle = element_text(size = 8),
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 8)
    )
  )

print(p_scatter_combined)
save_plot("ibex35_sentiment_lags_scatter", p_scatter_combined)

# 4.6 Momentum y aceleración
log_cat("Gráfico 6: Momentum y Aceleración del sentimiento\n")

p_momentum <- ggplot(recent_data, aes(x = date)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(aes(y = sentiment_change, color = "Cambio (momentum)"),
    alpha = 0.7, linewidth = 0.8
  ) +
  geom_line(aes(y = sentiment_accel, color = "Aceleración"),
    alpha = 0.7, linewidth = 0.8
  ) +
  scale_color_manual(values = c(
    "Cambio (momentum)" = "steelblue",
    "Aceleración" = "darkred"
  )) +
  labs(
    title = "Momentum y Aceleración del Sentimiento",
    subtitle = "Últimos 6 meses | Cambio = Δ sentimiento | Aceleración = Δ(Δ sentimiento)",
    x = "Fecha",
    y = "Valor",
    color = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10, hjust = 0.5),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    legend.position = "top"
  )

print(p_momentum)
save_plot("ibex35_sentiment_momentum_accel", p_momentum)

# 4.7 Desviación de la tendencia (MA)
log_cat("Gráfico 7: Desviación de la media móvil\n")

p_deviation <- ggplot(recent_data, aes(x = date)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(aes(y = sentiment_dev_ma7, color = "Desviación MA7"),
    alpha = 0.7, linewidth = 0.8
  ) +
  geom_line(aes(y = sentiment_dev_ma30, color = "Desviación MA30"),
    alpha = 0.7, linewidth = 0.8
  ) +
  scale_color_manual(values = c(
    "Desviación MA7" = "steelblue",
    "Desviación MA30" = "darkgreen"
  )) +
  labs(
    title = "Desviación del Sentimiento respecto a su Tendencia",
    subtitle = "Valores >0 = sentimiento por encima de tendencia | <0 = por debajo",
    x = "Fecha",
    y = "Desviación (puntos)",
    color = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10, hjust = 0.5),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    legend.position = "top"
  )

print(p_deviation)
save_plot("ibex35_sentiment_deviation_ma", p_deviation)


# 5. ESTADÍSTICAS DESCRIPTIVAS DE LAGS

log_cat("\n=== ESTADÍSTICAS DESCRIPTIVAS ===\n")

stats_lags <- sentiment_with_lags[, .(
  Variable = c(
    "Tone Score", "Lag 1", "Lag 7", "MA 7", "MA 30",
    "Cambio", "Aceleración"
  ),
  Media = c(
    mean(mean_tone_score, na.rm = TRUE),
    mean(sentiment_lag1, na.rm = TRUE),
    mean(sentiment_lag7, na.rm = TRUE),
    mean(sentiment_ma7, na.rm = TRUE),
    mean(sentiment_ma30, na.rm = TRUE),
    mean(sentiment_change, na.rm = TRUE),
    mean(sentiment_accel, na.rm = TRUE)
  ),
  SD = c(
    sd(mean_tone_score, na.rm = TRUE),
    sd(sentiment_lag1, na.rm = TRUE),
    sd(sentiment_lag7, na.rm = TRUE),
    sd(sentiment_ma7, na.rm = TRUE),
    sd(sentiment_ma30, na.rm = TRUE),
    sd(sentiment_change, na.rm = TRUE),
    sd(sentiment_accel, na.rm = TRUE)
  ),
  Min = c(
    min(mean_tone_score, na.rm = TRUE),
    min(sentiment_lag1, na.rm = TRUE),
    min(sentiment_lag7, na.rm = TRUE),
    min(sentiment_ma7, na.rm = TRUE),
    min(sentiment_ma30, na.rm = TRUE),
    min(sentiment_change, na.rm = TRUE),
    min(sentiment_accel, na.rm = TRUE)
  ),
  Max = c(
    max(mean_tone_score, na.rm = TRUE),
    max(sentiment_lag1, na.rm = TRUE),
    max(sentiment_lag7, na.rm = TRUE),
    max(sentiment_ma7, na.rm = TRUE),
    max(sentiment_ma30, na.rm = TRUE),
    max(sentiment_change, na.rm = TRUE),
    max(sentiment_accel, na.rm = TRUE)
  )
)]

log_cat("Estadísticas de variables con lags:\n")
print(stats_lags)


# 6. GUARDAR RESULTADOS

log_cat("\n=== GUARDANDO RESULTADOS ===\n")

# Guardar dataset completo con lags
write_parquet(
  sentiment_with_lags,
  file.path(output_dir, "ibex35_daily_sentiment_with_lags.parquet")
)

# Guardar correlaciones de lags
fwrite(
  lag_corr_long,
  file.path(output_dir, "ibex35_sentiment_lag_correlations.csv")
)

# Guardar ACF/PACF
fwrite(
  acf_data,
  file.path(output_dir, "ibex35_sentiment_acf.csv")
)
fwrite(
  pacf_data,
  file.path(output_dir, "ibex35_sentiment_pacf.csv")
)

log_cat("\n✓ Análisis de lags completado\n")
log_cat("Archivos generados:\n")
log_cat("  - ibex35_daily_sentiment_with_lags.parquet (dataset completo)\n")
log_cat("  - ibex35_sentiment_lag_correlations.csv\n")
log_cat("  - ibex35_sentiment_acf.csv\n")
log_cat("  - ibex35_sentiment_pacf.csv\n")
log_cat("  - 7 gráficos PNG\n")

log_cat("\n=== INTERPRETACIÓN ===\n")
log_cat("ACF: Mide correlación entre serie y sus lags\n")
log_cat("  - ACF alto en lag 1-7: persistencia del sentimiento\n")
log_cat("  - ACF decae lento: tendencia o estacionalidad\n")
log_cat("PACF: Correlación controlando lags intermedios\n")
log_cat("  - PACF significativo solo en lag 1: proceso AR(1)\n")
log_cat("  - PACF significativo hasta lag p: proceso AR(p)\n")
log_cat("\nVARIABLES ÚTILES PARA MODELO FINANCIERO:\n")
log_cat("  - sentiment_lag1, sentiment_lag7: efecto retardado\n")
log_cat("  - sentiment_ma7, sentiment_ma30: tendencia\n")
log_cat("  - sentiment_change: momentum\n")
log_cat("  - sentiment_dev_ma7: desviación de tendencia\n")
log_cat("  - volatility_lag1, volatility_lag7: incertidumbre previa\n")
