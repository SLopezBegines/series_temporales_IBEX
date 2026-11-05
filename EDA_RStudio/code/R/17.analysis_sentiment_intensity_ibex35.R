# GRÁFICOS DE FRECUENCIA - VARIABLES CATEGORIZADAS ####

log_cat("\n=== GRÁFICOS DE FRECUENCIA POR CATEGORÍAS ===\n")

# Función para categorizar variables
categorize_sentiment <- function(df) {
  df %>%
    mutate(
      # Categorías de mean_tone_score
      tone_category = case_when(
        mean_tone_score < -2 ~ "Muy negativo (<-2)",
        mean_tone_score >= -2 & mean_tone_score < 0 ~ "Ligeramente negativo (-2,0)",
        mean_tone_score >= 0 & mean_tone_score < 2 ~ "Ligeramente positivo (0,2)",
        mean_tone_score >= 2 ~ "Muy positivo (>2)"
      ) %>% factor(levels = c(
        "Muy negativo (<-2)",
        "Ligeramente negativo (-2,0)",
        "Ligeramente positivo (0,2)",
        "Muy positivo (>2)"
      )),

      # Categorías de mean_polarity
      polarity_category = case_when(
        mean_polarity < 5 ~ "Tranquilo (<5)",
        mean_polarity >= 5 & mean_polarity <= 10 ~ "Normal (5-10)",
        mean_polarity > 10 ~ "Alta carga (>10)"
      ) %>% factor(levels = c(
        "Tranquilo (<5)",
        "Normal (5-10)",
        "Alta carga (>10)"
      )),

      # Categorías de sd_tone_score
      volatility_category = case_when(
        sd_tone_score < 3 ~ "Baja (<3): Consenso",
        sd_tone_score >= 3 & sd_tone_score <= 5 ~ "Media (3-5)",
        sd_tone_score > 5 ~ "Alta (>5): Noticias mixtas"
      ) %>% factor(levels = c(
        "Baja (<3): Consenso",
        "Media (3-5)",
        "Alta (>5): Noticias mixtas"
      ))
    )
}

# Aplicar categorización
daily_cat <- categorize_sentiment(daily_sentiment)
monthly_cat <- categorize_sentiment(monthly_sentiment)
yearly_cat <- categorize_sentiment(yearly_sentiment)


# GRÁFICOS DIARIOS ####

log_cat("Generando gráficos de frecuencia diaria...\n")

# Tone Score - Diario
p_tone_daily <- daily_cat %>%
  count(tone_category) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = tone_category, y = n, fill = tone_category)) +
  geom_col(alpha = 0.8, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
    vjust = -0.5, size = 2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Muy negativo (<-2)" = "#D73027",
    "Ligeramente negativo (-2,0)" = "#FC8D59",
    "Ligeramente positivo (0,2)" = "#91BFDB",
    "Muy positivo (>2)" = "#4575B4"
  )) +
  labs(
    title = "Distribución de Sentimiento \nDiario",
    subtitle = sprintf("Total días: %d", nrow(daily_cat)),
    x = "Categoría Tone Score",
    y = "Frecuencia (días)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, max(table(daily_cat$tone_category)) * 1.15)

# Polarity - Diario
p_polarity_daily <- daily_cat %>%
  count(polarity_category) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = polarity_category, y = n, fill = polarity_category)) +
  geom_col(alpha = 0.8, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
    vjust = -0.5, size = 2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Tranquilo (<5)" = "#91CF60",
    "Normal (5-10)" = "#FEE08B",
    "Alta carga (>10)" = "#FC8D59"
  )) +
  labs(
    title = "Distribución de Intensidad Emocional \nDiario",
    subtitle = sprintf("Total días: %d", nrow(daily_cat)),
    x = "Categoría Polarity",
    y = "Frecuencia (días)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, max(table(daily_cat$polarity_category)) * 1.15)

# Volatility - Diario
p_volatility_daily <- daily_cat %>%
  count(volatility_category) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = volatility_category, y = n, fill = volatility_category)) +
  geom_col(alpha = 0.8, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
    vjust = -0.5, size = 2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Baja (<3): Consenso" = "#66C2A5",
    "Media (3-5)" = "#FFD92F",
    "Alta (>5): Noticias mixtas" = "#E78AC3"
  )) +
  labs(
    title = "Distribución de Volatilidad \nDiario",
    subtitle = sprintf("Total días: %d", nrow(daily_cat)),
    x = "Categoría SD Tone",
    y = "Frecuencia (días)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, max(table(daily_cat$volatility_category)) * 1.15)

# Panel combinado diario
p_daily_combined <- (p_tone_daily | p_polarity_daily | p_volatility_daily) +
  plot_annotation(
    title = "Distribuciones de Sentimiento IBEX35 \nNivel Diario",
    theme = theme(
      plot.title = element_text(size = 10, hjust = 0.5),
      plot.subtitle = element_text(size = 8),
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 8)
    )
  )

print(p_daily_combined)
save_plot("ibex35_sentiment_freq_daily", p_daily_combined)


# GRÁFICOS MENSUALES

log_cat("Generando gráficos de frecuencia mensual...\n")

# Tone Score - Mensual
p_tone_monthly <- monthly_cat %>%
  count(tone_category) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = tone_category, y = n, fill = tone_category)) +
  geom_col(alpha = 0.8, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
    vjust = -0.5, size = 2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Muy negativo (<-2)" = "#D73027",
    "Ligeramente negativo (-2,0)" = "#FC8D59",
    "Ligeramente positivo (0,2)" = "#91BFDB",
    "Muy positivo (>2)" = "#4575B4"
  )) +
  labs(
    title = "Distribución de Sentimiento \nMensual",
    subtitle = sprintf("Total meses: %d", nrow(monthly_cat)),
    x = "Categoría Tone Score",
    y = "Frecuencia (meses)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, max(table(monthly_cat$tone_category)) * 1.15)

# Polarity - Mensual
p_polarity_monthly <- monthly_cat %>%
  count(polarity_category) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = polarity_category, y = n, fill = polarity_category)) +
  geom_col(alpha = 0.8, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
    vjust = -0.5, size = 2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Tranquilo (<5)" = "#91CF60",
    "Normal (5-10)" = "#FEE08B",
    "Alta carga (>10)" = "#FC8D59"
  )) +
  labs(
    title = "Distribución de Intensidad Emocional \nMensual",
    subtitle = sprintf("Total meses: %d", nrow(monthly_cat)),
    x = "Categoría Polarity",
    y = "Frecuencia (meses)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, max(table(monthly_cat$polarity_category)) * 1.15)

# Volatility - Mensual
p_volatility_monthly <- monthly_cat %>%
  count(volatility_category) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = volatility_category, y = n, fill = volatility_category)) +
  geom_col(alpha = 0.8, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
    vjust = -0.5, size = 2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Baja (<3): Consenso" = "#66C2A5",
    "Media (3-5)" = "#FFD92F",
    "Alta (>5): Noticias mixtas" = "#E78AC3"
  )) +
  labs(
    title = "Distribución de Volatilidad \nMensual",
    subtitle = sprintf("Total meses: %d", nrow(monthly_cat)),
    x = "Categoría SD Tone",
    y = "Frecuencia (meses)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, max(table(monthly_cat$volatility_category)) * 1.15)

# Panel combinado mensual
p_monthly_combined <- (p_tone_monthly | p_polarity_monthly | p_volatility_monthly) +
  plot_annotation(
    title = "Distribuciones de Sentimiento IBEX35 \nNivel Mensual",
    theme = theme(
      plot.title = element_text(size = 10, hjust = 0.5),
      plot.subtitle = element_text(size = 8),
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 8)
    )
  )

print(p_monthly_combined)
save_plot("ibex35_sentiment_freq_monthly", p_monthly_combined)


# GRÁFICOS ANUALES

log_cat("Generando gráficos de frecuencia anual...\n")

# Tone Score - Anual
p_tone_yearly <- yearly_cat %>%
  count(tone_category) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = tone_category, y = n, fill = tone_category)) +
  geom_col(alpha = 0.8, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
    vjust = -0.5, size = 2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Muy negativo (<-2)" = "#D73027",
    "Ligeramente negativo (-2,0)" = "#FC8D59",
    "Ligeramente positivo (0,2)" = "#91BFDB",
    "Muy positivo (>2)" = "#4575B4"
  )) +
  labs(
    title = "Distribución de Sentimiento \nAnual",
    subtitle = sprintf("Total años: %d", nrow(yearly_cat)),
    x = "Categoría Tone Score",
    y = "Frecuencia (años)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, max(table(yearly_cat$tone_category)) * 1.2)

# Polarity - Anual
p_polarity_yearly <- yearly_cat %>%
  count(polarity_category) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = polarity_category, y = n, fill = polarity_category)) +
  geom_col(alpha = 0.8, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
    vjust = -0.5, size = 2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Tranquilo (<5)" = "#91CF60",
    "Normal (5-10)" = "#FEE08B",
    "Alta carga (>10)" = "#FC8D59"
  )) +
  labs(
    title = "Distribución de Intensidad Emocional \nAnual",
    subtitle = sprintf("Total años: %d", nrow(yearly_cat)),
    x = "Categoría Polarity",
    y = "Frecuencia (años)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, max(table(yearly_cat$polarity_category)) * 1.2)

# Volatility - Anual
p_volatility_yearly <- yearly_cat %>%
  count(volatility_category) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = volatility_category, y = n, fill = volatility_category)) +
  geom_col(alpha = 0.8, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
    vjust = -0.5, size = 2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Baja (<3): Consenso" = "#66C2A5",
    "Media (3-5)" = "#FFD92F",
    "Alta (>5): Noticias mixtas" = "#E78AC3"
  )) +
  labs(
    title = "Distribución de Volatilidad \nAnual",
    subtitle = sprintf("Total años: %d", nrow(yearly_cat)),
    x = "Categoría SD Tone",
    y = "Frecuencia (años)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, max(table(yearly_cat$volatility_category)) * 1.2)

# Panel combinado anual
p_yearly_combined <- (p_tone_yearly | p_polarity_yearly | p_volatility_yearly) +
  plot_annotation(
    title = "Distribuciones de Sentimiento IBEX35 \nNivel Anual",
    theme = theme(
      plot.title = element_text(size = 10, hjust = 0.5),
      plot.subtitle = element_text(size = 8),
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 8)
    )
  )

print(p_yearly_combined)
save_plot("ibex35_sentiment_freq_yearly", p_yearly_combined)


# RESUMEN ESTADÍSTICO

log_cat("\n=== RESUMEN ESTADÍSTICO POR CATEGORÍAS ===\n")

log_cat("\nDIARIO:\n")
log_cat("Tone Score:\n")
print(table(daily_cat$tone_category))
log_cat("\nPolarity:\n")
print(table(daily_cat$polarity_category))
log_cat("\nVolatilidad:\n")
print(table(daily_cat$volatility_category))

log_cat("\n\nMENSUAL:\n")
log_cat("Tone Score:\n")
print(table(monthly_cat$tone_category))
log_cat("\nPolarity:\n")
print(table(monthly_cat$polarity_category))
log_cat("\nVolatilidad:\n")
print(table(monthly_cat$volatility_category))

log_cat("\n\nANUAL:\n")
log_cat("Tone Score:\n")
print(table(yearly_cat$tone_category))
log_cat("\nPolarity:\n")
print(table(yearly_cat$polarity_category))
log_cat("\nVolatilidad:\n")
print(table(yearly_cat$volatility_category))

log_cat("\n✓ Gráficos de frecuencia generados\n")
