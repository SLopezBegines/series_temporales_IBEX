# 03_sentiment_analysis_parquet.R
# Análisis completo de sentimientos con todos los campos TONE

library(arrow)
library(dplyr)
library(data.table)
library(ggplot2)
library(lubridate)
library(DBI)
library(duckdb)
library(patchwork)

# CONFIGURACIÓN DE LOGGING ####
log_cat <- function(...) {
  args <- list(...)

  for (arg in args) {
    if (is.data.frame(arg)) {
      output <- capture.output(print(arg))
      cat(output, sep = "\n")
      cat(output, sep = "\n", file = log_file, append = TRUE)
    } else {
      msg <- as.character(arg)
      cat(msg, "\n")
      cat(msg, "\n", file = log_file, append = TRUE)
    }
  }
}

# input_parquet <- "data/gdelt_filtered/gdelt_ibex35_filtered.parquet"
# output_dir <- "data/gdelt_filtered/analysis_ibex35"

# Crear directorio de salida si no existe
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

log_cat("=== ANÁLISIS DE SENTIMIENTOS IBEX35 - COMPLETO ===\n\n")

con <- dbConnect(duckdb::duckdb())

# ESTADÍSTICAS BÁSICAS - TODOS LOS CAMPOS TONE ####
log_cat("=== ESTADÍSTICAS BÁSICAS - 6 CAMPOS TONE ===\n")
stats <- dbGetQuery(con, sprintf("
  SELECT
    COUNT(*) as total_records,
    COUNT(DISTINCT strptime(CAST(DATE AS VARCHAR), '%%Y%%m%%d')) as unique_days,
    SUM(NUMARTS) as total_articles,
    MIN(strptime(CAST(DATE AS VARCHAR), '%%Y%%m%%d')) as min_date,
    MAX(strptime(CAST(DATE AS VARCHAR), '%%Y%%m%%d')) as max_date,
    -- Tone Score (campo 1)
    AVG(CAST(string_split(TONE, ',')[1] AS DOUBLE)) as avg_tone_score,
    STDDEV(CAST(string_split(TONE, ',')[1] AS DOUBLE)) as sd_tone_score,
    MIN(CAST(string_split(TONE, ',')[1] AS DOUBLE)) as min_tone_score,
    MAX(CAST(string_split(TONE, ',')[1] AS DOUBLE)) as max_tone_score,
    -- Positive (campo 2)
    AVG(CAST(string_split(TONE, ',')[2] AS DOUBLE)) as avg_positive,
    STDDEV(CAST(string_split(TONE, ',')[2] AS DOUBLE)) as sd_positive,
    -- Negative (campo 3)
    AVG(CAST(string_split(TONE, ',')[3] AS DOUBLE)) as avg_negative,
    STDDEV(CAST(string_split(TONE, ',')[3] AS DOUBLE)) as sd_negative,
    -- Polarity (campo 4)
    AVG(CAST(string_split(TONE, ',')[4] AS DOUBLE)) as avg_polarity,
    STDDEV(CAST(string_split(TONE, ',')[4] AS DOUBLE)) as sd_polarity,
    -- Activity (campo 5)
    AVG(CAST(string_split(TONE, ',')[5] AS DOUBLE)) as avg_activity,
    STDDEV(CAST(string_split(TONE, ',')[5] AS DOUBLE)) as sd_activity,
    -- Self-reference (campo 6)
    AVG(CAST(string_split(TONE, ',')[6] AS DOUBLE)) as avg_selfref,
    STDDEV(CAST(string_split(TONE, ',')[6] AS DOUBLE)) as sd_selfref
  FROM read_parquet('%s')
  WHERE TONE IS NOT NULL AND TONE != ''
", input_parquet))
log_cat(stats)

## Distribución de sentimientos ####
log_cat("\n=== DISTRIBUCIÓN DE SENTIMIENTOS ===\n")
dist <- dbGetQuery(con, sprintf("
  WITH tones AS (
    SELECT CAST(string_split(TONE, ',')[1] AS DOUBLE) as tone_val
    FROM read_parquet('%s')
    WHERE TONE IS NOT NULL AND TONE != ''
  )
  SELECT
    SUM(CASE WHEN tone_val < -1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_negative,
    SUM(CASE WHEN tone_val >= -1 AND tone_val <= 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_neutral,
    SUM(CASE WHEN tone_val > 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_positive
  FROM tones
", input_parquet))
log_cat(sprintf("Negativo (<-1):  %.1f%%\n", dist$pct_negative))
log_cat(sprintf("Neutral (-1,1):  %.1f%%\n", dist$pct_neutral))
log_cat(sprintf("Positivo (>1):   %.1f%%\n", dist$pct_positive))

log_cat("\n=== COBERTURA TEMPORAL ===\n")
log_cat("Período:", paste(stats$min_date, "a", stats$max_date), "\n")
log_cat("Días únicos:", format(stats$unique_days, big.mark = ","), "\n")
log_cat("Total artículos:", format(stats$total_articles, big.mark = ","), "\n")


# AGREGACIONES TEMPORALES - TODOS LOS CAMPOS####
## Diarias ####
log_cat("\n=== AGREGACIONES TEMPORALES ===\n")

log_cat("Agregación diaria...\n")
daily_sentiment <- dbGetQuery(con, sprintf("
  SELECT
    strptime(CAST(DATE AS VARCHAR), '%%Y%%m%%d') as date,
    -- Tone Score
    AVG(CAST(string_split(TONE, ',')[1] AS DOUBLE)) as mean_tone_score,
    MEDIAN(CAST(string_split(TONE, ',')[1] AS DOUBLE)) as median_tone_score,
    STDDEV(CAST(string_split(TONE, ',')[1] AS DOUBLE)) as sd_tone_score,
    MIN(CAST(string_split(TONE, ',')[1] AS DOUBLE)) as min_tone_score,
    MAX(CAST(string_split(TONE, ',')[1] AS DOUBLE)) as max_tone_score,
    -- Positive
    AVG(CAST(string_split(TONE, ',')[2] AS DOUBLE)) as mean_positive,
    STDDEV(CAST(string_split(TONE, ',')[2] AS DOUBLE)) as sd_positive,
    -- Negative
    AVG(CAST(string_split(TONE, ',')[3] AS DOUBLE)) as mean_negative,
    STDDEV(CAST(string_split(TONE, ',')[3] AS DOUBLE)) as sd_negative,
    -- Polarity
    AVG(CAST(string_split(TONE, ',')[4] AS DOUBLE)) as mean_polarity,
    STDDEV(CAST(string_split(TONE, ',')[4] AS DOUBLE)) as sd_polarity,
    -- Activity
    AVG(CAST(string_split(TONE, ',')[5] AS DOUBLE)) as mean_activity,
    STDDEV(CAST(string_split(TONE, ',')[5] AS DOUBLE)) as sd_activity,
    -- Self-reference
    AVG(CAST(string_split(TONE, ',')[6] AS DOUBLE)) as mean_selfref,
    STDDEV(CAST(string_split(TONE, ',')[6] AS DOUBLE)) as sd_selfref,
    -- Contadores
    COUNT(*) as total_records,
    SUM(NUMARTS) as total_articles
  FROM read_parquet('%s')
  WHERE TONE IS NOT NULL AND TONE != ''
  GROUP BY DATE
  ORDER BY DATE
", input_parquet)) %>% as.data.table()


## Mensuales ####
log_cat("Agregación mensual...\n")
monthly_sentiment <- dbGetQuery(con, sprintf("
  WITH parsed AS (
    SELECT
      CAST(substr(CAST(DATE AS VARCHAR), 1, 4) AS INTEGER) AS Year,
      CAST(substr(CAST(DATE AS VARCHAR), 5, 2) AS INTEGER) AS Month,
      CAST(string_split(TONE, ',')[1] AS DOUBLE) as tone_score,
      CAST(string_split(TONE, ',')[2] AS DOUBLE) as positive,
      CAST(string_split(TONE, ',')[3] AS DOUBLE) as negative,
      CAST(string_split(TONE, ',')[4] AS DOUBLE) as polarity,
      CAST(string_split(TONE, ',')[5] AS DOUBLE) as activity,
      CAST(string_split(TONE, ',')[6] AS DOUBLE) as selfref,
      NUMARTS
    FROM read_parquet('%s')
    WHERE TONE IS NOT NULL AND TONE != ''
  )
  SELECT
    make_date(Year, Month, 1) as YearMonth,
    AVG(tone_score) as mean_tone_score,
    STDDEV(tone_score) as sd_tone_score,
    AVG(positive) as mean_positive,
    AVG(negative) as mean_negative,
    AVG(polarity) as mean_polarity,
    AVG(activity) as mean_activity,
    AVG(selfref) as mean_selfref,
    COUNT(*) as total_records,
    SUM(NUMARTS) as total_articles
  FROM parsed
  GROUP BY Year, Month
  ORDER BY Year, Month
", input_parquet)) %>% as.data.table()
## Anuales ####
log_cat("Agregación anual...\n")
yearly_sentiment <- dbGetQuery(con, sprintf("
  WITH parsed AS (
    SELECT
      CAST(substr(CAST(DATE AS VARCHAR), 1, 4) AS INTEGER) AS Year,
      CAST(string_split(TONE, ',')[1] AS DOUBLE) as tone_score,
      CAST(string_split(TONE, ',')[2] AS DOUBLE) as positive,
      CAST(string_split(TONE, ',')[3] AS DOUBLE) as negative,
      CAST(string_split(TONE, ',')[4] AS DOUBLE) as polarity,
      CAST(string_split(TONE, ',')[5] AS DOUBLE) as activity,
      CAST(string_split(TONE, ',')[6] AS DOUBLE) as selfref,
      NUMARTS
    FROM read_parquet('%s')
    WHERE TONE IS NOT NULL AND TONE != ''
  )
  SELECT
    Year,
    AVG(tone_score) as mean_tone_score,
    STDDEV(tone_score) as sd_tone_score,
    AVG(positive) as mean_positive,
    AVG(negative) as mean_negative,
    AVG(polarity) as mean_polarity,
    AVG(activity) as mean_activity,
    AVG(selfref) as mean_selfref,
    COUNT(*) as total_records,
    SUM(NUMARTS) as total_articles
  FROM parsed
  GROUP BY Year
  ORDER BY Year
", input_parquet)) %>% as.data.table()

write_parquet(daily_sentiment, file.path(output_dir, "ibex35_daily_sentiment.parquet"))
write_parquet(monthly_sentiment, file.path(output_dir, "ibex35_monthly_sentiment.parquet"))
write_parquet(yearly_sentiment, file.path(output_dir, "ibex35_yearly_sentiment.parquet"))


# TOP FUENTES ####

log_cat("\n=== TOP 10 FUENTES ===\n")
top_sources <- dbGetQuery(con, sprintf("
  SELECT
    SOURCES,
    COUNT(*) as records,
    SUM(NUMARTS) as articles,
    AVG(CAST(string_split(TONE, ',')[1] AS DOUBLE)) as mean_tone_score
  FROM read_parquet('%s')
  WHERE TONE IS NOT NULL AND TONE != ''
  GROUP BY SOURCES
  ORDER BY records DESC
  LIMIT 10
", input_parquet)) %>% as.data.table()
log_cat(top_sources)
write_parquet(top_sources, file.path(output_dir, "ibex35_top_sources.parquet"))


# EVENTOS EXTREMOS ####

log_cat("\n=== DÍAS MÁS NEGATIVOS ===\n")
most_negative <- daily_sentiment[order(mean_tone_score)][1:5]
log_cat(most_negative[, .(date, mean_tone_score, sd_tone_score, total_articles)])

log_cat("\n=== DÍAS MÁS POSITIVOS ===\n")
most_positive <- daily_sentiment[order(-mean_tone_score)][1:5]
log_cat(most_positive[, .(date, mean_tone_score, sd_tone_score, total_articles)])

log_cat("\n=== DÍAS MÁS VOLÁTILES (mayor sd_tone) ===\n")
most_volatile <- daily_sentiment[order(-sd_tone_score)][1:5]
log_cat(most_volatile[, .(date, mean_tone_score, sd_tone_score, total_articles)])


# TEMAS Y ORGANIZACIONES ####

log_cat("\n=== TEMAS ===\n")
theme_freq <- dbGetQuery(con, sprintf("
  WITH theme_split AS (
    SELECT unnest(string_split(THEMES, ';')) as theme
    FROM read_parquet('%s')
    WHERE THEMES IS NOT NULL AND THEMES != ''
  )
  SELECT theme, COUNT(*) as N
  FROM theme_split
  WHERE theme IS NOT NULL AND theme != ''
  GROUP BY theme
  ORDER BY N DESC
", input_parquet)) %>% as.data.table()

log_cat("Temas únicos:", format(nrow(theme_freq), big.mark = ","), "\n")
print(theme_freq[1:20])
reactViewTable(theme_freq)
write_parquet(theme_freq, file.path(output_dir, "ibex35_themes_frequency.parquet"))

log_cat("\n=== ORGANIZACIONES ===\n")
org_freq <- dbGetQuery(con, sprintf("
  WITH org_split AS (
    SELECT unnest(string_split(ORGANIZATIONS, ';')) as organization
    FROM read_parquet('%s')
    WHERE ORGANIZATIONS IS NOT NULL AND ORGANIZATIONS != ''
  )
  SELECT organization, COUNT(*) as N
  FROM org_split
  WHERE organization IS NOT NULL AND organization != ''
  GROUP BY organization
  ORDER BY N DESC
", input_parquet)) %>% as.data.table()

log_cat("Organizaciones únicas:", format(nrow(org_freq), big.mark = ","), "\n")
print(org_freq[1:20])

reactViewTable(org_freq)
write_parquet(org_freq, file.path(output_dir, "ibex35_organizations_frequency.parquet"))


# VISUALIZACIONES ####

log_cat("\n=== GENERANDO VISUALIZACIONES ===\n")

daily_sentiment[, date := as.Date(date)]
monthly_sentiment[, YearMonth := as.Date(YearMonth)]

## 1. TONE SCORE - Serie temporal con volatilidad ####
log_cat("Gráfico 1: Tone Score + Volatilidad\n")
p1 <- ggplot(daily_sentiment, aes(x = date)) +
  geom_ribbon(
    aes(
      ymin = mean_tone_score - sd_tone_score,
      ymax = mean_tone_score + sd_tone_score
    ),
    fill = "steelblue", alpha = 0.2
  ) +
  geom_line(aes(y = mean_tone_score), color = "steelblue", size = 0.8) +
  geom_smooth(aes(y = mean_tone_score),
    method = "loess",
    color = "red", se = FALSE, size = 0.6
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Tone Score IBEX35 - Serie Temporal",
    subtitle = "Línea azul: media diaria | Área: ±1 SD (volatilidad)",
    x = NULL, y = "Tone Score"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8)
  )
print(p1)
save_plot("ibex35_tone_score_volatility", p1)

## 2. PANEL: 6 campos TONE en diario ####
log_cat("Gráfico 2: Panel 6 campos TONE\n")
p2a <- ggplot(daily_sentiment, aes(x = date, y = mean_tone_score)) +
  geom_line(color = "steelblue", alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(title = "1. Tone Score", x = NULL, y = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p2b <- ggplot(daily_sentiment, aes(x = date, y = mean_positive)) +
  geom_line(color = "darkgreen", alpha = 0.7) +
  labs(title = "2. Positive %", x = NULL, y = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p2c <- ggplot(daily_sentiment, aes(x = date, y = mean_negative)) +
  geom_line(color = "darkred", alpha = 0.7) +
  labs(title = "3. Negative %", x = NULL, y = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p2d <- ggplot(daily_sentiment, aes(x = date, y = mean_polarity)) +
  geom_line(color = "purple", alpha = 0.7) +
  labs(title = "4. Polarity (intensidad)", x = NULL, y = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p2e <- ggplot(daily_sentiment, aes(x = date, y = mean_activity)) +
  geom_line(color = "orange", alpha = 0.7) +
  labs(title = "5. Activity Reference", x = NULL, y = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p2f <- ggplot(daily_sentiment, aes(x = date, y = mean_selfref)) +
  geom_line(color = "brown", alpha = 0.7) +
  labs(title = "6. Self Reference", x = "Fecha", y = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p2_combined <- (p2a | p2b) / (p2c | p2d) / (p2e | p2f) +
  plot_annotation(
    title = "6 Campos TONE - Serie Temporal Diaria",
    theme = theme(plot.title = element_text(size = 12))
  )
print(p2_combined)
save_plot("ibex35_6campos_panel", p2_combined)

## 3. VOLATILIDAD: sd_tone a lo largo del tiempo ####
log_cat("Gráfico 3: Volatilidad temporal\n")
p3 <- ggplot(daily_sentiment, aes(x = date, y = sd_tone_score)) +
  geom_line(color = "darkblue", alpha = 0.7) +
  geom_smooth(method = "loess", color = "red", se = FALSE) +
  labs(
    title = "Volatilidad del Sentimiento IBEX35",
    subtitle = "SD del Tone Score por día (mayor SD = mayor dispersión de opiniones)",
    x = "Fecha", y = "SD Tone Score"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))
print(p3)
save_plot("ibex35_volatility_timeseries", p3)

## 4. DISTRIBUCIONES: Histogramas de los 6 campos ####
log_cat("Gráfico 4: Distribuciones de 6 campos TONE\n")
tone_sample <- dbGetQuery(con, sprintf("
  SELECT
    CAST(string_split(TONE, ',')[1] AS DOUBLE) as tone_score,
    CAST(string_split(TONE, ',')[2] AS DOUBLE) as positive,
    CAST(string_split(TONE, ',')[3] AS DOUBLE) as negative,
    CAST(string_split(TONE, ',')[4] AS DOUBLE) as polarity,
    CAST(string_split(TONE, ',')[5] AS DOUBLE) as activity,
    CAST(string_split(TONE, ',')[6] AS DOUBLE) as selfref
  FROM read_parquet('%s')
  WHERE TONE IS NOT NULL AND TONE != ''
  USING SAMPLE 100000
", input_parquet)) %>% as.data.table()

p4a <- ggplot(tone_sample, aes(x = tone_score)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "1. Tone Score", x = NULL, y = "Freq") +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p4b <- ggplot(tone_sample, aes(x = positive)) +
  geom_histogram(bins = 50, fill = "darkgreen", alpha = 0.7) +
  labs(title = "2. Positive %", x = NULL, y = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p4c <- ggplot(tone_sample, aes(x = negative)) +
  geom_histogram(bins = 50, fill = "darkred", alpha = 0.7) +
  labs(title = "3. Negative %", x = NULL, y = "Freq") +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p4d <- ggplot(tone_sample, aes(x = polarity)) +
  geom_histogram(bins = 50, fill = "purple", alpha = 0.7) +
  labs(title = "4. Polarity", x = NULL, y = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p4e <- ggplot(tone_sample, aes(x = activity)) +
  geom_histogram(bins = 50, fill = "orange", alpha = 0.7) +
  labs(title = "5. Activity", x = "Valor", y = "Freq") +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p4f <- ggplot(tone_sample, aes(x = selfref)) +
  geom_histogram(bins = 50, fill = "brown", alpha = 0.7) +
  labs(title = "6. Self Reference", x = "Valor", y = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

p4_combined <- (p4a | p4b) / (p4c | p4d) / (p4e | p4f) +
  plot_annotation(
    title = "Distribución de 6 Campos TONE (muestra 100k registros)",
    theme = theme(
      plot.title = element_text(size = 10),
      plot.subtitle = element_text(size = 8),
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 8)
    )
  )

print(p4_combined)
save_plot("ibex35_6campos_distributions", p4_combined)

## 5. POSITIVE vs NEGATIVE scatter con densidad ####
log_cat("Gráfico 5: Positive vs Negative\n")
p5 <- ggplot(tone_sample, aes(x = positive, y = negative)) +
  geom_hex(bins = 50) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  scale_fill_viridis_c() +
  labs(
    title = "Relación Positive vs Negative",
    subtitle = "Hexbin plot (muestra 100k) | Línea: igualdad",
    x = "% Palabras Positivas", y = "% Palabras Negativas"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))

print(p5)
save_plot("ibex35_positive_vs_negative", p5)

## 6. MENSUAL: comparar tone_score con polarity ####
log_cat("Gráfico 6: Mensual - Tone vs Polarity\n")
p6 <- ggplot(monthly_sentiment, aes(x = YearMonth)) +
  geom_col(aes(y = mean_tone_score), fill = "steelblue", alpha = 0.7) +
  geom_line(aes(y = mean_polarity / 5), color = "purple", size = 1) +
  scale_y_continuous(
    name = "Tone Score (barras)",
    sec.axis = sec_axis(~ . * 5, name = "Polarity (línea)")
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Tone Score vs Polarity - Mensual",
    subtitle = "Barras: Tone Score | Línea morada: Polarity (escalada /5)",
    x = "Mes"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


print(p6)
save_plot("ibex35_monthly_tone_polarity", p6)

## 7. HEATMAP: correlación entre campos TONE ####
log_cat("Gráfico 7: Correlaciones entre campos TONE\n")
cor_matrix <- tone_sample[, .(tone_score, positive, negative, polarity, activity, selfref)] %>%
  cor(use = "complete.obs")
# Definir orden de variables (mantener orden original)
var_order <- c("tone_score", "positive", "negative", "polarity", "activity", "selfref")

# Convertir a formato long (matriz completa)
cor_long <- cor_matrix %>%
  as.data.table(keep.rownames = TRUE) %>%
  melt(id.vars = "rn", variable.name = "Variable2", value.name = "Correlation") %>%
  setnames("rn", "Variable1") %>%
  # Ordenar factores (invertir Y para que coincida con pheatmap)
  .[, Variable1 := factor(Variable1, levels = rev(var_order))] %>%
  .[, Variable2 := factor(Variable2, levels = var_order)]

# Visualización estilo pheatmap
p7 <- ggplot(cor_long, aes(x = Variable2, y = Variable1, fill = Correlation)) +
  geom_tile(color = "grey90", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", Correlation)),
    color = "black", size = 3
  ) +
  scale_fill_gradientn(
    colors = c(
      "#3B4CC0", "#6788EE", "#9ABBFF", "#C9D7F0",
      "#EDD1C2", "#F7A789", "#E26952", "#B40426"
    ),
    limits = c(-1, 1),
    name = "Correlation",
    breaks = seq(-1, 1, 0.25)
  ) +
  scale_x_discrete(position = "top") +
  scale_y_discrete(position = "left") +
  labs(
    title = "Matriz de Correlación - Campos TONE",
    x = NULL, y = NULL
  ) +
  coord_fixed() +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    axis.text.x = element_text(
      angle = 45, hjust = 0, vjust = 0.1,
      color = "black", face = "plain"
    ),
    axis.text.y = element_text(color = "black", face = "plain"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "grey50", fill = NA, linewidth = 1),
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9)
  )


print(p7)
save_plot("ibex35_tone_correlation_heatmap", p7)

dbDisconnect(con, shutdown = TRUE)

log_cat("\n=== COMPLETADO ===\n")
log_cat("Archivos generados en:", output_dir, "\n")
log_cat("- ibex35_daily_sentiment.parquet (con 6 campos TONE)\n")
log_cat("- ibex35_monthly_sentiment.parquet\n")
log_cat("- ibex35_yearly_sentiment.parquet\n")
log_cat("- 7 gráficos PNG\n")
