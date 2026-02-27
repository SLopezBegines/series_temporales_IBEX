# CONTEO DE NOTICIAS POSITIVAS/NEGATIVAS ####

log_cat("\n=== CONTEO DE NOTICIAS POSITIVAS/NEGATIVAS ===\n")

con <- dbConnect(duckdb::duckdb())


# 1. CONTEO GLOBAL DE REGISTROS ####

log_cat("Calculando conteo global de registros por sentimiento...\n")

conteo_global <- dbGetQuery(con, sprintf("
  WITH classified AS (
    SELECT
      CAST(string_split(TONE, ',')[1] AS DOUBLE) as tone_score,
      CASE
        WHEN CAST(string_split(TONE, ',')[1] AS DOUBLE) < -2 THEN 'Muy negativo'
        WHEN CAST(string_split(TONE, ',')[1] AS DOUBLE) >= -2
          AND CAST(string_split(TONE, ',')[1] AS DOUBLE) < 0 THEN 'Ligeramente negativo'
        WHEN CAST(string_split(TONE, ',')[1] AS DOUBLE) >= 0
          AND CAST(string_split(TONE, ',')[1] AS DOUBLE) < 2 THEN 'Ligeramente positivo'
        WHEN CAST(string_split(TONE, ',')[1] AS DOUBLE) >= 2 THEN 'Muy positivo'
      END as categoria,
      CASE
        WHEN CAST(string_split(TONE, ',')[1] AS DOUBLE) < 0 THEN 'Negativo'
        ELSE 'Positivo'
      END as sentimiento_simple,
      NUMARTS
    FROM read_parquet('%s')
    WHERE TONE IS NOT NULL AND TONE != ''
  )
  SELECT
    categoria,
    sentimiento_simple,
    COUNT(*) as num_registros,
    SUM(NUMARTS) as num_articulos,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct_registros,
    ROUND(100.0 * SUM(NUMARTS) / SUM(SUM(NUMARTS)) OVER (), 2) as pct_articulos
  FROM classified
  GROUP BY categoria, sentimiento_simple
  ORDER BY
    CASE sentimiento_simple
      WHEN 'Negativo' THEN 1
      ELSE 2
    END,
    CASE categoria
      WHEN 'Muy negativo' THEN 1
      WHEN 'Ligeramente negativo' THEN 2
      WHEN 'Ligeramente positivo' THEN 3
      WHEN 'Muy positivo' THEN 4
    END
", input_parquet)) %>% as.data.table()

log_cat("Conteo global:\n")
print(conteo_global)

# Resumen binario (Positivo/Negativo)
conteo_binario <- conteo_global[, .(
  num_registros = sum(num_registros),
  num_articulos = sum(num_articulos),
  pct_registros = sum(pct_registros),
  pct_articulos = sum(pct_articulos)
), by = sentimiento_simple]

log_cat("\nResumen Positivo vs Negativo:\n")
print(conteo_binario)


# 2. EVOLUCIÓN TEMPORAL - CONTEO DIARIO ####

log_cat("\nCalculando evolución temporal diaria...\n")

conteo_diario <- dbGetQuery(con, sprintf("
  WITH classified AS (
    SELECT
      strptime(CAST(DATE AS VARCHAR), '%%Y%%m%%d') as Date,
      CAST(string_split(TONE, ',')[1] AS DOUBLE) as tone_score,
      CASE
        WHEN CAST(string_split(TONE, ',')[1] AS DOUBLE) < 0 THEN 'Negativo'
        ELSE 'Positivo'
      END as sentimiento,
      NUMARTS
    FROM read_parquet('%s')
    WHERE TONE IS NOT NULL AND TONE != ''
  )
  SELECT
    Date,
    sentimiento,
    COUNT(*) as num_registros,
    SUM(NUMARTS) as num_articulos
  FROM classified
  GROUP BY Date, sentimiento
  ORDER BY Date, sentimiento
", input_parquet)) %>% as.data.table()

# Pivotar para tener positivo y negativo en columnas
conteo_diario_wide <- dcast(conteo_diario,
  Date ~ sentimiento,
  value.var = c("num_registros", "num_articulos"),
  fill = 0
)

# Calcular proporciones y ratio
conteo_diario_wide[, ":="(
  total_registros = num_registros_Negativo + num_registros_Positivo,
  total_articulos = num_articulos_Negativo + num_articulos_Positivo,
  pct_registros_negativos = 100 * num_registros_Negativo / (num_registros_Negativo + num_registros_Positivo),
  pct_registros_positivos = 100 * num_registros_Positivo / (num_registros_Negativo + num_registros_Positivo),
  ratio_pos_neg = num_registros_Positivo / pmax(num_registros_Negativo, 1)
)]

conteo_diario_wide[, Date := as.Date(Date)]

log_cat("Primeras filas del conteo diario:\n")
print(head(conteo_diario_wide, 10))


# 3. EVOLUCIÓN MENSUAL ####

log_cat("\nCalculando evolución mensual...\n")

conteo_mensual <- dbGetQuery(con, sprintf("
  WITH classified AS (
    SELECT
      CAST(substr(CAST(DATE AS VARCHAR), 1, 4) AS INTEGER) AS Year,
      CAST(substr(CAST(DATE AS VARCHAR), 5, 2) AS INTEGER) AS Month,
      CAST(string_split(TONE, ',')[1] AS DOUBLE) as tone_score,
      CASE
        WHEN CAST(string_split(TONE, ',')[1] AS DOUBLE) < 0 THEN 'Negativo'
        ELSE 'Positivo'
      END as sentimiento,
      NUMARTS
    FROM read_parquet('%s')
    WHERE TONE IS NOT NULL AND TONE != ''
  )
  SELECT
    make_date(Year, Month, 1) as YearMonth,
    sentimiento,
    COUNT(*) as num_registros,
    SUM(NUMARTS) as num_articulos
  FROM classified
  GROUP BY Year, Month, sentimiento
  ORDER BY Year, Month, sentimiento
", input_parquet)) %>% as.data.table()

conteo_mensual_wide <- dcast(conteo_mensual,
  YearMonth ~ sentimiento,
  value.var = c("num_registros", "num_articulos"),
  fill = 0
)

conteo_mensual_wide[, ":="(
  pct_registros_negativos = 100 * num_registros_Negativo / (num_registros_Negativo + num_registros_Positivo),
  pct_registros_positivos = 100 * num_registros_Positivo / (num_registros_Negativo + num_registros_Positivo),
  ratio_pos_neg = num_registros_Positivo / pmax(num_registros_Negativo, 1)
)]

conteo_mensual_wide[, YearMonth := as.Date(YearMonth)]


# 4. VISUALIZACIONES ####

log_cat("\n=== GENERANDO VISUALIZACIONES ===\n")

# 4.1 BAR CHART - Positivo vs Negativo
log_cat("Gráfico 2: Bar chart positivo vs negativo\n")

p_bar_simple <- ggplot(
  conteo_binario,
  aes(x = sentimiento_simple, y = num_registros, fill = sentimiento_simple)
) +
  geom_col(alpha = 0.8, color = "black", linewidth = 0.5) +
  geom_text(
    aes(label = sprintf(
      "%s\n(%.1f%%)",
      format(num_registros, big.mark = ","),
      pct_registros
    )),
    vjust = -0.2, size = 2, fontface = "bold"
  ) +
  scale_fill_manual(values = c("Negativo" = "#E74C3C", "Positivo" = "#3498DB")) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.1))) +
  labs(
    title = "Noticias Positivas vs Negativas - IBEX35",
    subtitle = sprintf(
      "Ratio Positivo/Negativo: %.2f",
      conteo_binario[sentimiento_simple == "Positivo", num_registros] /
        conteo_binario[sentimiento_simple == "Negativo", num_registros]
    ),
    x = NULL,
    y = "Número de registros"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  )

print(p_bar_simple)
save_plot("ibex35_sentiment_pos_neg_bar", p_bar_simple)

# 4.2 STACKED BAR - Categorías detalladas
log_cat("Gráfico 3: Stacked bar con 4 categorías\n")

p_bar_detailed <- ggplot(
  conteo_global,
  aes(x = "Total", y = num_registros, fill = categoria)
) +
  geom_col(position = "fill", color = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%s\n%.1f%%", categoria, pct_registros, num_registros)),
    position = position_fill(vjust = 0.5),
    size = 2, fontface = "bold", color = "white"
  ) +
  scale_fill_manual(values = c(
    "Muy negativo" = "#D73027",
    "Ligeramente negativo" = "#FC8D59",
    "Ligeramente positivo" = "#91BFDB",
    "Muy positivo" = "#4575B4"
  )) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Proporción de Sentimiento en Noticias IBEX35",
    subtitle = "Distribución porcentual por categoría",
    x = NULL,
    y = "Proporción",
    fill = "Categoría"
  ) +
  coord_flip() +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    legend.position = "bottom"
  )

print(p_bar_detailed)
save_plot("ibex35_sentiment_proportion_stacked", p_bar_detailed)

# 4.3 SERIE TEMPORAL - Conteo diario de positivas vs negativas
log_cat("Gráfico 4: Serie temporal diaria\n")

conteo_diario_long <- melt(conteo_diario_wide,
  id.vars = "Date",
  measure.vars = c("num_registros_Negativo", "num_registros_Positivo"),
  variable.name = "Sentimiento",
  value.name = "Conteo"
)

conteo_diario_long[, Sentimiento := ifelse(Sentimiento == "num_registros_Negativo",
  "Negativo", "Positivo"
)]

p_timeseries <- ggplot(conteo_diario_long, aes(x = Date, y = Conteo, color = Sentimiento)) +
  geom_line(alpha = 0.6) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
  scale_color_manual(values = c("Negativo" = "#E74C3C", "Positivo" = "#3498DB")) +
  labs(
    title = "Evolución Temporal de Noticias Positivas vs Negativas",
    subtitle = "Serie diaria con suavizado LOESS",
    x = "Fecha",
    y = "Número de registros diarios",
    color = "Sentimiento"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8),
    legend.position = "top"
  )

print(p_timeseries)
save_plot("ibex35_sentiment_timeseries_daily", p_timeseries)

# 4.4 RATIO POSITIVO/NEGATIVO
log_cat("Gráfico 6: Ratio Positivo/Negativo\n")

p_ratio <- ggplot(conteo_diario_wide, aes(x = Date, y = ratio_pos_neg)) +
  geom_line(alpha = 0.5, color = "steelblue") +
  geom_smooth(method = "loess", color = "darkblue", se = TRUE) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "Ratio Positivo/Negativo a lo Largo del Tiempo",
    subtitle = ">1 = más positivas | <1 = más negativas | línea roja = equilibrio",
    x = "Fecha",
    y = "Ratio (Positivo / Negativo)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 10),
    plot.subtitle = element_text(size = 8),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 8)
  )

print(p_ratio)
save_plot("ibex35_sentiment_ratio_timeseries", p_ratio)


# 5. GUARDAR RESULTADOS

write_parquet(conteo_diario_wide, file.path(output_dir, "ibex35_sentiment_count_daily.parquet"))
write_parquet(conteo_mensual_wide, file.path(output_dir, "ibex35_sentiment_count_monthly.parquet"))
fwrite(conteo_global, file.path(output_dir, "ibex35_sentiment_count_global.csv"))

dbDisconnect(con, shutdown = TRUE)
