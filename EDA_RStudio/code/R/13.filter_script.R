library(DBI)
library(duckdb)

# CONFIGURACIÓN DE LOGGING
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


# CONFIGURACIÓN GEOGRÁFICA

GEO_COUNTRIES_PRIMARY <- c("SP") # España
GEO_COUNTRIES_SECONDARY <- c(
  "UK", "FR", "GM", "IT", "PO", "NL", "BE", "AU", "SW",
  "NO", "FI", "DA", "GR", "PL", "SZ", "HU", "EI", "RO"
) # Europa

# Combinar en una lista completa de países válidos
geo_valid_countries <- c(GEO_COUNTRIES_PRIMARY, GEO_COUNTRIES_SECONDARY)
geo_pattern <- paste0("'", paste(geo_valid_countries, collapse = "','"), "'")


# INICIO PROCESO

# input_parquet <- "data/gdelt_all_consolidated.parquet"
# output_file <- "data/gdelt_filtered/gdelt_ibex35_filtered_low_noise_tracking.parquet"

log_cat("=== FILTRADO IBEX35 - REQUIERE 2+ CRITERIOS + GEOGRAFÍA ===\n\n")
log_cat("Archivo entrada:", input_parquet, "\n")
size_gb <- file.info(input_parquet)$size / (1024^3)
log_cat("Tamaño:", round(size_gb, 1), "GB\n")
log_cat("Países válidos:", paste(geo_valid_countries, collapse = ", "), "\n\n")

con <- dbConnect(duckdb::duckdb())

# Query con extracción de geografía y filtro
query <- sprintf("
SELECT
  *,
  -- Extraer país de LOCATIONS (formato: lat#lon#countrycode#...)
  CASE
    WHEN LOCATIONS IS NOT NULL AND LENGTH(LOCATIONS) > 0
    THEN SPLIT_PART(SPLIT_PART(LOCATIONS, ';', 1), '#', 3)
    ELSE NULL
  END as first_location_country,
  -- Indicadores de criterios cumplidos
  regexp_matches(THEMES, '(?i)(ECON_STOCKMARKET|WB_.*STOCK|SPAIN|SPANISH|IBEX|MADRID.*EXCHANGE|BME)') AS cumple_THEMES,
  regexp_matches(ORGANIZATIONS, '(?i)(santander|bbva|telefonica|iberdrola|inditex|repsol|caixabank|endesa|naturgy|ferrovial|amadeus|aena|grifols|mapfre|cellnex|colonial|sabadell|enagas|acerinox|merlin|logista|bankinter|indra|fluidra|melia|viscofan|solaria|ibex|bolsa.*madrid|mercado.*(español|espanol)|bme|latibex)') AS cumple_ORGANIZATIONS,
  (regexp_matches(SOURCEURLS, '(?i)\\.es') OR regexp_matches(SOURCES, '(?i)(elmundo|elpais|abc|lavanguardia|eleconomista|expansion|cincodias|vozpopuli|elconfidencial|publico|eldiario|20minutos|rtve|efe|europa.*press)')) AS cumple_SOURCES,
  -- Contador total
  (CASE WHEN regexp_matches(THEMES, '(?i)(ECON_STOCKMARKET|WB_.*STOCK|SPAIN|SPANISH|IBEX|MADRID.*EXCHANGE|BME)') THEN 1 ELSE 0 END +
   CASE WHEN regexp_matches(ORGANIZATIONS, '(?i)(santander|bbva|telefonica|iberdrola|inditex|repsol|caixabank|endesa|naturgy|ferrovial|amadeus|aena|grifols|mapfre|cellnex|colonial|sabadell|enagas|acerinox|merlin|logista|bankinter|indra|fluidra|melia|viscofan|solaria|ibex|bolsa.*madrid|mercado.*(español|espanol)|bme|latibex)') THEN 1 ELSE 0 END +
   CASE WHEN (regexp_matches(SOURCEURLS, '(?i)\\.es') OR regexp_matches(SOURCES, '(?i)(elmundo|elpais|abc|lavanguardia|eleconomista|expansion|cincodias|vozpopuli|elconfidencial|publico|eldiario|20minutos|rtve|efe|europa.*press)')) THEN 1 ELSE 0 END) AS num_criterios_cumplidos
FROM read_parquet('%s')
WHERE
  -- Criterio 1: Debe cumplir al menos 2 criterios IBEX35
  (CASE WHEN regexp_matches(THEMES, '(?i)(ECON_STOCKMARKET|WB_.*STOCK|SPAIN|SPANISH|IBEX|MADRID.*EXCHANGE|BME)') THEN 1 ELSE 0 END +
   CASE WHEN regexp_matches(ORGANIZATIONS, '(?i)(santander|bbva|telefonica|iberdrola|inditex|repsol|caixabank|endesa|naturgy|ferrovial|amadeus|aena|grifols|mapfre|cellnex|colonial|sabadell|enagas|acerinox|merlin|logista|bankinter|indra|fluidra|melia|viscofan|solaria|ibex|bolsa.*madrid|mercado.*(español|espanol)|bme|latibex)') THEN 1 ELSE 0 END +
   CASE WHEN (regexp_matches(SOURCEURLS, '(?i)\\.es') OR regexp_matches(SOURCES, '(?i)(elmundo|elpais|abc|lavanguardia|eleconomista|expansion|cincodias|vozpopuli|elconfidencial|publico|eldiario|20minutos|rtve|efe|europa.*press)')) THEN 1 ELSE 0 END) >= 2

  -- Criterio 2: Geografía válida (mantiene NULL o países Europa/España)
  AND (
    CASE
      WHEN LOCATIONS IS NOT NULL AND LENGTH(LOCATIONS) > 0
      THEN SPLIT_PART(SPLIT_PART(LOCATIONS, ';', 1), '#', 3)
      ELSE NULL
    END IS NULL
    OR
    CASE
      WHEN LOCATIONS IS NOT NULL AND LENGTH(LOCATIONS) > 0
      THEN SPLIT_PART(SPLIT_PART(LOCATIONS, ';', 1), '#', 3)
      ELSE NULL
    END IN (%s)
  )

  -- Criterio 3: Exclusión animales
  AND NOT (
    regexp_matches(THEMES, '(?i)(alpine.*ibex|goat|wildlife|zoo|hunting|endangered.*species|animal.*conservation|taxidermy|pyrenean.*ibex|nubian.*ibex|siberian.*ibex)') OR
    regexp_matches(ORGANIZATIONS, '(?i)(alpine.*ibex|goat|wildlife|zoo|hunting|endangered.*species|animal.*conservation|taxidermy|pyrenean.*ibex|nubian.*ibex|siberian.*ibex)')
  )
", input_parquet, geo_pattern)

log_cat("Ejecutando filtrado...\n")
start_time <- Sys.time()

dbExecute(con, sprintf(
  "COPY (%s) TO '%s' (FORMAT PARQUET, COMPRESSION SNAPPY)",
  query, output_file
))

elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
log_cat(sprintf("\n✓ Completado en %.1f minutos\n\n", elapsed))


# ESTADÍSTICAS

if (file.exists(output_file)) {
  output_size_mb <- file.info(output_file)$size / (1024^2)
  count_query <- sprintf("SELECT COUNT(*) as n FROM read_parquet('%s')", output_file)
  n_records <- dbGetQuery(con, count_query)$n

  log_cat("Archivo salida:", output_file, "\n")
  log_cat("Tamaño:", round(output_size_mb, 1), "MB\n")
  log_cat("Registros:", format(n_records, big.mark = ","), "\n\n")

  # Distribución de criterios IBEX35
  log_cat("=== DISTRIBUCIÓN DE CRITERIOS IBEX35 ===\n")
  dist <- dbGetQuery(con, sprintf("
    SELECT
      num_criterios_cumplidos,
      COUNT(*) as n,
      ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
    FROM read_parquet('%s')
    GROUP BY num_criterios_cumplidos
    ORDER BY num_criterios_cumplidos
  ", output_file))
  print(dist)

  # Distribución geográfica
  log_cat("\n=== DISTRIBUCIÓN GEOGRÁFICA ===\n")
  geo_dist <- dbGetQuery(con, sprintf("
    SELECT
      COALESCE(first_location_country, 'SIN_DATO') as pais,
      COUNT(*) as n,
      ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
    FROM read_parquet('%s')
    GROUP BY first_location_country
    ORDER BY n DESC
    LIMIT 20
  ", output_file))
  print(geo_dist)

  # Combinaciones de criterios
  log_cat("\n=== COMBINACIONES DE CRITERIOS ===\n")
  combos <- dbGetQuery(con, sprintf("
    SELECT
      cumple_THEMES,
      cumple_ORGANIZATIONS,
      cumple_SOURCES,
      COUNT(*) as n
    FROM read_parquet('%s')
    GROUP BY cumple_THEMES, cumple_ORGANIZATIONS, cumple_SOURCES
    ORDER BY n DESC
  ", output_file))
  print(combos)

  # Resumen final
  log_cat("\n=== RESULTADO FINAL ===\n")
  reduction_factor <- round(size_gb * 1024 / output_size_mb, 1)
  log_cat("Reducción:", size_gb, "GB →", round(output_size_mb / 1024, 2), "GB\n")
  log_cat("Factor:", reduction_factor, "x\n")
}

dbDisconnect(con, shutdown = TRUE)

log_cat("\n✓ Cada registro tiene columnas:\n")
log_cat("  - first_location_country (código país o NULL)\n")
log_cat("  - cumple_THEMES (true/false)\n")
log_cat("  - cumple_ORGANIZATIONS (true/false)\n")
log_cat("  - cumple_SOURCES (true/false)\n")
log_cat("  - num_criterios_cumplidos (2 o 3)\n")
log_cat("\nCarga el dataset filtrado con arrow:\n")
log_cat("gdelt_filter <- arrow::read_parquet('", output_file, "')\n")
