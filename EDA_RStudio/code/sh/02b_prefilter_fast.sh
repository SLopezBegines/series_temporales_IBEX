#!/bin/bash
# 02b_prefilter_fast.sh
# Pre-filtrado ultrarrápido con grep para reducir tamaño

INPUT_FILE="/mnt/tnas/gdelt_data/gdelt_all_consolidated.csv"
PREFILT_FILE="data/gdelt_prefiltrado.csv"
OUTPUT_FILE="data/gdelt_ibex35_filtered.csv"

echo "=== PRE-FILTRADO RÁPIDO ==="
echo ""

# Verificar archivo
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: No se encuentra $INPUT_FILE"
    exit 1
fi

FILE_SIZE=$(du -h "$INPUT_FILE" | cut -f1)
echo "Archivo entrada: $INPUT_FILE ($FILE_SIZE)"
echo ""

# FASE 1: Pre-filtrado agresivo con grep (MUY RÁPIDO)
# Busca CUALQUIER mención de términos clave
echo "FASE 1: Pre-filtrado con grep..."
echo "(Buscando registros con términos españoles/IBEX35)"
START=$(date +%s)

# Extraer header
head -1 "$INPUT_FILE" > "$PREFILT_FILE"

# Filtrar líneas que contengan AL MENOS UNO de estos términos
# Esto reduce el archivo de 250GB a ~5-20GB típicamente
# Incluye todos los patrones: THEMES, ORGANIZATIONS, SOURCES
grep -iE '\.es|ECON_STOCKMARKET|WB_.*STOCK|SPAIN|SPANISH|IBEX|MADRID.*EXCHANGE|BME|bolsa.*madrid|mercado.*(español|espanol)|latibex|santander|bbva|telefonica|telefónica|iberdrola|inditex|repsol|caixabank|endesa|naturgy|ferrovial|amadeus|aena|acs|grifols|mapfre|cellnex|colonial|sabadell|enagas|enagás|acerinox|merlin|siemens.*gamesa|logista|bankinter|indra|fluidra|melia|meliá|tecnicas.*reunidas|técnicas.*reunidas|viscofan|solaria|pharma.*mar|elmundo|elpais|el.*pais|abc\.es|lavanguardia|eleconomista|expansion|expansión|cincodias|vozpopuli|elconfidencial|publico\.es|eldiario\.es|20minutos|rtve|efe|europa.*press' "$INPUT_FILE" >> "$PREFILT_FILE"

ELAPSED=$(($(date +%s) - START))
PREFILT_SIZE=$(du -h "$PREFILT_FILE" | cut -f1)

echo "✓ Pre-filtrado completado en ${ELAPSED}s"
echo "  Tamaño reducido: $FILE_SIZE -> $PREFILT_SIZE"
echo ""

# Contar registros pre-filtrados
PREFILT_LINES=$(($(wc -l < "$PREFILT_FILE") - 1))
echo "  Registros pre-filtrados: $(printf "%'d" $PREFILT_LINES)"
echo ""

# FASE 2: Ahora R/DuckDB puede procesar el archivo reducido rápidamente
echo "FASE 2: Aplicando filtros precisos con R..."
echo ""

# Crear script R temporal
cat > /tmp/filter_phase2.R << 'RSCRIPT'
library(DBI)
library(duckdb)

input <- "data/gdelt_prefiltrado.csv"
output <- "data/gdelt_ibex35_filtered.csv"

cat("Conectando DuckDB...\n")
con <- dbConnect(duckdb::duckdb())

# Query simplificada - archivo ya pre-filtrado
query <- "
SELECT * 
FROM read_csv_auto('data/gdelt_prefiltrado.csv')
WHERE 
  -- Excluir falsos positivos (cabras ibex)
  NOT (
    THEMES ILIKE '%goat%' OR
    THEMES ILIKE '%wildlife%' OR
    THEMES ILIKE '%zoo%' OR
    THEMES ILIKE '%alpine%ibex%' OR
    ORGANIZATIONS ILIKE '%goat%' OR
    ORGANIZATIONS ILIKE '%wildlife%' OR
    ORGANIZATIONS ILIKE '%zoo%' OR
    ORGANIZATIONS ILIKE '%alpine%ibex%'
  )
"

cat("Aplicando exclusiones...\n")
start <- Sys.time()

dbExecute(con, sprintf("COPY (%s) TO '%s' (HEADER, DELIMITER ',')", 
                       query, output))

elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
cat(sprintf("✓ Filtrado completado en %.1f segundos\n\n", elapsed))

# Estadísticas
if (file.exists(output)) {
  size_mb <- file.info(output)$size / (1024^2)
  count_q <- sprintf("SELECT COUNT(*) as n FROM read_csv_auto('%s')", output)
  n_records <- dbGetQuery(con, count_q)$n
  
  cat("=== RESULTADO FINAL ===\n")
  cat("Archivo:", output, "\n")
  cat("Tamaño:", round(size_mb, 1), "MB\n")
  cat("Registros:", format(n_records, big.mark = ","), "\n")
}

dbDisconnect(con, shutdown = TRUE)
RSCRIPT

# Ejecutar fase 2
Rscript /tmp/filter_phase2.R

TOTAL_TIME=$(($(date +%s) - START))
echo ""
echo "=== TIEMPO TOTAL ==="
echo "Fase 1 (grep): ${ELAPSED}s"
echo "Fase 2 (DuckDB): ver arriba"
echo "Total: ${TOTAL_TIME}s ($(($TOTAL_TIME / 60)) minutos)"
echo ""
echo "✓ COMPLETADO"
