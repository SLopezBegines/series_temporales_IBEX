#!/bin/bash
# step1_zip_to_parquet.sh
# Convierte ZIP directamente a Parquet (OPTIMIZADO - sin descomprimir)

INPUT_DIR="/mnt/NTFS/gdelt_consolidated/gdelt_raw"
OUTPUT_DIR="/mnt/NTFS/gdelt_consolidated/gdelt_parquet"
LOG_FILE="step1_zip_to_parquet.log"
NUM_CORES=2  # 2 cores para 16GB RAM

echo "=== PASO 1: ZIP → PARQUET (directo, sin descomprimir) ===" | tee $LOG_FILE
echo "Input:  $INPUT_DIR" | tee -a $LOG_FILE
echo "Output: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "Cores:  $NUM_CORES" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Crear directorio de salida
mkdir -p "$OUTPUT_DIR"

# Contar archivos
TOTAL_FILES=$(find "$INPUT_DIR" -name "*.csv.zip" | wc -l)
echo "Total archivos: $TOTAL_FILES" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

if [ $TOTAL_FILES -eq 0 ]; then
    echo "❌ Error: No se encontraron archivos *.csv.zip en $INPUT_DIR" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    echo "💡 Obtén la ruta correcta con:" | tee -a $LOG_FILE
    echo "   cd /carpeta/con/zips" | tee -a $LOG_FILE
    echo "   pwd" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    echo "Luego edita INPUT_DIR en este script (línea 5)" | tee -a $LOG_FILE
    exit 1
fi

# Función para convertir un archivo
convert_file() {
    zip_file="$1"
    output_dir="$2"
    
    filename=$(basename "$zip_file")
    parquet_name="${filename%.csv.zip}.parquet"
    parquet_path="$output_dir/$parquet_name"
    
    # Llamar al script R
    Rscript ../R/convert_zip_to_parquet.R "$zip_file" "$parquet_path"
    return $?
}

export -f convert_file
export OUTPUT_DIR

START_TIME=$(date +%s)

# Convertir en paralelo
if command -v parallel &> /dev/null; then
    echo "Usando GNU parallel..." | tee -a $LOG_FILE
    
    find "$INPUT_DIR" -name "*.csv.zip" | \
        parallel -j $NUM_CORES --progress --joblog parallel_jobs.log \
        convert_file {} "$OUTPUT_DIR" 2>&1 | tee -a $LOG_FILE

else
    echo "Usando procesamiento secuencial..." | tee -a $LOG_FILE
    
    COUNTER=0
    find "$INPUT_DIR" -name "*.csv.zip" | while read zip_file; do
        convert_file "$zip_file" "$OUTPUT_DIR" 2>&1 | tee -a $LOG_FILE
        
        COUNTER=$((COUNTER + 1))
        if [ $((COUNTER % 50)) -eq 0 ]; then
            echo "Progreso: $COUNTER/$TOTAL_FILES" | tee -a $LOG_FILE
        fi
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "" | tee -a $LOG_FILE
echo "Tiempo: ${MINUTES}m ${SECONDS}s" | tee -a $LOG_FILE

# Verificar resultados
PARQUET_COUNT=$(find "$OUTPUT_DIR" -name "*.parquet" | wc -l)
echo "" | tee -a $LOG_FILE
echo "=== RESULTADO ===" | tee -a $LOG_FILE
echo "Archivos Parquet creados: $PARQUET_COUNT" | tee -a $LOG_FILE
echo "Esperados: $TOTAL_FILES" | tee -a $LOG_FILE

if [ $PARQUET_COUNT -eq $TOTAL_FILES ]; then
    echo "✓ CONVERSIÓN COMPLETA" | tee -a $LOG_FILE
else
    echo "⚠ Faltan archivos: $((TOTAL_FILES - PARQUET_COUNT))" | tee -a $LOG_FILE
fi

# Tamaño total
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
echo "Tamaño total: $TOTAL_SIZE" | tee -a $LOG_FILE

# Espacio ahorrado vs método antiguo
SAVED_GB=$((TOTAL_FILES * 30 / 1024))
echo "" | tee -a $LOG_FILE
echo "💾 Espacio ahorrado (vs descomprimir): ~${SAVED_GB}GB" | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "Siguiente paso: Rscript step2_consolidate_parquet.R" | tee -a $LOG_FILE
