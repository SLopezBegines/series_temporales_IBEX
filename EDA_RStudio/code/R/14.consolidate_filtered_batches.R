#!/usr/bin/env Rscript
# consolidate_filtered_batches.R
# Consolida los 22 batches filtrados en un solo archivo

library(arrow)
library(dplyr)

# Configuración
# input_dir <- "/mnt/NTFS/gdelt_consolidated/gdelt_filtered"
# output_file <- "/mnt/NTFS/gdelt_consolidated/gdelt_filtered_consolidated.parquet"
# main_log <- file.path("/mnt/NTFS/gdelt_consolidated", paste0("filter_gdelt_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))

log_cat <- function(...) {
  msg <- paste0(...)
  cat(msg)
  cat(msg, file = main_log, append = TRUE)
}

log_cat("╔══════════════════════════════════════════════════════════════╗\n")
log_cat("║  CONSOLIDACIÓN DE BATCHES FILTRADOS                          ║\n")
log_cat("╚══════════════════════════════════════════════════════════════╝\n\n")

# Verificar directorio
if (!dir.exists(input_dir)) {
  stop("❌ No existe directorio: ", input_dir)
}

# Listar archivos filtrados
filtered_files <- list.files(input_dir,
  pattern = "filtered_.*\\.parquet$",
  full.names = TRUE
)

n_files <- length(filtered_files)

if (n_files == 0) {
  stop("❌ No se encontraron archivos filtrados en ", input_dir)
}

log_cat("Input:  ", input_dir, "\n")
log_cat("Output: ", output_file, "\n")
log_cat("Archivos a consolidar: ", n_files, "\n\n")

# Mostrar tamaños
log_cat("Archivos encontrados:\n")
total_size_mb <- 0
for (f in filtered_files) {
  size_mb <- file.info(f)$size / (1024^2)
  total_size_mb <- total_size_mb + size_mb
  log_cat("  ", basename(f), " (", round(size_mb, 2), " MB)\n")
}
log_cat("\nTamaño total: ", round(total_size_mb, 2), " MB\n\n")

# Advertencia si el total es muy grande
if (total_size_mb > 500) {
  log_cat("⚠️  Advertencia: Tamaño total >500MB\n")
  log_cat("   Consolidación puede tomar varios minutos\n\n")
}

# ============================================================================
# CONSOLIDACIÓN
# ============================================================================

log_cat("=== CONSOLIDANDO ===\n\n")
log_cat("Abriendo dataset...\n")

start_time <- Sys.time()

tryCatch(
  {
    # Abrir como dataset Arrow (no carga en RAM)
    dataset <- open_dataset(filtered_files, format = "parquet")

    log_cat("✓ Dataset abierto\n")
    log_cat("Columnas: ", paste(names(dataset$schema), collapse = ", "), "\n\n")

    # Escribir consolidado
    log_cat("Escribiendo archivo consolidado...\n")

    write_parquet(
      dataset,
      sink = output_file,
      compression = "snappy",
      use_dictionary = TRUE
    )

    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    log_cat("✓ Consolidación completada en ", round(elapsed, 1), " segundos\n\n")
  },
  error = function(e) {
    log_cat("❌ ERROR durante consolidación:\n")
    log_cat(as.character(e), "\n\n")
    quit(status = 1)
  }
)

# ============================================================================
# VERIFICACIÓN
# ============================================================================

log_cat("=== VERIFICACIÓN ===\n\n")

if (file.exists(output_file)) {
  file_size_mb <- file.info(output_file)$size / (1024^2)

  log_cat("✅ Archivo consolidado creado\n")
  log_cat("   Ruta:   ", output_file, "\n")
  log_cat("   Tamaño: ", round(file_size_mb, 2), " MB\n\n")

  # Verificar integridad
  log_cat("Verificando integridad...\n")
  ds_final <- open_dataset(output_file, format = "parquet")

  # Contar registros (debería ser rápido con archivos pequeños)
  total_records <- ds_final %>%
    summarise(n = n()) %>%
    collect() %>%
    pull(n)

  log_cat("✓ Registros: ", format(total_records, big.mark = ","), "\n")

  # Rango de fechas
  date_stats <- ds_final %>%
    summarise(
      min_date = min(DATE, na.rm = TRUE),
      max_date = max(DATE, na.rm = TRUE)
    ) %>%
    collect()

  log_cat("✓ Fechas: ", date_stats$min_date, " → ", date_stats$max_date, "\n\n")

  # Muestra
  log_cat("Muestra de datos:\n")
  sample_data <- ds_final %>%
    dplyr::select(DATE, THEMES, LOCATIONS, TONE) %>%
    head(3) %>%
    collect()

  print(sample_data)
} else {
  log_cat("❌ ERROR: Archivo consolidado no se creó\n")
  quit(status = 1)
}

# ============================================================================
# RESUMEN FINAL
# ============================================================================

total_time <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

log_cat("\n╔══════════════════════════════════════════════════════════════╗\n")
log_cat("║  ✅ CONSOLIDACIÓN EXITOSA                                    ║\n")
log_cat("╚══════════════════════════════════════════════════════════════╝\n\n")

log_cat("Tiempo total: ", round(total_time, 2), " minutos\n\n")

log_cat("💡 Ahora puedes:\n")
log_cat("   1. Usar el archivo consolidado para análisis\n")
log_cat("   2. Eliminar batches filtrados individuales (ahorrar ~", round(total_size_mb, 0), "MB):\n")
log_cat("      rm ", input_dir, "/batch_filtered_*.parquet\n\n")

log_cat("Archivo final: ", output_file, "\n\n")
