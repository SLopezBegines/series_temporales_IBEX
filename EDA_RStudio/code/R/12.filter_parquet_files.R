#!/usr/bin/env Rscript
# filter_all_batches.R
# Aplica filtrado a todos los archivos parquet en input_dir

# Configuración
# input_dir <- "/mnt/NTFS/gdelt_consolidated/gdelt_parquet"
# output_dir <- "/mnt/NTFS/gdelt_consolidated/gdelt_filtered"
# filter_script <- "13.filter_script.R"

main_log <- file.path("/mnt/NTFS/gdelt_consolidated", paste0("filter_gdelt_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))

# Crear log principal
# main_log <- file.path("data", paste0("filter_parquet_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))

log_cat <- function(...) {
  msg <- paste0(...)
  cat(msg)
  cat(msg, file = main_log, append = TRUE)
}

log_cat("FILTRADO DE ARCHIVOS PARQUET (secuencial)\n")

# Verificar que existe el script de filtrado
if (!file.exists(filter_script)) {
  stop("No se encuentra el script de filtrado: ", filter_script)
}

# Crear directorio de salida
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  log_cat("Creado directorio: ", output_dir, "\n")
}

# Listar todos los archivos .parquet en input_dir
input_files <- list.files(input_dir, pattern = "\\.parquet$", full.names = FALSE)
total_files <- length(input_files)

if (total_files == 0) {
  stop("No se encontraron archivos .parquet en: ", input_dir)
}

log_cat("Input:  ", input_dir, "\n")
log_cat("Output: ", output_dir, "\n")
log_cat("Script: ", filter_script, "\n")
log_cat("Log:    ", main_log, "\n")
log_cat("Archivos encontrados: ", total_files, "\n\n")

# PROCESAR CADA ARCHIVO


start_time <- Sys.time()
processed <- 0
skipped <- 0
failed <- 0

for (i in seq_along(input_files)) {
  input_file <- input_files[i]
  input_parquet <- file.path(input_dir, input_file)

  # Generar nombre de output: agregar prefijo "filtered_"
  output_filename <- paste0("filtered_", input_file)
  output_file <- file.path(output_dir, output_filename)

  log_cat("\nArchivo ", i, "/", total_files, " \n")

  # Verificar si ya está procesado
  if (file.exists(output_file)) {
    log_cat("SKIP: Ya existe ", output_filename, "\n")
    skipped <- skipped + 1
    next
  }

  log_cat("Input:  ", input_file, "\n")
  log_cat("Output: ", output_filename, "\n")

  # Crear log específico para este archivo
  log_file <- file.path("/mnt/NTFS/gdelt_consolidated", paste0(
    "filter_log_", tools::file_path_sans_ext(input_file), "_",
    format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"
  ))

  log_cat("Log:    ", basename(log_file), "\n\n")
  log_cat("Procesando...\n")

  batch_start <- Sys.time()

  # Ejecutar script de filtrado
  result <- tryCatch(
    {
      # Establecer variables globales que el script espera
      assign("input_parquet", input_parquet, envir = .GlobalEnv)
      assign("output_file", output_file, envir = .GlobalEnv)
      assign("log_file", log_file, envir = .GlobalEnv)

      # Ejecutar script
      source(filter_script, local = FALSE)

      TRUE
    },
    error = function(e) {
      log_cat("ERROR: ", as.character(e), "\n")
      return(FALSE)
    }
  )

  batch_elapsed <- as.numeric(difftime(Sys.time(), batch_start, units = "secs"))

  if (result) {
    if (file.exists(output_file)) {
      file_size_mb <- file.info(output_file)$size / (1024^2)
      log_cat("ÉXITO en ", round(batch_elapsed, 1), " segundos\n")
      log_cat("   Tamaño: ", round(file_size_mb, 2), " MB\n")
      processed <- processed + 1
    } else {
      log_cat("Script completó pero no generó output\n")
      failed <- failed + 1
    }
  } else {
    log_cat("FALLÓ después de ", round(batch_elapsed, 1), " segundos\n")
    failed <- failed + 1
  }

  gc(verbose = FALSE)

  # Progress general
  total_elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  remaining_files <- total_files - i
  avg_time_per_file <- total_elapsed / i
  eta <- avg_time_per_file * remaining_files

  log_cat("\nProgreso: ", i, "/", total_files, " (", round((i / total_files) * 100, 1), "%)\n")
  log_cat("   Exitosos: ", processed, " | Saltados: ", skipped, " | Fallados: ", failed, "\n")
  log_cat("   Tiempo: ", round(total_elapsed, 1), " min | ETA: ", round(eta, 1), " min\n")
}


# RESUMEN FINAL


total_time <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))


log_cat("\nRESUMEN FINAL\n")


log_cat("Total archivos:    ", total_files, "\n")
log_cat("Total procesados:  ", processed, "\n")
log_cat("Total saltados:    ", skipped, "\n")
log_cat("Total fallados:    ", failed, "\n")
log_cat("Tiempo total:      ", round(total_time, 1), " minutos\n\n")

# Listar archivos generados
filtered_files <- list.files(output_dir, pattern = "^filtered_.*\\.parquet$", full.names = FALSE)
log_cat("Archivos filtrados generados: ", length(filtered_files), "\n")

if (length(filtered_files) > 0) {
  log_cat("\nArchivos:\n")
  for (f in filtered_files) {
    size_mb <- file.info(file.path(output_dir, f))$size / (1024^2)
    log_cat("  ", f, " (", round(size_mb, 2), " MB)\n")
  }
}

log_cat("\n")

if (processed == total_files) {
  log_cat("TODOS LOS ARCHIVOS PROCESADOS EXITOSAMENTE\n\n")
} else if (processed > 0) {
  log_cat("PROCESO INCOMPLETO\n\n")
  log_cat("Procesados: ", processed, "/", total_files, "\n")
  log_cat("Puedes re-ejecutar este script para procesar los faltantes\n")
  log_cat("(saltará automáticamente los ya procesados)\n\n")
} else {
  log_cat("NINGÚN ARCHIVO PROCESADO\n\n")
  log_cat("Verifica:\n")
  log_cat("  1. Ruta del script de filtrado: ", filter_script, "\n")
  log_cat("  2. Directorio de input: ", input_dir, "\n")
  log_cat("  3. Logs individuales en data/filter_log_*.txt\n\n")
}

log_cat("Log completo: ", main_log, "\n\n")
