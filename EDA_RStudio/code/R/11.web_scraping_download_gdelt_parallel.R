# 01_download_gdelt_parallel.R
library(data.table)
library(parallel)

# Configuración
base_url <- "http://data.gdeltproject.org/gkg/"
# start_date <- as.Date("2020-01-01")
# end_date <- as.Date("2025-10-16")
# output_dir <- "data/gdelt_raw"
# log_file <- file.path("data", paste0("download_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))
n_cores <- parallel::detectCores() - 1 # Dejar 1 core libre

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Generar fechas
dates <- seq(start_date, end_date, by = "day")
date_strings <- format(dates, "%Y%m%d")

# Calcular diferencia de fechas
total_days <- as.numeric(end_date - start_date) + 1
log_cat("=== CONFIGURACIÓN ===\n")
log_cat("Fecha inicio:", format(start_date, "%Y-%m-%d"), "\n")
log_cat("Fecha fin:   ", format(end_date, "%Y-%m-%d"), "\n")
log_cat("Total días:  ", total_days, "\n")
log_cat("Total archivos:", length(date_strings), "\n")
log_cat("Cores a usar:", n_cores, "\n\n")

# PRE-CHECK: Verificar archivos existentes
log_cat("=== PRE-CHECK DE ARCHIVOS EXISTENTES ===\n")
existing_files <- list.files(output_dir,
  pattern = "\\.gkg\\.csv\\.zip$",
  full.names = TRUE
)
existing_dates <- gsub(".*/([0-9]{8})\\.gkg\\.csv\\.zip", "\\1", existing_files)

# Validar archivos existentes (tamaño > 1KB)
valid_existing <- sapply(existing_files, function(f) {
  file.info(f)$size > 1000
})

existing_dates_valid <- existing_dates[valid_existing]
invalid_files <- existing_files[!valid_existing]

# Eliminar archivos inválidos
if (length(invalid_files) > 0) {
  log_cat("Eliminando", length(invalid_files), "archivos corruptos...\n")
  file.remove(invalid_files)
}

log_cat("Archivos válidos existentes:", length(existing_dates_valid), "\n")
log_cat("Archivos por descargar:     ", length(date_strings) - length(existing_dates_valid), "\n\n")

# Filtrar solo fechas faltantes
dates_to_download <- setdiff(date_strings, existing_dates_valid)

if (length(dates_to_download) == 0) {
  log_cat("¡Todos los archivos ya están descargados!\n")
  log_cat("Generando log de archivos existentes...\n")

  existing_info <- data.table(
    date = existing_dates_valid,
    status = "exists",
    size = sapply(
      file.path(output_dir, paste0(existing_dates_valid, ".gkg.csv.zip")),
      function(f) file.info(f)$size
    )
  )
  fwrite(existing_info, "data/download_log.csv")

  total_size_gb <- sum(existing_info$size, na.rm = TRUE) / (1024^3)
  log_cat("Tamaño total:", round(total_size_gb, 2), "GB\n")

  quit(save = "no")
}

# Función de descarga con reintentos
download_with_retry <- function(date_str, output_dir, base_url, max_retries = 3) {
  file_name <- paste0(date_str, ".gkg.csv.zip")
  url <- paste0(base_url, file_name)
  dest_file <- file.path(output_dir, file_name)

  # Intentar descarga con reintentos
  for (attempt in 1:max_retries) {
    result <- tryCatch(
      {
        download.file(url, dest_file,
          mode = "wb", quiet = TRUE,
          method = "libcurl", timeout = 120
        )
        file_size <- file.info(dest_file)$size

        if (file_size < 1000) {
          unlink(dest_file)
          stop("Archivo muy pequeño")
        }

        Sys.sleep(runif(1, 0.5, 1.5)) # Pausa aleatoria
        return(list(date = date_str, status = "downloaded", size = file_size))
      },
      error = function(e) {
        if (attempt < max_retries) {
          Sys.sleep(2^attempt) # Backoff exponencial
          return(NULL)
        } else {
          return(list(
            date = date_str, status = "error",
            message = conditionMessage(e)
          ))
        }
      }
    )

    if (!is.null(result)) break
  }

  return(result)
}

# Wrapper con barra de progreso
download_with_progress <- function(dates_vector, output_dir, base_url, n_cores) {
  total <- length(dates_vector)
  results <- vector("list", total)

  # Dividir en chunks para actualizar progreso
  chunk_size <- max(1, ceiling(total / (n_cores * 10)))
  n_chunks <- ceiling(total / chunk_size)

  log_cat("=== INICIANDO DESCARGA ===\n")
  log_cat("Archivos a descargar:", total, "\n")
  log_cat("Procesando en", n_chunks, "chunks de ~", chunk_size, "archivos\n\n")

  completed <- 0
  start_time <- Sys.time()

  for (i in 1:n_chunks) {
    # Índices del chunk
    start_idx <- (i - 1) * chunk_size + 1
    end_idx <- min(i * chunk_size, total)
    chunk_dates <- dates_vector[start_idx:end_idx]

    # Descargar chunk en paralelo
    chunk_results <- mclapply(chunk_dates, download_with_retry,
      output_dir = output_dir,
      base_url = base_url,
      mc.cores = n_cores
    )

    # Guardar resultados
    results[start_idx:end_idx] <- chunk_results

    # Actualizar progreso
    completed <- end_idx
    progress_pct <- round((completed / total) * 100, 1)

    # Calcular estadísticas
    downloaded_now <- sum(sapply(chunk_results, function(x) x$status == "downloaded"))
    errors_now <- sum(sapply(chunk_results, function(x) x$status == "error"))

    # Estimar tiempo restante
    elapsed_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    avg_time_per_file <- elapsed_time / completed
    remaining_files <- total - completed
    eta_secs <- avg_time_per_file * remaining_files

    # Barra de progreso visual
    bar_width <- 50
    filled <- round((completed / total) * bar_width)
    bar <- paste0(
      "[",
      paste(rep("=", filled), collapse = ""),
      paste(rep(" ", bar_width - filled), collapse = ""),
      "]"
    )

    # Imprimir progreso
    log_cat(
      "\r", bar, sprintf(" %5.1f%%", progress_pct),
      sprintf(" | %d/%d archivos", completed, total),
      sprintf(" | Desc: %d | Err: %d", downloaded_now, errors_now),
      sprintf(
        " | ETA: %s",
        if (eta_secs < 60) {
          sprintf("%.0fs", eta_secs)
        } else {
          sprintf("%.1fmin", eta_secs / 60)
        }
      )
    )

    flush.console()
  }

  log_cat("\n\n")

  return(results)
}

# Ejecutar descarga con progreso
start_time <- Sys.time()
results <- download_with_progress(dates_to_download, output_dir, base_url, n_cores)

# Combinar con archivos existentes
existing_results <- lapply(existing_dates_valid, function(d) {
  list(
    date = d,
    status = "exists",
    size = file.info(file.path(output_dir, paste0(d, ".gkg.csv.zip")))$size
  )
})

all_results <- c(results, existing_results)
results_dt <- rbindlist(all_results, fill = TRUE)
results_dt <- results_dt[order(date)]

# Guardar log
fwrite(results_dt, "data/download_log.csv")

# Análisis de resultados
log_cat("\n=== RESUMEN FINAL ===\n")
log_cat("Total fechas procesadas:", nrow(results_dt), "\n")
log_cat("Ya existían:           ", sum(results_dt$status == "exists"), "\n")
log_cat("Descargados ahora:     ", sum(results_dt$status == "downloaded"), "\n")
log_cat("Errores:               ", sum(results_dt$status == "error"), "\n")
log_cat(
  "Tasa de éxito:         ",
  sprintf("%.1f%%", 100 * (1 - sum(results_dt$status == "error") / nrow(results_dt))), "\n"
)

# Guardar lista de errores
errors <- results_dt[status == "error"]
if (nrow(errors) > 0) {
  fwrite(errors, "data/download_errors.csv")
  log_cat("\n⚠ Archivos con error guardados en: data/download_errors.csv\n")
  log_cat("Fechas con error:\n")
  log_cat(errors$date)
}

# Tamaño total
total_size_gb <- sum(results_dt$size, na.rm = TRUE) / (1024^3)
log_cat("\nTamaño total descargado:", round(total_size_gb, 2), "GB\n")

# Tiempo total
total_time <- difftime(Sys.time(), start_time, units = "mins")
log_cat("Tiempo total de descarga:", round(as.numeric(total_time), 1), "minutos\n")
