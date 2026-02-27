#!/usr/bin/env Rscript
# convert_zip_to_parquet.R
# Convierte ZIP directamente a Parquet (sin descomprimir)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  cat("Uso: Rscript convert_zip_to_parquet.R input.zip output.parquet\n")
  quit(status = 1)
}

input_zip <- args[1]
output_parquet <- args[2]

# Skip si ya existe
if (file.exists(output_parquet)) {
  cat("SKIP:", basename(output_parquet), "\n")
  quit(status = 0)
}

# Verificar input
if (!file.exists(input_zip)) {
  cat("ERROR: No existe", input_zip, "\n")
  quit(status = 1)
}

suppressMessages({
  library(data.table)
  library(arrow)
})

# Columnas GDELT 1.0
col_names <- c(
  "DATE", "NUMARTS", "COUNTS", "THEMES", "LOCATIONS",
  "PERSONS", "ORGANIZATIONS", "TONE", "CAMEOEVENTIDS",
  "SOURCES", "SOURCEURLS"
)

tryCatch({
  # Leer CSV directamente desde ZIP (sin descomprimir a disco)
  dt <- fread(cmd = paste0("unzip -p '", input_zip, "'"),
    sep = "\t",
    header = FALSE,
    quote = "",
    col.names = col_names,
    encoding = "UTF-8",
    showProgress = FALSE,
    skip = 1
  )
  
  # Escribir Parquet inmediatamente
  write_parquet(dt, output_parquet, compression = "snappy")
  
  cat("OK:", basename(output_parquet), "(", nrow(dt), "rows )\n")
  
  # Liberar memoria
  rm(dt)
  gc(verbose = FALSE)
  
  quit(status = 0)
  
}, error = function(e) {
  cat("ERROR:", basename(input_zip), "-", as.character(e), "\n")
  
  # Eliminar parquet corrupto si existe
  if (file.exists(output_parquet)) {
    unlink(output_parquet)
  }
  
  quit(status = 1)
})
