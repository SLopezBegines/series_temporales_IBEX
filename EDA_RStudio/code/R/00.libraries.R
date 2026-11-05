# --- Setup R + Python (reticulate) -------------------------------------------
# Ejecuta este chunk en R Markdown/Quarto
# Añade RSPM (Posit Package Manager), que incluye archivos binarios
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"))

# 1) Paquetes R requeridos (añade/quita según tu proyecto)
r_pkgs <- c(
  "reticulate",
  "tidyverse",
  "rmarkdown",
  "forecast",
  "prophet",
  "xgboost",
  "zoo",
  "lubridate",
  "corrplot",
  "arrow",
  "psych",
  "arsenal",
  "reactable",
  "plotly",
  "shiny",
  "shinyjs",
  "shinyWidgets",
  "shinycssloaders",
  "shinythemes",
  "patchwork",
  "reactable",
  "TTR", # runSD
  "pacman", # para gestión de paquetes
  "duckdb",
  "gert",
  "styler",
  "pak",
  "scales",
  "tseries",
  "moments",
  "FinTS",
  "urca",
  "vars",
  "lmtest",
  "esquisse",
  "ggrepel",
  "mgcv",
  "stringr",
  "tidytext",
  "ggnewscale",
  "caret",
  "recipes",
  "ggcorrplot",
  "DBI",
  "data.table",
  "randomForest",
  "descomponer",
  "janitor",
  "kableExtra",
  "fredr",
  "fastDummies"
)

# Instalación silenciosa de paquetes R faltantes
quiet_install <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace("pak", quietly = TRUE)) install.packages("pak")
    if (!requireNamespace(p, quietly = TRUE)) {
      pak::pkg_install(p, dependencies = TRUE)
    }
    suppressPackageStartupMessages(library(p, character.only = TRUE))
  }
}
quiet_install(r_pkgs)
# 2) (Opcional) Snapshot del entorno R del proyecto con renv
if (requireNamespace("renv", quietly = TRUE)) {
  # No captura paquetes Python, pero fija el setup R del proyecto.
  try(renv::snapshot(prompt = FALSE), silent = TRUE)
}

rm(r_pkgs, quiet_install)
message("✔ Entorno R configurado y verificado correctamente.")
