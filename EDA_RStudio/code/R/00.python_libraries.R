# 2) Asegurar conda disponible (usa Miniconda de reticulate si no hay conda del sistema)
library(reticulate)
Sys.unsetenv("RETICULATE_PYTHON")
Sys.unsetenv("RETICULATE_PYTHON_FALLBACK")

conda_ok <- tryCatch(
  {
    !is.null(reticulate::conda_binary())
  },
  error = function(e) FALSE
)

if (!conda_ok) {
  message("No se detecta conda. Instalando Miniconda para reticulate...")
  reticulate::install_miniconda()
}

# 3) Crear/actualizar entorno Python 'ts_pyenv' con Python 3.10
env_name <- "ts_pyenv"
py_ver <- "3.10"

envs <- tryCatch(reticulate::conda_list(), error = function(e) data.frame())
if (!env_name %in% envs$name) {
  message(sprintf(
    "Creando entorno conda '%s' (python=%s)...",
    env_name,
    py_ver
  ))
  reticulate::conda_create(
    envname = env_name,
    packages = paste0("python=", py_ver)
  )
  # Asegura pip
  reticulate::conda_install(envname = env_name, packages = "pip")
}

# 4) Instalar/actualizar numpy y pandas desde conda-forge (binarios estables)
reticulate::conda_install(
  envname = env_name,
  channel = "conda-forge",
  packages = c("numpy", "pandas", "tabulate", "tqdm")
)

# 5) Instalar/actualizar yfinance vía pip (recomendado por upstream)
reticulate::conda_install(
  envname = env_name,
  pip = TRUE,
  packages = c("pip", "yfinance")
)

# 6) Enlazar reticulate al entorno y robustez frente a curl_cffi
reticulate::use_condaenv(env_name, required = TRUE)
# Evita problemas reportados con curl_cffi en algunas builds
Sys.setenv(YF_USE_CURL_CFFI = "0")

# 7) Verificación de la configuración Python vista por reticulate
cfg <- reticulate::py_config()
message("Python detectado por reticulate: ", cfg$python)
message("NumPy path: ", cfg$numpy)
message("yfinance path: ", cfg$yfinance %||% "<no detectado por py_config()>")

# 8) Pruebas de import y versiones (sin usar red)
ok <- TRUE
tryCatch(
  {
    reticulate::py_run_string(
      "
import os, importlib, sys
os.environ['YF_USE_CURL_CFFI']=os.getenv('YF_USE_CURL_CFFI','0')
import numpy, pandas
yf = importlib.import_module('yfinance')
print('PYTHON:', sys.executable)
print('numpy:', numpy.__version__)
print('pandas:', pandas.__version__)
print('yfinance:', yf.__version__)
"
    )
  },
  error = function(e) {
    ok <<- FALSE
    message("ERROR importando módulos Python: ", conditionMessage(e))
    try(reticulate::py_last_error())
  }
)

if (!ok) {
  stop("Fallo en importación de módulos Python. Revisa mensajes previos.")
}


message("✔ Entorno Python configurado y verificado correctamente.")
