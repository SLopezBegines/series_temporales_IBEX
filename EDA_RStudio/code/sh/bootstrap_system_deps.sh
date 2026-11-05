#!/usr/bin/env bash
set -euo pipefail

# 0) Requisitos básicos
sudo apt update

# 1) Toolchain y utilidades
sudo apt install -y \
  build-essential gfortran pkg-config \
  git wget curl ca-certificates

# 2) Librerías de imagen / texto (ragg, textshaping, systemfonts, etc.)
sudo apt install -y \
  libfreetype6-dev libpng-dev libjpeg-dev libtiff-dev libwebp-dev \
  libharfbuzz-dev libfribidi-dev libfontconfig1-dev libglib2.0-dev \
  zlib1g-dev libbz2-dev

# 3) BLAS/LAPACK (fracdiff, forecast, etc.)
sudo apt install -y \
  libblas-dev liblapack-dev libopenblas-dev

# 4) Red/SSL/XML (curl/httr/readr/quantmod a veces tiran de esto)
sudo apt install -y \
  libcurl4-openssl-dev libssl-dev libxml2-dev

# 5) (Opcional) X11 headers (algunos paquetes gráficos antiguos lo requieren)
sudo apt install -y \
  libx11-dev libxext-dev libxrender-dev libxt-dev || true

# 6) Asegurar PKG_CONFIG_PATH para .pc de /usr/lib/x86_64-linux-gnu
PCP="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:/usr/share/pkgconfig"
if ! grep -q '^PKG_CONFIG_PATH=' "$HOME/.Renviron" 2>/dev/null; then
  echo "PKG_CONFIG_PATH=$PCP" >> "$HOME/.Renviron"
fi

echo ">> Dependencias del sistema instaladas."
echo ">> Cierra y reabre la sesión o ejecuta: export PKG_CONFIG_PATH=$PCP"
