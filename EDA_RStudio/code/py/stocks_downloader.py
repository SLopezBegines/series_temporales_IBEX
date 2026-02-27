# -------------------------------------------------------------
# IBEX35 Full Historical Downloader (Funciones)
# Autor: SLB
# -------------------------------------------------------------
# Requisitos: pip install yfinance pandas
# -------------------------------------------------------------

import yfinance as yf
import pandas as pd
import os
# -------------------------------------------------------------
# 1️⃣ Función: descargar histórico del IBEX35 (u otro índice)


def download_ibex(ticker, start_date, end_date, export_csv=True):
    print(f"⬇️ Descargando histórico de {ticker}...")
    
    # Descargar datos desde Yahoo Finance
    ibex_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if not ibex_data.empty:
        print(f"✅ Datos descargados para {ticker}.")
        
        if export_csv:
            # Crear carpeta si no existe
            os.makedirs("./data", exist_ok=True)
            filename = f"./data/{ticker}_Historical.csv"
            ibex_data.to_csv(filename)
            print(f"💾 Guardado en: {filename}")
    else:
        print(f"⚠️ Sin datos para {ticker}.")
    
    return ibex_data


# -------------------------------------------------------------
# 2️⃣ Función: descargar histórico de todas las empresas del IBEX35 (o lista similar)
def download_ibex_companies(companies_list, start_date, end_date, export_csv=True):
    print("Descargando históricos de empresas del IBEX35...")
    results = {}  # dict con claves = nombres de empresa

    for company in companies_list:
        ticker = company["ticker"]
        name = company["name"]
        print(f"Descargando {name} ({ticker})...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if not data.empty:
            print(f"Datos descargados para {name} ({ticker}).")
            if export_csv:
                # Crear carpeta si no existe
                os.makedirs("./data", exist_ok=True)
                # Guardar CSV
                filename = f"./data/{ticker}_Historical.csv"
                data.to_csv(filename)
                print(f"Guardado: {filename}")
        else:
            print(f"Sin datos para {ticker}.")

        # Guardar en dict con clave = name
        results[name] = {
            "ticker": ticker,
            "name": name,
            "df": data
        }

    # Guardar metadata
    companies_df = pd.DataFrame(companies_list)
    if export_csv:
        # Crear carpeta si no existe
        os.makedirs("./data", exist_ok=True)
        # Guardar CSV
        companies_df = pd.DataFrame(companies_list)
        companies_df.to_csv("./data/IBEX35_Companies_Sectors.csv", index=False)
        print("Archivo maestro guardado: IBEX35_Companies_Sectors.csv")
    else:
        print("Archivo maestro no guardado (export_csv=False).")
        print("Descarga completa.")
    # Devolver dict con resultados
    return results, companies_df

# -------------------------------------------------------------
