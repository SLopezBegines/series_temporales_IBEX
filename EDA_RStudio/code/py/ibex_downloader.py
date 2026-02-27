# -------------------------------------------------------------
# IBEX35 Full Historical Downloader
# Autor: ChatGPT
# -------------------------------------------------------------
# ⚙️ Requisitos: pip install yfinance pandas
# -------------------------------------------------------------

import yfinance as yf
import pandas as pd
import os

# -------------------------------------------------------------
# 1️⃣ Parámetros globales
IBEX_TICKER = "^IBEX"
#START_DATE = "1992-01-14"
#END_DATE = "2025-07-08"

# -------------------------------------------------------------
# 2️⃣ Lista de empresas + sectores
# ⚠️ Actualiza tickers y sectores según corresponda
ibex_companies = [
    {"ticker": "ANA.MC", "name": "Acciona", "sector": "Infraestructuras"},
    {"ticker": "ANE.MC", "name": "Acciona Energía", "sector": "Energia"},
    {"ticker": "ACX.MC", "name": "Acerinox", "sector": "Siderurgia"},
    {"ticker": "ACS.MC", "name": "ACS", "sector": "Construccion"},
    {"ticker": "Aena.MC", "name": "Aena", "sector": "Infraestructuras"},
    {"ticker": "AMS.MC", "name": "Amadeus", "sector": "Información y comunicaciones"},
    {"ticker": "MTS.MC", "name": "ArcelorMittal", "sector": "Siderurgia"},
    {"ticker": "BBVA.MC", "name": "BBVA", "sector": "Banca"},
    {"ticker": "SAB.MC", "name": "Banco Sabadell", "sector": "Banca"},
    {"ticker": "BKT.MC", "name": "Bankinter", "sector": "Banca"},
    {"ticker": "CABK.MC", "name": "CaixaBank", "sector": "Banca"},
    {"ticker": "CLNX.MC", "name": "Cellnex Telecom", "sector": "Información y comunicaciones"},
    {"ticker": "ENG.MC", "name": "Enagás", "sector": "Energia/Biomasa"},
    {"ticker": "ELE.MC", "name": "Endesa", "sector": "Energia"},
    {"ticker": "FER.MC", "name": "Ferrovial", "sector": "Construccion"},
    {"ticker": "FDR.MC", "name": "Fluidra", "sector": "Wellness"},
    {"ticker": "GRF.MC", "name": "Grifols", "sector": "Biotecnologia"},
    {"ticker": "IAG.MC", "name": "International Airlines Group", "sector": "Aerolineas"},
    {"ticker": "IBE.MC", "name": "Iberdrola", "sector": "Energia"},
    {"ticker": "ITX.MC", "name": "Inditex", "sector": "Retail/Moda"},
    {"ticker": "IDR.MC", "name": "Indra", "sector": "Información y comunicaciones"},
    {"ticker": "COL.MC", "name": "Colonial", "sector": "Inmobiliaria"},
    {"ticker": "LOG.MC", "name": "Logista", "sector": "Logistica"},
    {"ticker": "MAP.MC", "name": "MAPFRE", "sector": "Seguros"},
    {"ticker": "MRL.MC", "name": "Merlin Properties", "sector": "Inmobiliaria"},
    {"ticker": "NTGY.MC", "name": "Naturgy", "sector": "Energia"},
    {"ticker": "PUIG.MC", "name": "Puig Brands", "sector": "Actividades de las sociedades holding"},
    {"ticker": "REE.MC", "name": "Redeia", "sector": "Energia"},
    {"ticker": "REP.MC", "name": "Repsol", "sector": "Energia"},
    {"ticker": "ROVI.MC", "name": "Rovi", "sector": "Biotecnologia"},
    {"ticker": "SCYR.MC", "name": "Sacyr", "sector": "Construccion"},
    {"ticker": "SAN.MC", "name": "Santander", "sector": "Banca"},
    {"ticker": "SLR.MC", "name": "Solaria", "sector": "Energia"},
    {"ticker": "TEF.MC", "name": "Telefónica", "sector": "Información y comunicaciones"},
    {"ticker": "UNI.MC", "name": "Unicaja Banco", "sector": "Banca"},
    {"ticker": "MEL.MC", "name": "Meliá Hotels", "sector": "Turismo"},
    {"ticker": "PHM.MC", "name": "PharmaMar", "sector": "Biotecnología"},
    {"ticker": "ALM.MC", "name": "Almirall", "sector": "Farmacéutica"},
    {"ticker": "CIE.MC", "name": "CIE Automotive", "sector": "Automoción"},
    {"ticker": "SGRE.MC", "name": "Siemens Gamesa", "sector": "Energía Renovable"},
    {"ticker": "SOL.MC", "name": "Solaria", "sector": "Energía Renovable"},
    {"ticker": "VIS.MC", "name": "Viscofan", "sector": "Alimentación"},
    {"ticker": "RCI.MC", "name": "Renta Corporación", "sector": "Inmobiliaria"},
    {"ticker": "BKT.MC", "name": "Bankinter", "sector": "Banca"},
    {"ticker": "SNG.MC", "name": "Soltec", "sector": "Energía Renovable"},
    # ➕ Sustituye o completa tickers actuales según BME
]

# -------------------------------------------------------------
# 3️⃣ Descargar IBEX35 histórico
print(f"📈 Descargando histórico del IBEX35...")
ibex_data = yf.download(IBEX_TICKER, start=START_DATE, end=END_DATE, progress=False)
ibex_data.to_csv("./data/IBEX35_Historical.csv")
print("✅ Archivo guardado: IBEX35_Historical.csv")

# -------------------------------------------------------------
# 4️⃣ Descargar datos de cada empresa
for company in ibex_companies:
    ticker = company["ticker"]
    name = company["name"]
    print(f"⬇️ Descargando {name} ({ticker})...")
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if not data.empty:
        filename = f"{ticker}_Historical.csv"
        filename = f"./data/{filename}"
        data.to_csv(filename)
        print(f"✅ Guardado: {filename}")
    else:
        print(f"⚠️ Sin datos para {ticker}.")

# -------------------------------------------------------------
# 5️⃣ Crear archivo maestro de empresas + sectores
companies_df = pd.DataFrame(ibex_companies)
companies_df.to_csv("IBEX35_Companies_Sectors.csv", index=False)
print("✅ Archivo maestro guardado: IBEX35_Companies_Sectors.csv")

print("🚀 Todo completado.")
