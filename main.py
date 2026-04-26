import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pmdarima as pm
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings

# Warnungen für sauberen Output ignorieren
warnings.filterwarnings("ignore")

# ==========================================
# KONFIGURATION (Hier später anpassen!)
# ==========================================
GENESIS_USER = 'IHR_BENUTZERNAME'  # Hier später Ihren Destatis-User eintragen
GENESIS_PASS = 'IHR_PASSWORT'      # Hier später Ihr Destatis-Passwort eintragen

# ==========================================
# STUFE 1: DATENGEWINNUNG (EXTRACT)
# ==========================================
def fetch_genesis_data(table_code, start_year, end_year):
    """Holt Daten von der Destatis API. Fallback auf Dummy-Daten, falls kein Login existiert."""
    
    if GENESIS_USER == 'IHR_BENUTZERNAME':
        print(f"[*] Kein API-Login gefunden. Lade Dummy-Daten für Tabelle {table_code}...")
        return generate_dummy_data(table_code, start_year, end_year)

    url = "https://www-genesis.destatis.de/genesisWS/rest/2020/data/table"
    params = {
        'username': GENESIS_USER, 'password': GENESIS_PASS,
        'name': table_code, 'area': 'all', 'compress': 'false',
        'transpose': 'false', 'startyear': str(start_year),
        'endyear': str(end_year), 'language': 'de', 'format': 'csv'
    }
    
    print(f"[*] Sende API-Request für Tabelle {table_code}...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        # Destatis CSVs haben oft Metadaten oben und unten.
        # WICHTIG für Ihr Projekt: skiprows und skipfooter müssen Sie 
        # je nach exakter Tabelle später feintunen!
        df = pd.read_csv(StringIO(response.text), sep=';', skiprows=5, skipfooter=3, engine='python')
        return df
    else:
        raise Exception(f"API Fehler: {response.status_code} - {response.text}")

def generate_dummy_data(table_code, start_year, end_year):
    """Generiert realistische Testdaten für den Proof of Concept."""
    jahre = list(range(start_year, end_year + 1))
    if '61111' in table_code: # VPI Dummy (Inflation steigt extrem ab 2021)
        werte = [90 + i*1.2 if year < 2021 else 90 + i*1.2 + (year-2020)*5 for i, year in enumerate(jahre)]
        return pd.DataFrame({'Jahr': jahre, 'VPI': werte})
    else: # Lohn Dummy (Löhne steigen linear)
        werte = [92 + i*2.0 for i, year in enumerate(jahre)]
        return pd.DataFrame({'Jahr': jahre, 'Nominallohn': werte})


# ==========================================
# STUFE 2: DATENAUFBEREITUNG (TRANSFORM)
# ==========================================
def process_data(df_vpi, df_lohn):
    """Führt die Tabellen zusammen und berechnet die Kaufkraft (Reallohn)."""
    print("[*] Verarbeite und mergen der Daten...")
    
    # In der echten API heißen die Spalten oft kryptisch. 
    # Hier tun wir so, als hätten wir sie schon bereinigt auf 'Jahr', 'VPI' und 'Nominallohn'.
    
    # 1. Merge über das Jahr
    df_merged = pd.merge(df_lohn, df_vpi, on='Jahr', how='inner')
    
    # 2. Berechnung der Kaufkraft (Reallohnindex)
    df_merged['Reallohn'] = (df_merged['Nominallohn'] / df_merged['VPI']) * 100
    df_merged['Reallohn'] = df_merged['Reallohn'].round(2)
    
    # Index auf das Jahr setzen (wichtig für Zeitreihenanalyse!)
    df_merged.set_index('Jahr', inplace=True)
    return df_merged


# ==========================================
# STUFE 3: PROGNOSE (MODELING)
# ==========================================
def create_forecast(time_series, forecast_years=3):
    """Erstellt eine ARIMA-Prognose für die nächsten X Jahre."""
    print(f"[*] Trainiere ARIMA-Modell für {forecast_years} Jahre in die Zukunft...")
    
    # Auto-ARIMA sucht automatisch nach den besten statistischen Parametern (p,d,q)
    model = pm.auto_arima(time_series, 
                          seasonal=False, # Da wir Jahresdaten haben, gibt es keine klassische Saisonalität
                          stepwise=True, 
                          suppress_warnings=True,
                          error_action="ignore")
    
    # Prognose und Konfidenzintervalle (Unsicherheitsbereich) berechnen
    forecast, conf_int = model.predict(n_periods=forecast_years, return_conf_int=True)
    
    # Schönen Pandas DataFrame für die Ergebnisse bauen
    last_year = time_series.index[-1]
    future_years = np.arange(last_year + 1, last_year + 1 + forecast_years)
    
    df_forecast = pd.DataFrame({
        'Prognose': forecast,
        'Unteres_KI': conf_int[:, 0],
        'Oberes_KI': conf_int[:, 1]
    }, index=future_years)
    
    return df_forecast


# ==========================================
# STUFE 4: VISUALISIERUNG (REPORTING)
# ==========================================
def plot_pipeline_results(df_history, df_forecast):
    """Erstellt einen Publikations-reifen Graphen mit Seaborn/Matplotlib."""
    print("[*] Erstelle Visualisierung...")
    
    # Seaborn Style aktivieren für modernere Optik
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # 1. Historische Daten plotten
    plt.plot(df_history.index, df_history['Nominallohn'], label='Nominallohnindex', color='blue', marker='o')
    plt.plot(df_history.index, df_history['VPI'], label='Verbraucherpreisindex (VPI)', color='red', marker='o')
    plt.plot(df_history.index, df_history['Reallohn'], label='Reallohnindex (Kaufkraft)', color='green', linewidth=3, marker='s')
    
    # 2. Prognose plotten (gestrichelt)
    plt.plot(df_forecast.index, df_forecast['Prognose'], label='Prognose Kaufkraft', color='green', linestyle='--', marker='s')
    
    # Konfidenzintervall (Unsicherheit) als schattierten Bereich einzeichnen
    plt.fill_between(df_forecast.index, 
                     df_forecast['Unteres_KI'], 
                     df_forecast['Oberes_KI'], 
                     color='green', alpha=0.2, label='Konfidenzintervall (95%)')
    
    # 3. Graph aufhübschen
    plt.title('Entwicklung und Prognose der Kaufkraft in Deutschland', fontsize=16, fontweight='bold')
    plt.xlabel('Jahr', fontsize=12)
    plt.ylabel('Indexwert (Basis = 100)', fontsize=12)
    plt.axvline(x=df_history.index[-1], color='gray', linestyle=':', label='Ende Ist-Daten')
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    # Bild speichern und anzeigen
    plt.savefig('kaufkraft_analyse.png', dpi=300)
    print("[*] Graph als 'kaufkraft_analyse.png' gespeichert.")
    plt.show()


# ==========================================
# HAUPTPROGRAMM (Hier läuft die Pipeline ab)
# ==========================================
if __name__ == "__main__":
    print("=== START DATA PIPELINE ===")
    
    # Zeitraum festlegen
    START_JAHR = 2010
    END_JAHR = 2023
    
    # 1. Extract
    df_raw_vpi = fetch_genesis_data('61111-0002', START_JAHR, END_JAHR)
    df_raw_lohn = fetch_genesis_data('62231-0001', START_JAHR, END_JAHR)
    
    # 2. Transform
    df_clean = process_data(df_raw_vpi, df_raw_lohn)
    print("\nBereinigte historische Daten (Auszug):")
    print(df_clean.tail())
    
    # 3. Modeling (Prognose für 4 Jahre: 2024 - 2027)
    df_pred = create_forecast(df_clean['Reallohn'], forecast_years=4)
    print("\nPrognoseergebnisse:")
    print(df_pred.round(2))
    
    # 4. Visualization
    plot_pipeline_results(df_clean, df_pred)
    
    print("=== PIPELINE ERFOLGREICH BEENDET ===")