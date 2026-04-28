import pandas as pd
import pmdarima as pm
import numpy as np

"""
+--------------------------------+
| Datentransformation            |
+--------------------------------+
"""
def read_and_clean_data():
    """
    Importiert die CSV und wandelt sie mittels pandas in einen DataFrame um.
    Dieser DataFrame und die Daten in ihm werden anschließend bereinigt.

    Input:
        eine Datei namens 'cleaned_data.csv' im Verzeichnis von main.py
    
    Return:
        DataFrame
    """

    df = pd.read_csv('cleaned_data.csv', sep=";", skiprows=5, nrows=4)
    #df = df.drop(['2007', '2008', '2009'], axis=1)
    df.rename(columns={'Unnamed: 0' : 'Indikator', 'Unnamed: 1' : 'Einheit'}, inplace=True)
    df.loc[1, 'Indikator'] = 'Reallohnveränderung zum Vorjahr'
    df.loc[3, 'Indikator'] = 'Nominallohnveränderung zum Vorjahr'
    

    for col in df.columns[2:]:
        df[col] = df[col].astype(str).str.replace(',','.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df.loc[1, '2021'] = 0

    return df

"""
+--------------------------------+
| Datenberechnung                |
+--------------------------------+
"""
def calculate_vpi(df):
    """
    Berechnet den VPI + Veraenderung des VPIs aufgrundlage der Daten aus dem gegebenen DataFrame

    Input:
        DataFrame

    Return:
        DataFrame
    """
    
    vpi_row = {}

    for col in df.columns[2:]:
        # VPI = 100 * (Nominallohnindex / Reallohnindex)
        vpi_row[col] = round(100*(df[col][2]/df[col][0]), 1)
    
    df.loc[len(df)] = vpi_row
    df.loc[4, 'Indikator'] = 'Verbraucherpreisindex'
    df.loc[4, 'Einheit'] = '2025=100'

    vpi_change_row = {}

    previous_col = None
    for col in df.columns[2:]:
        if previous_col is not None:
            change = round(((float(df[col][4])/float(df[previous_col][4])) - 1) * 100, 1)
            vpi_change_row[col] = change
        else:
            vpi_change_row[col] = np.nan
        previous_col = col
    
    df.loc[len(df)] = vpi_change_row
    df.loc[5, 'Indikator'] = 'VPI-Veraenderung zum Vorjahr'
    df.loc[5, 'Einheit'] = 'in (%)'

    
    return df
"""
+--------------------------------+
| Prognose                       |
+--------------------------------+
"""
def create_forecast(df, years):
    df_nli = pd.to_numeric(df.loc[2, '2007':'2025'])
    df_vpi = pd.to_numeric(df.loc[4, '2007':'2025'])

    model_nli = pm.auto_arima(df_nli, stepwise=True)
    model_vpi = pm.auto_arima(df_vpi, stepwise=True)

    forecast_nli, conf_int_nli = model_nli.predict(n_periods=years, return_conf_int=True)
    forecast_vpi, conf_int_vpi = model_vpi.predict(n_periods=years, return_conf_int=True)

    # RLI = (NLI / VPI) * 100
    forecast_rli = (np.array(forecast_nli) / np.array(forecast_vpi)) * 100

    conf_int_rli_low = (conf_int_nli[:, 0] / conf_int_vpi[:, 1]) * 100
    conf_int_rli_high = (conf_int_nli[:, 1] / conf_int_vpi[:, 0]) * 100

    last_year = int(df_nli.index[-1])
    future_years = np.arange(last_year + 1, last_year + 1 + years)

    df_forecast_nli = pd.DataFrame({
        'Prognose': np.array(forecast_nli),
        'Unteres_Konfidenz_Intervall': conf_int_nli[:, 0],
        'Oberes_Konfidenz_Intervall': conf_int_nli[:, 1]
    }, index=future_years)

    df_forecast_vpi = pd.DataFrame({
        'Prognose': np.array(forecast_vpi),
        'Unteres_Konfidenz_Intervall': conf_int_vpi[:, 0],
        'Oberes_Konfidenz_Intervall': conf_int_vpi[:, 1]
    }, index=future_years)

    df_forecast_rli = pd.DataFrame({
        'Prognose': forecast_rli,
        'Unteres_Konfidenz_Intervall': conf_int_rli_low,
        'Oberes_Konfidenz_Intervall': conf_int_rli_high
    }, index=future_years)

    return (df_forecast_rli, df_forecast_nli, df_forecast_vpi)


if __name__ == "__main__":
    data = read_and_clean_data()
    data = calculate_vpi(data)
    print(data)
    forecast_data = create_forecast(data, 5)
    counter = 0
    for data in forecast_data:
        if counter == 0:
            print(f"================RLI================")
        elif counter == 1:
            print(f"================NLI================")
        else:
            print(f"================VPI================")
        print(data)
        
        counter += 1