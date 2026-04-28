import pandas as pd

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
    df.rename(columns={'Unnamed: 0' : 'Indikator', 'Unnamed: 1' : 'Einheit'}, inplace=True)
    df = df.drop(['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014'], axis=1)
    df.loc[1, 'Indikator'] = 'Reallohnveränderung zum Vorjahr'
    df.loc[3, 'Indikator'] = 'Nominallohnveränderung zum Vorjahr'

    for col in df.columns[2:]:
        df[col] = df[col].astype(str).str.replace(',','.')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.loc[1, '2021'] = 0

    return df

def calculate_vpi(df):
    """
    Berechnet den VPI aufgrundlage der Daten aus dem gegebenen DataFrame

    Input:
        DataFrame

    Return:
        DataFrame
    """
    
    vpi_row = {}

    for col in df.columns[2:]:
        # VPI = 100 * (Nominallohnindex / Reallohnindex)
        vpi_row[col] = str(round(100*(df[col][2]/df[col][0]), 1))
    
    df.loc[len(df)] = vpi_row
    df.loc[4, 'Indikator'] = 'Verbraucherpreisindex'
    df.loc[4, 'Einheit'] = '2025=100'
    
    return df

if __name__ == "__main__":
    data = read_and_clean_data()
    data = calculate_vpi(data)
    print(data)