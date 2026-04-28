import pandas as pd

def read_and_clean_data():
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

if __name__ == "__main__":
    data = read_and_clean_data()
    print(data)