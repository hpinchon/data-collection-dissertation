import pandas as pd
import csv

def detect_delimiter(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        try:
            return sniffer.sniff(sample).delimiter
        except csv.Error:
            return ',' 

files_info = [
    # ('filename.csv', year, 'country code column', 'cpi score column')
    ('/workspaces/data-collection-dissertation/CPI2012_Results 14.44.57.csv', 2012, 'WB Code', 'CPI 2012 Score'),
    ('CPI2013_Results_2022-01-20-183035_stnh.csv', 2013, 'WB Code', 'CPI 2013 Score'),
    ('CPI2014_Results.csv', 2014, 'WB Code', 'CPI 2014'),
    ('/workspaces/data-collection-dissertation/CPI_2015_FullDataSet_2022-01-18-145020_enyn_2022-01-20-180010_mabu.csv', 2015, 'Country Code', 'CPI 2015 Score'),
    ('CPI2016_Results.csv', 2016, 'WB Code', 'CPI2016'),
    ('/workspaces/data-collection-dissertation/CPI2017_Full_DataSet-1801.csv', 2017, 'ISO3', 'CPI Score 2017'),
    ('/workspaces/data-collection-dissertation/CPI2018_Full-Results_1801.csv', 2018, 'ISO3', 'CPI Score 2018'),
    ('/workspaces/data-collection-dissertation/CPI2019-1.csv', 2019, 'ISO3', 'CPI score 2019'),
]

dfs = []

for filename, year, code_col, score_col in files_info:
    delimiter = detect_delimiter(filename)
    df = pd.read_csv(filename, delimiter=delimiter)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = [col.strip() for col in df.columns]
    df = df[[code_col, score_col]].copy()
    df.columns = ['Country Code', 'CPI Score']
    df['Year'] = year
    df = df.dropna(subset=['Country Code', 'CPI Score'])
    df['CPI Score'] = pd.to_numeric(df['CPI Score'], errors='coerce')
    df = df.dropna(subset=['CPI Score'])
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv('CPI_Merged.csv', index=False)

