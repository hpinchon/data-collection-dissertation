import pandas as pd

df = pd.read_csv('merged_data.csv')
# List of ex-HIPC country codes
ex_hipc_iso3 = [
    'AFG', 'BEN', 'BOL', 'BFA', 'BDI', 'CMR', 'CAF', 'TCD', 'COM', 'COG', 'COD', 'CIV', 'ETH', 'GMB',
    'GHA', 'GIN', 'GNB', 'GUY', 'HTI', 'HND', 'LBR', 'MDG', 'MWI', 'MLI', 'MRT', 'MOZ', 'NIC', 'NER',
    'RWA', 'STP', 'SEN', 'SLE', 'TZA', 'TGO', 'UGA', 'ZMB'
]

# DataFrame with unique countries and HIPC status
hipc_df = pd.DataFrame({'country': list(set(df['country']))})
hipc_df['Former_HIPC'] = hipc_df['country'].isin(ex_hipc_iso3).astype(int)

df = df.merge(hipc_df, on='country', how='left')
df.to_csv('merged_data_with_hipc.csv', index=False)



