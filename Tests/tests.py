import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('your_dataset_interpolated.csv')

region_dict = {
    # Sub-Saharan Africa
    'AGO':'Sub-Saharan Africa', 'BDI':'Sub-Saharan Africa', 'BEN':'Sub-Saharan Africa', 'BFA':'Sub-Saharan Africa',
    'BWA':'Sub-Saharan Africa', 'CAF':'Sub-Saharan Africa', 'CIV':'Sub-Saharan Africa', 'CMR':'Sub-Saharan Africa',
    'COD':'Sub-Saharan Africa', 'COG':'Sub-Saharan Africa', 'COM':'Sub-Saharan Africa', 'CPV':'Sub-Saharan Africa',
    'DJI':'Sub-Saharan Africa', 'ERI':'Sub-Saharan Africa', 'ETH':'Sub-Saharan Africa', 'GAB':'Sub-Saharan Africa',
    'GHA':'Sub-Saharan Africa', 'GIN':'Sub-Saharan Africa', 'GMB':'Sub-Saharan Africa', 'GNB':'Sub-Saharan Africa',
    'KEN':'Sub-Saharan Africa', 'LBR':'Sub-Saharan Africa', 'MDG':'Sub-Saharan Africa', 'MLI':'Sub-Saharan Africa',
    'MOZ':'Sub-Saharan Africa', 'MRT':'Sub-Saharan Africa', 'MUS':'Sub-Saharan Africa', 'MWI':'Sub-Saharan Africa',
    'NER':'Sub-Saharan Africa', 'NGA':'Sub-Saharan Africa', 'RWA':'Sub-Saharan Africa', 'SEN':'Sub-Saharan Africa',
    'SLE':'Sub-Saharan Africa', 'SOM':'Sub-Saharan Africa', 'STP':'Sub-Saharan Africa', 'SWZ':'Sub-Saharan Africa',
    'TCD':'Sub-Saharan Africa', 'TGO':'Sub-Saharan Africa', 'TZA':'Sub-Saharan Africa', 'UGA':'Sub-Saharan Africa',
    'ZAF':'Sub-Saharan Africa', 'ZMB':'Sub-Saharan Africa', 'ZWE':'Sub-Saharan Africa',

    # East Asia & Pacific
    'CHN':'East Asia & Pacific', 'FJI':'East Asia & Pacific', 'IDN':'East Asia & Pacific', 'KHM':'East Asia & Pacific',
    'LAO':'East Asia & Pacific', 'MMR':'East Asia & Pacific', 'MNG':'East Asia & Pacific', 'PNG':'East Asia & Pacific',
    'PHL':'East Asia & Pacific', 'THA':'East Asia & Pacific', 'TLS':'East Asia & Pacific', 'TON':'East Asia & Pacific',
    'VNM':'East Asia & Pacific', 'VUT':'East Asia & Pacific', 'WSM':'East Asia & Pacific',

    # Europe & Central Asia
    'ALB':'Europe & Central Asia', 'ARM':'Europe & Central Asia', 'AZE':'Europe & Central Asia', 'BIH':'Europe & Central Asia',
    'BLR':'Europe & Central Asia', 'BGR':'Europe & Central Asia', 'GEO':'Europe & Central Asia', 'KAZ':'Europe & Central Asia',
    'KGZ':'Europe & Central Asia', 'MKD':'Europe & Central Asia', 'MDA':'Europe & Central Asia', 'MNE':'Europe & Central Asia',
    'ROU':'Europe & Central Asia', 'RUS':'Europe & Central Asia', 'SRB':'Europe & Central Asia', 'TJK':'Europe & Central Asia',
    'TKM':'Europe & Central Asia', 'TUR':'Europe & Central Asia', 'UKR':'Europe & Central Asia', 'UZB':'Europe & Central Asia',
    'XKX':'Europe & Central Asia',

    # Latin America & Caribbean
    'ARG':'Latin America & Caribbean', 'BLZ':'Latin America & Caribbean', 'BOL':'Latin America & Caribbean', 'BRA':'Latin America & Caribbean',
    'COL':'Latin America & Caribbean', 'CRI':'Latin America & Caribbean', 'DMA':'Latin America & Caribbean', 'DOM':'Latin America & Caribbean',
    'ECU':'Latin America & Caribbean', 'GRD':'Latin America & Caribbean', 'GTM':'Latin America & Caribbean', 'GUY':'Latin America & Caribbean',
    'HND':'Latin America & Caribbean', 'HTI':'Latin America & Caribbean', 'JAM':'Latin America & Caribbean', 'LCA':'Latin America & Caribbean',
    'MEX':'Latin America & Caribbean', 'NIC':'Latin America & Caribbean', 'PAN':'Latin America & Caribbean', 'PER':'Latin America & Caribbean',
    'PRY':'Latin America & Caribbean', 'SLV':'Latin America & Caribbean', 'SRB':'Europe & Central Asia', 'STP':'Sub-Saharan Africa',
    'SUR':'Latin America & Caribbean', 'TTO':'Latin America & Caribbean', 'URY':'Latin America & Caribbean', 'VCT':'Latin America & Caribbean',
    'VEN':'Latin America & Caribbean',

    # Middle East & North Africa
    'DZA':'Middle East & North Africa', 'EGY':'Middle East & North Africa', 'IRN':'Middle East & North Africa', 'JOR':'Middle East & North Africa',
    'LBN':'Middle East & North Africa', 'MAR':'Middle East & North Africa', 'SDN':'Sub-Saharan Africa', 'SYR':'Middle East & North Africa',
    'TUN':'Middle East & North Africa', 'YEM':'Middle East & North Africa',

    # South Asia
    'AFG':'South Asia', 'BGD':'South Asia', 'BTN':'South Asia', 'IND':'South Asia', 'LKA':'South Asia', 'MDV':'South Asia', 'NPL':'South Asia', 'PAK':'South Asia',
}

df['region'] = df['country'].map(region_dict)

# Calculate log GDP per capita in 2012 for each country
gdp2012 = df[df['year'] == 2012][['country', 'GDP_per_Capita']].copy()
gdp2012['log_gdp_pc_2012'] = np.log(gdp2012['GDP_per_Capita'])
df = df.merge(gdp2012[['country', 'log_gdp_pc_2012']], on='country', how='left')

# Average the outcome variable over 2012-2019 for each country
df_period = df[(df['year'] >= 2012) & (df['year'] <= 2019)]
agg = df_period.groupby('country').agg({
    'Rule_of_Law': 'mean',
    'Treatment_Group_x': 'last',  # or 'max', depending on your definition
    'log_gdp_pc_2012': 'first',
    'region': 'first'
}).reset_index()

model = smf.ols(
    "Rule_of_Law ~ Treatment_Group_x + log_gdp_pc_2012 + C(region)",
    data=agg
).fit()
print(model.summary())

