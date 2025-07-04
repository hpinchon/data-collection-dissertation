import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

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

df['GDP_per_Capita'] = pd.to_numeric(df['GDP_per_Capita'], errors='coerce')

gdp2012 = df[df['year'] == 2012][['country', 'GDP_per_Capita']].rename(columns={'GDP_per_Capita': 'GDP_pc_2012'})
gdp2012['log_gdp_pc_2012'] = np.log(gdp2012['GDP_pc_2012'])
df = df.merge(gdp2012[['country', 'log_gdp_pc_2012']], on='country', how='left')

# Create a composite institutional quality index
inst_cols = [
    'Control_of_Corruption', 'Government_Effectiveness',
    'Rule_of_Law', 'Voice_and_Accountability', 'Regulatory_Quality', 'cbi_index'
]
df['inst_quality'] = df[inst_cols].mean(axis=1)

# Select macro controls
macro_controls = [
    'GDP_Growth', 'Inflation_CPI', 'Terms_of_Trade',
    'Current_Account_Balance', 'realIR', 'broad_money_to_gdp',
    'Primary_fiscal_balance'
]

formula = (
    "inst_quality ~ Treatment_Group_x + log_gdp_pc_2012 + C(region)"
)

# Drop missing values
model_vars = ['inst_quality', 'Treatment_Group_x', 'region', 'log_gdp_pc_2012'] + macro_controls
df_model = df[model_vars].dropna()

# Run OLS regression
model = smf.ols(formula, data=df_model).fit(cov_type="cluster", cov_kwds={"groups": df_model["region"]})
print(model.summary())


