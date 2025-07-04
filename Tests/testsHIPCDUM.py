import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('ITHINKITSFULLNOW.csv')

if 'cbi_index' in df.columns:
    df['cbi_index'] = (
        df['cbi_index']
        .astype(str)
        .str.replace(',', '.', regex=False)
    )
    df['cbi_index'] = pd.to_numeric(df['cbi_index'], errors='coerce')

ex_hipc_iso3 = ['AFG', 'BEN', 'BOL', 'BFA', 'BDI', 'CMR', 'CAF', 'TCD', 'COM', 'COG', 'COD', 'CIV', 'ETH', 'GMB', 'GHA', 'GIN', 'GNB', 'GUY', 'HTI', 'HND', 'LBR', 'MDG', 'MWI', 'MLI', 'MRT', 'MOZ', 'NIC', 'NER', 'RWA', 'STP', 'SEN', 'SLE', 'TZA', 'TGO', 'UGA', 'ZMB']

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
    'PRY':'Latin America & Caribbean', 'SLV':'Latin America & Caribbean', 'SUR':'Latin America & Caribbean', 'TTO':'Latin America & Caribbean',
    'URY':'Latin America & Caribbean', 'VCT':'Latin America & Caribbean', 'VEN':'Latin America & Caribbean',
    # Middle East & North Africa
    'DZA':'Middle East & North Africa', 'EGY':'Middle East & North Africa', 'IRN':'Middle East & North Africa', 'JOR':'Middle East & North Africa',
    'LBN':'Middle East & North Africa', 'MAR':'Middle East & North Africa', 'SYR':'Middle East & North Africa',
    'TUN':'Middle East & North Africa', 'YEM':'Middle East & North Africa',
    # South Asia
    'AFG':'South Asia', 'BGD':'South Asia', 'BTN':'South Asia', 'IND':'South Asia', 'LKA':'South Asia', 'MDV':'South Asia', 'NPL':'South Asia', 'PAK':'South Asia',
}
df['region'] = df['country'].map(region_dict)


df_overindebted = df[df['Treatment_Group_x'] == 1].copy()


gdp2012 = df_overindebted[df_overindebted['year'] == 2012][['country', 'GDP_per_Capita']].copy()
gdp2012['log_gdp_pc_2012'] = np.log(gdp2012['GDP_per_Capita'])
gdp2012['region'] = gdp2012['country'].map(region_dict)
gdp2012['Former_HIPC'] = gdp2012['country'].isin(ex_hipc_iso3).astype(int)


outcome_vars = [
    'Control_of_Corruption','Government_Effectiveness','Rule_of_Law','Voice_and_Accountability',
    'Regulatory_Quality','cbi_index','Current_Account_Balance','External_Debt_to_GNI',
    'Debt_Service_to_Exports','GDP_Growth','GDP_per_Capita','Inflation_CPI','Terms_of_Trade',
    'cpiapadm','cpiamacr','cpiafinq','cpiadebt','realIR','democracy_index','broad_money_to_gdp',
    'Primary_fiscal_balance'
]


df_period = df_overindebted[(df_overindebted['year'] >= 2012) & (df_overindebted['year'] <= 2019)].copy()

results = []

for var in outcome_vars:
    avg = df_period.groupby('country')[var].mean().reset_index()
    merged = avg.merge(gdp2012[['country', 'log_gdp_pc_2012', 'region', 'Former_HIPC']], on='country', how='left')
    merged = merged.dropna(subset=[var, 'log_gdp_pc_2012', 'region', 'Former_HIPC'])
    if merged.shape[0] < 10:
        continue  # skip if too few observations
    formula = f"{var} ~ Former_HIPC + log_gdp_pc_2012 + C(region)"
    # Use robust SEs for external financing vars, else cluster by region
    external_financing_vars = ['External_Debt_to_GNI','Debt_Service_to_Exports', 'cbi_index', 'Polity2_Score', 'Voice_and_Accountability', 'Regulatory_Quality', 'cpiamacr', 'cpiadebt']
    if var in external_financing_vars:
        model = smf.ols(formula, data=merged).fit(cov_type='HC1')
    else:
        model = smf.ols(formula, data=merged).fit(cov_type='cluster', cov_kwds={'groups': merged['region']})
    est = model.params.get('Former_HIPC', np.nan)
    se = model.bse.get('Former_HIPC', np.nan)
    p = model.pvalues.get('Former_HIPC', np.nan)
    if p < 0.01:
        stars = '***'
    elif p < 0.05:
        stars = '**'
    elif p < 0.1:
        stars = '*'
    else:
        stars = ''
    results.append({
        'Variable': var,
        'Estimate': f"{est:.3f}",
        'StdErr': f"{se:.2f}{stars}",
        'R2': f"{model.rsquared:.3f}",
        'N': int(model.nobs)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
