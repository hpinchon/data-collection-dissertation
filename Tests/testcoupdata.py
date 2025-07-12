#this too was a failure but cool thing to look at
#basically trying to see if i could find a link between coup in countries and their subsequent rise in debt levels
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


group1_codes = [
    'ALB','DZA','ARG','ARM','AZE','BGD','BLR','BLZ','BEN','BTN','BOL','BIH','BWA','BRA',
    'BGR','BFA','KHM','CHN','COL','COM','COD','CRI','CIV','DOM','ECU','EGY','SLV','ERI',
    'SWZ','FJI','GAB','GEO','GTM','GIN','GNB','GUY','HND','IND','IDN','IRN','JOR','KAZ',
    'KEN','XKX','KGZ','LBN','LBR','MDG','MWI','MLI','MUS','MEX','MDA','MNG','MAR','MMR',
    'NPL','NIC','NER','NGA','MKD','PAK','PAN','PNG','PRY','PER','PHL','ROU','RUS','RWA',
    'SEN','SRB','SLB','ZAF','LCA','SYR','TZA','THA','TLS','TGO','TUN','TUR','TKM','UGA',
    'UKR','UZB','VUT','VEN','VNM','YEM'
]

group2_codes = [
    'AFG','BDI','CPV','CMR','CAF','TCD','COG','DJI','DMA','ETH','GMB','GHA','GRD','HTI',
    'LAO','MDV','MRT','MOZ','WSM','STP','SLE','SOM','VCT','SDN','TJK','TON','ZMB','ZWE',
    'AGO','LKA','JAM','MNE'
]


all_group_countries = set(group1_codes + group2_codes)

df_period = pd.read_csv('merged_data_enriched.csv')
df_period = df_period[df_period['year'] >= 2001].copy()
df_period = df_period[df_period['scode'].isin(all_group_countries)].copy()


outcome_vars = ['GDP_Growth', 'GDP_per_Capita', 'Inflation_CPI', 'Terms_of_Trade', 
                'cpiamacr', 'cpiadebt', 'Primary_fiscal_balance', 'EconomicFreedomIndex', 
                'Polity2_Score', 'external_debt_to_gni']

outcome_vars = [var for var in outcome_vars if var in df_period.columns]

treat = df_period[['scode', 'treatment']].drop_duplicates().rename(columns={'scode': 'country'})

if 'log_gdp_pc_2012' in df_period.columns and 'region' in df_period.columns:
    gdp2012 = df_period[['scode', 'log_gdp_pc_2012', 'region']].drop_duplicates().rename(columns={'scode': 'country'})
else:
    gdp2012 = df_period[['scode']].drop_duplicates().rename(columns={'scode': 'country'})
    gdp2012['log_gdp_pc_2012'] = np.random.normal(8, 1, len(gdp2012))  # Placeholder
    gdp2012['region'] = 'Unknown'  # Placeholder

results = []

for var in outcome_vars:
    
    avg = df_period.groupby('scode')[var].mean().reset_index().rename(columns={'scode': 'country'})
   
    merged = avg.merge(gdp2012[['country', 'log_gdp_pc_2012', 'region']], on='country', how='left')
    merged = merged.merge(treat, on='country', how='left')

    merged = merged.dropna(subset=[var, 'treatment', 'log_gdp_pc_2012', 'region'])
    
    if merged.shape[0] < 10:
        continue  # skip if too few observations

    external_financing_vars = ['external_debt_to_gni', 'Debt_Service_to_Exports', 'cbi_index', 
                              'Polity2_Score', 'Voice_and_Accountability', 'Regulatory_Quality', 
                              'cpiamacr', 'cpiadebt']
    
    formula = f"{var} ~ treatment + log_gdp_pc_2012 + C(region)"
    
    model = smf.ols(formula, data=merged).fit()

    
    # Extract results
    est = model.params.get('treatment', np.nan)
    se = model.bse.get('treatment', np.nan)
    p = model.pvalues.get('treatment', np.nan)
    
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
print(results_df)
