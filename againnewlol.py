import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from linearmodels import PanelOLS
from linearmodels.iv import IV2SLS
import warnings
warnings.filterwarnings("ignore")

# 1. Data loading & PCA
def load_and_prepare_data(file_path='ITHINKITSFULLNOW.csv'):
    df = pd.read_csv(file_path)
    wgi_vars = ['Control_of_Corruption', 'Government_Effectiveness',
                'Rule_of_Law', 'Voice_and_Accountability', 'Regulatory_Quality']
    wgi_vars = [v for v in wgi_vars if v in df.columns]
    df['country_year'] = df['country'].astype(str) + '_' + df['year'].astype(str)
    return df, wgi_vars

def create_wgi_composite(df, wgi_vars):
    wgi_data = df[wgi_vars + ['country', 'year']].dropna()
    scaler = StandardScaler()
    wgi_scaled = scaler.fit_transform(wgi_data[wgi_vars])
    pca = PCA(n_components=len(wgi_vars)).fit(wgi_scaled)
    wgi_df = pd.DataFrame({
        'country': wgi_data['country'],
        'year':    wgi_data['year'],
        'WGI_Composite': pca.transform(wgi_scaled)[:, 0]
    })
    return wgi_df, pca, scaler


# 2. Feature engineering
def create_analysis_variables(df, wgi_df):
    d = df.merge(wgi_df, on=['country', 'year'], how='left').sort_values(['country','year'])
    d['Delta_WGI']            = d.groupby('country')['WGI_Composite'].diff()
    d['Delta_Debt']           = d.groupby('country')['External_Debt_to_GNI'].diff()
    d['Delta_GDP_Growth']     = d.groupby('country')['GDP_Growth'].diff()
    d['Delta_GDP_per_Capita'] = d.groupby('country')['GDP_per_Capita'].pct_change()
    d['WGI_lag1']             = d.groupby('country')['WGI_Composite'].shift(1)
    d['WGI_lag2']             = d.groupby('country')['WGI_Composite'].shift(2)
    d['time_trend']           = d.groupby('country').cumcount()+1
    # Add crisis dummies if years 2008 or 2020 are present
    if 'year' in d.columns:
        d['crisis_2008'] = (d['year'] == 2008).astype(int)
        d['crisis_2020'] = (d['year'] == 2020).astype(int)
    # Example threshold: high/low initial WGI
    d['high_initial_WGI'] = (d.groupby('country')['WGI_Composite'].transform('first') > d['WGI_Composite'].median()).astype(int)
    return d

# 3. Estimation functions
def estimate_baseline_model(df):
    controls = ['Delta_GDP_Growth','Delta_GDP_per_Capita','Terms_of_Trade','Primary_fiscal_balance','Inflation_CPI']
    controls = [c for c in controls if c in df.columns]
    df_panel = df.dropna(subset=['Delta_Debt','Delta_WGI'] + controls)
    df_panel = df_panel.set_index(['country','year'])
    res = PanelOLS(df_panel['Delta_Debt'],
                   df_panel[['Delta_WGI']+controls],
                   entity_effects=True, time_effects=True
                  ).fit(cov_type='robust')
    return res, df_panel

def estimate_iv_model(df, use_region=False, use_year=False, instruments=['WGI_lag1','WGI_lag2']):
    controls = ['Delta_GDP_Growth','Delta_GDP_per_Capita','GDP_Growth','Terms_of_Trade','Primary_fiscal_balance','Inflation_CPI']
    controls = [c for c in controls if c in df.columns]
    needed = ['Delta_Debt','Delta_WGI'] + instruments + controls
    if use_region and 'region' in df.columns:
        region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)
        df = pd.concat([df, region_dummies], axis=1)
        controls += list(region_dummies.columns)
    if use_year and 'year' in df.columns:
        year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True)
        df = pd.concat([df, year_dummies], axis=1)
        controls += list(year_dummies.columns)
    iv = df.dropna(subset=needed)
    iv = iv.set_index(['country','year'])
    model = IV2SLS(iv['Delta_Debt'],
                   iv[controls],
                   iv[['Delta_WGI']],
                   iv[instruments])
    res = model.fit(cov_type='robust')
    return res, iv

# 4. Enhanced diagnostics
def first_stage_ftest(iv_res, iv_df, instruments=['WGI_lag1','WGI_lag2']):
    y = iv_df['Delta_WGI']
    X = sm.add_constant(iv_df[instruments + [c for c in iv_df.columns if c.startswith('Delta_') or c.startswith('GDP_')]])
    fs = sm.OLS(y, X).fit()
    f = fs.f_test([f"{inst} = 0" for inst in instruments])
    print(f"First-stage F-stat : {float(f.fvalue):.2f}  (p={float(f.pvalue):.4f})")

def leave_one_out_iv(df):
    print("\n--- Leave-one-out Sargan tests ---")
    for inst in ['WGI_lag1','WGI_lag2']:
        res, _ = estimate_iv_model(df, instruments=[inst])
        print(f"Sargan p-value with only {inst}: {res.sargan.pval:.4f}")

def endogeneity_tests(iv_res):
    print(f"Wu-Hausman p-value : {iv_res.wu_hausman().pval:.4f}")
    print(f"Durbin     p-value : {iv_res.durbin().pval:.4f}")

def overid_tests(iv_res):
    print(f"Sargan p-value     : {iv_res.sargan.pval:.4f}")

def anderson_rubin(iv_df, beta0=0):
    y = iv_df['Delta_Debt'].values
    x = iv_df['Delta_WGI'].values
    Z = np.column_stack([np.ones(len(iv_df)),
                         iv_df[['WGI_lag1','WGI_lag2']].values])
    n, k = Z.shape
    y_null = y - beta0*x
    Pz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    SSR = ((np.eye(n)-Pz) @ y_null).T @ ((np.eye(n)-Pz) @ y_null)
    df_num, df_den = 2-1, n-k
    ar_F = (SSR/df_num)/(SSR/df_den)
    pval = 1 - stats.f.cdf(ar_F, df_num, df_den)
    print(f"Anderson-Rubin F   : {ar_F:.2f}  (p={pval:.4f})")

def residual_diagnostics(iv_res):
    e = iv_res.resids
    fitted = iv_res.fitted_values.values.reshape(-1, 1)
    exog_with_const = sm.add_constant(fitted)
    try:
        bp_stat, bp_pvalue, lm_stat, lm_pvalue = het_breuschpagan(e, exog_with_const)
        print(f"Breusch-Pagan p-val: {bp_pvalue:.4f}")
    except Exception as error:
        print(f"Breusch-Pagan test failed: {error}")
    dw_stat = durbin_watson(e)
    print(f"Durbin-Watson      : {dw_stat:.2f}")

def run_iv_diagnostics(iv_res, iv_df, df):
    print("\n---------------------------")
    print("IV DIAGNOSTICS")
    print("---------------------------")
    first_stage_ftest(iv_res, iv_df)
    endogeneity_tests(iv_res)
    overid_tests(iv_res)
    anderson_rubin(iv_df)
    residual_diagnostics(iv_res)
    leave_one_out_iv(df)

# 5. Main analysis driver
def main_analysis():
    print("Loading data …")
    df, wgi_vars = load_and_prepare_data()
    print("Building composite WGI …")
    wgi_df, pca, _ = create_wgi_composite(df, wgi_vars)
    print("Engineering features …")
    dfA = create_analysis_variables(df, wgi_df)
    results = {}
    print("Estimating FE baseline …")
    fe_res, fe_df = estimate_baseline_model(dfA)
    results['Fixed-Effects'] = fe_res
    print("Estimating IV model …")
    iv_res, iv_df = estimate_iv_model(dfA)
    results['IV'] = iv_res
    run_iv_diagnostics(iv_res, iv_df, dfA)
    for name, res in results.items():
        print(f"\n{name} SUMMARY\n{'-'*60}")
        print(res.summary)
    return results, dfA

# 6. Recommendations / further work
def additional_robustness_checks():
    print("\nFURTHER ROBUSTNESS IDEAS")
    print("• Alternative lag structures for instruments")
    print("• System-GMM dynamic panel (Arellano-Bover/Blundell-Bond)")
    print("• Structural breaks around major crises (2008, 2020)")
    print("• Threshold effects: is there a governance level where debt effects change?")
    print("• Different WGI aggregation (factor analysis, simple average)")
    print("• Country-specific time trends")

if __name__ == "__main__":
    res, data = main_analysis()
    additional_robustness_checks()


import pandas as pd
from linearmodels.panel import DifferenceGMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load and preprocess your data
df = pd.read_csv('ITHINKITSFULLNOW.csv')
wgi_vars = ['Control_of_Corruption', 'Government_Effectiveness', 'Rule_of_Law', 'Voice_and_Accountability', 'Regulatory_Quality']
wgi_vars = [v for v in wgi_vars if v in df.columns]
scaler = StandardScaler()
wgi_scaled = scaler.fit_transform(df[wgi_vars].dropna())
pca = PCA(n_components=1).fit(wgi_scaled)
df['WGI_Composite'] = pd.Series(pca.transform(scaler.transform(df[wgi_vars])), index=df.dropna(subset=wgi_vars).index)

# 2. Sort and set index for panel
df = df.sort_values(['country', 'year'])
df = df.set_index(['country', 'year'])

# 3. Prepare variables
df['L.External_Debt_to_GNI'] = df.groupby(level=0)['External_Debt_to_GNI'].shift(1)
exog_vars = ['GDP_Growth', 'Terms_of_Trade', 'Primary_fiscal_balance', 'Inflation_CPI']
exog_vars = [v for v in exog_vars if v in df.columns]

# 4. Drop missing values for variables used
panel = df.dropna(subset=['External_Debt_to_GNI', 'L.External_Debt_to_GNI', 'WGI_Composite'] + exog_vars)

# 5. Specify endogenous and instrument variables
dependent = panel['External_Debt_to_GNI']
exog = panel[exog_vars]
endog = panel[['L.External_Debt_to_GNI', 'WGI_Composite']]
instruments = {}

# 6. Fit Difference-GMM model
model = DifferenceGMM(dependent, exog, endog, panel.index.get_level_values(0), panel.index.get_level_values(1))
results = model.fit()
print(results.summary)

