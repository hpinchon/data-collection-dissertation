#This did not work but was worth trying
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        'WGI_Composite': pca.transform(wgi_scaled)[:, 0]        # PC-1
    })
    return wgi_df, pca, scaler

def create_analysis_variables(df, wgi_df):
    d = df.merge(wgi_df, on=['country', 'year'], how='left').sort_values(['country','year'])
    d['Delta_WGI']            = d.groupby('country')['WGI_Composite'].diff()
    d['Delta_Debt']           = d.groupby('country')['External_Debt_to_GNI'].diff()
    d['Delta_GDP_Growth']     = d.groupby('country')['GDP_Growth'].diff()
    d['Delta_GDP_per_Capita'] = d.groupby('country')['GDP_per_Capita'].pct_change()
    d['WGI_lag1']             = d.groupby('country')['WGI_Composite'].shift(1)
    d['WGI_lag2']             = d.groupby('country')['WGI_Composite'].shift(2)
    d['time_trend']           = d.groupby('country').cumcount()+1
    return d

def estimate_baseline_model(df):
    df_panel = df.dropna(subset=['Delta_Debt','Delta_WGI','Delta_GDP_Growth',
                                 'Delta_GDP_per_Capita'])
    df_panel = df_panel.set_index(['country','year'])
    res = PanelOLS(df_panel['Delta_Debt'],
                   df_panel[['Delta_WGI','Delta_GDP_Growth','Delta_GDP_per_Capita']],
                   entity_effects=True, time_effects=True
                  ).fit(cov_type='robust')
    return res, df_panel


def estimate_iv_model(df):
    iv = df.dropna(subset=['Delta_Debt','Delta_WGI','WGI_lag1','WGI_lag2',
                           'Delta_GDP_Growth','Delta_GDP_per_Capita','GDP_Growth'])
    iv = iv.set_index(['country','year'])
    model = IV2SLS(iv['Delta_Debt'],
                   iv[['Delta_GDP_Growth','Delta_GDP_per_Capita','GDP_Growth']],
                   iv[['Delta_WGI']],
                   iv[['WGI_lag1','WGI_lag2']])
    res = model.fit(cov_type='robust')
    return res, iv

def first_stage_ftest(iv_res, iv_df):
    y = iv_df['Delta_WGI']
    X = sm.add_constant(iv_df[['WGI_lag1','WGI_lag2',
                               'Delta_GDP_Growth','Delta_GDP_per_Capita','GDP_Growth']])
    fs = sm.OLS(y, X).fit()
    f = fs.f_test(['WGI_lag1 = 0', 'WGI_lag2 = 0'])
    print(f"First-stage F-stat : {float(f.fvalue):.2f}  (p={float(f.pvalue):.4f})")

def kleibergen_paap(iv_res):
    try:
        kp = iv_res.first_stage.f_statistic
        print(f"Kleibergen-Paap rk : {kp:.2f}")
    except AttributeError:
        print("K-P statistic not available.")

def endogeneity_tests(iv_res):
    print(f"Wu-Hausman p-value : {iv_res.wu_hausman().pval:.4f}")
    print(f"Durbin     p-value : {iv_res.durbin().pval:.4f}")

def overid_tests(iv_res):
    print(f"Sargan p-value     : {iv_res.sargan.pval:.4f}")

def anderson_rubin(iv_df, beta0=0):
    y = iv_df['Delta_Debt'].values
    x = iv_df['Delta_WGI'].values
    Z = np.column_stack([np.ones(len(iv_df)),
                         iv_df[['WGI_lag1','WGI_lag2',
                                'Delta_GDP_Growth','Delta_GDP_per_Capita',
                                'GDP_Growth']].values])
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
    print(f"Durbin-Watson      : {durbin_watson(e):.2f}")

def run_iv_diagnostics(iv_res, iv_df):
    first_stage_ftest(iv_res, iv_df)
    kleibergen_paap(iv_res)
    endogeneity_tests(iv_res)
    overid_tests(iv_res)
    anderson_rubin(iv_df)
    residual_diagnostics(iv_res)

def main_analysis():
    df, wgi_vars = load_and_prepare_data()
    wgi_df, pca, _ = create_wgi_composite(df, wgi_vars)
    dfA = create_analysis_variables(df, wgi_df)

    results = {}
    fe_res, fe_df = estimate_baseline_model(dfA)
    results['Fixed-Effects'] = fe_res
    iv_res, iv_df = estimate_iv_model(dfA)
    results['IV'] = iv_res

    run_iv_diagnostics(iv_res, iv_df)

    for name, res in results.items():
        print(f"\n{name} SUMMARY\n{'-'*60}")
        print(res.summary)

    return results, dfA

if __name__ == "__main__":
    res, data = main_analysis()
