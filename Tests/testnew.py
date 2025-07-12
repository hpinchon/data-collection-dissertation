import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from linearmodels import PanelOLS, PooledOLS
from linearmodels.iv import IV2SLS
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(file_path='ITHINKITSFULLNOW.csv'):
    """
    Load and prepare the dataset for analysis
    """
    df = pd.read_csv(file_path)
    
    # Define WGI variables
    wgi_vars = ['Control_of_Corruption', 'Government_Effectiveness', 
                'Rule_of_Law', 'Voice_and_Accountability', 'Regulatory_Quality']
    
    # Check if all WGI variables exist
    missing_vars = [var for var in wgi_vars if var not in df.columns]
    if missing_vars:
        print(f"Warning: Missing WGI variables: {missing_vars}")
        wgi_vars = [var for var in wgi_vars if var in df.columns]
    
    # Create country-year identifier
    df['country_year'] = df['country'].astype(str) + '_' + df['year'].astype(str)
    
    return df, wgi_vars

# =============================================================================
# 2. WGI AGGREGATION USING PCA
# =============================================================================

def create_wgi_composite(df, wgi_vars):
    """
    Create composite WGI indicator using Principal Component Analysis
    Following World Bank methodology for WGI aggregation
    """
    # Remove rows with missing WGI data
    wgi_data = df[wgi_vars + ['country', 'year']].dropna()
    
    # Standardize the WGI variables
    scaler = StandardScaler()
    wgi_scaled = scaler.fit_transform(wgi_data[wgi_vars])
    
    # Apply PCA
    pca = PCA(n_components=len(wgi_vars))
    wgi_pca = pca.fit_transform(wgi_scaled)
    
    # Create composite index using first principal component
    # This captures the most variance in institutional quality
    wgi_composite = wgi_pca[:, 0]
    
    # Create DataFrame with composite indicator
    wgi_df = pd.DataFrame({
        'country': wgi_data['country'],
        'year': wgi_data['year'],
        'WGI_Composite': wgi_composite
    })
    
    # Print PCA results
    print("PCA Results for WGI Aggregation:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"First component explains {pca.explained_variance_ratio_[0]:.3f} of total variance")
    
    # Print component loadings
    loadings = pd.DataFrame(
        pca.components_[0], 
        index=wgi_vars, 
        columns=['PC1_Loading']
    )
    print("\nFirst Principal Component Loadings:")
    print(loadings)
    
    return wgi_df, pca, scaler

# =============================================================================
# 3. VARIABLE TRANSFORMATIONS AND FEATURE ENGINEERING
# =============================================================================

def create_analysis_variables(df, wgi_df):
    """
    Create variables for analysis including changes and lags
    """
    # Merge WGI composite with main dataset
    df_analysis = df.merge(wgi_df, on=['country', 'year'], how='left')
    
    # Sort by country and year for time series operations
    df_analysis = df_analysis.sort_values(['country', 'year'])
    
    # Create change variables
    df_analysis['Delta_WGI'] = df_analysis.groupby('country')['WGI_Composite'].diff()
    df_analysis['Delta_Debt'] = df_analysis.groupby('country')['External_Debt_to_GNI'].diff()
    df_analysis['Delta_GDP_Growth'] = df_analysis.groupby('country')['GDP_Growth'].diff()
    df_analysis['Delta_GDP_per_Capita'] = df_analysis.groupby('country')['GDP_per_Capita'].pct_change()
    
    # Create lagged variables for instruments
    df_analysis['WGI_Composite_lag1'] = df_analysis.groupby('country')['WGI_Composite'].shift(1)
    df_analysis['WGI_Composite_lag2'] = df_analysis.groupby('country')['WGI_Composite'].shift(2)
    df_analysis['External_Debt_to_GNI_lag1'] = df_analysis.groupby('country')['External_Debt_to_GNI'].shift(1)
    
    # Create time trend
    df_analysis['time_trend'] = df_analysis.groupby('country').cumcount() + 1
    
    return df_analysis

# =============================================================================
# 4. ECONOMETRIC MODELS
# =============================================================================

def estimate_baseline_model(df_analysis):
    """
    Estimate baseline fixed effects model
    """
    # Define the model specification
    formula = """
    Delta_Debt ~ Delta_WGI + Delta_GDP_Growth + Delta_GDP_per_Capita + 
                 lag(GDP_Growth, 1) + lag(Terms_of_Trade, 1) + 
                 lag(Primary_fiscal_balance, 1) + lag(Inflation_CPI, 1)
                 
    """
    
    # Prepare data for panel estimation
    df_panel = df_analysis.dropna(subset=['Delta_Debt', 'Delta_WGI', 'Delta_GDP_Growth', 
                                         'Delta_GDP_per_Capita', 'GDP_Growth', 
                                         'Terms_of_Trade', 'Primary_fiscal_balance', 
                                         'Inflation_CPI'])
    
    # Set index for panel data
    df_panel = df_panel.set_index(['country', 'year'])
    
    # Estimate fixed effects model
    model = PanelOLS(
        dependent=df_panel['Delta_Debt'],
        exog=df_panel[['Delta_WGI', 'Delta_GDP_Growth', 'Delta_GDP_per_Capita']],
        entity_effects=False,
        time_effects=False
    )
    
    results = model.fit(cov_type='HC1')
    
    return results, df_panel

def estimate_iv_model(df_analysis):
    """
    Estimate IV model to address endogeneity
    Uses lagged WGI values as instruments
    """
    # Prepare data
    df_iv = df_analysis.dropna(subset=['Delta_Debt', 'Delta_WGI', 'WGI_Composite_lag1', 
                                      'WGI_Composite_lag2', 'Delta_GDP_Growth', 
                                      'Delta_GDP_per_Capita', 'GDP_Growth'])
    
    df_iv = df_iv.set_index(['country', 'year'])
    
    # IV estimation using lagged WGI as instrument
    model = IV2SLS(
        dependent=df_iv['Delta_Debt'],
        exog=df_iv[['Delta_GDP_Growth', 'Delta_GDP_per_Capita', 'GDP_Growth']],
        endog=df_iv[['Delta_WGI']],
        instruments=df_iv[['WGI_Composite_lag1', 'WGI_Composite_lag2']]
    )
    
    results = model.fit()
    
    return results, df_iv

def estimate_regional_heterogeneity(df_analysis):
    """
    Estimate model with regional heterogeneity
    """
    # Create regional interaction terms
    df_analysis['Delta_WGI_x_Region'] = df_analysis['Delta_WGI'] * pd.get_dummies(df_analysis['region'])
    
    # Estimate model with regional interactions
    df_regional = df_analysis.dropna(subset=['Delta_Debt', 'Delta_WGI', 'region'])
    df_regional = df_regional.set_index(['country', 'year'])
    
    # Add regional dummies
    regional_dummies = pd.get_dummies(df_regional['region'], prefix='region')
    
    exog_vars = pd.concat([
        df_regional[['Delta_WGI', 'Delta_GDP_Growth', 'Delta_GDP_per_Capita']],
        regional_dummies
    ], axis=1)
    
    model = PanelOLS(
        dependent=df_regional['Delta_Debt'],
        exog=exog_vars,
        entity_effects=True,
        time_effects=True
    )
    
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results, df_regional

# =============================================================================
# 5. ROBUSTNESS CHECKS AND DIAGNOSTICS
# =============================================================================

def diagnostic_tests(results, df_panel):
    """
    Perform diagnostic tests on the model
    """
    print("=== DIAGNOSTIC TESTS ===")
    
    # Test for heteroscedasticity
    residuals = results.resids
    fitted = results.fitted_values
    
    # Breusch-Pagan test
    try:
        bp_test = het_breuschpagan(residuals, fitted)
        print(f"Breusch-Pagan test p-value: {bp_test[1]:.4f}")
    except:
        print("Could not perform Breusch-Pagan test")
    
    # Durbin-Watson test for autocorrelation
    try:
        dw_stat = durbin_watson(residuals)
        print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    except:
        print("Could not perform Durbin-Watson test")
    
    # First-stage F-statistic for IV (if applicable)
    if hasattr(results, 'first_stage'):
        print(f"First-stage F-statistic: {results.first_stage.fstat:.4f}")
        print(f"First-stage F-statistic p-value: {results.first_stage.fstat_pvalue:.4f}")

def placebo_test(df_analysis):
    """
    Perform placebo test using future institutional quality changes
    """
    # Create lead variables (future changes)
    df_analysis['Delta_WGI_lead1'] = df_analysis.groupby('country')['Delta_WGI'].shift(-1)
    
    # Estimate model with future changes
    df_placebo = df_analysis.dropna(subset=['Delta_Debt', 'Delta_WGI_lead1', 'Delta_GDP_Growth'])
    df_placebo = df_placebo.set_index(['country', 'year'])
    
    model = PanelOLS(
        dependent=df_placebo['Delta_Debt'],
        exog=df_placebo[['Delta_WGI_lead1', 'Delta_GDP_Growth', 'Delta_GDP_per_Capita']],
        entity_effects=True,
        time_effects=True
    )
    
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results

# =============================================================================
# 6. VISUALIZATION AND RESULTS
# =============================================================================

def create_visualizations(df_analysis, results):
    """
    Create visualizations for the analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot of WGI vs Debt changes
    axes[0, 0].scatter(df_analysis['Delta_WGI'], df_analysis['Delta_Debt'], alpha=0.6)
    axes[0, 0].set_xlabel('Change in WGI Composite')
    axes[0, 0].set_ylabel('Change in External Debt/GNI')
    axes[0, 0].set_title('Institutional Quality vs Debt Changes')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add regression line
    x = df_analysis['Delta_WGI'].dropna()
    y = df_analysis['Delta_Debt'].dropna()
    common_idx = x.index.intersection(y.index)
    if len(common_idx) > 0:
        z = np.polyfit(x[common_idx], y[common_idx], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(x[common_idx], p(x[common_idx]), "r--", alpha=0.8)
    
    # 2. Regional variation
    if 'region' in df_analysis.columns:
        regional_data = df_analysis.groupby('region').agg({
            'Delta_WGI': 'mean',
            'Delta_Debt': 'mean'
        }).reset_index()
        
        axes[0, 1].scatter(regional_data['Delta_WGI'], regional_data['Delta_Debt'])
        for i, region in enumerate(regional_data['region']):
            axes[0, 1].annotate(region, (regional_data['Delta_WGI'].iloc[i], 
                                        regional_data['Delta_Debt'].iloc[i]))
        axes[0, 1].set_xlabel('Mean Change in WGI')
        axes[0, 1].set_ylabel('Mean Change in Debt/GNI')
        axes[0, 1].set_title('Regional Averages')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals vs Fitted
    if hasattr(results, 'resids') and hasattr(results, 'fitted_values'):
        axes[1, 0].scatter(results.fitted_values, results.resids, alpha=0.6)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Fitted Values')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
    
    # 4. Time series of key variables
    if 'year' in df_analysis.columns:
        yearly_data = df_analysis.groupby('year').agg({
            'Delta_WGI': 'mean',
            'Delta_Debt': 'mean'
        }).reset_index()
        
        ax2 = axes[1, 1]
        ax2.plot(yearly_data['year'], yearly_data['Delta_WGI'], 'b-', label='Δ WGI')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Change in WGI', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        ax3 = ax2.twinx()
        ax3.plot(yearly_data['year'], yearly_data['Delta_Debt'], 'r-', label='Δ Debt')
        ax3.set_ylabel('Change in Debt/GNI', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        
        axes[1, 1].set_title('Time Series of Key Variables')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_results_summary(results_dict):
    """
    Print comprehensive results summary
    """
    print("=" * 80)
    print("INSTITUTIONAL QUALITY AND SOVEREIGN DEBT ANALYSIS")
    print("=" * 80)
    
    for model_name, results in results_dict.items():
        print(f"\n{model_name.upper()} RESULTS:")
        print("-" * 50)
        print(results.summary)
        
        # Extract key coefficient
        try:
            wgi_coef = results.params['Delta_WGI']
            wgi_pval = results.pvalues['Delta_WGI']
            wgi_se = results.std_errors['Delta_WGI']
            
            print(f"\nKEY FINDING:")
            print(f"WGI Coefficient: {wgi_coef:.4f}")
            print(f"Standard Error: {wgi_se:.4f}")
            print(f"P-value: {wgi_pval:.4f}")
            
            if wgi_pval < 0.01:
                significance = "***"
            elif wgi_pval < 0.05:
                significance = "**"
            elif wgi_pval < 0.10:
                significance = "*"
            else:
                significance = ""
                
            print(f"Significance: {significance}")
            
            # Economic interpretation
            if wgi_coef < 0:
                print(f"INTERPRETATION: A 1-unit improvement in institutional quality")
                print(f"is associated with a {abs(wgi_coef):.2f} percentage point")
                print(f"REDUCTION in external debt-to-GNI ratio.")
            else:
                print(f"INTERPRETATION: A 1-unit improvement in institutional quality")
                print(f"is associated with a {wgi_coef:.2f} percentage point")
                print(f"INCREASE in external debt-to-GNI ratio.")
                
        except KeyError:
            print("WGI coefficient not found in results")

# =============================================================================
# 7. MAIN ANALYSIS FUNCTION
# =============================================================================

def main_analysis():
    """
    Execute the complete analysis
    """
    print("Loading and preparing data...")
    
    # Load data
    df, wgi_vars = load_and_prepare_data()
    
    # Create WGI composite
    print("Creating WGI composite indicator...")
    wgi_df, pca, scaler = create_wgi_composite(df, wgi_vars)
    
    # Create analysis variables
    print("Creating analysis variables...")
    df_analysis = create_analysis_variables(df, wgi_df)
    
    # Store results
    results_dict = {}
    
    # Estimate baseline model
    print("Estimating baseline model...")
    try:
        baseline_results, df_panel = estimate_baseline_model(df_analysis)
        results_dict['Baseline Fixed Effects'] = baseline_results
    except Exception as e:
        print(f"Error in baseline model: {e}")
    
    # Estimate IV model
    print("Estimating IV model...")
    try:
        iv_results, df_iv = estimate_iv_model(df_analysis)
        results_dict['IV Model'] = iv_results
    except Exception as e:
        print(f"Error in IV model: {e}")
    
    # Estimate regional heterogeneity
    print("Estimating regional heterogeneity...")
    try:
        regional_results, df_regional = estimate_regional_heterogeneity(df_analysis)
        results_dict['Regional Heterogeneity'] = regional_results
    except Exception as e:
        print(f"Error in regional model: {e}")
    
    # Diagnostic tests
    print("Performing diagnostic tests...")
    if 'Baseline Fixed Effects' in results_dict:
        diagnostic_tests(results_dict['Baseline Fixed Effects'], df_panel)
    
    # Placebo test
    print("Performing placebo test...")
    try:
        placebo_results = placebo_test(df_analysis)
        results_dict['Placebo Test'] = placebo_results
    except Exception as e:
        print(f"Error in placebo test: {e}")
    
    # Create visualizations
    print("Creating visualizations...")
    if 'Baseline Fixed Effects' in results_dict:
        create_visualizations(df_analysis, results_dict['Baseline Fixed Effects'])
    
    # Print results
    print_results_summary(results_dict)
    
    return results_dict, df_analysis

# =============================================================================
# 8. ADDITIONAL SUGGESTIONS FOR ROBUSTNESS
# =============================================================================

def additional_robustness_checks(df_analysis):
    """
    Additional robustness checks you might want to implement
    """
    print("\nADDITIONAL ROBUSTNESS SUGGESTIONS:")
    print("1. Test for different lag structures")
    print("2. Use GMM estimation for dynamic panel models")
    print("3. Test for structural breaks around financial crises")
    print("4. Include interaction terms with income levels")
    print("5. Test alternative WGI aggregation methods")
    print("6. Use bilateral exchange rate instruments for debt")
    print("7. Test for threshold effects in institutional quality")

# =============================================================================
# 9. EXECUTE ANALYSIS
# =============================================================================

if __name__ == "__main__":
    # Run the complete analysis
    results, df_analysis = main_analysis()
    
    # Additional suggestions
    additional_robustness_checks(df_analysis)


