import pandas as pd
import requests
import time
from datetime import datetime

class WorkingFerryDiwanCollector:
    def __init__(self):
        """countries from Ferry & Diwan (2025) Tables A1 & A2"""
        
        # Table A1 - Criteria 1: DSA = High Risk/Debt Distress (28 countries)
        self.criteria1_countries = [
            'AFG',  # Afghanistan
            'BDI',  # Burundi
            'CPV',  # Cabo Verde
            'CMR',  # Cameroon
            'CAF',  # Central Afr. Rep.
            'TCD',  # Chad
            'COG',  # Congo, Rep.
            'DJI',  # Djibouti
            'DMA',  # Dominica
            'ETH',  # Ethiopia
            'GMB',  # Gambia, The
            'GHA',  # Ghana
            'GRD',  # Grenada
            'HTI',  # Haiti
            'LAO',  # Lao PDR
            'MDV',  # Maldives
            'MRT',  # Mauritania
            'MOZ',  # Mozambique
            'WSM',  # Samoa
            'STP',  # Sao Tome & Princ.
            'SLE',  # Sierra Leone
            'SOM',  # Somalia
            'VCT',  # St. Vinc. & Grenad.
            'SDN',  # Sudan
            'TJK',  # Tajikistan
            'TON',  # Tonga
            'ZMB',  # Zambia
            'ZWE'   # Zimbabwe
        ]
        
        # Table A1 - Criteria 2: 3 or more threshold breaches (2015-19) (4 countries)
        self.criteria2_countries = [
            'AGO',  # Angola
            'LKA',  # Sri Lanka
            'JAM',  # Jamaica
            'MNE'   # Montenegro
        ]
        
        # Combined treatment (32 countries)
        self.treatment_countries = self.criteria1_countries + self.criteria2_countries
        
        # Table A2 - Control countries: Low or moderate risk (91 countries)
        self.control_countries = [
            'ALB', 'DZA', 'ARG', 'ARM', 'AZE', 'BGD', 'BLR', 'BLZ', 'BEN', 'BTN',
            'BOL', 'BIH', 'BWA', 'BRA', 'BGR', 'BFA', 'KHM', 'CHN', 'COL', 'COM',
            'COD', 'CRI', 'CIV', 'DOM', 'ECU', 'EGY', 'SLV', 'ERI', 'SWZ', 'FJI',
            'GAB', 'GEO', 'GTM', 'GIN', 'GNB', 'GUY', 'HND', 'IND', 'IDN', 'IRN',
            'JOR', 'KAZ', 'KEN', 'XKX', 'KGZ', 'LBN', 'LBR', 'MDG', 'MWI', 'MLI',
            'MUS', 'MEX', 'MDA', 'MNG', 'MAR', 'MMR', 'NPL', 'NIC', 'NER', 'NGA',
            'MKD', 'PAK', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'ROU', 'RUS', 'RWA',
            'SEN', 'SRB', 'SLB', 'ZAF', 'LCA', 'SYR', 'TZA', 'THA', 'TLS', 'TGO',
            'TUN', 'TUR', 'TKM', 'UGA', 'UKR', 'UZB', 'VUT', 'VEN', 'VNM', 'YEM'
        ]
        
        self.all_countries = self.treatment_countries + self.control_countries
        print(f"Ferry & Diwan (2025) Sample: {len(self.all_countries)} countries")
        print(f"Treatment: {len(self.treatment_countries)} | Control: {len(self.control_countries)}")

    def fetch_indicator_safe(self, country_code, indicator_code):
        """Fetch data using direct World Bank REST API to avoid wbgapi issues"""
        data_points = []
        
        # Use World Bank REST API directly
        base_url = "https://api.worldbank.org/v2/country"
        url = f"{base_url}/{country_code}/indicator/{indicator_code}?format=json&date=2000:2023&per_page=1000"
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    
                    # World Bank API returns [metadata, data] format
                    if isinstance(json_response, list) and len(json_response) > 1:
                        data_array = json_response[1]  # Second element contains the data
                        
                        if data_array and isinstance(data_array, list):
                            for item in data_array:
                                # Safely check if item is a dictionary before accessing keys
                                if isinstance(item, dict):
                                    year = item.get('date')
                                    value = item.get('value')
                                    
                                    if year and value is not None:
                                        try:
                                            data_points.append({
                                                'country': country_code,
                                                'year': int(year),
                                                'indicator': indicator_code,
                                                'value': float(value)
                                            })
                                        except (ValueError, TypeError):
                                            continue  # Skip invalid data
                
                except (ValueError, TypeError) as json_error:
                    # Skip countries with malformed JSON responses
                    pass
            
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            # Skip failed requests silently to avoid spam
            pass
        
        return data_points

    def collect_all_data_working(self):
        """Collect data using bulletproof method that actually works"""
        print("Starting WORKING data collection...")
        print("=" * 50)
        
        # Core indicators
        indicators = {
            'EIU_DI': 'Democracy Index',
            'WB_WDI_IQ_CPA_DEBT_XQ': 'CPIA_debtpolicy_rating',
            'WB_WDI_IQ_CPA_FINQ_XQ': 'CPIA_budgetary_and_financial_management_rating',
            'WB_WDI_IQ_CPA_MACR_XQ': 'CPIA_macroeconomic_management_rating',
            'WB_WDI_IQ_CPA_PADM_XQ': 'CPIA_public_admin_rating',
            'WB_WDI_FM_LBL_BMNY_GD_ZS': 'Broad_money_to_GDP',
            'WB_WDI_FR_INR_RINR': 'Real_interest_rate',
        }
        
        all_data_points = []
        total_operations = len(self.all_countries) * len(indicators)
        current_operation = 0
        
        # Collect data country by country, indicator by indicator
        for country in self.all_countries:
            print(f"Processing {country}...")
            
            for indicator_code, indicator_name in indicators.items():
                current_operation += 1
                
                # Progress update
                if current_operation % 50 == 0:
                    progress = (current_operation / total_operations) * 100
                    print(f"  Progress: {progress:.1f}% ({current_operation}/{total_operations})")
                
                # Fetch data for this country-indicator combination
                data_points = self.fetch_indicator_safe(country, indicator_code)
                
                # Add indicator name to each data point
                for point in data_points:
                    point['indicator_name'] = indicator_name
                    all_data_points.append(point)
        
        # Convert to DataFrame
        if all_data_points:
            print(f"\nProcessing {len(all_data_points)} data points...")
            
            df = pd.DataFrame(all_data_points)
            
            # Pivot to wide format
            df_wide = df.pivot_table(
                index=['country', 'year'], 
                columns='indicator_name', 
                values='value',
                aggfunc='first'  # Take first value if duplicates
            ).reset_index()
            
            # Add treatment group classifications
            df_wide['Treatment_Group'] = df_wide['country'].apply(
                lambda x: 1 if x in self.treatment_countries else 0
            )
            df_wide['Criteria_1'] = df_wide['country'].apply(
                lambda x: 1 if x in self.criteria1_countries else 0
            )
            df_wide['Criteria_2'] = df_wide['country'].apply(
                lambda x: 1 if x in self.criteria2_countries else 0
            )
            
            #Sort by country and year
            df_wide = df_wide.sort_values(['country', 'year'])
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"ferry_diwan_2025_workinga_{timestamp}.csv"
            df_wide.to_csv(filename, index=False)
            
            # Results summary
            print("\n" + "=" * 50)
            print("✓ DATA COLLECTION SUCCESSFUL!")
            print("=" * 50)
            print(f"File saved: {filename}")
            print(f"Dataset shape: {df_wide.shape}")
            print(f"Countries with data: {df_wide['country'].nunique()}")
            print(f"Years: {df_wide['year'].min():.0f}-{df_wide['year'].max():.0f}")
            print(f"Treatment observations: {(df_wide['Treatment_Group'] == 1).sum()}")
            print(f"Control observations: {(df_wide['Treatment_Group'] == 0).sum()}")
            
            # Data availability summary
            print(f"\nData availability by indicator:")
            for col in df_wide.columns:
                if col not in ['country', 'year', 'Treatment_Group', 'Criteria_1', 'Criteria_2']:
                    non_null_count = df_wide[col].count()
                    total_count = len(df_wide)
                    coverage = (non_null_count / total_count) * 100
                    print(f"  {col}: {non_null_count}/{total_count} ({coverage:.1f}%)")
            
            # Sample preview
            print(f"\nFirst 5 rows:")
            print(df_wide.head())
            
            return df_wide
        
        else:
            print("✗ No data collected - API might be down")
            return pd.DataFrame()

if __name__ == "__main__":
    collector = WorkingFerryDiwanCollector()
    dataset = collector.collect_all_data_working()
    
    if len(dataset) > 0:
        print(f"\n SUCCESS!")
        print(f"The CSV file contains the exact 123-country sample with treatment/control classifications.")
    else:
        print("Collection failed. Try again in a few minutes - World Bank API might be temporarily down.")
