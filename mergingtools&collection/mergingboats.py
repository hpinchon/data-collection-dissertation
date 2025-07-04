import pandas as pd

# Load datasets
original = pd.read_csv('/workspaces/data-collection-dissertation/AlmostFULLDATA.csv')
datacontract = pd.read_csv('/workspaces/data-collection-dissertation/Copy of Enforcing Contracts.csv', delimiter=';')
datapolity = pd.read_csv('/workspaces/data-collection-dissertation/Copy of PolityDATA.csv', delimiter=';')

datacontract_subset = datacontract[['WB Code', 'Enforcing Contracts score']].copy()
datacontract_subset['year'] = 2019  # Assuming the year is constant for this dataset
datacontract_subset = datacontract_subset.rename(
    columns={
        'WB Code': 'country',
        'Enforcing Contracts score': 'Contract_Enforcement_Score'
    }
)

datapolity_subset = datapolity[['scode', 'year', 'polity2']].copy()
datapolity_subset = datapolity_subset.rename(
    columns={
        'scode': 'country',
        'year': 'year',
        'polity2': 'Polity2_Score'
    }
)

merged = pd.merge(original, datacontract_subset, how='left', on=['country', 'year'])

merged = pd.merge(merged, datapolity_subset, how='left', on=['country', 'year'])
merged.to_csv('merged_data.csv', index=False)