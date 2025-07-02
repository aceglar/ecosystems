import pandas as pd
import numpy as np

# Define new version of firm-level dependency computation
def compute_dependencies(
    firms: pd.DataFrame,
    encore_ds_direct: pd.DataFrame,
    leontief_inverse: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute direct, indirect, and total dependency scores for each firm based on:
    - Direct scores from ENCORE
    - Indirect scores using Leontief inverse weighted average

    Parameters:
    - firms: DataFrame with columns ['firm_id', 'country', 'sector'] representing (c, j)
    - encore_ds_direct: DataFrame indexed by (country, sector) with columns for ecosystem services
    - leontief_inverse: DataFrame indexed and columned by MultiIndex (country, sector) representing L[(c,j),(c',j')]

    Returns:
    - firms_with_scores: firms DataFrame with direct, indirect, and total scores appended
    """
    firms = firms.copy()
    firm_keys = list(zip(firms['country'], firms['sector']))

    es_services = encore_ds_direct.columns
    results = []

    for (c, j) in firm_keys:
        firm_row = {'country': c, 'sector': j}

        # Direct dependency
        try:
            ds_direct = encore_ds_direct.loc[(c, j)]
        except KeyError:
            ds_direct = pd.Series([0.0]*len(es_services), index=es_services)

        # Indirect dependency
        try:
            # Extract row from L for firm (c,j)
            row_L = leontief_inverse.loc[(c, j)]
            row_sum = row_L.sum()
            if row_sum > 0:
                weights = row_L / row_sum
            else:
                weights = row_L.copy()
                weights[:] = 0.0

            # Weighted average of direct dependencies of all suppliers
            indirect_score = encore_ds_direct.mul(weights.values, axis=0).sum()
        except KeyError:
            indirect_score = pd.Series([0.0]*len(es_services), index=es_services)

        # Total dependency
        total_score = ds_direct + (1 - ds_direct) * indirect_score

        for es in es_services:
            firm_row[f'{es}_direct'] = ds_direct[es]
            firm_row[f'{es}_indirect'] = indirect_score[es]
            firm_row[f'{es}_total'] = total_score[es]

        results.append(firm_row)

    return pd.DataFrame(results)


# Generate synthetic inputs to test the new function
# Ecosystem services
ecoservices = ['pollination', 'flood_protection', 'water_purification']
countries = ['DE', 'FR', 'IT']
sectors = ['A', 'B', 'C']

# Firms (20 random firms)
np.random.seed(0)
firms_synthetic = pd.DataFrame({
    'firm_id': [f'FIRM_{i}' for i in range(20)],
    'country': np.random.choice(countries, 20),
    'sector': np.random.choice(sectors, 20)
})

# Direct dependency scores from ENCORE
idx = pd.MultiIndex.from_product([countries, sectors], names=['country', 'sector'])
encore_direct_synthetic = pd.DataFrame(np.random.rand(len(idx), len(ecoservices)), index=idx, columns=ecoservices)

# Leontief inverse (same index/columns)
L_idx = pd.MultiIndex.from_product([countries, sectors], names=['country', 'sector'])
L_inverse_synthetic = pd.DataFrame(np.random.rand(len(L_idx), len(L_idx)), index=L_idx, columns=L_idx)

# Compute dependencies
firm_dependency_scores = compute_dependencies(
    firms=firms_synthetic,
    encore_ds_direct=encore_direct_synthetic,
    leontief_inverse=L_inverse_synthetic
)

print(firm_dependency_scores)