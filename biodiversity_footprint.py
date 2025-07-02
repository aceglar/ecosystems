import pandas as pd
import numpy as np

def compute_bank_biodiversity_footprint(
    loan_data: pd.DataFrame,
    firm_info: pd.DataFrame,
    direct_intensity_vector: pd.DataFrame,
    leontief_inverse: pd.DataFrame,
    msa_ghg_factor: float = 4.37e-5
) -> pd.DataFrame:
    """
    Compute biodiversity footprint (MSA-loss) for banks based on land use and GHG pressures.

    Parameters:
    - loan_data: DataFrame with columns ['bank_id', 'firm_id', 'loan_amount']
    - firm_info: DataFrame with ['firm_id', 'country', 'sector']
    - direct_intensity_vector: DataFrame indexed by (country, sector) with columns:
        ['msa_ghg_per_eur', 'msa_lu_per_eur']
    - leontief_inverse: Leontief inverse matrix indexed/columned by (country, sector)
    - msa_ghg_factor: MSA loss per kg CO₂ for 100-year integration (default GLOBIO: 4.37e-5)

    Returns:
    - DataFrame: bank-level MSA loss (GHG, LU, total)
    """
    # Merge loan and firm metadata
    df = loan_data.merge(firm_info, on='firm_id', how='left')

    # Match sector-level MSA intensities (F vector)
    df['key'] = list(zip(df['country'], df['sector']))
    direct_msa = direct_intensity_vector.copy()
    direct_msa['key'] = direct_msa.index
    df = df.merge(direct_msa, on='key', how='left')

    # Compute firm-level direct footprint (MSA-loss * loan exposure)
    df['msa_ghg_direct'] = df['msa_ghg_per_eur'] * df['loan_amount']
    df['msa_lu_direct'] = df['msa_lu_per_eur'] * df['loan_amount']

    # --- Indirect footprint using F_tot = L' x diag(F) ---
    # Construct diagonal matrix of direct intensities
    F_ghg = np.diag(direct_intensity_vector['msa_ghg_per_eur'].values)
    F_lu = np.diag(direct_intensity_vector['msa_lu_per_eur'].values)

    # Total impact per euro produced (upstream-inclusive)
    Ftot_ghg = leontief_inverse.T @ F_ghg
    Ftot_lu = leontief_inverse.T @ F_lu

    # Aggregate across inputs to get per-country-sector footprints
    indirect_ghg = pd.Series(Ftot_ghg.sum(axis=1), index=leontief_inverse.index)
    indirect_lu = pd.Series(Ftot_lu.sum(axis=1), index=leontief_inverse.index)

    indirect_df = pd.DataFrame({
        'msa_ghg_total_per_eur': indirect_ghg,
        'msa_lu_total_per_eur': indirect_lu
    })
    indirect_df['key'] = indirect_df.index

    df = df.merge(indirect_df, on='key', how='left')
    df['msa_ghg_total'] = df['msa_ghg_total_per_eur'] * df['loan_amount']
    df['msa_lu_total'] = df['msa_lu_total_per_eur'] * df['loan_amount']

    # Final aggregation at bank level
    bank_fp = df.groupby('bank_id').agg({
        'msa_ghg_total': 'sum',
        'msa_lu_total': 'sum'
    }).reset_index()

    bank_fp['msa_total'] = bank_fp['msa_ghg_total'] + bank_fp['msa_lu_total']
    return bank_fp

# Create synthetic data to test `compute_bank_biodiversity_footprint`

# Define small set of countries and sectors
countries = ['DE', 'FR', 'IT']
sectors = ['A', 'B', 'C']
idx = pd.MultiIndex.from_product([countries, sectors], names=['country', 'sector'])

# Synthetic firm info (30 firms)
np.random.seed(42)
firm_info = pd.DataFrame({
    'firm_id': [f'FIRM_{i}' for i in range(30)],
    'country': np.random.choice(countries, 30),
    'sector': np.random.choice(sectors, 30)
})

# Synthetic loan data: 10 banks lend to 30 firms
banks = [f'BANK_{i}' for i in range(10)]
loan_data = pd.DataFrame({
    'bank_id': np.random.choice(banks, 60),
    'firm_id': np.random.choice(firm_info['firm_id'], 60),
    'loan_amount': np.random.uniform(1e5, 1e6, size=60)
})

# Direct intensity vector: MSA loss per €1 (synthetic but positive)
direct_intensity_vector = pd.DataFrame({
    'msa_ghg_per_eur': np.random.uniform(1e-8, 1e-6, size=len(idx)),
    'msa_lu_per_eur': np.random.uniform(1e-7, 1e-5, size=len(idx))
}, index=idx)

# Synthetic Leontief inverse matrix: positive semi-random values
leontief_inverse = pd.DataFrame(
    np.random.uniform(0.1, 2.0, size=(len(idx), len(idx))),
    index=idx,
    columns=idx
)

# Run the biodiversity footprint function
bank_footprints = compute_bank_biodiversity_footprint(
    loan_data=loan_data,
    firm_info=firm_info,
    direct_intensity_vector=direct_intensity_vector,
    leontief_inverse=leontief_inverse
)

print(bank_footprints)