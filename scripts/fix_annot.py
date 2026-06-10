import pandas as pd
import numpy as np
from pathlib import Path

mapping = pd.read_csv('/Users/beshoyarmanios/ncrna_platform/data/raw/ensembl_to_symbol.csv')
tcga_summary = pd.read_csv('/Users/beshoyarmanios/ncrna_platform/data/processed/tcga_pancan_expression_summary_features.csv')

# Merge mapping with TCGA summary by symbol
annot = mapping.merge(tcga_summary, left_on='symbol', right_on='gene_symbol', how='left')
if 'gene_symbol' in annot.columns:
    annot = annot.drop(columns=['gene_symbol'])

# Add placeholder GEO expression columns as NaN (honest - GEO data not available for panel)
for col in ['expr_mean','expr_std','expr_median','expr_max',
            'expr_nonzero_fraction','expr_cv','expr_log1p_mean','expr_log1p_max']:
    annot[col] = np.nan

# Reorder columns
col_order = ['ensembl_id', 'symbol',
             'expr_mean', 'expr_std', 'expr_median', 'expr_max',
             'expr_nonzero_fraction', 'expr_cv', 'expr_log1p_mean', 'expr_log1p_max',
             'expr_mean_tcga_pancan', 'expr_median_tcga_pancan', 'expr_std_tcga_pancan',
             'expr_min_tcga_pancan', 'expr_max_tcga_pancan', 'expr_q1_tcga_pancan',
             'expr_q3_tcga_pancan', 'expr_iqr_tcga_pancan',
             'expr_prevalence_tcga_pancan', 'in_tcga_pancan_expr']
annot = annot[[c for c in col_order if c in annot.columns]]

out_path = Path('/Users/beshoyarmanios/ncrna_platform/data/processed/expression_summary_features_annot.csv')
annot.to_csv(out_path, index=False)

print('Saved:', out_path)
print('Shape:', annot.shape)
print()
print(annot[['ensembl_id', 'symbol', 'expr_mean_tcga_pancan', 'in_tcga_pancan_expr']].to_string(index=False))
