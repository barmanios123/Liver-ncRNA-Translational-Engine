"""
ncRNA Target Intelligence Platform — Feature Engineering
Phase 2: includes curated target evidence + GEO/TCGA expression summary features
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPR_SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "expression_summary_features_merged.csv"
)


def _safe_read_sql(
    query: str,
    conn: sqlite3.Connection,
    params: dict | None = None,
) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn, params=params or {})
    except Exception:
        return pd.DataFrame()


def _load_expression_summary_features() -> pd.DataFrame:
    """
    Load merged expression summary features.

    Expected columns include:
      - ensembl_id
      - symbol or gene_symbol
      - expr_mean
      - expr_std
      - expr_median
      - expr_max
      - expr_nonzero_fraction
      - expr_cv
      - expr_log1p_mean
      - expr_log1p_max
      - expr_mean_tcga_pancan
      - expr_median_tcga_pancan
      - expr_std_tcga_pancan
      - expr_min_tcga_pancan
      - expr_max_tcga_pancan
      - expr_q1_tcga_pancan
      - expr_q3_tcga_pancan
      - expr_iqr_tcga_pancan
      - expr_prevalence_tcga_pancan
      - in_tcga_pancan_expr
    """
    if not EXPR_SUMMARY_PATH.exists():
        return pd.DataFrame(
            columns=[
                "ensembl_id",
                "symbol",
                "expr_mean",
                "expr_std",
                "expr_median",
                "expr_max",
                "expr_nonzero_fraction",
                "expr_cv",
                "expr_log1p_mean",
                "expr_log1p_max",
                "expr_mean_tcga_pancan",
                "expr_median_tcga_pancan",
                "expr_std_tcga_pancan",
                "expr_min_tcga_pancan",
                "expr_max_tcga_pancan",
                "expr_q1_tcga_pancan",
                "expr_q3_tcga_pancan",
                "expr_iqr_tcga_pancan",
                "expr_prevalence_tcga_pancan",
                "in_tcga_pancan_expr",
            ]
        )

    df = pd.read_csv(EXPR_SUMMARY_PATH)

    if "symbol" not in df.columns:
        for cand in ["gene_symbol", "GeneSymbol"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "symbol"})
                break

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip()

    if "ensembl_id" in df.columns:
        df["ensembl_id"] = df["ensembl_id"].astype(str).str.strip()

    return df


def build_feature_matrix(
    db_path: str = "ncrna_platform.db",
    disease_id: str = "DIS_001",
    context_id: str = "CTX_001",
) -> pd.DataFrame:
    """
    Build a per-ncRNA feature matrix by merging evidence from multiple tables.
    """
    conn = sqlite3.connect(db_path)
    params = {"disease_id": disease_id, "context_id": context_id}

    # Aggregated expression features (precomputed)
    expr_feat_q = """
        SELECT
            ef.ncrna_id,
            ef.disease_id,
            ef.context_id,
            ef.n_expression_studies,
            ef.mean_abs_log2fc,
            ef.mean_log2fc,
            ef.max_abs_log2fc,
            ef.sig_rate,
            ef.de_consistency,
            ef.min_padj,
            ef.n_sig_expression_studies,
            ef.n_up_sig_expression_studies,
            ef.n_down_sig_expression_studies,
            ef.mean_tpm_disease,
            ef.mean_tpm_healthy,
            ef.mean_tau,
            ef.log_tpm_disease
        FROM expression_features ef
        WHERE ef.disease_id = :disease_id
          AND ef.context_id = :context_id
    """
    expr_feat_df = _safe_read_sql(expr_feat_q, conn, params=params)

    # Fallback: direct aggregation from expression_evidence
    if expr_feat_df.empty:
        expr_q = """
            SELECT
                e.ncrna_id,
                AVG(ABS(e.log2fc)) AS mean_abs_log2fc,
                AVG(e.log2fc) AS mean_log2fc,
                MAX(ABS(e.log2fc)) AS max_abs_log2fc,
                AVG(e.specificity_tau) AS mean_tau,
                SUM(CASE WHEN e.padj <= 0.05 THEN 1 ELSE 0 END) AS n_sig_expression_studies,
                COUNT(*) AS n_expression_studies,
                AVG(e.tpm_disease) AS mean_tpm_disease,
                AVG(e.tpm_healthy) AS mean_tpm_healthy,
                MIN(e.padj) AS min_padj,
                SUM(CASE WHEN e.padj <= 0.05 AND LOWER(e.direction) = 'up' THEN 1 ELSE 0 END) AS n_up_sig_expression_studies,
                SUM(CASE WHEN e.padj <= 0.05 AND LOWER(e.direction) = 'down' THEN 1 ELSE 0 END) AS n_down_sig_expression_studies
            FROM expression_evidence e
            WHERE e.disease_id = :disease_id
              AND e.context_id = :context_id
            GROUP BY e.ncrna_id
        """
        expr_feat_df = _safe_read_sql(expr_q, conn, params=params)

        if not expr_feat_df.empty:
            expr_feat_df["sig_rate"] = (
                expr_feat_df["n_sig_expression_studies"]
                / expr_feat_df["n_expression_studies"].clip(lower=1)
            )
            expr_feat_df["de_consistency"] = expr_feat_df.apply(
                lambda r: (
                    max(r["n_up_sig_expression_studies"], r["n_down_sig_expression_studies"])
                    / max(r["n_sig_expression_studies"], 1)
                    if r["n_sig_expression_studies"] > 0
                    else 0.0
                ),
                axis=1,
            )
            expr_feat_df["log_tpm_disease"] = np.log1p(
                expr_feat_df["mean_tpm_disease"].fillna(0.0)
            )

    # Tractability / modality
    tract_q = """
        SELECT
            ncrna_id,
            localization,
            isoform_count,
            gc_content,
            secondary_structure_score,
            aso_accessible,
            sirna_compatible,
            small_mol_bindable,
            crispr_feasible,
            best_modality
        FROM tractability_features
    """
    tract_df = _safe_read_sql(tract_q, conn)
    if not tract_df.empty:
        tract_df["modality_breadth"] = (
            tract_df["aso_accessible"].fillna(0)
            + tract_df["sirna_compatible"].fillna(0)
            + tract_df["small_mol_bindable"].fillna(0)
            + tract_df["crispr_feasible"].fillna(0)
        )
        tract_df["nuclear_flag"] = (
            tract_df["localization"].fillna("").astype(str).str.lower() == "nuclear"
        ).astype(int)
        tract_df["isoform_complexity"] = np.log1p(tract_df["isoform_count"].fillna(0))

    # Binding features (PaRPI / iDeepB summary)
    binding_q = """
        SELECT
            ncrna_id,
            n_high_conf_rbps,
            max_rbp_score,
            mean_top5_rbp_score,
            liver_rbp_score,
            max_peak_score,
            peak_density,
            n_binding_hotspots,
            hotspot_5prime_fraction,
            hotspot_3prime_fraction,
            binding_mechanism_score,
            binding_targetability_score,
            binding_risk_score
        FROM binding_features
    """
    binding_df = _safe_read_sql(binding_q, conn)
    if binding_df.empty:
        binding_df = pd.DataFrame(
            columns=[
                "ncrna_id",
                "n_high_conf_rbps",
                "max_rbp_score",
                "mean_top5_rbp_score",
                "liver_rbp_score",
                "max_peak_score",
                "peak_density",
                "n_binding_hotspots",
                "hotspot_5prime_fraction",
                "hotspot_3prime_fraction",
                "binding_mechanism_score",
                "binding_targetability_score",
                "binding_risk_score",
            ]
        )

    # Perturbation evidence
    pert_q = """
        SELECT
            ncrna_id,
            COUNT(*) AS n_perturbation_studies,
            AVG(ABS(effect_size)) AS mean_pert_effect,
            SUM(CASE WHEN confidence = 'high' THEN 1 ELSE 0 END) AS n_high_conf_perts
        FROM perturbation_evidence
        GROUP BY ncrna_id
    """
    pert_df = _safe_read_sql(pert_q, conn)

    # Clinical links
    clin_q = """
        SELECT
            ncrna_id,
            COUNT(*) AS n_clinical_studies,
            AVG(ABS(correlation_r)) AS mean_clinical_r,
            SUM(CASE WHEN pvalue < 0.01 THEN 1 ELSE 0 END) AS n_sig_clinical,
            SUM(CASE WHEN biomarker_type = 'prognostic' THEN 1 ELSE 0 END) AS n_prognostic
        FROM clinical_links
        GROUP BY ncrna_id
    """
    clin_df = _safe_read_sql(clin_q, conn)

    # Pathway / network
    pw_q = """
        SELECT
            ncrna_id,
            COUNT(*) AS n_pathways,
            MIN(enrichment_fdr) AS best_pathway_fdr,
            MAX(ABS(pearson_r)) AS max_hub_correlation
        FROM pathway_links
        GROUP BY ncrna_id
    """
    pw_df = _safe_read_sql(pw_q, conn)

    # Literature
    lit_q = """
        SELECT
            ncrna_id,
            COUNT(*) AS n_lit_statements,
            AVG(confidence_score) AS mean_lit_confidence,
            SUM(is_contradictory) AS n_contradictory,
            COUNT(DISTINCT pubmed_id) AS n_unique_papers,
            MAX(year) AS most_recent_year
        FROM literature_evidence
        GROUP BY ncrna_id
    """
    lit_df = _safe_read_sql(lit_q, conn)

    # NEW: downstream effects (mechanism + phenotype)
    effects_q = """
        SELECT
            ncrna_id,
            COUNT(*) AS n_effects,
            AVG(evidence_score) AS mean_effect_confidence,
            SUM(CASE WHEN phenotype_direction = 'improves' THEN 1 ELSE 0 END) AS n_improving_effects,
            SUM(CASE WHEN phenotype_direction = 'worsens' THEN 1 ELSE 0 END) AS n_worsening_effects,
            SUM(CASE WHEN phenotype_category = 'steatosis' AND phenotype_direction = 'improves' THEN 1 ELSE 0 END) AS n_steatosis_improves,
            SUM(CASE WHEN phenotype_category = 'steatosis' AND phenotype_direction = 'worsens' THEN 1 ELSE 0 END) AS n_steatosis_worsens,
            SUM(CASE WHEN phenotype_category = 'fibrosis' AND phenotype_direction = 'improves' THEN 1 ELSE 0 END) AS n_fibrosis_improves,
            SUM(CASE WHEN phenotype_category = 'fibrosis' AND phenotype_direction = 'worsens' THEN 1 ELSE 0 END) AS n_fibrosis_worsens,
            SUM(CASE WHEN phenotype_category = 'proliferation' AND phenotype_direction = 'improves' THEN 1 ELSE 0 END) AS n_proliferation_improves,
            SUM(CASE WHEN phenotype_category = 'proliferation' AND phenotype_direction = 'worsens' THEN 1 ELSE 0 END) AS n_proliferation_worsens
        FROM downstream_effects
        GROUP BY ncrna_id
    """
    effects_df = _safe_read_sql(effects_q, conn)

    # Master ncRNA registry
    master_q = """
        SELECT
            ncrna_id,
            symbol,
            biotype,
            conservation_score,
            transcript_count
        FROM ncrna_master
    """
    master_df = pd.read_sql_query(master_q, conn)

    # Curated liver-disease targets (if present)
    curated_q = """
        SELECT
            symbol,
            MAX(
                CASE
                    WHEN evidence_tier = 'Tier1' THEN 1.00
                    WHEN evidence_tier = 'Tier2' THEN 0.66
                    WHEN evidence_tier = 'Tier3' THEN 0.33
                    ELSE 0
                END
            ) AS curated_tier_score,
            MAX(CASE WHEN is_contradictory = 1 THEN 1 ELSE 0 END) AS curated_contradiction_flag,
            MAX(CASE WHEN disease_stage = 'HCC' THEN 1 ELSE 0 END) AS curated_hcc_flag,
            MAX(CASE WHEN disease_stage = 'MASLD' THEN 1 ELSE 0 END) AS curated_masld_flag,
            MAX(CASE WHEN disease_stage = 'Fibrosis' THEN 1 ELSE 0 END) AS curated_fibrosis_flag,
            MAX(CASE WHEN disease_stage = 'MASH' THEN 1 ELSE 0 END) AS curated_mash_flag,
            COUNT(*) AS curated_record_count
        FROM curated_targets
        GROUP BY symbol
    """
    curated_df = _safe_read_sql(curated_q, conn)

    conn.close()

    # Start from master list
    df = master_df.copy()

    # Merge by ncrna_id: expression, tractability, binding, perturbation, clinical, pathways, literature, downstream effects
    for sub_df in [
        expr_feat_df,
        tract_df,
        binding_df,
        pert_df,
        clin_df,
        pw_df,
        lit_df,
        effects_df,
    ]:
        if not sub_df.empty:
            merge_cols = [
                c for c in sub_df.columns
                if c not in ["disease_id", "context_id"]
            ]
            sub_df = sub_df[merge_cols].copy()
            df = df.merge(sub_df, on="ncrna_id", how="left")

    # Merge curated evidence by symbol
    if not curated_df.empty:
        curated_df["symbol"] = curated_df["symbol"].astype(str).str.strip()
        df["symbol"] = df["symbol"].astype(str).str.strip()
        df = df.merge(curated_df, on="symbol", how="left")

    # Merge expression summary (GEO/TCGA) by symbol
    expr_summary_df = _load_expression_summary_features()
    if not expr_summary_df.empty:
        df["symbol"] = df["symbol"].astype(str).str.strip()
        expr_summary_df["symbol"] = expr_summary_df["symbol"].astype(str).str.strip()
        df = df.merge(expr_summary_df, on="symbol", how="left", suffixes=("", "_expr"))

        if "ensembl_id_expr" in df.columns:
            df = df.drop(columns=["ensembl_id_expr"])

    # Basic numeric cleaning
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    if "in_tcga_pancan_expr" in df.columns:
        df["in_tcga_pancan_expr"] = df["in_tcga_pancan_expr"].fillna(False).astype(int)

    df["curated_exists"] = (
        (df["curated_record_count"] > 0).astype(int)
        if "curated_record_count" in df.columns
        else 0
    )

    # Backfill key columns used in scoring / risk
    required_defaults = {
        "n_expression_studies": 0.0,
        "mean_abs_log2fc": 0.0,
        "mean_log2fc": 0.0,
        "max_abs_log2fc": 0.0,
        "sig_rate": 0.0,
        "de_consistency": 0.0,
        "min_padj": 1.0,
        "n_sig_expression_studies": 0.0,
        "n_up_sig_expression_studies": 0.0,
        "n_down_sig_expression_studies": 0.0,
        "mean_tpm_disease": 0.0,
        "mean_tpm_healthy": 0.0,
        "mean_tau": 0.0,
        "log_tpm_disease": 0.0,
        "n_contradictory": 0.0,
        "n_lit_statements": 0.0,
        "isoform_count": 0.0,
        "aso_accessible": 0.0,
        "sirna_compatible": 0.0,
        "small_mol_bindable": 0.0,
        "crispr_feasible": 0.0,
        "modality_breadth": 0.0,
        "n_pathways": 0.0,
        "max_hub_correlation": 0.0,
        "mean_lit_confidence": 0.0,
        "n_unique_papers": 0.0,
        "n_perturbation_studies": 0.0,
        "mean_pert_effect": 0.0,
        "n_high_conf_perts": 0.0,
        "n_clinical_studies": 0.0,
        "mean_clinical_r": 0.0,
        "n_sig_clinical": 0.0,
        "n_prognostic": 0.0,
        "conservation_score": 0.0,
        "transcript_count": 0.0,
        "curated_tier_score": 0.0,
        "curated_contradiction_flag": 0.0,
        "curated_hcc_flag": 0.0,
        "curated_masld_flag": 0.0,
        "curated_fibrosis_flag": 0.0,
        "curated_mash_flag": 0.0,
        "curated_record_count": 0.0,
        # NEW: downstream-effect summary defaults
        "n_effects": 0.0,
        "mean_effect_confidence": 0.0,
        "n_improving_effects": 0.0,
        "n_worsening_effects": 0.0,
        "n_steatosis_improves": 0.0,
        "n_steatosis_worsens": 0.0,
        "n_fibrosis_improves": 0.0,
        "n_fibrosis_worsens": 0.0,
        "n_proliferation_improves": 0.0,
        "n_proliferation_worsens": 0.0,
    }

    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)

    # Binding feature defaults (explicitly ensure 0.0 for missing values)
    binding_fill_cols = [
        "n_high_conf_rbps",
        "max_rbp_score",
        "mean_top5_rbp_score",
        "liver_rbp_score",
        "max_peak_score",
        "peak_density",
        "n_binding_hotspots",
        "hotspot_5prime_fraction",
        "hotspot_3prime_fraction",
        "binding_mechanism_score",
        "binding_targetability_score",
        "binding_risk_score",
    ]
    for col in binding_fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # Risk flags
    df["risk_ubiquitous"] = (df["mean_tau"] < 0.4).astype(int)
    df["risk_contradictory_lit"] = (
        (df["n_contradictory"] / df["n_lit_statements"].clip(lower=1)) > 0.3
    ).astype(int)
    df["risk_high_isoforms"] = (df["isoform_count"] > 5).astype(int)

    print(f"✅ Feature matrix: {df.shape[0]} ncRNAs × {df.shape[1]} features")
    return df


if __name__ == "__main__":
    feat_df = build_feature_matrix()
    print(feat_df.head().to_string(index=False))