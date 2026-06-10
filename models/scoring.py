"""
ncRNA Target Intelligence Platform — Translational Scoring Engine
Phase 3: biologic coherence scoring + contradiction-aware perturbation handling
"""

import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception as e:
    xgb = None
    HAS_XGB = False
    print(f"⚠️ XGBoost unavailable; using GradientBoostingRegressor instead. Reason: {e}")

from models.features import build_feature_matrix
from models.biologic_coherence import (
    score_biologic_coherence,
    contradiction_summary,
    aggregate_target_scores,
)

SCORE_WEIGHTS = {
    "relevance": 0.20,
    "specificity": 0.10,
    "mechanism": 0.14,
    "tractability": 0.10,
    "human_evidence": 0.14,
    "curated": 0.13,
    "perturbation_biology": 0.14,
    "risk": 0.05,
}

RELEVANCE_FEATURES = [
    "mean_abs_log2fc",
    "sig_rate",
    "de_consistency",
    "log_tpm_disease",
]

SPECIFICITY_FEATURES = [
    "mean_tau",
    "log_tpm_disease",
]

MECHANISM_FEATURES = [
    "n_perturbation_studies",
    "mean_pert_effect",
    "n_high_conf_perts",
    "n_pathways",
    "max_hub_correlation",
]

TRACTABILITY_FEATURES = [
    "modality_breadth",
    "aso_accessible",
    "sirna_compatible",
    "crispr_feasible",
]

HUMAN_EVIDENCE_FEATURES = [
    "n_clinical_studies",
    "mean_clinical_r",
    "n_sig_clinical",
    "n_prognostic",
    "mean_lit_confidence",
    "n_unique_papers",
]

CURATED_FEATURES = [
    "curated_exists",
    "curated_tier_score",
    "curated_hcc_flag",
    "curated_masld_flag",
    "curated_fibrosis_flag",
    "curated_mash_flag",
]

PERTURBATION_BIOLOGY_FEATURES = [
    "biologic_coherence_score",
    "best_biologic_evidence",
    "mean_evidence_quality",
    "perturbation_study_count",
    "final_perturbation_biology_score",
]

RISK_FEATURES = [
    "risk_ubiquitous",
    "risk_contradictory_lit",
    "risk_high_isoforms",
    "curated_contradiction_flag",
    "contradiction_flag",
    "contradiction_penalty",
    "mean_safety_penalty",
]

ALL_FEATURE_COLS = (
    RELEVANCE_FEATURES
    + SPECIFICITY_FEATURES
    + MECHANISM_FEATURES
    + TRACTABILITY_FEATURES
    + HUMAN_EVIDENCE_FEATURES
    + CURATED_FEATURES
    + PERTURBATION_BIOLOGY_FEATURES
    + RISK_FEATURES
)

TCGA_FEATURES = [
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

GF_REGULATORY_FEATURES = [
    "n_pathways",
    "max_hub_correlation",
    "mean_lit_confidence",
    "n_unique_papers",
    "conservation_score",
]

GF_PERTURBATION_FEATURES = [
    "n_perturbation_studies",
    "n_high_conf_perts",
    "mean_pert_effect",
    "de_consistency",
    "final_perturbation_biology_score",
    "biologic_coherence_score",
    "best_biologic_evidence",
]

GF_DISEASE_SHIFT_FEATURES = [
    "mean_abs_log2fc",
    "sig_rate",
    "mean_clinical_r",
    "n_sig_clinical",
]

GF_CONTEXT_FEATURES = [
    "mean_tau",
    "log_tpm_disease",
    "expr_prevalence_tcga_pancan",
    "in_tcga_pancan_expr",
]

GF_RISK_FEATURES = [
    "risk_ubiquitous",
    "risk_contradictory_lit",
    "risk_high_isoforms",
    "curated_contradiction_flag",
    "contradiction_flag",
    "contradiction_penalty",
    "mean_safety_penalty",
]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _scale_series(series: pd.Series) -> pd.Series:
    s = _safe_numeric(series)
    non_null = s.dropna()

    if non_null.empty:
        return pd.Series(0.0, index=series.index, dtype=float)

    if non_null.nunique() <= 1:
        out = pd.Series(0.0, index=series.index, dtype=float)
        out.loc[non_null.index] = 1.0
        return out

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(non_null.to_numpy().reshape(-1, 1)).flatten()

    out = pd.Series(0.0, index=series.index, dtype=float)
    out.loc[non_null.index] = scaled
    return out.clip(0.0, 1.0)


def compute_component_score(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) == 0:
        return pd.Series(0.0, index=df.index, dtype=float)

    sub = df[cols].copy()
    for col in cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    valid_cols = [c for c in cols if sub[c].notna().any()]
    if len(valid_cols) == 0:
        return pd.Series(0.0, index=df.index, dtype=float)

    scaled_sub = pd.DataFrame(index=sub.index)
    for col in valid_cols:
        scaled_sub[col] = _scale_series(sub[col])

    return scaled_sub.mean(axis=1, skipna=True).fillna(0.0).clip(0.0, 1.0)


def compute_relevance_score(df: pd.DataFrame) -> pd.Series:
    mean_abs_log2fc_scaled = (
        _scale_series(df["mean_abs_log2fc"])
        if "mean_abs_log2fc" in df.columns
        else pd.Series(0.0, index=df.index, dtype=float)
    )
    sig_rate = (
        _safe_numeric(df["sig_rate"]).fillna(0.0).clip(0.0, 1.0)
        if "sig_rate" in df.columns
        else pd.Series(0.0, index=df.index, dtype=float)
    )
    de_consistency = (
        _safe_numeric(df["de_consistency"]).fillna(0.0).clip(0.0, 1.0)
        if "de_consistency" in df.columns
        else pd.Series(0.0, index=df.index, dtype=float)
    )
    log_tpm_scaled = (
        _scale_series(df["log_tpm_disease"])
        if "log_tpm_disease" in df.columns
        else pd.Series(0.0, index=df.index, dtype=float)
    )

    score = (
        0.40 * mean_abs_log2fc_scaled
        + 0.30 * sig_rate
        + 0.20 * de_consistency
        + 0.10 * log_tpm_scaled
    )

    return score.fillna(0.0).clip(0.0, 1.0)


def compute_specificity_score(df: pd.DataFrame) -> pd.Series:
    tau = (
        _safe_numeric(df["mean_tau"]).fillna(0.0).clip(0.0, 1.0)
        if "mean_tau" in df.columns
        else pd.Series(0.0, index=df.index, dtype=float)
    )
    log_tpm_scaled = (
        _scale_series(df["log_tpm_disease"])
        if "log_tpm_disease" in df.columns
        else pd.Series(0.0, index=df.index, dtype=float)
    )

    score = 0.70 * tau + 0.30 * log_tpm_scaled
    return score.fillna(0.0).clip(0.0, 1.0)


def compute_risk_penalty(df: pd.DataFrame) -> pd.Series:
    risk_raw = compute_component_score(df, RISK_FEATURES)
    return (1.0 - risk_raw).clip(0.0, 1.0)


def load_biologic_coherence_features(
    db_path: str,
    disease_id: str | None = None,
    context_id: str | None = None,
) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        expected_df = pd.read_sql_query("SELECT * FROM expected_biology", conn)
        perturb_df = pd.read_sql_query("SELECT * FROM perturbation_evidence", conn)
    except Exception as e:
        print(f"⚠️ Biologic coherence tables missing or unreadable: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

    if expected_df.empty or perturb_df.empty:
        print("⚠️ No biologic coherence data available (expected_biology or perturbation_evidence empty).")
        return pd.DataFrame()

    if disease_id and "disease_context" in expected_df.columns and "disease_context" in perturb_df.columns:
        mask_exp = expected_df["disease_context"].astype(str).str.contains(disease_id, case=False, na=False) | (
            expected_df["disease_context"].astype(str) == disease_id
        )
        mask_pert = perturb_df["disease_context"].astype(str).str.contains(disease_id, case=False, na=False) | (
            perturb_df["disease_context"].astype(str) == disease_id
        )
        expected_df = expected_df[mask_exp]
        perturb_df = perturb_df[mask_pert]

    if context_id and "cell_context" in expected_df.columns and "cell_context" in perturb_df.columns:
        mask_exp = expected_df["cell_context"].astype(str).str.contains(context_id, case=False, na=False) | (
            expected_df["cell_context"].astype(str) == context_id
        )
        mask_pert = perturb_df["cell_context"].astype(str).str.contains(context_id, case=False, na=False) | (
            perturb_df["cell_context"].astype(str) == context_id
        )
        expected_df = expected_df[mask_exp]
        perturb_df = perturb_df[mask_pert]

    if expected_df.empty or perturb_df.empty:
        print("⚠️ Biologic coherence: no rows after disease/context filtering.")
        return pd.DataFrame()

    scored = score_biologic_coherence(expected_df, perturb_df)
    contrad = contradiction_summary(scored)
    agg = aggregate_target_scores(scored, contrad)

    if "target_id" in agg.columns and "ncrna_id" not in agg.columns:
        agg = agg.rename(columns={"target_id": "ncrna_id"})

    return agg


def merge_biologic_coherence_features(
    feature_df: pd.DataFrame,
    db_path: str,
    disease_id: str,
    context_id: str,
) -> pd.DataFrame:
    out = feature_df.copy()
    bc_df = load_biologic_coherence_features(db_path, disease_id, context_id)

    fill_cols = [
        "biologic_coherence_score",
        "best_biologic_evidence",
        "mean_evidence_quality",
        "mean_safety_penalty",
        "perturbation_study_count",
        "contradiction_flag",
        "contradiction_penalty",
        "final_perturbation_biology_score",
    ]

    if bc_df.empty:
        for col in fill_cols:
            if col not in out.columns:
                out[col] = 0.0
        return out

    join_keys = [c for c in ["ncrna_id", "gene_symbol", "disease_context", "cell_context"] if c in out.columns and c in bc_df.columns]

    if not join_keys and "ncrna_id" in out.columns and "ncrna_id" in bc_df.columns:
        join_keys = ["ncrna_id"]

    if not join_keys and "symbol" in out.columns and "gene_symbol" in bc_df.columns:
        bc_df = bc_df.rename(columns={"gene_symbol": "symbol"})
        join_keys = ["symbol"]

    if not join_keys:
        for col in fill_cols:
            if col not in out.columns:
                out[col] = 0.0
        return out

    out = out.merge(bc_df, on=join_keys, how="left")

    for col in fill_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if "contradiction_flag" in out.columns:
        out["contradiction_flag"] = out["contradiction_flag"].astype(int)

    return out


def compute_translational_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["relevance_score"] = compute_relevance_score(df)
    df["specificity_score"] = compute_specificity_score(df)
    df["mechanism_score"] = compute_component_score(df, MECHANISM_FEATURES)
    df["tractability_score"] = compute_component_score(df, TRACTABILITY_FEATURES)
    df["human_evidence_score"] = compute_component_score(df, HUMAN_EVIDENCE_FEATURES)
    df["curated_score"] = compute_component_score(df, CURATED_FEATURES)

    for col, default in {
        "final_perturbation_biology_score": 0.0,
        "biologic_coherence_score": 0.0,
        "best_biologic_evidence": 0.0,
        "mean_evidence_quality": 0.0,
        "mean_safety_penalty": 0.0,
        "perturbation_study_count": 0.0,
        "contradiction_flag": 0,
        "contradiction_penalty": 0.0,
        "mean_lit_confidence": 0.0,
        "n_unique_papers": 0.0,
        "n_pathways": 0.0,
        "n_high_conf_perts": 0.0,
    }.items():
        if col not in df.columns:
            df[col] = default

    lit_conf = _safe_numeric(df["mean_lit_confidence"]).fillna(0.0).clip(0.0, 1.0)
    paper_count = _safe_numeric(df["n_unique_papers"]).fillna(0.0)
    pathway_count = _safe_numeric(df["n_pathways"]).fillna(0.0)
    high_conf_perts = _safe_numeric(df["n_high_conf_perts"]).fillna(0.0)

    paper_flag = (paper_count > 0).astype(float)
    mech_flag = (pathway_count > 0).astype(float)
    pert_flag = (high_conf_perts > 0).astype(float)
    high_conf_lit_flag = (lit_conf >= 0.8).astype(float)

    df["mechanism_score"] = (
        0.70 * df["mechanism_score"].fillna(0.0)
        + 0.20 * lit_conf
        + 0.10 * mech_flag
    ).clip(0.0, 1.0)

    df["human_evidence_score"] = (
        0.70 * df["human_evidence_score"].fillna(0.0)
        + 0.20 * paper_flag
        + 0.10 * lit_conf
    ).clip(0.0, 1.0)

    df["perturbation_biology_score"] = compute_component_score(df, PERTURBATION_BIOLOGY_FEATURES)
    df["risk_score"] = compute_risk_penalty(df)

    df["evidence_bonus"] = (
        0.08 * paper_flag
        + 0.06 * high_conf_lit_flag
        + 0.04 * mech_flag
        + 0.04 * pert_flag
    ).clip(0.0, 0.18)

    df["translational_score"] = (
        SCORE_WEIGHTS["relevance"] * df["relevance_score"]
        + SCORE_WEIGHTS["specificity"] * df["specificity_score"]
        + SCORE_WEIGHTS["mechanism"] * df["mechanism_score"]
        + SCORE_WEIGHTS["tractability"] * df["tractability_score"]
        + SCORE_WEIGHTS["human_evidence"] * df["human_evidence_score"]
        + SCORE_WEIGHTS["curated"] * df["curated_score"]
        + SCORE_WEIGHTS["perturbation_biology"] * df["perturbation_biology_score"]
        + SCORE_WEIGHTS["risk"] * df["risk_score"]
        + df["evidence_bonus"]
    ).clip(0.0, 1.0)

    def assign_tier(score: float) -> str:
        if score >= 0.68:
            return "Tier 1 — High Confidence"
        if score >= 0.42:
            return "Tier 2 — Moderate Confidence"
        return "Tier 3 — Exploratory"

    df["confidence_tier"] = df["translational_score"].apply(assign_tier)
    return df.sort_values("translational_score", ascending=False).reset_index(drop=True)


def compute_geneformer_like_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, default in {
        "mean_lit_confidence": 0.0,
        "n_unique_papers": 0.0,
    }.items():
        if col not in df.columns:
            df[col] = default

    df["gf_regulatory_centrality"] = compute_component_score(df, GF_REGULATORY_FEATURES)
    df["gf_perturbation_impact"] = compute_component_score(df, GF_PERTURBATION_FEATURES)
    df["gf_disease_shift"] = compute_component_score(df, GF_DISEASE_SHIFT_FEATURES)
    df["gf_context_support"] = compute_component_score(df, GF_CONTEXT_FEATURES)

    lit_conf = _safe_numeric(df["mean_lit_confidence"]).fillna(0.0).clip(0.0, 1.0)
    paper_flag = (_safe_numeric(df["n_unique_papers"]).fillna(0.0) > 0).astype(float)

    df["gf_regulatory_centrality"] = (
        0.75 * df["gf_regulatory_centrality"].fillna(0.0)
        + 0.15 * lit_conf
        + 0.10 * paper_flag
    ).clip(0.0, 1.0)

    risk_raw = compute_component_score(df, GF_RISK_FEATURES)
    df["gf_risk_adjustment"] = (1.0 - risk_raw).clip(0.0, 1.0)

    df["gf_evidence_bonus"] = (
        0.10 * paper_flag
        + 0.10 * (lit_conf >= 0.8).astype(float)
    ).clip(0.0, 0.20)

    df["gf_geneformer_like_score"] = (
        0.28 * df["gf_regulatory_centrality"].fillna(0.0)
        + 0.32 * df["gf_perturbation_impact"].fillna(0.0)
        + 0.22 * df["gf_disease_shift"].fillna(0.0)
        + 0.10 * df["gf_context_support"].fillna(0.0)
        + 0.08 * df["gf_risk_adjustment"].fillna(0.0)
        + df["gf_evidence_bonus"]
    ).clip(0.0, 1.0)

    return df


def train_ranking_model(df: pd.DataFrame, label_col: str = "translational_score") -> dict:
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    if not feature_cols:
        return {
            "model": None,
            "model_name": "none",
            "feature_importances": {},
            "cv_r2_mean": 0.0,
            "cv_r2_std": 0.0,
            "feature_cols": [],
        }

    X = df[feature_cols].fillna(0).values
    y = df[label_col].values

    if HAS_XGB and xgb is not None:
        try:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
            model_name = "XGBoost"
        except Exception:
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            model_name = "GradientBoostingRegressor"
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model_name = "GradientBoostingRegressor"

    cv = min(5, len(df))
    if cv >= 2:
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception:
            cv_mean = 0.0
            cv_std = 0.0
    else:
        cv_mean = 0.0
        cv_std = 0.0

    model.fit(X, y)

    importances = getattr(model, "feature_importances_", [0] * len(feature_cols))
    importances = dict(zip(feature_cols, importances))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    print(f"✅ Model trained with {model_name} | CV R² = {cv_mean:.3f} ± {cv_std:.3f}")

    return {
        "model": model,
        "model_name": model_name,
        "feature_importances": importances,
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "feature_cols": feature_cols,
    }


EXPERIMENT_LOGIC = {
    "high_curated_high_relevance": [
        "Validate expression in independent liver cohort",
        "ASO knockdown in PHH or HepG2 with lipid accumulation readout",
        "qPCR confirmation across disease-stage liver samples",
    ],
    "high_mechanism_low_clinical": [
        "Add orthogonal clinical validation cohort",
        "Run perturbation RNA-seq to connect target to pathway mechanism",
        "Test biomarker detectability in serum/exosomes",
    ],
    "contradictory_target": [
        "Manual literature review before wet-lab commitment",
        "Stage-specific validation in hepatocyte and stellate models",
        "Check isoform-specific expression before ASO design",
    ],
    "high_biology_low_consensus": [
        "Repeat perturbation in orthogonal liver-relevant model",
        "Add rescue experiment to confirm directionality",
        "Profile off-target and hepatocyte identity liabilities",
    ],
    "default": [
        "Validate differential expression in an independent cohort",
        "Review pathway support and perturbation evidence",
        "Prioritize for targeted follow-up assay panel",
    ],
}


def recommend_experiments(row: pd.Series) -> list:
    if row.get("curated_tier_score", 0) >= 0.66 and row.get("relevance_score", 0) >= 0.5:
        return EXPERIMENT_LOGIC["high_curated_high_relevance"]
    if row.get("curated_contradiction_flag", 0) == 1 or row.get("contradiction_flag", 0) == 1:
        return EXPERIMENT_LOGIC["contradictory_target"]
    if row.get("biologic_coherence_score", 0) >= 0.6 and row.get("contradiction_penalty", 0) >= 0.25:
        return EXPERIMENT_LOGIC["high_biology_low_consensus"]
    if row.get("mechanism_score", 0) > 0.5 and row.get("human_evidence_score", 0) < 0.3:
        return EXPERIMENT_LOGIC["high_mechanism_low_clinical"]
    return EXPERIMENT_LOGIC["default"]


def build_risk_flags(row: pd.Series) -> list:
    flags = []

    if row.get("risk_ubiquitous", 0):
        flags.append("Low tissue specificity (Tau < 0.4) — systemic liability risk")
    if row.get("risk_contradictory_lit", 0):
        flags.append("High contradictory literature fraction")
    if row.get("risk_high_isoforms", 0):
        flags.append("High isoform complexity may complicate oligo design")
    if row.get("curated_contradiction_flag", 0):
        flags.append("Manual curation flagged this target as contradictory/context-dependent")
    if row.get("contradiction_flag", 0):
        flags.append("Perturbation evidence contains supportive and opposing studies")
    if row.get("contradiction_penalty", 0) >= 0.25:
        flags.append(f"Contradiction penalty elevated ({row.get('contradiction_penalty', 0):.2f})")
    if row.get("mean_safety_penalty", 0) >= 0.25:
        flags.append(f"Biology-aware safety penalty elevated ({row.get('mean_safety_penalty', 0):.2f})")
    if row.get("curated_exists", 0) == 0:
        flags.append("No curated liver-disease evidence currently linked")

    return flags if flags else ["No major risk flags identified"]


def ensure_target_scores_schema(conn: sqlite3.Connection):
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS target_scores (
            score_id TEXT PRIMARY KEY,
            ncrna_id TEXT,
            disease_id TEXT,
            context_id TEXT,
            relevance_score REAL,
            specificity_score REAL,
            mechanism_score REAL,
            tractability_score REAL,
            human_evidence_score REAL,
            curated_score REAL,
            perturbation_biology_score REAL,
            risk_score REAL,
            translational_score REAL,
            confidence_tier TEXT,
            top_evidence TEXT,
            risk_flags TEXT,
            recommended_experiments TEXT,
            model_version TEXT,
            scored_date TEXT,
            expr_mean_tcga_pancan REAL,
            expr_median_tcga_pancan REAL,
            expr_std_tcga_pancan REAL,
            expr_min_tcga_pancan REAL,
            expr_max_tcga_pancan REAL,
            expr_q1_tcga_pancan REAL,
            expr_q3_tcga_pancan REAL,
            expr_iqr_tcga_pancan REAL,
            expr_prevalence_tcga_pancan REAL,
            in_tcga_pancan_expr INTEGER,
            gf_geneformer_like_score REAL,
            gf_regulatory_centrality REAL,
            gf_perturbation_impact REAL,
            gf_disease_shift REAL,
            gf_context_support REAL,
            gf_risk_adjustment REAL,
            gf_model_version TEXT,
            n_pathways REAL,
            max_hub_correlation REAL,
            mean_lit_confidence REAL,
            n_unique_papers REAL,
            conservation_score REAL,
            n_perturbation_studies REAL,
            n_high_conf_perts REAL,
            mean_pert_effect REAL,
            de_consistency REAL,
            mean_abs_log2fc REAL,
            sig_rate REAL,
            mean_clinical_r REAL,
            n_sig_clinical REAL,
            mean_tau REAL,
            log_tpm_disease REAL,
            risk_ubiquitous INTEGER,
            risk_contradictory_lit INTEGER,
            risk_high_isoforms INTEGER,
            curated_contradiction_flag INTEGER,
            biologic_coherence_score REAL,
            best_biologic_evidence REAL,
            mean_evidence_quality REAL,
            mean_safety_penalty REAL,
            perturbation_study_count REAL,
            contradiction_flag INTEGER,
            contradiction_penalty REAL,
            final_perturbation_biology_score REAL
        )
        """
    )

    existing_cols = pd.read_sql_query("PRAGMA table_info(target_scores);", conn)["name"].tolist()

    extra_schema = {
        "curated_score": "REAL",
        "perturbation_biology_score": "REAL",
        "scored_date": "TEXT",
        "expr_mean_tcga_pancan": "REAL",
        "expr_median_tcga_pancan": "REAL",
        "expr_std_tcga_pancan": "REAL",
        "expr_min_tcga_pancan": "REAL",
        "expr_max_tcga_pancan": "REAL",
        "expr_q1_tcga_pancan": "REAL",
        "expr_q3_tcga_pancan": "REAL",
        "expr_iqr_tcga_pancan": "REAL",
        "expr_prevalence_tcga_pancan": "REAL",
        "in_tcga_pancan_expr": "INTEGER",
        "gf_geneformer_like_score": "REAL",
        "gf_regulatory_centrality": "REAL",
        "gf_perturbation_impact": "REAL",
        "gf_disease_shift": "REAL",
        "gf_context_support": "REAL",
        "gf_risk_adjustment": "REAL",
        "gf_model_version": "TEXT",
        "n_pathways": "REAL",
        "max_hub_correlation": "REAL",
        "mean_lit_confidence": "REAL",
        "n_unique_papers": "REAL",
        "conservation_score": "REAL",
        "n_perturbation_studies": "REAL",
        "n_high_conf_perts": "REAL",
        "mean_pert_effect": "REAL",
        "de_consistency": "REAL",
        "mean_abs_log2fc": "REAL",
        "sig_rate": "REAL",
        "mean_clinical_r": "REAL",
        "n_sig_clinical": "REAL",
        "mean_tau": "REAL",
        "log_tpm_disease": "REAL",
        "risk_ubiquitous": "INTEGER",
        "risk_contradictory_lit": "INTEGER",
        "risk_high_isoforms": "INTEGER",
        "curated_contradiction_flag": "INTEGER",
        "biologic_coherence_score": "REAL",
        "best_biologic_evidence": "REAL",
        "mean_evidence_quality": "REAL",
        "mean_safety_penalty": "REAL",
        "perturbation_study_count": "REAL",
        "contradiction_flag": "INTEGER",
        "contradiction_penalty": "REAL",
        "final_perturbation_biology_score": "REAL",
    }

    for col, col_type in extra_schema.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE target_scores ADD COLUMN {col} {col_type}")


def save_scores_to_db(
    scored_df: pd.DataFrame,
    db_path="ncrna_platform.db",
    disease_id="DIS_001",
    context_id="CTX_001",
    model_version="v3.1_biocoherence_evidenceaware",
    gf_model_version="gf_v0.7",
):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    ensure_target_scores_schema(conn)

    for _, row in scored_df.iterrows():
        nid = row["ncrna_id"]
        exps = recommend_experiments(row)
        flags = build_risk_flags(row)

        top_ev = []
        if row.get("curated_exists", 0):
            top_ev.append(f"Curated liver evidence tier score = {row.get('curated_tier_score', 0):.2f}")
        if row.get("mean_abs_log2fc", 0) > 0:
            top_ev.append(f"Mean |log2FC| = {row.get('mean_abs_log2fc', 0):.2f}")
        if row.get("sig_rate", 0) > 0:
            top_ev.append(f"Significant dataset fraction = {row.get('sig_rate', 0):.2f}")
        if row.get("de_consistency", 0) > 0:
            top_ev.append(f"DE direction consistency = {row.get('de_consistency', 0):.2f}")
        if row.get("mean_clinical_r", 0) > 0:
            top_ev.append(f"Clinical correlation r = {row.get('mean_clinical_r', 0):.2f}")
        if row.get("n_high_conf_perts", 0) > 0:
            top_ev.append(f"{int(row.get('n_high_conf_perts', 0))} high-confidence perturbation studies")
        if row.get("n_unique_papers", 0) > 0:
            top_ev.append(f"{int(row.get('n_unique_papers', 0))} curated paper(s); mean confidence = {row.get('mean_lit_confidence', 0):.2f}")
        if row.get("biologic_coherence_score", 0) > 0:
            top_ev.append(f"Biologic coherence score = {row.get('biologic_coherence_score', 0):.2f}")
        if row.get("contradiction_penalty", 0) > 0:
            top_ev.append(f"Contradiction penalty = {row.get('contradiction_penalty', 0):.2f}")
        if row.get("in_tcga_pancan_expr", 0):
            top_ev.append(f"TCGA pan-cancer mean expression = {row.get('expr_mean_tcga_pancan', 0):.2f}")
        if row.get("gf_geneformer_like_score", 0) > 0:
            top_ev.append(f"Geneformer-like score = {row.get('gf_geneformer_like_score', 0):.2f}")

        if not top_ev:
            top_ev = ["Limited evidence available"]

        insert_cols = [
            "score_id",
            "ncrna_id",
            "disease_id",
            "context_id",
            "relevance_score",
            "specificity_score",
            "mechanism_score",
            "tractability_score",
            "human_evidence_score",
            "curated_score",
            "perturbation_biology_score",
            "risk_score",
            "translational_score",
            "confidence_tier",
            "top_evidence",
            "risk_flags",
            "recommended_experiments",
            "model_version",
            "scored_date",
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
            "gf_geneformer_like_score",
            "gf_regulatory_centrality",
            "gf_perturbation_impact",
            "gf_disease_shift",
            "gf_context_support",
            "gf_risk_adjustment",
            "gf_model_version",
            "n_pathways",
            "max_hub_correlation",
            "mean_lit_confidence",
            "n_unique_papers",
            "conservation_score",
            "n_perturbation_studies",
            "n_high_conf_perts",
            "mean_pert_effect",
            "de_consistency",
            "mean_abs_log2fc",
            "sig_rate",
            "mean_clinical_r",
            "n_sig_clinical",
            "mean_tau",
            "log_tpm_disease",
            "risk_ubiquitous",
            "risk_contradictory_lit",
            "risk_high_isoforms",
            "curated_contradiction_flag",
            "biologic_coherence_score",
            "best_biologic_evidence",
            "mean_evidence_quality",
            "mean_safety_penalty",
            "perturbation_study_count",
            "contradiction_flag",
            "contradiction_penalty",
            "final_perturbation_biology_score",
        ]

        values = (
            f"SCR_{nid}_{disease_id}_{context_id}",
            nid,
            disease_id,
            context_id,
            round(float(row.get("relevance_score", 0)), 4),
            round(float(row.get("specificity_score", 0)), 4),
            round(float(row.get("mechanism_score", 0)), 4),
            round(float(row.get("tractability_score", 0)), 4),
            round(float(row.get("human_evidence_score", 0)), 4),
            round(float(row.get("curated_score", 0)), 4),
            round(float(row.get("perturbation_biology_score", 0)), 4),
            round(float(row.get("risk_score", 0)), 4),
            round(float(row.get("translational_score", 0)), 4),
            row.get("confidence_tier", "Tier 3 — Exploratory"),
            json.dumps(top_ev),
            json.dumps(flags),
            json.dumps(exps),
            model_version,
            pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S"),
            round(float(row.get("expr_mean_tcga_pancan", 0)), 6),
            round(float(row.get("expr_median_tcga_pancan", 0)), 6),
            round(float(row.get("expr_std_tcga_pancan", 0)), 6),
            round(float(row.get("expr_min_tcga_pancan", 0)), 6),
            round(float(row.get("expr_max_tcga_pancan", 0)), 6),
            round(float(row.get("expr_q1_tcga_pancan", 0)), 6),
            round(float(row.get("expr_q3_tcga_pancan", 0)), 6),
            round(float(row.get("expr_iqr_tcga_pancan", 0)), 6),
            round(float(row.get("expr_prevalence_tcga_pancan", 0)), 6),
            int(row.get("in_tcga_pancan_expr", 0)),
            round(float(row.get("gf_geneformer_like_score", 0)), 6),
            round(float(row.get("gf_regulatory_centrality", 0)), 6),
            round(float(row.get("gf_perturbation_impact", 0)), 6),
            round(float(row.get("gf_disease_shift", 0)), 6),
            round(float(row.get("gf_context_support", 0)), 6),
            round(float(row.get("gf_risk_adjustment", 0)), 6),
            gf_model_version,
            round(float(row.get("n_pathways", 0)), 6),
            round(float(row.get("max_hub_correlation", 0)), 6),
            round(float(row.get("mean_lit_confidence", 0)), 6),
            round(float(row.get("n_unique_papers", 0)), 6),
            round(float(row.get("conservation_score", 0)), 6),
            round(float(row.get("n_perturbation_studies", 0)), 6),
            round(float(row.get("n_high_conf_perts", 0)), 6),
            round(float(row.get("mean_pert_effect", 0)), 6),
            round(float(row.get("de_consistency", 0)), 6),
            round(float(row.get("mean_abs_log2fc", 0)), 6),
            round(float(row.get("sig_rate", 0)), 6),
            round(float(row.get("mean_clinical_r", 0)), 6),
            round(float(row.get("n_sig_clinical", 0)), 6),
            round(float(row.get("mean_tau", 0)), 6),
            round(float(row.get("log_tpm_disease", 0)), 6),
            int(row.get("risk_ubiquitous", 0)),
            int(row.get("risk_contradictory_lit", 0)),
            int(row.get("risk_high_isoforms", 0)),
            int(row.get("curated_contradiction_flag", 0)),
            round(float(row.get("biologic_coherence_score", 0)), 6),
            round(float(row.get("best_biologic_evidence", 0)), 6),
            round(float(row.get("mean_evidence_quality", 0)), 6),
            round(float(row.get("mean_safety_penalty", 0)), 6),
            round(float(row.get("perturbation_study_count", 0)), 6),
            int(row.get("contradiction_flag", 0)),
            round(float(row.get("contradiction_penalty", 0)), 6),
            round(float(row.get("final_perturbation_biology_score", 0)), 6),
        )

        sql = f"""
            INSERT OR REPLACE INTO target_scores ({', '.join(insert_cols)})
            VALUES ({', '.join(['?'] * len(insert_cols))})
        """
        c.execute(sql, values)

    conn.commit()
    conn.close()
    print(f"✅ {len(scored_df)} scores saved to database")


def run_scoring_pipeline(
    db_path="ncrna_platform.db",
    disease_id="DIS_001",
    context_id="CTX_001",
):
    print("\n── ncRNA Target Intelligence Platform ──")
    print(f"   Disease: {disease_id}  |  Context: {context_id}\n")

    feat_df = build_feature_matrix(db_path, disease_id, context_id)
    feat_df = merge_biologic_coherence_features(feat_df, db_path, disease_id, context_id)

    scored_df = compute_translational_scores(feat_df)
    scored_df = compute_geneformer_like_scores(scored_df)

    model_out = train_ranking_model(scored_df)
    save_scores_to_db(scored_df, db_path, disease_id, context_id)

    print("\n── Top 10 Ranked ncRNA Targets ──")
    display_cols = [
        "symbol",
        "translational_score",
        "gf_geneformer_like_score",
        "confidence_tier",
        "curated_tier_score",
        "relevance_score",
        "specificity_score",
        "mechanism_score",
        "perturbation_biology_score",
        "biologic_coherence_score",
        "contradiction_penalty",
        "human_evidence_score",
        "n_unique_papers",
        "mean_lit_confidence",
    ]
    if "expr_mean_tcga_pancan" in scored_df.columns:
        display_cols.append("expr_mean_tcga_pancan")

    present_cols = [c for c in display_cols if c in scored_df.columns]
    print(scored_df[present_cols].head(10).to_string(index=False))
    return scored_df, model_out


if __name__ == "__main__":
    scored_df, model_out = run_scoring_pipeline()