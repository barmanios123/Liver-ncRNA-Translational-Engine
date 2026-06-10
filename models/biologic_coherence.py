from __future__ import annotations

import numpy as np
import pandas as pd

MODALITY_WEIGHTS = {
    "ASO": 1.00,
    "siRNA": 0.95,
    "CRISPRi": 0.95,
    "CRISPRa": 0.85,
    "KO": 0.90,
    "OE": 0.80,
}

MODEL_WEIGHTS = {
    "primary_hepatocyte": 1.00,
    "primary_hsc": 1.00,
    "organoid": 0.95,
    "in_vivo_mouse": 0.90,
    "human_biopsy_derived": 1.00,
    "cell_line_liver": 0.70,
    "non_liver_cell_line": 0.35,
    "unknown": 0.50,
}


def _norm_text(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()


def _split_set(x):
    if pd.isna(x) or x is None or str(x).strip() == "":
        return set()
    return {i.strip().lower() for i in str(x).split("|") if i.strip()}


def _safe_mean(vals):
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def compute_expected_match(row):
    baseline = _norm_text(row.get("baseline_direction"))
    preferred = _norm_text(row.get("preferred_intervention"))
    expected_pheno = _norm_text(row.get("expected_phenotype_direction"))
    perturb = _norm_text(row.get("perturbation_direction"))
    observed = _norm_text(row.get("observed_primary_effect"))

    intervention_match = 0.5
    if preferred in {"either", None, "unknown"}:
        intervention_match = 1.0
    elif preferred == perturb:
        intervention_match = 1.0
    else:
        intervention_match = 0.0

    phenotype_match = 0.5
    if expected_pheno in {None, "unknown", "mixed"} or observed in {None, "unknown", "mixed"}:
        phenotype_match = 0.5
    elif expected_pheno == observed:
        phenotype_match = 1.0
    else:
        phenotype_match = 0.0

    return 0.4 * intervention_match + 0.6 * phenotype_match


def compute_marker_coherence(row):
    expected_pos = _split_set(row.get("expected_marker_positive"))
    expected_neg = _split_set(row.get("expected_marker_negative"))
    observed_pos = _split_set(row.get("observed_marker_positive"))
    observed_neg = _split_set(row.get("observed_marker_negative"))

    pos_hits = len(expected_pos & observed_pos)
    neg_hits = len(expected_neg & observed_neg)

    pos_total = max(len(expected_pos), 1)
    neg_total = max(len(expected_neg), 1)

    pos_score = pos_hits / pos_total if expected_pos else 0.5
    neg_score = neg_hits / neg_total if expected_neg else 0.5

    return 0.5 * pos_score + 0.5 * neg_score


def compute_safety_penalty(row):
    liability = _split_set(row.get("safety_liability_marker"))
    observed_neg = _split_set(row.get("observed_marker_negative"))
    observed_pos = _split_set(row.get("observed_marker_positive"))

    liability_hits = len(liability & (observed_neg | observed_pos))
    if not liability:
        return 0.0

    frac = liability_hits / len(liability)
    return min(frac, 1.0)


def compute_evidence_quality(row):
    modality = row.get("modality", "unknown")
    model = row.get("model_system", "unknown")

    modality_w = MODALITY_WEIGHTS.get(str(modality), 0.60)
    model_w = MODEL_WEIGHTS.get(str(model), 0.50)

    on_target = row.get("on_target_confidence", 0.5)
    off_target_risk = row.get("off_target_risk", 0.5)
    liver_relevance = row.get("liver_relevance", 0.5)
    replicate_count = row.get("replicate_count", 1)

    rep_score = min(np.log1p(replicate_count) / np.log(4), 1.0)

    quality = (
        0.20 * modality_w
        + 0.20 * model_w
        + 0.20 * float(on_target)
        + 0.15 * (1 - float(off_target_risk))
        + 0.15 * float(liver_relevance)
        + 0.10 * rep_score
    )
    return float(np.clip(quality, 0, 1))


def score_biologic_coherence(expected_df, perturb_df):
    join_cols = ["target_id", "gene_symbol", "disease_context", "cell_context"]
    usable_join_cols = [c for c in join_cols if c in expected_df.columns and c in perturb_df.columns]

    if not usable_join_cols:
        raise ValueError(
            "No shared join columns found between expected_df and perturb_df. "
            f"Expected one or more of: {join_cols}"
        )

    merged = perturb_df.merge(expected_df, on=usable_join_cols, how="left", suffixes=("", "_exp"))

    merged["expected_match_score"] = merged.apply(compute_expected_match, axis=1)
    merged["marker_coherence_score"] = merged.apply(compute_marker_coherence, axis=1)
    merged["safety_penalty"] = merged.apply(compute_safety_penalty, axis=1)
    merged["evidence_quality"] = merged.apply(compute_evidence_quality, axis=1)

    merged["biologic_coherence_evidence_score"] = (
        0.35 * merged["expected_match_score"]
        + 0.30 * merged["marker_coherence_score"]
        + 0.25 * merged["evidence_quality"]
        - 0.20 * merged["safety_penalty"]
    ).clip(0, 1)

    # Alias expected by scoring.py
    merged["biologic_coherence_score"] = merged["biologic_coherence_evidence_score"]

    return merged


def contradiction_summary(scored_df):
    def classify_direction(x):
        x = _norm_text(x)
        if x in {"beneficial", "improve"}:
            return "supportive"
        if x in {"harmful", "worsen"}:
            return "opposing"
        if x in {"neutral", "mixed", None, "unknown"}:
            return "unclear"
        return "unclear"

    df = scored_df.copy()
    df["direction_class"] = df["observed_primary_effect"].map(classify_direction)

    group_cols = ["target_id", "gene_symbol", "disease_context", "cell_context"]
    usable_group_cols = [c for c in group_cols if c in df.columns]
    rows = []

    for keys, g in df.groupby(usable_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_map = dict(zip(usable_group_cols, keys))

        supportive = (g["direction_class"] == "supportive").sum()
        opposing = (g["direction_class"] == "opposing").sum()
        unclear = (g["direction_class"] == "unclear").sum()
        total = len(g)

        contradiction_ratio = min(supportive, opposing) / max(supportive + opposing, 1)
        contradiction_flag = int(supportive > 0 and opposing > 0)

        weighted_support = _safe_mean(
            g.loc[g["direction_class"] == "supportive", "biologic_coherence_evidence_score"]
        )
        weighted_oppose = _safe_mean(
            g.loc[g["direction_class"] == "opposing", "biologic_coherence_evidence_score"]
        )

        row = {
            "n_studies": total,
            "n_supportive": int(supportive),
            "n_opposing": int(opposing),
            "n_unclear": int(unclear),
            "contradiction_flag": contradiction_flag,
            "contradiction_ratio": float(contradiction_ratio),
            "mean_supportive_score": weighted_support if not np.isnan(weighted_support) else 0.0,
            "mean_opposing_score": weighted_oppose if not np.isnan(weighted_oppose) else 0.0,
        }
        row.update(key_map)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["contradiction_penalty"] = (
        0.6 * out["contradiction_ratio"]
        + 0.4 * (out["mean_opposing_score"] > out["mean_supportive_score"]).astype(float)
    ).clip(0, 1)

    return out


def aggregate_target_scores(scored_df, contradiction_df):
    group_cols = ["target_id", "gene_symbol", "disease_context", "cell_context"]
    usable_group_cols = [c for c in group_cols if c in scored_df.columns]

    count_col = "evidence_id" if "evidence_id" in scored_df.columns else usable_group_cols[0]

    agg = (
        scored_df.groupby(usable_group_cols, dropna=False)
        .agg(
            biologic_coherence_score=("biologic_coherence_score", "mean"),
            best_biologic_evidence=("biologic_coherence_evidence_score", "max"),
            mean_evidence_quality=("evidence_quality", "mean"),
            mean_safety_penalty=("safety_penalty", "mean"),
            perturbation_study_count=(count_col, "count"),
        )
        .reset_index()
    )

    if contradiction_df is not None and not contradiction_df.empty:
        merge_cols = [c for c in usable_group_cols if c in contradiction_df.columns]
        contradiction_keep = merge_cols + ["contradiction_flag", "contradiction_penalty"]

        final = agg.merge(
            contradiction_df[contradiction_keep],
            on=merge_cols,
            how="left",
        )
    else:
        final = agg.copy()
        final["contradiction_flag"] = 0
        final["contradiction_penalty"] = 0.0

    final["contradiction_flag"] = final["contradiction_flag"].fillna(0).astype(int)
    final["contradiction_penalty"] = final["contradiction_penalty"].fillna(0.0)

    final["final_perturbation_biology_score"] = (
        0.70 * final["biologic_coherence_score"]
        + 0.20 * final["best_biologic_evidence"]
        + 0.10 * np.clip(np.log1p(final["perturbation_study_count"]) / np.log(5), 0, 1)
        - 0.35 * final["contradiction_penalty"]
    ).clip(0, 1)

    return final