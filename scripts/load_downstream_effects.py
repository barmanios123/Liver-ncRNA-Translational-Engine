"""
Load downstream ncRNA effect annotations from CSV into downstream_effects table.

Expected CSV columns:
- symbol
- target_gene
- target_ensembl_id
- effect_direction
- effect_type
- phenotype_category
- phenotype_direction
- evidence_type
- evidence_source
- evidence_score
- pubmed_id
- dataset_id
- notes

Usage:
    python -m scripts.load_downstream_effects --csv data/downstream_effects_seed.csv
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


VALID_EFFECT_DIRECTIONS = {"up", "down", "mixed", "neutral"}
VALID_EFFECT_TYPES = {
    "expression",
    "splicing",
    "epigenetic",
    "protein_stability",
    "translation",
    "localization",
    "unknown",
}
VALID_PHENOTYPE_DIRECTIONS = {"improves", "worsens", "neutral", "mixed"}
VALID_EVIDENCE_SOURCES = {"literature", "perturbation_dataset", "inferred", "internal"}
VALID_EVIDENCE_TYPES = {
    "ASO",
    "siRNA",
    "CRISPRi",
    "CRISPRa",
    "overexpression",
    "correlation",
    "knockdown",
    "knockout",
    "unknown",
}


def normalize_text(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return x if x else None


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "symbol",
        "target_gene",
        "target_ensembl_id",
        "effect_direction",
        "effect_type",
        "phenotype_category",
        "phenotype_direction",
        "evidence_type",
        "evidence_source",
        "evidence_score",
        "pubmed_id",
        "dataset_id",
        "notes",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df = df[required_cols].copy()

    for col in df.columns:
        if col != "evidence_score":
            df[col] = df[col].apply(normalize_text)

    df["effect_direction"] = df["effect_direction"].fillna("unknown").str.lower()
    df["effect_type"] = df["effect_type"].fillna("unknown").str.lower()
    df["phenotype_direction"] = df["phenotype_direction"].fillna("neutral").str.lower()
    df["evidence_type"] = df["evidence_type"].fillna("unknown")
    df["evidence_source"] = df["evidence_source"].fillna("internal").str.lower()
    df["phenotype_category"] = df["phenotype_category"].fillna("unknown").str.lower()

    df["evidence_score"] = pd.to_numeric(df["evidence_score"], errors="coerce").fillna(0.0)
    df["evidence_score"] = df["evidence_score"].clip(lower=0.0, upper=1.0)

    bad_effect_dir = ~df["effect_direction"].isin(VALID_EFFECT_DIRECTIONS)
    if bad_effect_dir.any():
        raise ValueError(
            f"Invalid effect_direction values: {sorted(df.loc[bad_effect_dir, 'effect_direction'].dropna().unique())}"
        )

    bad_effect_type = ~df["effect_type"].isin(VALID_EFFECT_TYPES)
    if bad_effect_type.any():
        raise ValueError(
            f"Invalid effect_type values: {sorted(df.loc[bad_effect_type, 'effect_type'].dropna().unique())}"
        )

    bad_pheno_dir = ~df["phenotype_direction"].isin(VALID_PHENOTYPE_DIRECTIONS)
    if bad_pheno_dir.any():
        raise ValueError(
            f"Invalid phenotype_direction values: {sorted(df.loc[bad_pheno_dir, 'phenotype_direction'].dropna().unique())}"
        )

    bad_source = ~df["evidence_source"].isin(VALID_EVIDENCE_SOURCES)
    if bad_source.any():
        raise ValueError(
            f"Invalid evidence_source values: {sorted(df.loc[bad_source, 'evidence_source'].dropna().unique())}"
        )

    bad_ev_type = ~df["evidence_type"].isin(VALID_EVIDENCE_TYPES)
    if bad_ev_type.any():
        raise ValueError(
            f"Invalid evidence_type values: {sorted(df.loc[bad_ev_type, 'evidence_type'].dropna().unique())}"
        )

    df = df[df["symbol"].notna() & df["target_gene"].notna()].copy()
    return df


def load_symbol_map(conn: sqlite3.Connection) -> pd.DataFrame:
    q = """
        SELECT ncrna_id, symbol
        FROM ncrna_master
    """
    m = pd.read_sql_query(q, conn)
    m["symbol"] = m["symbol"].astype(str).str.strip()
    return m


def get_next_effect_id(conn: sqlite3.Connection) -> int:
    q = """
        SELECT effect_id
        FROM downstream_effects
        WHERE effect_id LIKE 'EFF_%'
    """
    cur = conn.cursor()
    rows = cur.execute(q).fetchall()
    max_num = 0
    for (eid,) in rows:
        try:
            num = int(str(eid).replace("EFF_", ""))
            max_num = max(max_num, num)
        except Exception:
            continue
    return max_num + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to downstream effects CSV")
    parser.add_argument("--db", default="ncrna_platform.db", help="Path to SQLite DB")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = validate_dataframe(df)

    conn = sqlite3.connect(args.db)

    symbol_map = load_symbol_map(conn)
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.merge(symbol_map, on="symbol", how="left")

    missing = df[df["ncrna_id"].isna()].copy()
    if not missing.empty:
        print("\n⚠️ Symbols not found in ncrna_master:")
        print(missing[["symbol", "target_gene", "phenotype_category"]].drop_duplicates().to_string(index=False))

    df = df[df["ncrna_id"].notna()].copy()
    if df.empty:
        print("No valid rows to load after symbol mapping.")
        conn.close()
        return

    next_id = get_next_effect_id(conn)
    df = df.reset_index(drop=True)
    df["effect_id"] = [f"EFF_{i:03d}" for i in range(next_id, next_id + len(df))]

    insert_rows = df[
        [
            "effect_id",
            "ncrna_id",
            "target_gene",
            "target_ensembl_id",
            "effect_direction",
            "effect_type",
            "phenotype_category",
            "phenotype_direction",
            "evidence_type",
            "evidence_source",
            "evidence_score",
            "pubmed_id",
            "dataset_id",
            "notes",
        ]
    ].itertuples(index=False, name=None)

    conn.executemany(
        """
        INSERT OR REPLACE INTO downstream_effects (
            effect_id,
            ncrna_id,
            target_gene,
            target_ensembl_id,
            effect_direction,
            effect_type,
            phenotype_category,
            phenotype_direction,
            evidence_type,
            evidence_source,
            evidence_score,
            pubmed_id,
            dataset_id,
            notes,
            added_date
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
        """,
        list(insert_rows),
    )
    conn.commit()

    print(f"✅ Loaded {len(df)} downstream_effects rows into {args.db}")

    summary = (
        df.groupby(["symbol", "phenotype_category", "phenotype_direction"])
        .size()
        .reset_index(name="n")
        .sort_values(["symbol", "phenotype_category"])
    )
    print("\nLoaded summary:")
    print(summary.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()