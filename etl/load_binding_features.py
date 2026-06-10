"""
Load RNA-binding features (PaRPI / iDeepB summaries) into the binding_features table.

Expected input CSV (example columns):

    ncrna_id               # REQUIRED: must match ncrna_master.ncrna_id
    n_high_conf_rbps       # optional numeric
    max_rbp_score          # optional numeric
    mean_top5_rbp_score    # optional numeric
    liver_rbp_score        # optional numeric
    max_peak_score         # optional numeric
    peak_density           # optional numeric
    n_binding_hotspots     # optional numeric
    hotspot_5prime_fraction
    hotspot_3prime_fraction
    binding_mechanism_score
    binding_targetability_score
    binding_risk_score
    parpi_model_version    # optional text
    ideepb_model_version   # optional text

Any missing columns will be filled with 0.0 (for numeric) or left as NULL (for text).
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


DEFAULT_DB_PATH = Path("ncrna_platform.db")


NUMERIC_COLS: List[str] = [
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

TEXT_COLS: List[str] = [
    "parpi_model_version",
    "ideepb_model_version",
]


def load_binding_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Binding features CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "ncrna_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'ncrna_id' column matching ncrna_master.ncrna_id")

    # Normalize ncrna_id
    df["ncrna_id"] = df["ncrna_id"].astype(str).str.strip()

    # Ensure all expected columns exist
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0

    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = None

    # Coerce numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Deduplicate by ncrna_id (keep last occurrence)
    df = df.sort_values("ncrna_id").drop_duplicates(subset=["ncrna_id"], keep="last")

    return df[
        ["ncrna_id"]
        + NUMERIC_COLS
        + TEXT_COLS
    ].copy()


def upsert_binding_features(
    df: pd.DataFrame,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    if df.empty:
        print("No binding features to load (input DataFrame is empty).")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check table exists
    cur.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='binding_features'
        """
    )
    if cur.fetchone() is None:
        raise RuntimeError("Table 'binding_features' does not exist. Run db/schema.py:create_database first.")

    # Prepare upsert statement
    # binding_id is generated as BND_<ncrna_id> for simplicity
    sql = """
    INSERT INTO binding_features (
        binding_id,
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
        binding_risk_score,
        parpi_model_version,
        ideepb_model_version,
        last_updated
    )
    VALUES (
        :binding_id,
        :ncrna_id,
        :n_high_conf_rbps,
        :max_rbp_score,
        :mean_top5_rbp_score,
        :liver_rbp_score,
        :max_peak_score,
        :peak_density,
        :n_binding_hotspots,
        :hotspot_5prime_fraction,
        :hotspot_3prime_fraction,
        :binding_mechanism_score,
        :binding_targetability_score,
        :binding_risk_score,
        :parpi_model_version,
        :ideepb_model_version,
        CURRENT_TIMESTAMP
    )
    ON CONFLICT(binding_id) DO UPDATE SET
        n_high_conf_rbps            = excluded.n_high_conf_rbps,
        max_rbp_score               = excluded.max_rbp_score,
        mean_top5_rbp_score         = excluded.mean_top5_rbp_score,
        liver_rbp_score             = excluded.liver_rbp_score,
        max_peak_score              = excluded.max_peak_score,
        peak_density                = excluded.peak_density,
        n_binding_hotspots          = excluded.n_binding_hotspots,
        hotspot_5prime_fraction     = excluded.hotspot_5prime_fraction,
        hotspot_3prime_fraction     = excluded.hotspot_3prime_fraction,
        binding_mechanism_score     = excluded.binding_mechanism_score,
        binding_targetability_score = excluded.binding_targetability_score,
        binding_risk_score          = excluded.binding_risk_score,
        parpi_model_version         = excluded.parpi_model_version,
        ideepb_model_version        = excluded.ideepb_model_version,
        last_updated                = CURRENT_TIMESTAMP
    ;
    """

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "binding_id": f"BND_{row['ncrna_id']}",
                "ncrna_id": row["ncrna_id"],
                **{col: float(row[col]) for col in NUMERIC_COLS},
                "parpi_model_version": row.get("parpi_model_version"),
                "ideepb_model_version": row.get("ideepb_model_version"),
            }
        )

    cur.executemany(sql, rows)
    conn.commit()

    # Quick summary
    cur.execute("SELECT COUNT(*) FROM binding_features")
    total_rows = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*)
        FROM binding_features
        WHERE
            COALESCE(ABS(n_high_conf_rbps), 0)
          + COALESCE(ABS(max_rbp_score), 0)
          + COALESCE(ABS(mean_top5_rbp_score), 0)
          + COALESCE(ABS(liver_rbp_score), 0)
          + COALESCE(ABS(max_peak_score), 0)
          + COALESCE(ABS(peak_density), 0)
          + COALESCE(ABS(n_binding_hotspots), 0)
          + COALESCE(ABS(hotspot_5prime_fraction), 0)
          + COALESCE(ABS(hotspot_3prime_fraction), 0)
          + COALESCE(ABS(binding_mechanism_score), 0)
          + COALESCE(ABS(binding_targetability_score), 0)
          + COALESCE(ABS(binding_risk_score), 0)
          > 0
        """
    )
    nonzero_rows = cur.fetchone()[0]

    conn.close()

    print(f"✅ Upserted {len(rows)} binding feature rows into binding_features.")
    print(f"   Total rows in binding_features: {total_rows}")
    print(f"   Rows with non-zero binding signal: {nonzero_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load PaRPI / iDeepB binding feature summaries into SQLite."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to binding features CSV (one row per ncrna_id).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    db_path = Path(args.db)

    df = load_binding_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    upsert_binding_features(df, db_path=db_path)


if __name__ == "__main__":
    main()