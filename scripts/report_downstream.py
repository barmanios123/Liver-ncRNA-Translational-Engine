"""
Downstream phenotype report for ncRNA platform.

Example usage:

  # All ncRNAs that improve fibrosis in DIS_003, liver hepatocyte context
  python -m scripts.report_downstream \
    --disease DIS_003 \
    --context CTX_001 \
    --phenotype fibrosis \
    --direction improves \
    --limit 50 \
    --csv output/downstream_fibrosis_improves_DIS_003_CTX_001.csv
"""

import argparse
import csv
import sqlite3
from pathlib import Path
from textwrap import shorten


def parse_args():
    p = argparse.ArgumentParser(
        description="Query downstream phenotype evidence joined to target scores."
    )
    p.add_argument("--db", default="ncrna_platform.db", help="Path to SQLite DB")
    p.add_argument("--disease", required=True, help="Disease ID (e.g., DIS_001)")
    p.add_argument("--context", required=True, help="Context ID (e.g., CTX_001)")
    p.add_argument(
        "--phenotype",
        required=True,
        help="Phenotype category (e.g., fibrosis, steatosis, glucose)",
    )
    p.add_argument(
        "--direction",
        choices=["improves", "worsens", "mixed", "neutral"],
        required=True,
        help="Phenotype direction filter",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of rows to display/export",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write CSV report",
    )
    p.add_argument(
        "--biotype",
        type=str,
        default=None,
        help="Optional biotype filter (e.g., lncRNA)",
    )
    p.add_argument(
        "--dedup-symbol",
        action="store_true",
        help="Keep only the top row per symbol after sorting",
    )
    return p.parse_args()


def fmt(x, nd=4):
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main():
    args = parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(db_path.as_posix())
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = """
    SELECT
      n.ncrna_id,
      n.symbol,
      n.biotype,
      t.disease_id,
      t.context_id,
      t.translational_score,
      t.relevance_score,
      t.specificity_score,
      t.mechanism_score,
      t.human_evidence_score,
      t.risk_score,
      t.confidence_tier,
      d.effect_id,
      d.phenotype_category,
      d.phenotype_direction,
      d.target_gene,
      d.effect_direction,
      d.effect_type,
      d.evidence_type,
      d.evidence_source,
      d.evidence_score,
      d.pubmed_id,
      d.dataset_id,
      d.notes
    FROM target_scores t
    JOIN ncrna_master n
      ON n.ncrna_id = t.ncrna_id
    JOIN downstream_effects d
      ON d.ncrna_id = t.ncrna_id
    WHERE t.disease_id = ?
      AND t.context_id = ?
      AND d.phenotype_category = ?
      AND d.phenotype_direction = ?
    """

    params = [
        args.disease,
        args.context,
        args.phenotype,
        args.direction,
    ]

    if args.biotype:
        sql += " AND n.biotype = ?"
        params.append(args.biotype)

    sql += """
    ORDER BY
      t.translational_score DESC,
      d.evidence_score DESC,
      n.symbol ASC
    """

    rows = cur.execute(sql, tuple(params)).fetchall()

    if args.dedup_symbol:
        seen = set()
        deduped = []
        for r in rows:
            sym = r["symbol"]
            if sym not in seen:
                deduped.append(r)
                seen.add(sym)
        rows = deduped

    rows = rows[: args.limit]

    if not rows:
        print(
            f"No downstream matches for disease={args.disease}, "
            f"context={args.context}, phenotype={args.phenotype}, "
            f"direction={args.direction}"
            + (f", biotype={args.biotype}" if args.biotype else "")
        )
        conn.close()
        return

    print(
        "\n── Downstream phenotype report ──\n"
        f"Disease: {args.disease} | Context: {args.context} | "
        f"Phenotype: {args.phenotype} ({args.direction})"
        + (f" | Biotype: {args.biotype}" if args.biotype else "")
        + ("\nMode: deduplicated by symbol" if args.dedup_symbol else "")
        + "\n"
    )

    header = [
        "symbol",
        "biotype",
        "trans_score",
        "relevance",
        "specificity",
        "mechanism",
        "human_evid",
        "risk",
        "tier",
        "target_gene",
        "effect_dir",
        "effect_type",
        "evid_type",
        "evid_source",
        "evid_score",
        "pubmed_id",
        "dataset_id",
        "note_snippet",
    ]
    print("\t".join(header))

    output_rows = []
    for r in rows:
        note_snip = shorten(r["notes"] or "", width=80, placeholder="…")
        line = [
            r["symbol"] or "",
            r["biotype"] or "",
            fmt(r["translational_score"], 4),
            fmt(r["relevance_score"], 4),
            fmt(r["specificity_score"], 4),
            fmt(r["mechanism_score"], 4),
            fmt(r["human_evidence_score"], 4),
            fmt(r["risk_score"], 4),
            r["confidence_tier"] or "",
            r["target_gene"] or "",
            r["effect_direction"] or "",
            r["effect_type"] or "",
            r["evidence_type"] or "",
            r["evidence_source"] or "",
            fmt(r["evidence_score"], 2),
            r["pubmed_id"] or "",
            r["dataset_id"] or "",
            note_snip,
        ]
        output_rows.append(line)
        print("\t".join(line))

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(output_rows)
        print(f"\n✅ Wrote CSV report to {out_path}")

    conn.close()


if __name__ == "__main__":
    main()