"""
Inspect a single ncRNA target: scores + downstream effects.

Example usage:

  python -m scripts.inspect_target --symbol HNF1A-AS1

  python -m scripts.inspect_target --symbol HULC

If multiple ncrna_ids share the same symbol, you can disambiguate with:

  python -m scripts.inspect_target --symbol HNF1A-AS1 --ncrna-id LNCRNA_001
"""

import argparse
import sqlite3
from pathlib import Path
from textwrap import shorten


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "ncrna_platform.db"


def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect scores and downstream effects for a single ncRNA."
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite DB")
    p.add_argument(
        "--symbol",
        required=True,
        help="ncRNA symbol (e.g., HNF1A-AS1, HULC, MALAT1)",
    )
    p.add_argument(
        "--ncrna-id",
        default=None,
        help="Optional ncrna_id to disambiguate if symbol is not unique",
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

    # 1. Resolve symbol → ncrna_id(s)
    nm_rows = cur.execute(
        """
        SELECT ncrna_id, symbol, biotype, chrom, start_pos, end_pos
        FROM ncrna_master
        WHERE UPPER(symbol) = UPPER(?)
        """,
        (args.symbol,),
    ).fetchall()

    if not nm_rows:
        print(f"No entries in ncrna_master for symbol='{args.symbol}'")
        conn.close()
        return

    if args.ncrna_id:
        nm_rows = [r for r in nm_rows if r["ncrna_id"] == args.ncrna_id]
        if not nm_rows:
            print(
                f"Symbol '{args.symbol}' found, but no match for ncrna_id='{args.ncrna_id}'"
            )
            conn.close()
            return

    if len(nm_rows) > 1 and not args.ncrna_id:
        print("Multiple ncrna_ids found for this symbol:")
        for r in nm_rows:
            print(
                f"  {r['ncrna_id']} | {r['symbol']} | {r['biotype']} "
                f"| {r['chrom']}:{r['start_pos']}-{r['end_pos']}"
            )
        print("\nRe-run with --ncrna-id to inspect a specific transcript.")
        conn.close()
        return

    master = nm_rows[0]
    ncrna_id = master["ncrna_id"]

    print("── Target overview ──")
    print(
        f"Symbol: {master['symbol']}  |  ncrna_id: {ncrna_id}  |  "
        f"Biotype: {master['biotype']}"
    )
    print(
        f"Genomic: {master['chrom']}:{master['start_pos']}-{master['end_pos']}\n"
    )

    # 2. Show all target_scores rows for this ncrna_id
    score_rows = cur.execute(
        """
        SELECT
          disease_id,
          context_id,
          translational_score,
          relevance_score,
          specificity_score,
          mechanism_score,
          human_evidence_score,
          risk_score,
          tractability_score,
          confidence_tier
        FROM target_scores
        WHERE ncrna_id = ?
        ORDER BY disease_id, context_id
        """,
        (ncrna_id,),
    ).fetchall()

    print("── Target scores across disease/context ──")
    if not score_rows:
        print("No entries in target_scores for this ncrna_id.\n")
    else:
        header = [
            "disease",
            "context",
            "transl",
            "relevance",
            "specificity",
            "mechanism",
            "human_evid",
            "risk",
            "tract",
            "tier",
        ]
        print("\t".join(header))
        for r in score_rows:
            line = [
                r["disease_id"],
                r["context_id"],
                fmt(r["translational_score"], 4),
                fmt(r["relevance_score"], 4),
                fmt(r["specificity_score"], 4),
                fmt(r["mechanism_score"], 4),
                fmt(r["human_evidence_score"], 4),
                fmt(r["risk_score"], 4),
                fmt(r["tractability_score"], 4),
                r["confidence_tier"] or "",
            ]
            print("\t".join(line))
        print()

    # 3. Show downstream_effects with explicit interpretation
    eff_rows = cur.execute(
        """
        SELECT
          effect_id,
          phenotype_category,
          phenotype_direction,
          target_gene,
          effect_direction,
          effect_type,
          evidence_type,
          evidence_source,
          evidence_score,
          pubmed_id,
          dataset_id,
          notes
        FROM downstream_effects
        WHERE ncrna_id = ?
        ORDER BY phenotype_category, phenotype_direction, target_gene
        """,
        (ncrna_id,),
    ).fetchall()

    print("── Downstream effects (ncRNA perturbation → gene/phenotype) ──")
    if not eff_rows:
        print("No downstream_effects entries for this ncrna_id.\n")
        conn.close()
        return

    header = [
        "phenotype",
        "phenotype_dir",
        "target_gene",
        "gene_effect_dir",
        "effect_type",
        "evid_type",
        "evid_source",
        "evid_score",
        "pubmed_id",
        "dataset_id",
        "ncRNA_modulation",
        "ncRNA_modulation_effect_on_phenotype",
        "note_snippet",
    ]
    print("\t".join(header))

    for r in eff_rows:
        # Interpretation of ncRNA modulation:
        # - effect_direction == "down" typically comes from knockdown/inhibition of ncRNA
        # - effect_direction == "up" typically comes from overexpression of ncRNA
        # This is based on how you seeded the table (ASO/siRNA vs OE).
        gene_dir = (r["effect_direction"] or "").lower()
        phen_dir = (r["phenotype_direction"] or "").lower()

        if gene_dir == "down":
            ncrna_modulation = "ncRNA downregulation (e.g., ASO/siRNA)"
        elif gene_dir == "up":
            ncrna_modulation = "ncRNA upregulation/overexpression"
        else:
            ncrna_modulation = "ncRNA modulation (direction ambiguous)"

        if phen_dir == "improves":
            phen_effect = "ncRNA modulation improves phenotype"
        elif phen_dir == "worsens":
            phen_effect = "ncRNA modulation worsens phenotype"
        else:
            phen_effect = f"ncRNA modulation: phenotype_direction={phen_dir or 'NA'}"

        note_snip = shorten(r["notes"] or "", width=80, placeholder="…")

        line = [
            r["phenotype_category"] or "",
            r["phenotype_direction"] or "",
            r["target_gene"] or "",
            r["effect_direction"] or "",
            r["effect_type"] or "",
            r["evidence_type"] or "",
            r["evidence_source"] or "",
            fmt(r["evidence_score"], 2),
            r["pubmed_id"] or "",
            r["dataset_id"] or "",
            ncrna_modulation,
            phen_effect,
            note_snip,
        ]
        print("\t".join(line))

    print()
    conn.close()


if __name__ == "__main__":
    main()