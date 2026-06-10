from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "ncrna_platform.db"
APP_PATH = PROJECT_ROOT / "app" / "dashboard.py"

REQUIRED_TABLES = {
    "target_scores",
    "ncrna_master",
    "curated_targets",
}

TARGET_SCORE_REQUIRED_COLS = {
    "ncrna_id",
    "disease_id",
    "context_id",
    "confidence_tier",
    "translational_score",
    "relevance_score",
    "mechanism_score",
    "human_evidence_score",
}

TCGA_EXPECTED_COLS = {
    "expr_mean_tcga_pancan",
    "expr_median_tcga_pancan",
    "expr_prevalence_tcga_pancan",
    "in_tcga_pancan_expr",
}

CURATED_EXPECTED_COLS = {
    "symbol",
    "disease_stage",
    "evidence_tier",
    "direction",
    "mechanism_note",
    "source_ref",
}

def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)

def warn(msg):
    print(f"WARN: {msg}")

def ok(msg):
    print(f"OK: {msg}")

def get_columns(conn, table_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}

def main():
    print("=== ncRNA Platform Health Check ===")

    if not APP_PATH.exists():
        fail(f"dashboard.py not found at {APP_PATH}")
    ok(f"App file found: {APP_PATH}")

    text = APP_PATH.read_text(encoding="utf-8", errors="ignore")

    if "<br>" in text or "&lt;br&gt;" in text:
        fail("dashboard.py appears corrupted with HTML break tags")

    if 'if __name__ == "__main__":' not in text:
        fail('dashboard.py missing main entrypoint')

    if "st.set_page_config" not in text:
        warn("dashboard.py missing st.set_page_config")

    if "TCGA_COLS" not in text:
        warn("dashboard.py missing TCGA_COLS definition")

    ok("dashboard.py basic structure looks valid")

    if not DB_PATH.exists():
        fail(f"Database not found: {DB_PATH}")
    ok(f"Database found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    missing_tables = REQUIRED_TABLES - tables
    if missing_tables:
        fail(f"Missing required tables: {sorted(missing_tables)}")
    ok(f"Required tables present: {sorted(REQUIRED_TABLES)}")

    target_cols = get_columns(conn, "target_scores")
    missing_target_cols = TARGET_SCORE_REQUIRED_COLS - target_cols
    if missing_target_cols:
        fail(f"target_scores missing columns: {sorted(missing_target_cols)}")
    ok("target_scores required columns present")

    missing_tcga_cols = TCGA_EXPECTED_COLS - target_cols
    if missing_tcga_cols:
        warn(f"target_scores missing TCGA columns: {sorted(missing_tcga_cols)}")
    else:
        ok("target_scores TCGA columns present")

    curated_cols = get_columns(conn, "curated_targets")
    missing_curated_cols = CURATED_EXPECTED_COLS - curated_cols
    if missing_curated_cols:
        warn(f"curated_targets missing columns: {sorted(missing_curated_cols)}")
    else:
        ok("curated_targets expected columns present")

    n_target_scores = conn.execute("SELECT COUNT(*) FROM target_scores").fetchone()[0]
    n_curated = conn.execute("SELECT COUNT(*) FROM curated_targets").fetchone()[0]
    n_master = conn.execute("SELECT COUNT(*) FROM ncrna_master").fetchone()[0]

    if n_target_scores == 0:
        fail("target_scores is empty")
    ok(f"target_scores rows: {n_target_scores}")

    if n_master == 0:
        fail("ncrna_master is empty")
    ok(f"ncrna_master rows: {n_master}")

    if n_curated == 0:
        warn("curated_targets is empty")
    else:
        ok(f"curated_targets rows: {n_curated}")

    disease_context_counts = conn.execute("""
        SELECT COUNT(DISTINCT disease_id), COUNT(DISTINCT context_id)
        FROM target_scores
    """).fetchone()
    ok(f"Distinct disease IDs: {disease_context_counts[0]}, context IDs: {disease_context_counts[1]}")

    tcga_nonnull = 0
    if "expr_mean_tcga_pancan" in target_cols:
        tcga_nonnull = conn.execute("""
            SELECT COUNT(*)
            FROM target_scores
            WHERE expr_mean_tcga_pancan IS NOT NULL
        """).fetchone()[0]

        if tcga_nonnull == 0:
            warn("TCGA column exists but all expr_mean_tcga_pancan values are NULL")
        else:
            ok(f"Rows with TCGA mean values: {tcga_nonnull}")

    curated_linked = 0
    if "symbol" in curated_cols:
        curated_linked = conn.execute("""
            SELECT COUNT(*)
            FROM curated_targets
            WHERE symbol IS NOT NULL AND TRIM(symbol) != ''
        """).fetchone()[0]
        ok(f"Curated rows with symbol values: {curated_linked}")

    conn.close()
    print("=== Health check complete ===")

if __name__ == "__main__":
    main()