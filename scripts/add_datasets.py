from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


DATASETS = [
    # Uncomment this only if the original raw files are actually present.
    # {
    #     "dataset_id": "GSE126848",
    #     "description": "Original MASLD/MASH liver bulk RNA-seq cohort",
    #     "counts_path": DATA_DIR / "GSE126848_Gene_counts_raw.txt.gz",
    #     "metadata_path": DATA_DIR / "GSE126848_series_matrix.txt.gz",
    #     "symbol_map_path": DATA_DIR / "ensembl_to_symbol.csv",
    #     "platform": "bulk_rnaseq",
    #     "disease_id": "DIS_001",
    #     "context_id": "CTX_001",
    # },

    # Fill these in once you download the real files into ./data
    {
        "dataset_id": "GSE130970",
        "description": "MASLD/MASH liver bulk RNA-seq cohort",
        "counts_path": DATA_DIR / "GSE130970_counts_raw.txt.gz",
        "metadata_path": DATA_DIR / "GSE130970_series_matrix.txt.gz",
        "symbol_map_path": DATA_DIR / "ensembl_to_symbol.csv",
        "platform": "bulk_rnaseq",
        "disease_id": "DIS_002",
        "context_id": "CTX_001",
    },
    {
        "dataset_id": "GSE135251",
        "description": "Additional liver bulk RNA-seq cohort",
        "counts_path": DATA_DIR / "GSE135251_counts_raw.txt.gz",
        "metadata_path": DATA_DIR / "GSE135251_series_matrix.txt.gz",
        "symbol_map_path": DATA_DIR / "ensembl_to_symbol.csv",
        "platform": "bulk_rnaseq",
        "disease_id": "DIS_001",
        "context_id": "CTX_001",
    },
]


def validate_dataset(ds: dict) -> list[str]:
    errors: list[str] = []
    required = ["dataset_id", "counts_path", "metadata_path", "symbol_map_path"]

    for key in required:
        if key not in ds:
            errors.append(f"Missing required key: {key}")

    for path_key in ["counts_path", "metadata_path", "symbol_map_path"]:
        if path_key in ds:
            path = Path(ds[path_key])
            if not path.exists():
                errors.append(f"{path_key} not found for {ds['dataset_id']}: {path}")

    return errors


def run_ingestion(ds: dict) -> None:
    cmd = [
        sys.executable,
        "-m",
        "etl.ingest_bulk",
        "--dataset-id",
        ds["dataset_id"],
        "--counts",
        str(ds["counts_path"]),
        "--metadata",
        str(ds["metadata_path"]),
        "--symbol-map",
        str(ds["symbol_map_path"]),
        "--platform",
        ds.get("platform", "bulk_rnaseq"),
        "--disease-id",
        ds.get("disease_id", "DIS_001"),
        "--context-id",
        ds.get("context_id", "CTX_001"),
    ]

    print(f"\n=== Ingesting {ds['dataset_id']} ===")
    if ds.get("description"):
        print(ds["description"])
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    if not DATASETS:
        print("No datasets configured in DATASETS.")
        print("Edit scripts/add_datasets.py and add at least one dataset entry.")
        return

    ingested = 0
    skipped = 0

    for ds in DATASETS:
        print(f"\n--- Checking {ds['dataset_id']} ---")
        errors = validate_dataset(ds)

        if errors:
            print(f"Skipping {ds['dataset_id']} because of missing/invalid inputs:")
            for err in errors:
                print(f"  - {err}")
            skipped += 1
            continue

        try:
            run_ingestion(ds)
            ingested += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Ingestion failed for {ds['dataset_id']}: {e}")
            skipped += 1

    print("\nSummary:")
    print(f"  Ingested: {ingested}")
    print(f"  Skipped:  {skipped}")

    print("\nNext run:")
    print("  python -m scripts.build_expression_features")
    print("  python -m models.scoring")
    print("  python -m models.train")


if __name__ == "__main__":
    main()