# Liver ncRNA Translational Engine

A translational prioritization platform for liver noncoding RNA targets in MASLD/MASH, fibrosis, and HCC.

It combines curated literature evidence, perturbation data, disease-context expression from public GEO cohorts, and a Geneformer-like component score to rank ncRNA targets, then presents them in an interactive Streamlit dashboard with full evidence traceability.

## Why

Static target lists hide *why* a target ranks where it does. This engine keeps every rank tied to its evidence, so a reviewer can tell paper-backed findings apart from model-derived or cohort-level hypotheses and judge whether a target is experimentally actionable.

## Features

- **Ranked target table** for a chosen disease and tissue/cell context
- **Two ranking views**: a baseline translational score and a Geneformer-like score, with delta rank between them
- **Target dossiers** that separate literature-backed evidence (with PMIDs), quantitative support, model-integrated hypotheses, and curated contradictions
- **Mechanism interpretation** with explicit evidence typing; ASO, siRNA, and CRISPRi perturbations are treated as intervention-level evidence
- **Shortlist comparison** across selected targets
- **Provenance and methodology views** for every feature and score component

## Data

Expression and clinical context come from public cohorts, including:

| Dataset | Description |
|---|---|
| GSE135251 | MASLD biopsy cohort (HTSeq counts) |
| GSE130970 | NAFLD/NASH cohort |
| GSE126848 | NAFLD/NASH bulk RNA-seq |
| GSE193066 | NAFLD cohort |
| TCGA Pan-Cancer | Expression context for HCC relevance |

Ingestion scripts in `etl/` load each cohort into a single SQLite database, `ncrna_platform.db`.

## Ranking logic

The **baseline translational score** reflects the overall prioritization framework. The **Geneformer-like score** adds an interpretable component structure:

- regulatory centrality
- perturbation impact
- disease shift
- context support
- risk adjustment

**Delta rank** = baseline rank − Geneformer-like rank. A positive delta means the target rises under the component model.

An ablation run that drops `confidence_tier` from the model features (to reduce label leakage) shifts importance toward biologically meaningful signals such as disease shift, perturbation impact, and expression change. The comparison is saved in `output/comparisons/`.

## Example: HNF1A-AS1

For HNF1A-AS1 in the MASLD/MASH liver context, the dossier shows the overall rank, confidence and curated tiers, literature statements with PMIDs, differential expression and clinical correlation, downstream mechanisms tied to specific perturbation modalities, curated contradictions, and the component scores behind the rank.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The dashboard reads `ncrna_platform.db` from the repository root, which is included so the app runs out of the box. To rebuild or extend it, run the cohort ingestion scripts in `etl/` (for example `python etl/ingest_GSE135251.py`).

## Repository structure

```text
app/        Streamlit dashboard
configs/    Dataset and feature configuration
data/       Source and processed cohort data
db/         Schema, loaders, and literature seed
etl/        Ingestion pipelines, one per cohort
models/     Feature engineering, scoring, knowledge graph, training
output/     Model outputs, benchmarks, and run comparisons
outputs/    Curated target summary tables
scripts/    Utilities for validation, reporting, and dataset prep
```

## Roadmap

- Multi-disease and multi-tissue prioritization
- Richer perturbation metadata and causal evidence grading
- Interactive target–gene–phenotype graphs
- Versioned model and evidence snapshots for reproducible dashboard states

## License

MIT. See [LICENSE](LICENSE).
