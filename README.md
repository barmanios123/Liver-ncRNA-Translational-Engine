# Liver ncRNA Translational Engine

A translational prioritization platform for liver noncoding RNA targets in MASLD/MASH, fibrosis, and HCC contexts.

This project combines curated evidence, perturbation support, disease-context expression features, and a Geneformer-like scoring framework to rank ncRNA targets and present them through an interactive Streamlit dashboard. The goal is to move beyond static target lists and provide a more interpretable, evidence-aware prioritization workflow for translational target discovery.

## Project Overview

The Liver ncRNA Translational Engine is designed to help identify and compare liver-relevant ncRNA targets by integrating:

- curated liver disease evidence,
- perturbation and downstream mechanism data,
- translational scoring components,
- disease and context specificity features,
- Geneformer-like component scoring for rank refinement.

The dashboard is built to support both exploratory target discovery and presentation-ready target dossiers, with clear traceability between ranking, evidence, and mechanism interpretation.

## What the Dashboard Does

The Streamlit dashboard provides an interactive interface for reviewing ranked ncRNA targets in a selected liver disease and tissue or cell context.

Core dashboard capabilities include:

- compact ranked target table,
- baseline vs Geneformer-like rank comparison,
- delta-rank traceability,
- shortlist comparison across selected targets,
- focused target dossiers,
- evidence-stratified summaries,
- downstream mechanism interpretation,
- raw provenance and methodology inspection.

## What’s New in the Updated Dashboard

The current dashboard version introduces a more presentation-ready and biologically interpretable target review workflow.

### Evidence stratification

Target dossiers now separate:

- literature-backed evidence,
- quantitative support,
- model-integrated or cohort-level hypotheses,
- supported mechanism statements,
- unspecified evidence statements.

This prevents paper-backed findings from being mixed with model-derived or observational hypotheses.

### Mechanism interpretation

Downstream mechanism summaries now use explicit evidence typing to distinguish:

- direct perturbation or supported mechanistic evidence,
- inference-level or cohort-derived support,
- rows with missing or unclear evidence class.

Supported perturbation modalities such as **ASO**, **siRNA**, and **CRISPRi** are treated as intervention-level evidence classes.

### Improved traceability

The dashboard now makes it easier to understand why a target ranks highly by exposing:

- baseline rank,
- Geneformer-like rank,
- delta rank,
- component score table,
- raw feature provenance,
- model comparison summary.

### Cleaner presentation flow

The dossier layout is streamlined for review with collaborators, hiring managers, or scientific stakeholders. Evidence categories are easier to explain, duplicate mechanism rows are collapsed before rendering, and quantitative score support is shown separately from paper-backed statements.

## Example Use Case

For a target such as **HNF1A-AS1** in the MASLD/MASH liver context, the dashboard can show:

- overall target rank,
- confidence tier and curated tier,
- literature-backed statements with PMIDs,
- quantitative support such as differential expression magnitude or clinical correlation,
- supported downstream mechanisms tied to specific perturbation modalities,
- curated contradictions,
- expression-context availability,
- interpretable ranking components.

This allows a reviewer to quickly assess whether a target is promising, evidence-backed, and experimentally actionable.

## Ranking Logic

The platform exposes two ranking views:

- **Baseline translational score**
- **Geneformer-like score**

The baseline score reflects the broader translational prioritization framework. The Geneformer-like score adds an interpretable component structure that emphasizes signals such as:

- regulatory centrality,
- perturbation impact,
- disease shift,
- context support,
- risk adjustment.

The dashboard also shows **delta rank**, defined as:

- baseline rank minus Geneformer-like rank.

A positive delta indicates that a target rises under the Geneformer-like prioritization framework.

## Model Interpretation

The dashboard includes a saved run comparison summary that helps explain model behavior.

In the latest comparison, the project distinguishes between:

- an earlier baseline run with `confidence_tier` present in model features,
- a newer ablation run that excludes `confidence_tier` to reduce label leakage.

This improves interpretability by shifting model importance toward more biologically meaningful signals such as disease shift, perturbation impact, expression change, and relevance or specificity features.

## Repository Structure

```text
ncrna_platform/
├── app/                    # Streamlit dashboard application
├── configs/                # Configuration files
├── data/                   # Source and intermediate data assets
├── db/                     # SQLite database(s), seed files, and DB utilities
├── etl/                    # ETL pipelines for feature assembly and ingestion
├── misc/                   # Temporary or scratch utilities
├── models/                 # Feature engineering and model code
├── notebooks/              # Exploratory and development notebooks
├── output/                 # Current model outputs and benchmark artifacts
├── outputs/                # Prior output artifacts and saved comparisons
├── scripts/                # Utility and reporting scripts
├── LICENSE
├── README.md
├── requirements.txt
└── ncrna_platform.db       # Legacy root-level DB file, if still present
```

## Key Data and Evidence Layers

The platform can incorporate multiple categories of evidence and feature provenance, including:

- curated target annotations,
- literature evidence,
- downstream effects,
- perturbation studies,
- disease-shift features,
- TCGA pan-cancer expression context,
- risk flags and contradiction indicators.

These layers are surfaced in the dashboard through evidence sections, component tables, and raw provenance views.

## Database Notes

The main SQLite database should live at:

```text
db/ncrna_platform.db
```

Backup database files may also exist in `db/`, for example development snapshots created before targeted fixes. These backup files are not intended for normal dashboard runs and should not be used as the default application database.

If scripts previously referenced `ncrna_platform.db` from the repository root, they should be updated to use the `db/` path instead.

## Running the App Locally

Install dependencies and launch the Streamlit dashboard:

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

If the local SQLite database is missing or incomplete, regenerate it through the ETL and scoring pipeline before launching the app.

By default, the dashboard should point to `db/ncrna_platform.db` rather than a root-level database file.

## Recommended Demo Flow

For a quick walkthrough:

1. Open the dashboard.
2. Filter to the MASLD/MASH liver context.
3. Sort by Geneformer-like score.
4. Review the top-ranked targets.
5. Compare a short list of candidates.
6. Open the target dossier for a top hit such as HNF1A-AS1.
7. Walk through:
   - evidence sections,
   - component score table,
   - downstream mechanism interpretation,
   - methodology and provenance view.

This sequence works well for scientific review, stakeholder demos, and interview presentations.

## Future Improvements

Potential next steps for the platform include:

- broader multi-disease and multi-tissue ncRNA prioritization,
- stronger deployment synchronization between local artifacts and hosted app state,
- richer perturbation metadata and causal evidence grading,
- interactive visualization of target-gene-phenotype relationships,
- versioned model and evidence snapshots for reproducible dashboard states.

## License

See the `LICENSE` file for licensing information.