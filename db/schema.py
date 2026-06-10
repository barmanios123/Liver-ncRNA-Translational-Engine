"""
ncRNA Target Intelligence Platform — Data Schema & Seed Data
Phase 1 Wedge: Liver MASLD/MASH → Fibrosis/HCC
"""

import sqlite3
import json
import numpy as np
import random

random.seed(42)
np.random.seed(42)


def create_database(db_path="ncrna_platform.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS ncrna_master (
        ncrna_id            TEXT PRIMARY KEY,
        symbol              TEXT,
        aliases             TEXT,
        biotype             TEXT,
        chrom               TEXT,
        start_pos           INTEGER,
        end_pos             INTEGER,
        strand              TEXT,
        transcript_count    INTEGER,
        conservation_score  REAL,
        ensembl_id          TEXT,
        added_date          TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS disease_context (
        disease_id      TEXT PRIMARY KEY,
        disease_name    TEXT NOT NULL,
        doid            TEXT,
        mondo_id        TEXT,
        mesh_id         TEXT,
        stage           TEXT,
        subtype         TEXT,
        species         TEXT DEFAULT "Homo sapiens",
        cohort_size     INTEGER,
        data_source     TEXT,
        added_date      TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS tissue_cell_context (
        context_id      TEXT PRIMARY KEY,
        tissue          TEXT NOT NULL,
        cell_type       TEXT,
        cell_state      TEXT,
        platform        TEXT,
        atlas_source    TEXT,
        sample_count    INTEGER,
        added_date      TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS expression_evidence (
        evidence_id         TEXT PRIMARY KEY,
        ncrna_id            TEXT REFERENCES ncrna_master(ncrna_id),
        disease_id          TEXT REFERENCES disease_context(disease_id),
        context_id          TEXT REFERENCES tissue_cell_context(context_id),
        log2fc              REAL,
        pvalue              REAL,
        padj                REAL,
        basemean            REAL,
        tpm_disease         REAL,
        tpm_healthy         REAL,
        specificity_tau     REAL,
        direction           TEXT,
        dataset_id          TEXT,
        added_date          TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS pathway_links (
        link_id             TEXT PRIMARY KEY,
        ncrna_id            TEXT REFERENCES ncrna_master(ncrna_id),
        pathway_name        TEXT,
        pathway_db          TEXT,
        pathway_id          TEXT,
        enrichment_pval     REAL,
        enrichment_fdr      REAL,
        coexp_module        TEXT,
        hub_gene            TEXT,
        pearson_r           REAL,
        added_date          TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS perturbation_evidence (
        pert_id             TEXT PRIMARY KEY,
        ncrna_id            TEXT REFERENCES ncrna_master(ncrna_id),
        perturbation_type   TEXT,
        target_gene_effect  TEXT,
        effect_size         REAL,
        phenotype           TEXT,
        cell_model          TEXT,
        species             TEXT DEFAULT "Homo sapiens",
        pubmed_id           TEXT,
        confidence          TEXT,
        added_date          TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS clinical_links (
        clin_id             TEXT PRIMARY KEY,
        ncrna_id            TEXT REFERENCES ncrna_master(ncrna_id),
        disease_id          TEXT REFERENCES disease_context(disease_id),
        clinical_metric     TEXT,
        correlation_r       REAL,
        pvalue              REAL,
        biomarker_type      TEXT,
        sample_type         TEXT,
        cohort_id           TEXT,
        pubmed_id           TEXT,
        added_date          TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS tractability_features (
        tract_id                    TEXT PRIMARY KEY,
        ncrna_id                    TEXT REFERENCES ncrna_master(ncrna_id),
        localization                TEXT,
        isoform_count               INTEGER,
        gc_content                  REAL,
        secondary_structure_score   REAL,
        aso_accessible              INTEGER,
        sirna_compatible            INTEGER,
        small_mol_bindable          INTEGER,
        crispr_feasible             INTEGER,
        best_modality               TEXT,
        added_date                  TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS binding_features (
        binding_id                  TEXT PRIMARY KEY,
        ncrna_id                    TEXT REFERENCES ncrna_master(ncrna_id),
        n_high_conf_rbps            INTEGER,
        max_rbp_score               REAL,
        mean_top5_rbp_score         REAL,
        liver_rbp_score             REAL,
        max_peak_score              REAL,
        peak_density                REAL,
        n_binding_hotspots          INTEGER,
        hotspot_5prime_fraction     REAL,
        hotspot_3prime_fraction     REAL,
        binding_mechanism_score     REAL,
        binding_targetability_score REAL,
        binding_risk_score          REAL,
        parpi_model_version         TEXT,
        ideepb_model_version        TEXT,
        last_updated                TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS literature_evidence (
        lit_id              TEXT PRIMARY KEY,
        ncrna_id            TEXT REFERENCES ncrna_master(ncrna_id),
        disease_id          TEXT REFERENCES disease_context(disease_id),
        statement           TEXT,
        direction           TEXT,
        confidence_score    REAL,
        pubmed_id           TEXT,
        journal             TEXT,
        year                INTEGER,
        is_contradictory    INTEGER DEFAULT 0,
        added_date          TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS downstream_effects (
        effect_id           TEXT PRIMARY KEY,
        ncrna_id            TEXT REFERENCES ncrna_master(ncrna_id),
        target_gene         TEXT,
        target_ensembl_id   TEXT,
        effect_direction    TEXT,
        effect_type         TEXT,
        phenotype_category  TEXT,
        phenotype_direction TEXT,
        evidence_type       TEXT,
        evidence_source     TEXT,
        evidence_score      REAL,
        pubmed_id           TEXT,
        dataset_id          TEXT,
        notes               TEXT,
        added_date          TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS target_scores (
        score_id                    TEXT PRIMARY KEY,
        ncrna_id                    TEXT REFERENCES ncrna_master(ncrna_id),
        disease_id                  TEXT REFERENCES disease_context(disease_id),
        context_id                  TEXT REFERENCES tissue_cell_context(context_id),
        relevance_score             REAL,
        specificity_score           REAL,
        mechanism_score             REAL,
        tractability_score          REAL,
        human_evidence_score        REAL,
        risk_score                  REAL,
        translational_score         REAL,
        confidence_tier             TEXT,
        top_evidence                TEXT,
        risk_flags                  TEXT,
        recommended_experiments     TEXT,
        model_version               TEXT DEFAULT "v1.0",
        scored_date                 TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    print("✅ Schema created:", db_path)
    return conn


def seed_liver_masld_data(conn):
    c = conn.cursor()

    ncrnas = [
        ("LNCRNA_001", "HNF1A-AS1", json.dumps(["HNF1A-OS1", "HNF1AOS"]), "lncRNA", "chr12", 121416552, 121428540, "+", 3, 0.82, "ENSG00000197253"),
        ("LNCRNA_002", "NEAT1", json.dumps(["MEN epsilon/beta"]), "lncRNA", "chr11", 65422774, 65445540, "+", 2, 0.91, "ENSG00000245532"),
        ("LNCRNA_003", "MALAT1", json.dumps(["NEAT2", "MALAT-1"]), "lncRNA", "chr11", 65497686, 65506516, "+", 1, 0.95, "ENSG00000251562"),
        ("LNCRNA_004", "SNHG12", json.dumps(["NCRNA00040"]), "lncRNA", "chr1", 192647882, 192650956, "+", 4, 0.71, "ENSG00000099834"),
        ("LNCRNA_005", "HULC", json.dumps(["LINC00078"]), "lncRNA", "chr6", 74432849, 74436026, "+", 2, 0.55, "ENSG00000251164"),
        ("LNCRNA_006", "HOTAIR", json.dumps(["HOXC11AS2"]), "lncRNA", "chr12", 54356593, 54368819, "-", 3, 0.88, "ENSG00000228630"),
        ("LNCRNA_007", "GAS5", json.dumps(["SNHG2"]), "lncRNA", "chr1", 173860855, 173865006, "-", 6, 0.77, "ENSG00000234741"),
        ("LNCRNA_008", "LINC01116", json.dumps([]), "lncRNA", "chr2", 96524110, 96535200, "+", 2, 0.43, "ENSG00000230876"),
        ("LNCRNA_009", "MIAT", json.dumps(["GOMAFU"]), "lncRNA", "chr22", 26669592, 26797219, "+", 5, 0.68, "ENSG00000225783"),
        ("LNCRNA_010", "XIST", json.dumps(["LINC00023"]), "lncRNA", "chrX", 73820651, 73852723, "+", 8, 0.93, "ENSG00000229807"),
        ("LNCRNA_011", "HNF4A-AS1", json.dumps(["HNF4A-AS", "HNF4A antisense 1"]), "lncRNA", "chr20", 42980000, 43020000, "+", 2, 0.81, "ENSG000002XXXX1"),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO ncrna_master VALUES (?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        ncrnas
    )

    diseases = [
        ("DIS_001", "MASLD", "DOID:0080208", "MONDO:0013209", "D065626", "steatosis", "NAS 1-3", "Homo sapiens", 420, "GSE48452"),
        ("DIS_002", "MASH", "DOID:0080208", "MONDO:0013209", "D065626", "steatohepatitis", "NAS 4-6", "Homo sapiens", 210, "GSE130970"),
        ("DIS_003", "Liver Fibrosis", "DOID:5082", "MONDO:0016264", "D008103", "fibrosis", "F2-F4", "Homo sapiens", 180, "GSE84044"),
        ("DIS_004", "HCC", "DOID:684", "MONDO:0007256", "D006528", "advanced", "BCLC B-C", "Homo sapiens", 350, "TCGA-LIHC"),
        ("DIS_005", "MASLD-Progression", "DOID:0080208", "MONDO:0013209", "D065626", "progression", "longitudinal", "Homo sapiens", 95, "GSE163211"),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO disease_context VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        diseases
    )

    contexts = [
        ("CTX_001", "liver", "hepatocyte", "lipid-loaded", "bulk_rnaseq", "GTEx", 120),
        ("CTX_002", "liver", "hepatocyte", "healthy", "bulk_rnaseq", "GTEx", 200),
        ("CTX_003", "liver", "stellate_cell", "activated", "scRNAseq", "HLCA", 45),
        ("CTX_004", "liver", "kupffer_cell", "inflammatory", "scRNAseq", "HLCA", 38),
        ("CTX_005", "liver", "hepatocyte", "PHH_in_vitro", "bulk_rnaseq", "in-house", 24),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO tissue_cell_context VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        contexts
    )

    ncrna_ids = [n[0] for n in ncrnas]
    disease_ids = [d[0] for d in diseases]
    exp_rows, eid = [], 1

    for nid in ncrna_ids:
        for did in disease_ids:
            log2fc = float(np.random.normal(1.6 if nid in ["LNCRNA_001", "LNCRNA_005", "LNCRNA_006", "LNCRNA_011"] else -0.4, 0.9))
            pval = float(np.random.uniform(0.0001, 0.05))
            padj = pval * 10
            tpm_d = float(np.random.uniform(5, 200))
            tpm_h = tpm_d / (2 ** log2fc)
            tau = float(np.random.uniform(0.3, 0.95))
            direc = "up" if log2fc > 0.5 else ("down" if log2fc < -0.5 else "mixed")

            exp_rows.append((
                f"EXP_{eid:04d}",
                nid,
                did,
                "CTX_001",
                round(log2fc, 3),
                round(pval, 6),
                round(padj, 6),
                round((tpm_d + tpm_h) / 2, 2),
                round(tpm_d, 2),
                round(tpm_h, 2),
                round(tau, 3),
                direc,
                f"GSE{random.randint(10000, 99999)}"
            ))
            eid += 1

    c.executemany(
        "INSERT OR IGNORE INTO expression_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        exp_rows
    )

    tract_data = [
        ("TR_001", "LNCRNA_001", "cytoplasmic", 3, 0.58, -42.1, 1, 1, 0, 1, "ASO"),
        ("TR_002", "LNCRNA_002", "nuclear", 2, 0.61, -55.3, 1, 0, 0, 1, "ASO"),
        ("TR_003", "LNCRNA_003", "nuclear", 1, 0.55, -61.2, 1, 0, 0, 1, "ASO"),
        ("TR_004", "LNCRNA_004", "cytoplasmic", 4, 0.47, -38.7, 1, 1, 0, 1, "siRNA"),
        ("TR_005", "LNCRNA_005", "cytoplasmic", 2, 0.52, -29.4, 1, 1, 1, 1, "siRNA"),
        ("TR_006", "LNCRNA_006", "nuclear", 3, 0.62, -58.1, 1, 0, 0, 1, "ASO"),
        ("TR_007", "LNCRNA_007", "cytoplasmic", 6, 0.49, -33.6, 1, 1, 0, 1, "ASO"),
        ("TR_008", "LNCRNA_008", "nuclear", 2, 0.44, -22.1, 0, 0, 0, 0, "biomarker"),
        ("TR_009", "LNCRNA_009", "nuclear", 5, 0.59, -47.8, 1, 0, 0, 1, "ASO"),
        ("TR_010", "LNCRNA_010", "nuclear", 8, 0.67, -64.3, 1, 0, 0, 1, "ASO"),
        ("TR_011", "LNCRNA_011", "nuclear", 2, 0.57, -46.0, 1, 0, 0, 1, "ASO"),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO tractability_features VALUES (?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        tract_data
    )

    pert_data = [
        ("PERT_001", "LNCRNA_001", "ASO", "FASN", 2.1, "lipid_accumulation", "PHH", "Homo sapiens", "37234567", "high"),
        ("PERT_002", "LNCRNA_001", "siRNA", "SREBP1C", 1.8, "lipid_droplet", "HepG2", "Homo sapiens", "35123456", "medium"),
        ("PERT_003", "LNCRNA_002", "ASO", "TGF-b1", 1.5, "fibrosis_marker", "LX-2", "Homo sapiens", "38765432", "high"),
        ("PERT_004", "LNCRNA_005", "siRNA", "AKT1", 1.2, "cell_viability", "HepG2", "Homo sapiens", "36543210", "medium"),
        ("PERT_005", "LNCRNA_006", "CRISPRi", "EZH2", 2.4, "chromatin_compact", "HepG2", "Homo sapiens", "39876543", "high"),
        ("PERT_006", "LNCRNA_007", "ASO", "p21", -1.6, "cell_cycle_arrest", "PHH", "Homo sapiens", "34567890", "medium"),
        ("PERT_007", "LNCRNA_003", "ASO", "PCNA", 1.1, "proliferation", "HepG2", "Homo sapiens", "33456789", "low"),
        ("PERT_008", "LNCRNA_009", "siRNA", "SRSF1", 0.9, "RNA_splicing", "HepG2", "Homo sapiens", "32345678", "low"),
        ("PERT_009", "LNCRNA_011", "ASO", "HNF4A", 1.4, "hepatocyte_identity", "PHH", "Homo sapiens", "39999991", "medium"),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO perturbation_evidence VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        pert_data
    )

    clin_data = [
        ("CLIN_001", "LNCRNA_001", "DIS_001", "NAS_score", 0.71, 0.001, "prognostic", "tissue_biopsy", "COHORT_UK", "37234567"),
        ("CLIN_002", "LNCRNA_001", "DIS_003", "fibrosis_stage", 0.68, 0.002, "prognostic", "tissue_biopsy", "COHORT_EU", "37234568"),
        ("CLIN_003", "LNCRNA_002", "DIS_002", "ALT", 0.55, 0.01, "diagnostic", "serum", "COHORT_US", "38765432"),
        ("CLIN_004", "LNCRNA_005", "DIS_004", "OS_months", -0.61, 0.003, "prognostic", "tissue_biopsy", "TCGA_LIHC", "36543210"),
        ("CLIN_005", "LNCRNA_006", "DIS_004", "tumor_stage", 0.74, 0.001, "prognostic", "tissue_biopsy", "TCGA_LIHC", "39876543"),
        ("CLIN_006", "LNCRNA_007", "DIS_001", "insulin_resist", -0.48, 0.02, "predictive", "tissue_biopsy", "COHORT_DE", "34567890"),
        ("CLIN_007", "LNCRNA_011", "DIS_001", "steatosis_grade", 0.44, 0.03, "predictive", "tissue_biopsy", "COHORT_UK", "39999991"),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO clinical_links VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        clin_data
    )

    lit_data = [
        ("LIT_001", "LNCRNA_001", "DIS_001", "HNF1A-AS1 promotes hepatic lipid accumulation via FASN regulation", "promotes", 0.92, "37234567", "Hepatology", 2023, 0),
        ("LIT_002", "LNCRNA_001", "DIS_003", "HNF1A-AS1 knockdown reduces fibrosis markers in MASH", "suppresses", 0.87, "35123456", "J Hepatol", 2022, 0),
        ("LIT_003", "LNCRNA_002", "DIS_002", "NEAT1 activates stellate cells via TGF-b signaling", "promotes", 0.83, "38765432", "Gut", 2023, 0),
        ("LIT_004", "LNCRNA_003", "DIS_004", "MALAT1 overexpression correlates with HCC recurrence", "promotes", 0.79, "33456789", "Cancer Res", 2021, 0),
        ("LIT_005", "LNCRNA_005", "DIS_004", "HULC drives HCC growth through AKT/mTOR axis", "promotes", 0.85, "36543210", "Oncogene", 2022, 0),
        ("LIT_006", "LNCRNA_006", "DIS_004", "HOTAIR epigenetically silences tumor suppressors in HCC", "promotes", 0.91, "39876543", "Nat Commun", 2023, 0),
        ("LIT_007", "LNCRNA_007", "DIS_001", "GAS5 restrains hepatic glucose production", "suppresses", 0.76, "34567890", "Cell Metab", 2021, 0),
        ("LIT_008", "LNCRNA_008", "DIS_002", "LINC01116 expression unchanged in MASH cohort — contradictory finding", "mixed", 0.42, "38888888", "PLoS ONE", 2022, 1),
        ("LIT_009", "LNCRNA_011", "DIS_001", "HNF4A-AS1 is associated with hepatocyte differentiation and lipid metabolism programs", "mixed", 0.72, "39999991", "Hepatology Reports", 2024, 0),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO literature_evidence VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        lit_data
    )

    pathway_data = [
        ("PW_001", "LNCRNA_001", "Fatty acid biosynthesis", "KEGG", "hsa00061", 0.001, 0.005, "turquoise", "FASN", 0.82),
        ("PW_002", "LNCRNA_001", "PPAR signaling", "KEGG", "hsa03320", 0.003, 0.012, "turquoise", "PPARA", 0.74),
        ("PW_003", "LNCRNA_002", "TGF-beta signaling", "KEGG", "hsa04350", 0.001, 0.004, "blue", "TGFB1", 0.79),
        ("PW_004", "LNCRNA_003", "Spliceosome", "KEGG", "hsa03040", 0.002, 0.008, "brown", "SRSF1", 0.71),
        ("PW_005", "LNCRNA_005", "PI3K-Akt signaling", "KEGG", "hsa04151", 0.004, 0.015, "red", "AKT1", 0.68),
        ("PW_006", "LNCRNA_006", "Polycomb repression", "Reactome", "R-HSA-212300", 0.001, 0.003, "green", "EZH2", 0.88),
        ("PW_007", "LNCRNA_007", "p53 signaling", "KEGG", "hsa04115", 0.005, 0.018, "yellow", "TP53", 0.65),
        ("PW_008", "LNCRNA_011", "Hepatocyte differentiation", "Reactome", "R-HSA-9000001", 0.004, 0.017, "purple", "HNF4A", 0.73),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO pathway_links VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        pathway_data
    )

    conn.commit()
    print("✅ Seed data inserted")
    return conn


def seed_downstream_effects(conn):
    c = conn.cursor()

    seed_rows = [
        (
            "EFF_001", "LNCRNA_001", "FASN", None, "down", "expression",
            "steatosis", "improves", "ASO", "perturbation_dataset", 0.90,
            "37234567", "GSE130970",
            "HNF1A-AS1 ASO knockdown reduces FASN and hepatic lipid accumulation"
        ),
        (
            "EFF_002", "LNCRNA_001", "COL1A1", None, "down", "expression",
            "fibrosis", "improves", "siRNA", "perturbation_dataset", 0.80,
            "35123456", "GSE135251",
            "HNF1A-AS1 knockdown reduces fibrosis markers in MASH/fibrotic context"
        ),
        (
            "EFF_003", "LNCRNA_002", "TGFB1", None, "up", "expression",
            "fibrosis", "worsens", "ASO", "literature", 0.85,
            "38765432", None,
            "NEAT1 activates stellate cells via TGF-beta signaling"
        ),
        (
            "EFF_004", "LNCRNA_003", "PCNA", None, "up", "expression",
            "proliferation", "worsens", "ASO", "literature", 0.80,
            "33456789", "TCGA-LIHC",
            "MALAT1 overexpression associated with proliferation and HCC recurrence"
        ),
        (
            "EFF_005", "LNCRNA_005", "AKT1", None, "up", "expression",
            "proliferation", "worsens", "siRNA", "literature", 0.85,
            "36543210", "TCGA-LIHC",
            "HULC drives HCC growth via AKT/mTOR signaling"
        ),
        (
            "EFF_006", "LNCRNA_006", "EZH2", None, "up", "epigenetic",
            "tumor_suppression", "worsens", "CRISPRi", "literature", 0.90,
            "39876543", None,
            "HOTAIR recruits EZH2 and silences tumor suppressor programs in HCC"
        ),
        (
            "EFF_007", "LNCRNA_007", "G6PC", None, "down", "expression",
            "glucose", "improves", "ASO", "literature", 0.75,
            "34567890", None,
            "GAS5 restrains hepatic glucose production"
        ),
        (
            "EFF_008", "LNCRNA_011", "HNF4A", None, "up", "expression",
            "hepatocyte_identity", "improves", "ASO", "literature", 0.78,
            "39999991", None,
            "HNF4A-AS1 supports hepatocyte identity programs linked to metabolic homeostasis"
        ),
    ]

    c.executemany(
        """
        INSERT OR IGNORE INTO downstream_effects (
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
        seed_rows
    )

    conn.commit()
    print("✅ Downstream effects seeded")
    return conn


if __name__ == "__main__":
    conn = create_database("ncrna_platform.db")
    seed_liver_masld_data(conn)
    seed_downstream_effects(conn)
    conn.close()
    print("Database ready.")