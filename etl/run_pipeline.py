"""
ncRNA Target Intelligence Platform — One-command launcher.
Run: python run_platform.py
"""

from schema import create_database, seed_liver_masld_data
from scoring import run_scoring_pipeline
from knowledge_graph import build_knowledge_graph, extract_graph_features, predict_novel_associations

DB_PATH = "ncrna_platform.db"

print("=" * 60)
print("  ncRNA Target Intelligence Platform — Setup & Launch")
print("=" * 60)

print("\n[1/3] Building database and seeding MASLD/liver data...")
conn = create_database(DB_PATH)
seed_liver_masld_data(conn)
conn.close()

print("\n[2/3] Running translational scoring pipeline...")
scored_df, model_out = run_scoring_pipeline(DB_PATH)

print("\n[3/3] Building knowledge graph...")
G = build_knowledge_graph(DB_PATH)
ncrna_ids = [n for n, d in G.nodes(data=True) if d.get("node_type") == "ncRNA"]
disease_ids = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Disease"]

graph_feats = extract_graph_features(G, ncrna_ids)
novel_links = predict_novel_associations(G, ncrna_ids, disease_ids, top_k=5)

print("\n── Novel Predicted Associations ──")
print(novel_links.to_string(index=False))

print("\n" + "=" * 60)
print("✅ Setup complete!")
print("Launch dashboard: streamlit run dashboard.py")
print("Database: ncrna_platform.db")
print("=" * 60)