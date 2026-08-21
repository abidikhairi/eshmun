"""
Load the remaining KG subsets into the same local Neo4j instance already
seeded by load_neo4j_task_a_annotation.py (Protein/Term nodes keyed by
entry / (type, value), so this merges cleanly on top of what's there):

1. PPI edges (`interacts_with`, held in annotation_kg.parquet alongside
   task_a's relations but excluded from that first load) -- Protein-to-
   Protein, not Protein-to-Term, since its `value` column holds a partner
   accession, not a free-text term (verified: 100% of values match the
   UniProt accession pattern).
2. SCOP KG (scop_kg.parquet) -- scop_fold/scop_superfamily/scop_family.
3. All-organism KG (all_organism_kg.parquet) -- same relation set as
   task_a's annotation KG, just built from a different source (all
   reviewed SwissProt entries, all_swissprot_fields.tsv) instead of the
   human-only exports, so entry/value overlap with the first load is
   expected and desired (enriches shared nodes rather than duplicating,
   since MERGE keys are identical).

Usage:
    python3 scripts/data/thinking/load_neo4j_remaining.py
"""

import os

import pandas as pd
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")
BATCH_SIZE = 2000

_HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking", "processed")
ANNOTATION_KG_FILE = os.path.join(PROCESSED_DIR, "annotation_kg.parquet")
SCOP_KG_FILE = os.path.join(PROCESSED_DIR, "scop_kg.parquet")
ALL_ORGANISM_KG_FILE = os.path.join(PROCESSED_DIR, "all_organism_kg.parquet")

TERM_REL_TO_TYPE = {
    "member_of": "MEMBER_OF",
    "has_function": "HAS_FUNCTION",
    "involved_in": "INVOLVED_IN",
    "located_in": "LOCATED_IN",
    "catalyzes": "CATALYZES",
    "has_domain": "HAS_DOMAIN",
    "has_region": "HAS_REGION",
    "has_motif": "HAS_MOTIF",
    "has_length": "HAS_LENGTH",
    "scop_fold": "SCOP_FOLD",
    "scop_superfamily": "SCOP_SUPERFAMILY",
    "scop_family": "SCOP_FAMILY",
}


def load_term_batch(tx, rel_type: str, rows: list[dict]) -> None:
    query = f"""
    UNWIND $rows AS row
    MERGE (p:Protein {{entry: row.entry}})
    MERGE (t:Term {{type: row.relation, value: row.value}})
    MERGE (p)-[:{rel_type}]->(t)
    """
    tx.run(query, rows=rows)


def load_ppi_batch(tx, rows: list[dict]) -> None:
    query = """
    UNWIND $rows AS row
    MERGE (a:Protein {entry: row.entry})
    MERGE (b:Protein {entry: row.value})
    MERGE (a)-[:INTERACTS_WITH]->(b)
    """
    tx.run(query, rows=rows)


def load_term_kg(session, df: pd.DataFrame, label: str) -> None:
    print(f"loading {label}: {len(df)} triples across {df['entry'].nunique()} entries")
    for relation, group in df.groupby("relation"):
        rel_type = TERM_REL_TO_TYPE[relation]
        records = group[["entry", "relation", "value"]].to_dict("records")
        for i in range(0, len(records), BATCH_SIZE):
            session.execute_write(load_term_batch, rel_type, records[i : i + BATCH_SIZE])
        print(f"  {relation} ({rel_type}): {len(records)} loaded")


def main() -> None:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        session.run("CREATE INDEX protein_entry_idx IF NOT EXISTS FOR (p:Protein) ON (p.entry)")
        session.run("CREATE INDEX term_value_idx IF NOT EXISTS FOR (t:Term) ON (t.value)")

        annotation_kg = pd.read_parquet(ANNOTATION_KG_FILE)
        ppi = annotation_kg[annotation_kg["relation"] == "interacts_with"]
        print(f"loading ppi_interacts_with: {len(ppi)} edges across {ppi['entry'].nunique()} entries")
        records = ppi[["entry", "value"]].to_dict("records")
        for i in range(0, len(records), BATCH_SIZE):
            session.execute_write(load_ppi_batch, records[i : i + BATCH_SIZE])
        print(f"  interacts_with (INTERACTS_WITH): {len(records)} loaded")

        scop_kg = pd.read_parquet(SCOP_KG_FILE)
        load_term_kg(session, scop_kg, "scop_kg")

        all_organism_kg = pd.read_parquet(ALL_ORGANISM_KG_FILE)
        load_term_kg(session, all_organism_kg, "all_organism_kg")

        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"\ntotal nodes: {node_count}, total relationships: {rel_count}")

    driver.close()


if __name__ == "__main__":
    main()
