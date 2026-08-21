"""
Load task family A's annotation KG (data/thinking/processed/annotation_kg.parquet)
into the local Neo4j instance (bolt://localhost:7687) as a graph: one (:Protein)
node per entry, one (:Term) node per distinct (relation, value) pair, connected by
a relationship typed after the KG relation (MEMBER_OF, HAS_FUNCTION, ...).

Excludes the `interacts_with` relation -- that's PPI task family data that happens
to share this same KG parquet file (see build_annotation_kg.py's
build_human_ppi_triples), not part of task_a's annotation direction.

Usage:
    python3 scripts/data/thinking/load_neo4j_task_a_annotation.py
"""

import os

import pandas as pd
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")
BATCH_SIZE = 2000

_HERE = os.path.dirname(os.path.abspath(__file__))
ANNOTATION_KG_FILE = os.path.join(_HERE, "..", "..", "..", "data", "thinking", "processed", "annotation_kg.parquet")

REL_TO_TYPE = {
    "member_of": "MEMBER_OF",
    "has_function": "HAS_FUNCTION",
    "involved_in": "INVOLVED_IN",
    "located_in": "LOCATED_IN",
    "catalyzes": "CATALYZES",
    "has_domain": "HAS_DOMAIN",
    "has_region": "HAS_REGION",
    "has_motif": "HAS_MOTIF",
    "has_length": "HAS_LENGTH",
}


def load_batch(tx, rel_type: str, rows: list[dict]) -> None:
    query = f"""
    UNWIND $rows AS row
    MERGE (p:Protein {{entry: row.entry}})
    MERGE (t:Term {{type: row.relation, value: row.value}})
    MERGE (p)-[:{rel_type}]->(t)
    """
    tx.run(query, rows=rows)


def main() -> None:
    df = pd.read_parquet(ANNOTATION_KG_FILE)
    df = df[df["relation"] != "interacts_with"]
    print(f"loading {len(df)} triples across {df['entry'].nunique()} entries (interacts_with excluded)")

    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        session.run("CREATE INDEX protein_entry_idx IF NOT EXISTS FOR (p:Protein) ON (p.entry)")
        session.run("CREATE INDEX term_value_idx IF NOT EXISTS FOR (t:Term) ON (t.value)")

        for relation, group in df.groupby("relation"):
            rel_type = REL_TO_TYPE[relation]
            records = group[["entry", "relation", "value"]].to_dict("records")
            for i in range(0, len(records), BATCH_SIZE):
                session.execute_write(load_batch, rel_type, records[i : i + BATCH_SIZE])
            print(f"  {relation} ({rel_type}): {len(records)} loaded")

        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"\ntotal nodes: {node_count}, total relationships: {rel_count}")

    driver.close()


if __name__ == "__main__":
    main()
