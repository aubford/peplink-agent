# %%
from ragas.testset.persona import Persona
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import random
import numpy as np
from typing import Optional

load_dotenv()

network_persona = Persona(
    name="Technical Network Engineer",
    role_description="A professional specializing in network configurations and troubleshooting, particularly with experience in WAN setups and integrating satellite internet systems like Starlink with networking hardware such as Peplink Balance and Juniper SRX. This individual actively participates in technical forums to share knowledge, solve problems, and seek advice on complex network-related issues.",
)
full_time_rv_persona = Persona(
    name="Full-Time RV Owner",
    role_description="A full-time RV owner who is interested in using Pepwave devices to connect to the internet while traveling.",
)


LLM_MODEL = "gpt-4o"
TESTSET_SIZE = 50
output_dir = Path(__file__).parent / "output"
kg_path = output_dir / "kg_output__LATEST.json"

df = pd.read_parquet(output_dir / "__relationships_MERGED__LATEST.parquet")
sibling_df = df[df['relationship_type'].str.contains('sibling')]
main_df = df[~df['relationship_type'].str.contains('sibling')]

print(f"info: {df.info()}")
print(
    f"\n\nrelationship_type value counts:\n {sibling_df['relationship_type'].value_counts()}"
)
print(
    f"\nnon-sibling relationship_type value counts:\n {main_df['relationship_type'].value_counts()}"
)


def find_relationship_clusters(
    relationship_df: pd.DataFrame,
    n: int,
    cluster_size: int,
    random_seed: Optional[int] = None,
) -> set[frozenset]:
    """
    Find n clusters of relationships by traversing the knowledge graph using dataframe operations.

    Args:
        relationship_df: DataFrame containing relationships with source_id and target_id columns
        n: Number of clusters to find
        cluster_size: Size of each cluster
        random_seed: Optional seed for reproducibility

    Returns:
        Set of frozensets, where each frozenset contains the IDs of nodes in a cluster/path
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Create a shuffled copy of the dataframe
    graph_df = relationship_df.sample(frac=1.0, random_state=random_seed)
    # create simplified graph df
    graph_df = graph_df[
        ['source_id', 'target_id', 'relationship_type']
    ].drop_duplicates()
    # create separate df for merged relationships
    multi_df = graph_df[graph_df['relationship_type'] == 'multi']
    # remove merged relationships from graph_df
    graph_df = graph_df.drop(multi_df.index)

    clusters: set = set()
    while len(clusters) < n and len(graph_df) > 1:

        cluster_df = graph_df.iloc[[0]]
        graph_df = graph_df.iloc[1:]

        while len(cluster_df) < cluster_size:
            neighbors = multi_df[
                ~multi_df.index.isin(cluster_df.index)
                & (
                    (multi_df["source_id"].isin(cluster_df['target_id']))
                    | (multi_df["target_id"].isin(cluster_df['source_id']))
                )
            ]
            if neighbors.empty:
                neighbors = graph_df[
                    graph_df["source_id"].isin(cluster_df['target_id'])
                    | graph_df["target_id"].isin(cluster_df['source_id'])
                ]
            if neighbors.empty:
                break
            filtered_neighbors = neighbors.drop_duplicates(subset="target_id")
            filtered_neighbors = filtered_neighbors.drop_duplicates(subset="source_id")
            filtered_neighbors = filtered_neighbors.iloc[
                : cluster_size - len(cluster_df)
            ]
            graph_df = graph_df.drop(filtered_neighbors.index, errors="ignore")
            cluster_df = pd.concat([cluster_df, filtered_neighbors])

        node_ids = pd.unique(cluster_df[["source_id", "target_id"]].values.ravel())
        clusters.add(frozenset(node_ids))

    return clusters


# %% ########################## FIND RELATIONSHIP CLUSTERS ######################################################################################################################################

relationship_clusters: set[frozenset[str]] = find_relationship_clusters(main_df, 50, 5)
print("\n" + "-" * 100 + "\n")
print(f"Found {len(relationship_clusters)} relationship clusters:")
for i, cluster in enumerate(relationship_clusters, 1):
    print(f"Cluster {i}: {sorted(cluster)}")

# Get the set of all unique ids from relationship clusters
all_unique_ids = set()
for cluster in relationship_clusters:
    all_unique_ids.update(cluster)

print("\n" + "-" * 100 + "\n")
print(
    f"Total elements in relationship clusters: {sum(len(cluster) for cluster in relationship_clusters)}"
)
print(f"Total unique IDs across all clusters: {len(all_unique_ids)}")

# %%

nodes_df = pd.read_parquet(output_dir / "__nodes_LATEST.parquet")
node_clusters = [
    nodes_df[nodes_df['node_id'].isin(cluster)] for cluster in relationship_clusters
]

first_cluster_df = node_clusters[0]
first_cluster_df.to_parquet(output_dir / "testset_cluster_0.parquet")

# Create a text document with each page content separated by double newlines
with open(output_dir / "testset_cluster_0_content.txt", "w") as f:
    joiner = "\n\n" + "=" * 100 + "\n\n"
    f.write(joiner.join(first_cluster_df["page_content"].tolist()))


# %%
