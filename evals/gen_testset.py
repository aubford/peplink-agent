# %%
from ragas.testset.persona import Persona
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import random
import numpy as np
from typing import Optional

"""
TODO:
- Add sibling relationships to the clusters
    - 1. Ask LLM if it's a good cluster
    - 2. If so, find the most common node
    - 3. Append all the siblings of that node to the cluster
- Handicap same-data-type relationships
"""


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

# print(f"info: {df.info()}")
print(
    f"\n\nrelationship_type value counts:\n {main_df['relationship_type'].value_counts()}"
)


cluster_dfs_sample: list[pd.DataFrame] = []


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
    graph_df = graph_df.drop(columns=["bidirectional", "num_noisy_items"])
    graph_df = graph_df.drop_duplicates(
        subset=['source_id', 'target_id', 'relationship_type']
    )
    # create separate df for merged relationships
    multi_df = graph_df[graph_df['relationship_type'] == 'multi']
    # Sort graph_df so that rows in multi_df come first, preserving relative order within each group
    is_in_multi = graph_df.index.isin(multi_df.index)
    graph_df = pd.concat(
        [
            graph_df[is_in_multi],
            graph_df[~is_in_multi],
        ]
    )

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

        cluster_dfs_sample.append(cluster_df)
        node_ids = pd.unique(cluster_df[["source_id", "target_id"]].values.ravel())
        clusters.add(frozenset(node_ids))

    return clusters


def tack_on_node_col(
    clusters_df: pd.DataFrame, nodes_df: pd.DataFrame, col: str
) -> pd.DataFrame:
    clusters_df[f"source_{col}"] = clusters_df["source_id"].map(
        nodes_df.set_index("node_id")[col]
    )
    clusters_df[f"target_{col}"] = clusters_df["target_id"].map(
        nodes_df.set_index("node_id")[col]
    )
    return clusters_df


########################## FIND RELATIONSHIP CLUSTERS ######################################################################################################################################

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

nodes_df = pd.read_parquet(output_dir / "__nodes_LATEST.parquet")
for i, cdf in enumerate(cluster_dfs_sample):
    cdf["cluster"] = i

sample_df = pd.concat(cluster_dfs_sample)
sample_df = tack_on_node_col(sample_df, nodes_df, "page_content")
sample_df = tack_on_node_col(sample_df, nodes_df, "title")
sample_df = tack_on_node_col(sample_df, nodes_df, "technical_summary")
sample_df.to_parquet(output_dir / f"__clusters_sample.parquet")


# %%

# nodes_df = pd.read_parquet(output_dir / "__nodes_LATEST.parquet")
# node_clusters = [
#     nodes_df[nodes_df['node_id'].isin(cluster)] for cluster in relationship_clusters
# ]

# # Sample a random cluster
# sample_cluster_index = random.randint(0, len(node_clusters) - 1)
# sample_cluster_df = node_clusters[sample_cluster_index]
# sample_cluster_df.to_parquet(output_dir / "testset_cluster_0.parquet")

# # Create a text document with each page content separated by double newlines
# with open(output_dir / "testset_cluster_0_content.txt", "w") as f:
#     joiner = "\n\n" + "=" * 100 + "\n\n"
#     f.write(joiner.join(sample_cluster_df["page_content"].tolist()))
