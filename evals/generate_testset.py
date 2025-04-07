from pathlib import Path
import pandas as pd
import random
import numpy as np
from typing import Optional, Set, List, FrozenSet


"""
TODO:
- Add sibling relationships to the clusters
    - 1. Ask LLM if it's a good cluster
    - 2. If so, find the most common node
    - 3. Append all the siblings of that node to the cluster
- Handicap same-data-type relationships
"""


def tack_on_node_col(
    clusters_df: pd.DataFrame, node_df: pd.DataFrame, col: str
) -> pd.DataFrame:
    clusters_df[f"source_{col}"] = clusters_df["source_id"].map(
        node_df.set_index("node_id")[col]
    )
    clusters_df[f"target_{col}"] = clusters_df["target_id"].map(
        node_df.set_index("node_id")[col]
    )
    return clusters_df


class GenerateTestSet:
    def __init__(
        self,
        output_dir: Path,
        llm_model: str = "gpt-4o",
        testset_size: int = 50,
        cluster_size: int = 7,
        min_cluster_size: int = 3,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the GenerateTestSet class.

        Args:
            nodes_df_path: Path to the nodes dataframe parquet file
            relationships_df_path: Path to the relationships dataframe parquet file
            output_dir: Directory to store output files
            llm_model: LLM model to use
            testset_size: Number of clusters to generate for the test set
        """
        self.output_dir = Path(output_dir)
        nodes_df_path = output_dir / "__nodes_LATEST.parquet"
        relationships_df_path = (
            self.output_dir / "__relationships_MERGED__LATEST.parquet"
        )
        self.llm_model = llm_model
        self.testset_size = testset_size
        self.cluster_size = cluster_size
        self.min_cluster_size = min_cluster_size
        self.random_seed = random_seed

        # Load data
        self.nodes_df = pd.read_parquet(nodes_df_path)
        print(f"nodes: {len(self.nodes_df)}")
        self.relationships_df = pd.read_parquet(relationships_df_path)
        print(f"relationships: {len(self.relationships_df)}")

        # Split relationships dataframe
        self.sibling_df = self.relationships_df[
            self.relationships_df['relationship_type'].str.contains('sibling')
        ]
        self.main_df = self.create_main_df()

        # Thresholds for non-multi relationships
        self.single_rel_thresholds = {
            "summary_similarity": 0.815,
            "title_similarity": 0.805,
            "themes_overlap_score": 0.24,
            "entities_overlap_score": 0.24,
        }

        print(
            f"\n\nrelationship_type value counts:\n {self.main_df['relationship_type'].value_counts()}"
        )

        # Initialize storage for clusters
        self.cluster_dfs_sample: List[pd.DataFrame] = []
        self.found_clusters: Set[FrozenSet[str]] = set()

    def create_main_df(self):
        df = self.relationships_df[
            ~self.relationships_df['relationship_type'].str.contains('sibling')
        ]

        # Add data type relationship flag
        return df.assign(
            is_same_data_type=lambda df: df["source_id"].map(
                self.nodes_df.set_index("node_id")["type"]
            )
            == df["target_id"].map(self.nodes_df.set_index("node_id")["type"])
        )

    def find_relationship_clusters(self) -> Set[FrozenSet[str]]:
        """
        Find n clusters of relationships by traversing the knowledge graph using dataframe operations.

        Args:
            relationship_df: DataFrame containing relationships with source_id and target_id columns
            n: Number of clusters to find
            cluster_size: Size of each cluster
            min_cluster_size: Minimum size for a valid cluster
            random_seed: Optional seed for reproducibility

        Returns:
            Set of frozensets, where each frozenset contains the IDs of nodes in a cluster/path
        """
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Create a shuffled copy of the dataframe
        graph_df = self.main_df.sample(frac=1.0, random_state=self.random_seed)
        # create simplified graph df
        graph_df = graph_df.drop(columns=["bidirectional", "num_noisy_items"])
        graph_df = graph_df.drop_duplicates(
            subset=['source_id', 'target_id', 'relationship_type']
        )
        # create separate df for merged relationships
        multi_df = graph_df[graph_df['relationship_type'] == 'multi']
        non_multi_df = graph_df[~graph_df.index.isin(multi_df.index)]

        # filter non_multi_df to only include rows that pass single_rel_thresholds
        # null values (values that are not floats) should pass
        for col, threshold in self.single_rel_thresholds.items():
            # apply higher threshold for same-data-type relationships
            threshold_boost = 0.02 if col == "title_similarity" else 0.01
            effective_threshold = non_multi_df["is_same_data_type"].map(
                {True: threshold + threshold_boost, False: threshold}
            )
            non_multi_df = non_multi_df[
                non_multi_df[col].isna() | (non_multi_df[col] >= effective_threshold)
            ]

        # Sort graph_df so that rows in multi_df come first, preserving relative order within each group
        graph_df = pd.concat(
            [
                multi_df,
                non_multi_df,
            ]
        )

        print(
            f"\n\nPost-filtering non-multi {graph_df['relationship_type'].value_counts()}"
        )

        clusters: set = set()
        while len(clusters) < self.testset_size and len(graph_df) > 1:

            cluster_df = graph_df.iloc[[0]]
            graph_df = graph_df.iloc[1:]

            while len(cluster_df) < self.cluster_size:
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
                filtered_neighbors = filtered_neighbors.drop_duplicates(
                    subset="source_id"
                )
                filtered_neighbors = filtered_neighbors.iloc[
                    : self.cluster_size - len(cluster_df)
                ]
                graph_df = graph_df.drop(filtered_neighbors.index, errors="ignore")
                cluster_df = pd.concat([cluster_df, filtered_neighbors])

            if len(cluster_df) >= self.min_cluster_size:
                self.cluster_dfs_sample.append(cluster_df)
                node_ids = pd.unique(
                    cluster_df[["source_id", "target_id"]].values.ravel()
                )
                clusters.add(frozenset(node_ids))

        return clusters

    def _generate_cluster_info_parquet(self):
        """
        Generate a parquet file with the cluster information.
        This may have duplicate clusters, but it's whatever...
        """
        df = self.cluster_dfs_sample
        for i, cdf in enumerate(df):
            cdf["cluster"] = i

        sample_df = pd.concat(df)
        sample_df = tack_on_node_col(sample_df, self.nodes_df, "page_content")
        sample_df = tack_on_node_col(sample_df, self.nodes_df, "title")
        sample_df = tack_on_node_col(sample_df, self.nodes_df, "technical_summary")
        sample_df.to_parquet(self.output_dir / f"__clusters_sample.parquet")

    def _cluster_reporting(self, stage: str):
        print("\n" + "-" * 50 + f" {stage} " + "-" * 50 + "\n")
        print(f"Clusters count: {len(self.found_clusters)}")
        print(
            f"Total elements in relationship clusters: {sum(len(cluster) for cluster in self.found_clusters)}"
        )

        # Get the set of all unique ids from relationship clusters
        all_unique_ids = set()
        for cluster in self.found_clusters:
            all_unique_ids.update(cluster)

        print(f"Total unique IDs across all clusters: {len(all_unique_ids)}")

    def get_clusters(self) -> None:
        """
        Generate a test set by finding relationship clusters and enhancing them with siblings.

        Args:
            cluster_size: Size of each cluster
            min_cluster_size: Minimum size for a valid cluster
            random_seed: Optional seed for reproducibility
        """
        # Find relationship clusters
        self.found_clusters = self.find_relationship_clusters()
        self._cluster_reporting("init")
        self._generate_cluster_info_parquet()

    def llm_filter_clusters(self):
        """
        Filter clusters based on LLM's assessment.
        """
        pass

    def get_siblings_of_primary_node(self):
        """
        For each cluster, find the most common node and append its siblings.

        Returns:
            Dictionary mapping cluster index to a DataFrame of enhanced clusters with siblings
        """
        pass
