from datetime import datetime
from pathlib import Path
import dotenv
import pandas as pd
import random
import numpy as np
from typing import Optional, Set, List, FrozenSet
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from evals.evals_utils import output_nodes_path, output_relationships_path
import shutil

dotenv.load_dotenv()

"""
TODO:
- Add sibling relationships to the clusters
    - 1. Ask LLM if it's a good cluster
    - 2. If so, find the most common node
    - 3. Append all the siblings of that node to the cluster
"""


class GenerateTestSet:

    main_prompt = "# Documents:\n\n{documents}\n\n# Instructions:\nGenerate a query and answer based on the documents provided above according to the provided instructions"

    system_prompt = """
You are a technical content analyst specializing in IT networking and Pepwave products.
Your task is to analyze a set of documents and generate a multifaceted multi-hop query that a technician might ask based solely on the information in those documents.
Each set of documents contains related information. Identify a topic, product, technology, or concept that exists in many of the documents. There should be a diversity of useful technical information about that topic in multiple documents.
Create a query from the technical facts that can be gleaned from the documents that incorporates different information from as many documents as possible and connects them meaningfully.
Some of the documents are forum posts and contain various sections like ## Content, and ## Comments. The ### Content section is the original post and usually contains a question. Information presented as part of a question is not factual information and should not used to create the query. The ### Comments section contains responses to the original post and should be used as a source of technical information.
Once you have created the query, write a detailed answer to the query using only the information in the documents.
The query and answer should be based solely on the documents provided.
"""

    examples = [
        {
            "documents": '''
<DOCUMENT 1>
"""
## Post

### Title: My connection IS SLOW

### Content:

I connected a 50mps fibre optic internet connection to port 1 and a 10mbps phone internet service to port two.
Then cabled from Lan 1 to a wi fi router and I connect my devices via wi fi to that router.
However this is slower and more unstable than if I just connect my 50mps straight to my wifi router, overriding the Peplink.
I can see this when I test on Speedtest.net.

## Comments:

<comment>
Speedtest.net does not give an accurate reading when using multiple WANs because it doesn't take the Pepwave's load balancing into account. You must configure outbound policy rules in order to implement your scenario. Set your outbound policy to use the low-latency option and it will use the fastest connection.
  <reply>
  Hi Ron!
Does this mean that if I am using Strong DNS, this might be slowing it down?
    <reply>
    No, you just need to configure outbound policy rules.
  </reply>
</comment>
"""

<DOCUMENT 2>
"""
hi this is Dan and in this video I want to explain how to configure the outbound policy rules there are several options available and they all have different use cases when in doubt the best option is the power-fusion option that will use the fastest connection based on the FQDN protocol it is the best option to use for most simple use cases or if you are unsure I would recommend taking a look at the manual to learn about the other options available to you
"""

<DOCUMENT 3>
"""
You know about classes, but you may be thinking which traffic goes into each class? And how many classes should I configure? Well, let me help you simplify this You don't need to use every combination of class and drop probability. There is a real time class for voice and interactive video. This is a high priority class next, a class for critical data. This is for your business applications, databases, website, traffic. We could split this into two classes. If you do end up splitting your high priority class into two classes, make sure you set the outbound policy to use low-latency otherwise your equipment will not be able to handle the load and you may encounter a "unable to handle load" error. I would recommend starting with the default settings and then adding classes as needed. In all classes are not as complex as they may seem.
"""
''',
            "output": """
{
    "query": "What outbound policy settings should be attempted to circumvent a slow connection when using multiple wired connections or when encountering an 'unable to handle load' error? Can you rely on Speedtest.net to accurately measure performance in this scenario when using multiple wired connections?",
    "answer": "First, try the low-latency option. This is the best option when using multiple wired connections and is a solution to the 'unable to handle load' error. If that doesn't work, try the power-fusion option. Speedtest.net is not a reliable tool for measuring performance when using multiple wired connections because it does not take the Pepwave's load balancing into account."
}
""",
        },
        {
            "documents": '''
<DOCUMENT 1>
"""
## Post

### Title: How to optimize for MLRPV protocol?

### Content:

Our setup is using MLRPV protocol with 3 patchworked units in a BAFTA cluster.  The BAFTA cluster tends to be the best performing cluster for our use case with a class 5 ensemble transportation layer.  How do I optimize for this protocol?

## Comments:

<comment>
To optimize for MLRPV protocol, first make sure that the patchwork is grounded into at least 3 different antenna pods. Then, configure the system controller to use MLRPV-ensemble mode.
</comment>
"""

<DOCUMENT 2>
"""
## Post

### Title: What does the MLRPV-ensemble mode do?

### Content:

MLRPV protocol is a protocol that is typically used for section 9434 robotics implementations. It is an expert-only feature.  If you're not working in distributed robotics, you probably don't need to worry about it and can just leave it off. You would turn it on if you have a robot farm and want to make sure that all the robots are working together correctly. It does this by synchronizing the clocks of the robots day-night entrainment cycles.

## Comments:

<comment>
To optimize for MLRPV protocol, first make sure that the patchwork is grounded into at least 3 different antenna pods. Then, configure the system controller to use MLRPV-ensemble mode.
</comment>
"""

<DOCUMENT 3>
"""
one new feature we just addeed is the MLRPV-ensemble mode this is a new mode for section 9434 robotics situations it was added to the system controller in version 6.1 and can be used with any type of robot cluster you might come across and what have you I would recommend also taking a look at the ensemble controller section because that also has some features that are relevant to that protocol at the end of the day it just makes it really easy to get your robots synchronized well
"""
''',
            "output": """
{
    "query": "What is MLRPV-ensemble mode, how does it work, and how should you set up your BAFTA cluster patchwork when using it?",
    "answer": "MLRPV-ensemble mode is a new mode for section 9434 robotics situations that was added to the system controller in version 6.1 and can be used with any type of robot cluster when working with the MLRPV protocol. Enable it if you have a robot farm and want to make sure that all the robots are working together correctly. It works by synchronizing the clocks of the robots day-night entrainment cycles. To use MLRPV-ensmble mode with a patchworked BAFTA cluster, you need to make sure the patchwork is grounded into at least 3 different antenna pods before you turn it on in the system controller."
}
""",
        },
    ]

    def __init__(
        self,
        output_dir: Path,
        llm_model: str = "gpt-4o",
        testset_size: int = 50,
        cluster_size: int = 7,
        min_cluster_size: int = 3,
        random_seed: Optional[int] = None,
    ):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.llm_model = llm_model
        self.testset_size = testset_size
        self.cluster_size = cluster_size
        self.min_cluster_size = min_cluster_size
        self.random_seed = random_seed

        # Load data
        self.nodes_df = pd.read_parquet(output_nodes_path)
        print(f"nodes: {len(self.nodes_df)}")
        self.relationships_df = pd.read_parquet(output_relationships_path)
        print(f"relationships: {len(self.relationships_df)}")

        # Split relationships dataframe
        self.sibling_df = self.relationships_df[
            self.relationships_df["relationship_type"].str.contains("sibling")
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
        self.found_relationship_clusters: Set[FrozenSet[str]] = set()
        self.relationship_cluster_info_dfs: List[pd.DataFrame] = []
        self.node_clusters: List[pd.DataFrame] = []

    def create_main_df(self):
        main_df = self.relationships_df[
            ~self.relationships_df["relationship_type"].str.contains("sibling")
        ]

        # Add data type relationship flag
        return main_df.assign(
            is_same_data_type=lambda df: df["source_id"].map(
                self.nodes_df.set_index("node_id")["type"]
            )
            == df["target_id"].map(self.nodes_df.set_index("node_id")["type"])
        )

    def find_relationship_clusters(self) -> Set[FrozenSet[str]]:
        """
        Find n clusters of relationships by traversing the knowledge graph using dataframe operations.
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
            subset=["source_id", "target_id", "relationship_type"]
        )
        # create separate df for merged relationships
        multi_df = graph_df[graph_df["relationship_type"] == "multi"]
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
        while len(clusters) <= self.testset_size and len(graph_df) > 1:

            cluster_df = graph_df.iloc[[0]]
            graph_df = graph_df.iloc[1:]

            while len(cluster_df) < self.cluster_size:
                neighbors = multi_df[
                    ~multi_df.index.isin(cluster_df.index)
                    & (
                        (multi_df["source_id"].isin(cluster_df["target_id"]))
                        | (multi_df["target_id"].isin(cluster_df["source_id"]))
                    )
                ]
                if neighbors.empty:
                    neighbors = graph_df[
                        graph_df["source_id"].isin(cluster_df["target_id"])
                        | graph_df["target_id"].isin(cluster_df["source_id"])
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
                self.relationship_cluster_info_dfs.append(cluster_df)
                node_ids = pd.unique(
                    cluster_df[["source_id", "target_id"]].values.ravel()
                )
                clusters.add(frozenset(node_ids))

        return clusters

    def tack_on_node_col(self, clusters_df: pd.DataFrame, col: str) -> None:
        clusters_df[f"source_{col}"] = clusters_df["source_id"].map(
            self.nodes_df.set_index("node_id")[col]
        )
        clusters_df[f"target_{col}"] = clusters_df["target_id"].map(
            self.nodes_df.set_index("node_id")[col]
        )

    def _generate_cluster_info_parquet(self, dfs: List[pd.DataFrame], filename: str):
        """
        Generate a parquet file with the cluster information for inspection.
        This may have duplicate clusters that don't exist in the found_clusters set.
        """
        for i, cdf in enumerate(dfs):
            cdf["cluster"] = i

        sample_df = pd.concat(dfs)
        is_relationship_cluster = "source_id" in sample_df.columns
        if is_relationship_cluster:
            self.tack_on_node_col(sample_df, "page_content")
            self.tack_on_node_col(sample_df, "title")
            self.tack_on_node_col(sample_df, "technical_summary")
        sample_df.to_parquet(self.output_dir / f"{filename}.parquet")

    def _cluster_reporting(self):
        print("\n" + "-" * 50 + "\n")
        print(f"Clusters count: {len(self.found_relationship_clusters)}")
        print(
            f"Total elements in relationship clusters: {
                sum(
                    len(cluster) for cluster in self.found_relationship_clusters)}"
        )

        # Get the set of all unique ids from relationship clusters
        all_unique_ids = set()
        for cluster in self.found_relationship_clusters:
            all_unique_ids.update(cluster)

        print(f"Total unique IDs across all clusters: {len(all_unique_ids)}")

    def llm_filter_clusters(self):
        """
        Filter clusters based on LLM's assessment.
        """
        pass

    def _tokens_under_threshold(self, nodes_df: pd.DataFrame) -> bool:
        """
        Calculate token count and return whether it exceeds the threshold.
        Args:
            nodes_df: DataFrame containing nodes
        Returns:
            Boolean indicating whether token count exceeds the threshold.
        """
        max_token_count = 30_000
        token_count = nodes_df["page_content"].apply(len).sum()
        return token_count < max_token_count

    def get_node_clusters(self):
        """
        For each cluster, get the corresponding nodes and append siblings up to a max token count.
        """
        for cluster in self.found_relationship_clusters:
            nodes_df = self.nodes_df[self.nodes_df["node_id"].isin(cluster)].copy()
            sibling_relationships = self.sibling_df[
                (self.sibling_df["source_id"].isin(cluster))
                | (self.sibling_df["target_id"].isin(cluster))
            ]
            sibling_nodes = self.nodes_df[
                self.nodes_df["node_id"].isin(
                    pd.concat(
                        [
                            sibling_relationships["source_id"],
                            sibling_relationships["target_id"],
                        ]
                    ).unique()
                )
            ]
            while self._tokens_under_threshold(nodes_df) and not sibling_nodes.empty:
                # Pop a sibling node and add it to nodes_df
                sibling_node = sibling_nodes.iloc[0]
                sibling_nodes = sibling_nodes.iloc[1:]
                # Only add if not already in nodes_df
                if sibling_node["node_id"] not in nodes_df["node_id"].values:
                    nodes_df.loc[len(nodes_df)] = sibling_node
            self.node_clusters.append(nodes_df)

    def llm_generate_testset(self):
        query_schema = ResponseSchema(
            name="query",
            description="A multifaceted multi-hop query based on the provided documents",
        )
        answer_schema = ResponseSchema(
            name="answer",
            description="A detailed answer to the query using only information from the documents",
        )
        response_schemas = [query_schema, answer_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        example_prompt = ChatPromptTemplate.from_messages(
            [("human", self.main_prompt), ("ai", "{output}")]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.examples,
        )

        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                few_shot_prompt,
                ("human", self.main_prompt),
            ]
        )

        # Chain
        chain = final_prompt | self.llm | output_parser

        results = []
        for i, node_cluster_df in enumerate(self.node_clusters):
            documents_text = ""
            for idx, (_, row) in enumerate(node_cluster_df.iterrows(), 1):
                documents_text += (
                    f'<DOCUMENT {idx}>\n"""\n{row["page_content"]}\n"""\n\n'
                )

            try:
                result = chain.invoke(
                    {
                        "documents": documents_text,
                    }
                )

                results.append(
                    {
                        "cluster_id": i,
                        "documents_text": documents_text,
                        "query": result["query"],
                        "answer": result["answer"],
                        "document_ids": node_cluster_df["id"].tolist(),
                        "node_ids": node_cluster_df["node_id"].tolist(),
                    }
                )
            except Exception as e:
                print(f"Error processing cluster {i}: {e}")

        # Save results to a file
        output_path = self.output_dir / "generated_testset.parquet"
        pd.DataFrame(results).to_parquet(output_path)

        print(f"Generated testset saved to {output_path}")
        return results

    def copy_kg_data_to_testset_dir(self):
        """
        Copy the knowledge graph data files to the output directory.
        """
        shutil.copy(output_nodes_path, self.output_dir / output_nodes_path.name)
        shutil.copy(
            output_relationships_path, self.output_dir / output_relationships_path.name
        )
        print(f"Copied knowledge graph data files to {self.output_dir}")

    def create_testset(self):
        self.found_relationship_clusters = self.find_relationship_clusters()
        self._cluster_reporting()
        self._generate_cluster_info_parquet(
            self.relationship_cluster_info_dfs, "relationship_clusters_info"
        )
        self.get_node_clusters()
        self._generate_cluster_info_parquet(self.node_clusters, "node_clusters_for_llm")
        self.copy_kg_data_to_testset_dir()
        self.llm_generate_testset()


this_dir = Path(__file__).parent
if __name__ == "__main__":
    testset_size = 1
    generate_testset = GenerateTestSet(
        output_dir=this_dir
        / "testsets"
        / f"testset_{testset_size}__{datetime.now().strftime('%y-%m-%d_%H_%M')}",
        llm_model="gpt-4o",
        testset_size=testset_size,
    )
    generate_testset.create_testset()
