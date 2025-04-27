from datetime import datetime
from pathlib import Path
import dotenv
import pandas as pd
from typing import FrozenSet
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from pydantic import BaseModel, Field
import json
import torch
from evals.evals_utils import output_nodes_path, output_relationships_path
from util.util_main import count_tokens
import shutil
from evals.prompts.prompts import load_prompts
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

dotenv.load_dotenv()
this_dir = Path(__file__).parent
testset_dir = this_dir / "testsets"
GPT_4_1_MODEL = "gpt-4.1"

PROMPTS = load_prompts()


class FactCitation(BaseModel):
    claim: str = Field(description="An atomic claim from your answer")
    document_names: list[str] = Field(
        description="The names of the documents that you used as the basis for the claim"
    )


class ResponseModel(BaseModel):
    query: str = Field(
        description="A multifaceted multi-hop query based on the provided documents"
    )
    answer: str = Field(
        description="A detailed answer to the query using only information from the documents"
    )
    citations: list[FactCitation] = Field(
        description="A list of atomic claims from your answer along with supporting document names"
    )


class GenerateTestSet:
    """
    GenerateTestSet constructs multi-hop RAG evaluation testsets by leveraging a knowledge graph (KG) of documents
    and their relationships, and generating queries/answers using an LLM.

    Core Workflow:
    --------------
    - Loads nodes and relationships from pre-built KG parquet files.
    - Applies column-specific thresholds (`single_rel_thresholds`) to filter non-multi relationships, with adaptive
      thresholding for same-type node pairs.
    - Clusters are formed by traversing the KG using a modified breadth-first strategy:
        - First pass excludes 'sibling' relationships in order to generate a diverse base set. It also favors "multi"
          relationships or nodes connected with more than one type of relationship as these are considered more reliable.
        - Clusters are grown up to `non_sibling_target_cluster_size` nodes, with a minimum enforced by
          `min_cluster_size`.
        - Only clusters meeting the minimum size are retained; process continues until `testset_size` clusters are
          found.
    - For each cluster, additional sibling nodes (sharing `parent_doc_id` or `post_id`) are appended, up to a token
      budget (`max_context_token_count`) or a proportional cap (1.5x the non-sibling cluster size). Sibling nodes are
      nodes that share a parent document or were forum comments/responses to the same original post.
    - For each node cluster, a prompt is constructed and sent to the LLM to generate:
        - A multi-hop query requiring synthesis of information from multiple documents.
        - An answer grounded in the provided documents, with each atomic claim cited to supporting documents so the quality
        of the answer can be inspected and assessed.
        - A prompting strategy is used that uses a tiered approach to requesting the LLM meet evidence requirements.
        - The LLM is required to return empty fields if none of the evidence requirements tiers (minimum facts + breadth of documents)
        are not met.
    - Results are saved as structured JSON, including queries, answers, citations, and the underlying document/node IDs as metadata.
    - Cluster metadata and a copy of the KG data are exported for quality analysis and reproducibility.
    - Git tags are used to track the codebase state at the time of testset generation for reproducibility.

    Notable Implementation Details:
    ------------------------------
    - Relationship filtering uses both static and adaptive thresholds, with higher thresholds for same-type node pairs.
    - Clustering prioritizes multi-relationship edges, and avoids cycles/redundant expansion by deduplication.
    - Sibling node augmentation is strictly bounded by both token count and proportional size to avoid exceeding LLM
      context limits.
    - Prompting uses a system prompt, main prompt, and few-shot examples, enforcing strict evidence and citation
      requirements.
    - All configuration is parameterized via the constructor: `testset_name`, `testset_size`,
      `non_sibling_target_cluster_size`, `min_cluster_size`, `llm_model`, `max_context_token_count`, `temperature`,
      and `doc_text_column`.
    """

    def __init__(
        self,
        testset_name: str,
        llm_model: str = GPT_4_1_MODEL,
        testset_size: int = 100,
        non_sibling_target_cluster_size: int = 7,
        min_cluster_size: int = 3,
        max_context_token_count: int = 30_000,
        temperature: float = 0.5,
        doc_text_column: str = "page_content",
    ):
        # Initialize LLM and output directory
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
        )

        self.testset_name = testset_name
        testset_dirname = f"testset-{testset_size}_{testset_name}_{datetime.now().strftime('%y-%m-%d')}"

        self.output_dir = testset_dir / testset_dirname
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.llm_model = llm_model
        self.max_context_token_count = max_context_token_count
        self.testset_size = testset_size
        self.cluster_size = non_sibling_target_cluster_size
        self.min_cluster_size = min_cluster_size
        self.doc_text_column = doc_text_column

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
            "summary_similarity": 0.805,
            "title_similarity": 0.79,
            "themes_overlap_score": 0.22,
            "entities_overlap_score": 0.22,
        }

        print(
            f"\n\nrelationship_type value counts:\n {self.main_df['relationship_type'].value_counts()}"
        )

        # Initialize storage for clusters
        self.found_relationship_clusters: set[FrozenSet[str]] = set()
        self.relationship_cluster_info_dfs: list[pd.DataFrame] = []
        self.node_clusters: list[pd.DataFrame] = []

    main_prompt = PROMPTS["generate_testset/main_prompt"]

    system_prompt = PROMPTS["generate_testset/system_prompt"]

    examples = [
        {
            "documents": PROMPTS["generate_testset/exampleA_documents"],
            "output": PROMPTS["generate_testset/exampleA_output"],
        },
        {
            "documents": PROMPTS["generate_testset/exampleB_documents"],
            "output": PROMPTS["generate_testset/exampleB_output"],
        },
    ]

    def create_main_df(self):
        """
        Create a DataFrame of non-sibling relationships.
        Add a flag to indicate whether source/target nodes are of the same data type so we can require
        higher threshold later.
        """
        main_df = self.relationships_df[
            ~self.relationships_df["relationship_type"].str.contains("sibling")
        ]

        # Add same data type relationship flag for `find_relationship_clusters`
        return main_df.assign(
            is_same_data_type=lambda df: df["source_id"].map(
                self.nodes_df.set_index("node_id")["type"]
            )
            == df["target_id"].map(self.nodes_df.set_index("node_id")["type"])
        )

    def find_relationship_clusters(self) -> set[FrozenSet[str]]:
        """
        Find n clusters of relationships by traversing the knowledge graph using dataframe operations.
        Returns:
            Set of frozensets, where each frozenset contains the IDs of nodes in a cluster/path
        """
        # Create a shuffled copy of the non-sibling relationships dataframe
        graph_df = self.main_df.sample(frac=1.0)
        # create simplified graph df
        graph_df = graph_df.drop(columns=["bidirectional", "num_noisy_items"])
        graph_df = graph_df.drop_duplicates(
            subset=["source_id", "target_id", "relationship_type"]
        )
        # create separate df for multi-relationships (merged multiple relationships into one)
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
        # Main loop: build clusters until testset size or graph is exhausted
        while len(clusters) <= self.testset_size and len(graph_df) > 1:

            # pop first row from graph_df and init the cluster_df
            cluster_df = graph_df.iloc[[0]]
            graph_df = graph_df.iloc[1:]

            # expand the cluster using breadth-first until it reaches the cluster_size
            while len(cluster_df) < self.cluster_size:
                # Try to expand the cluster by adding neighbors from multi relationships first
                neighbors = multi_df[
                    ~multi_df.index.isin(cluster_df.index)
                    & (
                        (multi_df["source_id"].isin(cluster_df["target_id"]))
                        | (multi_df["target_id"].isin(cluster_df["source_id"]))
                    )
                ]
                if neighbors.empty:
                    # If no multi neighbors, try any remaining edges in the graph
                    neighbors = graph_df[
                        graph_df["source_id"].isin(cluster_df["target_id"])
                        | graph_df["target_id"].isin(cluster_df["source_id"])
                    ]
                if neighbors.empty:
                    break
                # Remove duplicate neighbors to avoid cycles and redundant expansion
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
        """
        Add a column to the clusters DataFrame with node attributes (e.g., title, content) for both source and target nodes.
        """
        clusters_df[f"source_{col}"] = clusters_df["source_id"].map(
            self.nodes_df.set_index("node_id")[col]
        )
        clusters_df[f"target_{col}"] = clusters_df["target_id"].map(
            self.nodes_df.set_index("node_id")[col]
        )

    def _generate_cluster_info_parquet(self, dfs: list[pd.DataFrame], filename: str):
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
            self.tack_on_node_col(sample_df, "primary_content")
        sample_df.to_parquet(self.output_dir / f"{filename}.parquet")

    def _cluster_reporting(self):
        """
        Print summary statistics about the clusters found.
        """
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
        token_count = nodes_df[self.doc_text_column].apply(count_tokens).sum()
        is_under_threshold = token_count < self.max_context_token_count
        if not is_under_threshold:
            print(
                f"FYI: A token count of {token_count} exceeded threshold {self.max_context_token_count}"
            )
        return is_under_threshold

    def get_node_clusters(self):
        """
        For each cluster, get the corresponding nodes and append siblings up to a max token count.
        Nodes are considered siblings if they share the same 'parent_doc_id' or 'post_id'.
        """
        for cluster in self.found_relationship_clusters:
            nodes_df = self.nodes_df[self.nodes_df["node_id"].isin(cluster)].copy()
            non_sibling_cluster_size = len(nodes_df)
            print(f"Non-sibling cluster size: {non_sibling_cluster_size}")

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
            print(f"Found {len(sibling_nodes)} sibling nodes.")
            # add siblings until the token count exceeds the threshold or number of siblings added is
            # greater than 70% of the initial cluster size.
            while (
                self._tokens_under_threshold(nodes_df)
                and not sibling_nodes.empty
                and len(nodes_df) < non_sibling_cluster_size * 1.5
            ):
                # Pop a sibling node and add it to nodes_df
                sibling_node = sibling_nodes.iloc[0]
                sibling_nodes = sibling_nodes.iloc[1:]
                # Only add if not already in nodes_df
                if sibling_node["node_id"] not in nodes_df["node_id"].values:
                    nodes_df.loc[len(nodes_df)] = sibling_node
            print(f"Added {len(nodes_df) - non_sibling_cluster_size} sibling nodes")
            self.node_clusters.append(nodes_df)

    def llm_generate_testset(self):
        """
        Use the LLM to generate queries and answers for each node cluster, then save the results.
        """
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

        chain = final_prompt | self.llm.with_structured_output(ResponseModel)
        return self.invoke_chain(chain, self.doc_text_column)

    def invoke_chain(self, chain, doc_text_column: str):
        results = []
        for i, node_cluster_df in enumerate(self.node_clusters):
            documents_text = ""
            documents_text_list = []
            for idx, (_, row) in enumerate(node_cluster_df.iterrows(), 1):
                row_doc_text = row[doc_text_column]
                if row_doc_text:
                    documents_text += f'## Document_{idx}\n\n{row_doc_text}\n\n'
                    documents_text_list.append(row_doc_text)
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
                        "context_docs_list": documents_text_list,
                        "query": result.query,
                        "answer": result.answer,
                        "citations": [c.model_dump() for c in result.citations],
                        "document_ids": node_cluster_df["id"].tolist(),
                        "node_ids": node_cluster_df["node_id"].tolist(),
                    }
                )
            except Exception as e:
                print(f"Error processing cluster {i}: {e}")

        # Save results to a file
        output_path = self.output_dir / f"generated_testset_{self.testset_name}.json"
        pd.DataFrame(results).to_json(output_path, orient="records", indent=2)

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
        """
        Main entry point: find clusters, generate node clusters, run LLM, and save all outputs.
        """
        self.found_relationship_clusters = self.find_relationship_clusters()
        self._cluster_reporting()
        self._generate_cluster_info_parquet(
            self.relationship_cluster_info_dfs, "relationship_clusters_info"
        )
        self.get_node_clusters()
        self._generate_cluster_info_parquet(self.node_clusters, "node_clusters_for_llm")
        self.copy_kg_data_to_testset_dir()
        self.llm_generate_testset()


class TransformTestset:
    def __init__(self, testset_dirname: str):
        self.output_dir = testset_dir / testset_dirname
        testset_file_path = self.output_dir / "generated_testset.json"
        if not testset_file_path.exists():
            raise FileNotFoundError(f"Testset file not found: {testset_file_path}")
        with open(testset_file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def create_subsample_testset(self) -> None:
        """
        Create a subset of the testset taking every other sample (even cluster_id only),
        and save as {name}__100_subsample.json in the same output directory.
        """
        # Keep only even cluster_id
        subsample = [item for item in self.data if item.get("cluster_id", 0) % 2 == 0]

        # Save to new file
        subsample_file = self.output_dir / f"generated_testset__100_subsample.json"
        with open(subsample_file, "w", encoding="utf-8") as f:
            json.dump(subsample, f, indent=2, ensure_ascii=False)

        print(f"Subset testset saved to {subsample_file}")

    def create_query_set(self):
        "Get all the queries from the testset and save as a json array of strings to a file"
        queries = [item["query"] + "\n\n" + item["answer"] for item in self.data]
        with open(self.output_dir / "queries.json", "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)

    def create_query_set_text_file(self):
        "Get all the queries from the testset and save as a json array of strings to a file"
        queries = [item["query"] + "\n\n" + item["answer"] for item in self.data]
        with open(self.output_dir / "queries.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(queries))

    def paraphrase_questions(
        self,
        questions: list[str],
        model_name: str = "kalpeshk2011/dipper-paraphraser-xxl",
        lexical_diversity: float = 0.5,
        order_diversity: float = 0.3,
        max_length: int = 512,
        num_return_sequences: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> list[str]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        wrapped_questions = [f"<sent>{q}</sent>" for q in questions]

        inputs = tokenizer(
            wrapped_questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            diversity_penalty=lexical_diversity,  # Controls lexical diversity
            repetition_penalty=1.2
            + order_diversity,  # Indirectly controls order diversity
        )

        paraphrased_questions = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        return paraphrased_questions

    if __name__ == "__main__":
        input_questions = [
            "How do the technical specifications, WAN connectivity options, and throughput limitations compare between the Peplink Balance 20, Balance 30, and Balance 50 models, and what are the implications for users considering an upgrade to support higher-speed connections like Starlink?",
            "How can a network administrator centrally manage firewall rules across multiple Peplink routers using InControl2, including country-based blocking, and what are key operational requirements and verification steps?",
            "How can a technician monitor and interpret the power supply input voltage and GPIO status on a Peplink router, and what steps should they take if voltage-related instability or restarts occur?",
        ]

        paraphrased = paraphrase_questions(input_questions)

        for idx, q in enumerate(paraphrased):
            print(f"Original:\n{input_questions[idx]}\n")
            print(f"Paraphrased:\n{q}\n")
            print("-" * 100)
