from pathlib import Path
import pandas as pd
import random
import numpy as np
import json
import re
from typing import Optional, Set, List, FrozenSet, Union
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


"""
TODO:
- Add sibling relationships to the clusters
    - 1. Ask LLM if it's a good cluster
    - 2. If so, find the most common node
    - 3. Append all the siblings of that node to the cluster
"""


class GenerateTestSet:

    system_prompt = """
    You are a technical content analyst specializing in IT networking and Pepwave products.
    Your task is to analyze a set of documents and generate a multifaceted multi-hop query that a technician might ask based solely on the information in those documents.
    Each set of documents contains related information. Identify a topic, product, technology, or concept that exists in many of the documents. There should be a diversity of useful technical information about that topic in multiple documents.
    Create a query from the technical information shared on that topic in the documents that incorporates different information from as many documents as possible and connects them meaningfully.
    The query should be based solely on the documents provided.
    Once you have created the query, write a detailed answer to the query using only the information in the documents.
    """

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
        self.node_clusters: List[pd.DataFrame] = []

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

    def tack_on_node_col(self, clusters_df: pd.DataFrame, col: str) -> pd.DataFrame:
        clusters_df[f"source_{col}"] = clusters_df["source_id"].map(
            self.nodes_df.set_index("node_id")[col]
        )
        clusters_df[f"target_{col}"] = clusters_df["target_id"].map(
            self.nodes_df.set_index("node_id")[col]
        )
        return clusters_df

    def _generate_cluster_info_parquet(self):
        """
        Generate a parquet file with the cluster information for inspection.
        This may have duplicate clusters that don't exist in the found_clusters set.
        """
        df = self.cluster_dfs_sample
        for i, cdf in enumerate(df):
            cdf["cluster"] = i

        sample_df = pd.concat(df)
        sample_df = self.tack_on_node_col(sample_df, "page_content")
        sample_df = self.tack_on_node_col(sample_df, "title")
        sample_df = self.tack_on_node_col(sample_df, "technical_summary")
        sample_df.to_parquet(self.output_dir / f"__clusters_sample.parquet")

    def _cluster_reporting(self):
        print("\n" + "-" * 50 + "\n")
        print(f"Clusters count: {len(self.found_clusters)}")
        print(
            f"Total elements in relationship clusters: {sum(len(cluster) for cluster in self.found_clusters)}"
        )

        # Get the set of all unique ids from relationship clusters
        all_unique_ids = set()
        for cluster in self.found_clusters:
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
        For each cluster, get the corresponding nodes andappend siblings up to a max token count.
        """
        for cluster in self.found_clusters:
            nodes_df = self.nodes_df[self.nodes_df["node_id"].isin(cluster)]
            sibling_relationships = self.sibling_df[
                (self.sibling_df["source_id"].isin(cluster))
                | (self.sibling_df["target_id"].isin(cluster))
            ]
            sibling_nodes = self.nodes_df[
                self.nodes_df["node_id"].isin(sibling_relationships["source_id"])
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
        """
        Query the LLM to create test data for each cluster.
        """
        # Define the expected output format
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
        format_instructions = output_parser.get_format_instructions()

        # Initialize the LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

        # Create four examples with different devices and use cases
        example_1_documents = """
DOCUMENT 1:
Title: MAX Transit Duo Router Specifications
Content: The Pepwave MAX Transit Duo supports dual cellular modems with carrier aggregation capability. It features two cellular modems that can operate simultaneously on different carriers, with bandwidth of up to 600 Mbps (300 Mbps per modem). The router includes two ethernet WAN/LAN ports and built-in GPS for vehicle tracking capabilities.

DOCUMENT 2:
Title: MAX Transit Duo Configuration Guide
Content: To enable the MAX Transit Duo's dual cellular capabilities, navigate to Network > WAN > Cellular Settings. In this menu, you can configure each modem independently, including carrier settings, APN configuration, and SIM card priority. For optimal performance, enable bandwidth bonding by going to SpeedFusion > Profiles and creating a new bonding profile.

DOCUMENT 3:
Title: SpeedFusion Bandwidth Bonding Technology
Content: Peplink's SpeedFusion technology allows multiple WAN connections to be combined into a single high-bandwidth connection. This technology works by splitting data packets across multiple paths, then reassembling them at the destination SpeedFusion endpoint. This provides not only increased bandwidth but also connection redundancy, as the session remains active even if one connection fails.

DOCUMENT 4:
Title: Vehicle Deployment Recommendations
Content: For optimal cellular performance in mobile deployments, mount the MAX Transit Duo's antennas on the roof of the vehicle with clear line of sight to the sky. Position the GPS antenna with unobstructed view for accurate location tracking. Configure bandwidth bonding with Hot Failover enabled to maintain session persistence during connection transitions between cell towers or when entering areas with poor coverage.
"""

        example_1_output = """
{
    "query": "How can the Pepwave MAX Transit Duo be configured for optimal performance in mobile vehicles using SpeedFusion technology?",
    "answer": "For optimal performance of the Pepwave MAX Transit Duo in mobile vehicles, several configuration steps are recommended. First, enable the dual cellular capabilities by navigating to Network > WAN > Cellular Settings, where each modem can be configured independently with appropriate carrier settings, APN configuration, and SIM card priority. To utilize the full bandwidth potential of up to 600 Mbps (300 Mbps per modem), enable SpeedFusion bandwidth bonding by going to SpeedFusion > Profiles and creating a new bonding profile. This technology combines multiple WAN connections by splitting data packets across multiple paths and reassembling them at the destination, providing both increased bandwidth and connection redundancy. For physical deployment, mount the antennas on the vehicle roof with clear line of sight, and position the GPS antenna with an unobstructed view for accurate location tracking. Finally, configure Hot Failover within the bandwidth bonding settings to maintain session persistence during connection transitions between cell towers or when entering areas with poor coverage."
}
"""

        example_2_documents = """
DOCUMENT 1:
Title: Balance 20X Router Overview
Content: The Peplink Balance 20X is a dual-WAN router designed for small businesses and branch offices. It features one Gigabit Ethernet WAN port, one cellular WAN module slot, and four Gigabit LAN ports. The router supports Peplink's InControl2 cloud management platform, allowing for remote configuration and monitoring. The Balance 20X can handle up to 900Mbps throughput and supports up to 60 concurrent VPN connections.

DOCUMENT 2:
Title: InControl2 Management Platform
Content: Peplink's InControl2 is a cloud-based management platform that provides centralized configuration, monitoring, and reporting for all Peplink devices. Administrators can remotely manage device settings, view bandwidth usage statistics, and set up alerts for connectivity issues. InControl2 also offers fleet management capabilities, allowing for mass configuration updates and firmware upgrades across multiple devices.

DOCUMENT 3:
Title: OutBound Policy Configuration
Content: The Outbound Policy feature in Peplink routers allows administrators to specify which WAN connection should be used for specific types of traffic. To configure this, go to Network > Outbound Policy in the admin interface. You can create rules based on source IP, destination IP, protocol, or application. For each rule, you can specify whether to use a specific WAN connection, load balance across multiple connections, or enable failover priority.

DOCUMENT 4:
Title: Balance 20X VPN Capabilities
Content: The Balance 20X supports multiple VPN technologies, including Peplink's SpeedFusion VPN, IPsec, OpenVPN, and L2TP. SpeedFusion VPN allows for bandwidth bonding, hot failover, and WAN smoothing across multiple connections. To set up a VPN, go to Network > VPN and select the desired protocol. For SpeedFusion, you'll need to configure a PepVPN profile and specify encryption settings, bandwidth allocation, and failover preferences.
"""

        example_2_output = """
{
    "query": "How can a small business configure outbound traffic policies and VPN connections on a Balance 20X router using InControl2?",
    "answer": "A small business can configure outbound traffic policies on a Balance 20X router by accessing Network > Outbound Policy in the admin interface. This feature allows creation of specific rules based on source IP, destination IP, protocol, or application, with options to use specific WAN connections, load balance across multiple connections, or enable failover priority. For VPN configuration, the Balance 20X supports multiple technologies including SpeedFusion VPN, IPsec, OpenVPN, and L2TP, accessible via Network > VPN in the interface. SpeedFusion VPN setup requires configuring a PepVPN profile with encryption settings, bandwidth allocation, and failover preferences. The router supports up to 60 concurrent VPN connections with throughput up to 900Mbps. All these configurations can be managed remotely using InControl2, Peplink's cloud-based management platform, which provides centralized configuration, monitoring, and reporting capabilities. InControl2 also offers fleet management features for mass configuration updates and firmware upgrades across multiple devices."
}
"""

        example_3_documents = """
DOCUMENT 1:
Title: MAX HD4 Router Features
Content: The Pepwave MAX HD4 is a high-performance router designed for demanding environments. It features four integrated LTE-A Pro cellular modems, eight configurable WAN/LAN ports, and support for Wi-Fi as WAN. The HD4 can deliver up to 1.2 Gbps of combined cellular bandwidth and supports Peplink's SpeedFusion bandwidth bonding and hot failover technologies. It includes a built-in GPS receiver for location tracking in mobile deployments.

DOCUMENT 2:
Title: Band Locking Configuration Guide
Content: To optimize cellular connectivity on Pepwave routers, band locking allows you to restrict modems to specific frequency bands. Navigate to Network > WAN > Cellular Settings, select the modem, and under 'Cellular Band Selection,' choose 'Custom' to manually select bands. In areas with known good coverage on specific bands, locking to those bands can improve connection stability and speed by preventing unnecessary band switching and focusing on the strongest signals.

DOCUMENT 3:
Title: Multi-Cellular Signal Optimization
Content: When deploying multiple cellular connections, antenna placement is critical. For the MAX HD4, use at least two high-gain MIMO antennas per modem, positioned at least 50cm apart to reduce interference. For vehicles, roof-mounted antennas provide the best reception. When operating in rural areas with limited coverage, connect the primary modem to a directional antenna pointed toward the nearest cell tower, while using omnidirectional antennas for the remaining modems to maintain connectivity in changing environments.

DOCUMENT 4:
Title: SpeedFusion Cloud Setup
Content: SpeedFusion Cloud provides an alternative to traditional site-to-site VPN deployments by offering cloud-based SpeedFusion endpoints. To configure, go to SpeedFusion > SpeedFusion Cloud and enter your subscription key. Select the WAN connections to be bonded and the optimal datacenter location. Enable traffic steering to select which traffic types use the SpeedFusion tunnel. For latency-sensitive applications like VoIP, enable WAN Smoothing to reduce jitter and packet loss across connections.
"""

        example_3_output = """
{
    "query": "What steps should be taken to optimize cellular performance on a MAX HD4 router in rural areas using band locking and proper antenna configuration?",
    "answer": "To optimize cellular performance on a MAX HD4 router in rural areas, several configuration steps should be implemented. First, configure band locking by navigating to Network > WAN > Cellular Settings, selecting each modem, and under 'Cellular Band Selection,' choosing 'Custom' to manually select bands. In rural areas with limited coverage, this prevents unnecessary band switching and focuses on the strongest signals for improved connection stability and speed. For antenna configuration, use at least two high-gain MIMO antennas per modem (the HD4 has four integrated LTE-A Pro modems), positioning them at least 50cm apart to reduce interference. The primary modem should be connected to a directional antenna pointed toward the nearest cell tower for strongest signal acquisition, while using omnidirectional antennas for the remaining modems to maintain connectivity as the environment changes. For mobile deployments, roof-mounted antennas provide optimal reception. These configurations can help maximize the HD4's potential to deliver up to 1.2 Gbps of combined cellular bandwidth even in challenging rural environments."
}
"""

        example_4_documents = """
DOCUMENT 1:
Title: BR1 Mini Hardware Specifications
Content: The Pepwave BR1 Mini is a compact single-cellular router designed for IoT and small vehicle applications. It features one cellular modem supporting LTE CAT-6 (300Mbps), one WAN Ethernet port, one LAN port, and 2.4GHz Wi-Fi. The router operates on 12-48V DC power input making it suitable for vehicle installations. The BR1 Mini includes a built-in GPS receiver and supports PepVPN connectivity with SpeedFusion capabilities when connecting to compatible Peplink devices.

DOCUMENT 2:
Title: IoT Deployment Scenarios
Content: For IoT deployments, the BR1 Mini serves as a reliable communication gateway. Common applications include connecting remote sensors, industrial control systems, and surveillance equipment. To optimize power consumption in solar-powered installations, configure the Wi-Fi schedule feature to disable Wi-Fi during non-operational hours. For security, create a custom firewall rule under Network > Firewall to restrict inbound connections and only allow traffic to specific ports needed by your IoT devices.

DOCUMENT 3:
Title: Remote Management for Unmanned Sites
Content: Configure InControl2 for remote management of BR1 Mini routers at unmanned sites by first registering the device using its serial number in the InControl2 portal. Enable the 'Remote Web Admin' feature under System > Admin Security to allow secure remote access to the device's admin interface through InControl2. Set up bandwidth monitoring alerts by configuring thresholds under InControl2 > Monitoring to receive notifications when data usage approaches plan limits, which is essential for cellular-connected devices with limited data plans.

DOCUMENT 4:
Title: GPS and Geofencing Configuration
Content: The BR1 Mini's built-in GPS capabilities allow for location tracking and geofencing applications. To enable GPS, go to System > GPS and check 'Enable GPS'. Configure the GPS reporting interval based on your tracking needs and power constraints. For geofencing, access InControl2 > Monitoring > Map View and create boundary zones by defining coordinates on the map. Configure alerts to trigger when devices enter or leave these predefined zones, which is useful for asset tracking and security purposes.
"""

        example_4_output = """
{
    "query": "How should a BR1 Mini be configured for remote IoT deployments with GPS tracking and power optimization features?",
    "answer": "For remote IoT deployments with GPS tracking and power optimization, the BR1 Mini should be configured with several specific settings. First, enable the GPS functionality by going to System > GPS and checking 'Enable GPS', then configure the GPS reporting interval based on tracking needs and power constraints. For power optimization, especially in solar-powered installations, use the Wi-Fi schedule feature to disable Wi-Fi during non-operational hours. Security should be enhanced by creating custom firewall rules under Network > Firewall to restrict inbound connections and only allow traffic to specific ports needed by the IoT devices. For remote management, register the device using its serial number in the InControl2 portal and enable the 'Remote Web Admin' feature under System > Admin Security. This allows secure remote access to the device's admin interface through InControl2. To monitor data usage on cellular connections with limited data plans, set up bandwidth monitoring alerts by configuring thresholds under InControl2 > Monitoring. Additionally, geofencing can be configured through InControl2 > Monitoring > Map View by creating boundary zones with coordinates, which can trigger alerts when devices enter or leave these predefined zones—useful for asset tracking and security purposes."
}
"""

        # Create separate system and human message templates
        system_message_template = SystemMessagePromptTemplate.from_template(
            "{system_prompt}"
        )

        human_template = """
Here are the documents to analyze:

{documents}

{format_instructions}

Here are some examples:

Example 1:
Documents:
{example_1_documents}

Output:
{example_1_output}

Example 2:
Documents:
{example_2_documents}

Output:
{example_2_output}

Generate a query and answer based on the documents provided above according to the guidelines provided.
"""
        human_message_template = HumanMessagePromptTemplate.from_template(
            human_template
        )

        # Combine into a chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                system_message_template,
                human_message_template,
            ]
        )

        # Process each cluster
        results = []
        for i, nodes_df in enumerate(self.node_clusters):
            # Simply report the index and total count without arithmetic
            print(f"Processing cluster {i} of {len(self.node_clusters) - 1}")

            # Create a formatted string of documents
            documents_text = ""
            for idx, row in nodes_df.iterrows():
                # Use string formatting that doesn't involve arithmetic on idx
                doc_num = str(idx)
                title = row.get('title', '') or row.get(
                    'technical_summary', f'Document {doc_num}'
                )
                documents_text += f"DOCUMENT {doc_num}:\nTitle: {title}\nContent: {row['page_content']}\n\n"

            # Prepare the messages
            chat_messages = prompt.format_messages(
                system_prompt=self.system_prompt,
                documents=documents_text,
                format_instructions=format_instructions,
                example_1_documents=example_1_documents,
                example_1_output=example_1_output,
                example_2_documents=example_2_documents,
                example_2_output=example_2_output,
                example_3_documents=example_3_documents,
                example_3_output=example_3_output,
                example_4_documents=example_4_documents,
                example_4_output=example_4_output,
            )

            # Call the LLM with structured messages
            response = llm.invoke(chat_messages)

            try:
                # Parse the response
                if hasattr(response, 'content') and isinstance(response.content, str):
                    parsed_response = output_parser.parse(response.content)
                    query = parsed_response.get("query", "")
                    answer = parsed_response.get("answer", "")
                else:
                    raise ValueError("Response content is not a string")
            except Exception as e:
                print(f"Error parsing response for cluster {i}: {e}")
                if hasattr(response, 'content'):
                    print(f"Raw response: {response.content}")
                    response_text = response.content
                else:
                    response_text = str(response)
                    print(f"Raw response: {response_text}")

                # Try a simpler JSON parsing as fallback
                try:
                    # Find JSON-like content between curly braces
                    if isinstance(response_text, str):
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        if json_match:
                            json_str = json_match.group(0)
                            parsed_json = json.loads(json_str)
                            query = parsed_json.get("query", "")
                            answer = parsed_json.get("answer", "")
                        else:
                            query = ""
                            answer = ""
                    else:
                        query = ""
                        answer = ""
                except Exception:
                    query = ""
                    answer = ""

            # Store the result
            results.append(
                {
                    "cluster_id": i,
                    "query": query,
                    "answer": answer,
                    "document_ids": nodes_df["node_id"].tolist(),
                }
            )

        # Save results to a file
        output_path = self.output_dir / "generated_testset.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Generated testset saved to {output_path}")
        return results

    def create_testset(self):
        """
        Create a test set by filtering clusters and appending siblings.
        """
        self.found_clusters = self.find_relationship_clusters()
        self._cluster_reporting()
        self._generate_cluster_info_parquet()
        self.get_node_clusters()
        self.llm_generate_testset()
