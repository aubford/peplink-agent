# %% ##########################################
from langchain_core.documents import Document
import pandas as pd
from util.document_utils import df_to_documents
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from datetime import datetime
from ragas.utils import num_tokens_from_string

from load.reddit_general.reddit_general_load import RedditGeneralLoad
from load.reddit.reddit_load import RedditLoad
from load.html.html_load import HtmlLoad
from load.web.web_load import WebLoad
from load.youtube.youtube_load import YoutubeLoad
from load.mongo.mongo_load import MongoLoad
import json

latest_kg_path = "evals/output/kg_output_LATEST.json"
latest_nodes_path = "evals/output/__nodes_LATEST.json"
isolated_nodes_path = "evals/output/__isolated_nodes.json"
duplicate_nodes_path = "evals/output/__duplicate_nodes.json"


# %% ##################  CREATE NODES-ONLY FILE ########################
def extract_nodes_to_file(input_path: str, output_path: str) -> None:
    """Extract nodes from knowledge graph JSON and save to new file."""
    with open(input_path) as f:
        kg_data = json.load(f)
        nodes = kg_data["nodes"]

    with open(output_path, "w") as f:
        json.dump(nodes, f, indent=2)


# Extract nodes from latest KG
extract_nodes_to_file(latest_kg_path, latest_nodes_path)

# %% ##################  COUNT TRANSFORMED NODES ########################

with open(latest_nodes_path) as f:
    nodes_list = json.load(f)

nodes_with_multiple_props = sum(
    1 for node in nodes_list if len(node.get("properties", {})) > 2
)
print(f"Total number of nodes: {len(nodes_list)}")
print(f"Transformed nodes with >2 properties: {nodes_with_multiple_props}")


# %% #################  VISUALIZER ######################################################################
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any


def visualize_knowledge_graph(kg_data: Dict[str, Any]) -> None:
    """Create a minimal visual representation of the knowledge graph.

    Args:
        kg_data: Dictionary containing nodes and relationships data
        max_nodes: Maximum number of nodes to display to prevent overcrowding
    """
    G = nx.DiGraph()

    # Add nodes (limited to max_nodes)
    nodes = kg_data["nodes"]
    node_mapping = {}  # Map node objects to their IDs

    for node in nodes:
        node_id = node.get("id", "")
        node_type = node.get("type", "")
        G.add_node(node_id, type=node_type)
        node_mapping[str(node)] = node_id

    # Add relationships only between included nodes
    node_ids = {node.get("id", "") for node in nodes}
    for rel in kg_data["relationships"]:
        source_id = rel.get("source", {}).get("id", "")
        target_id = rel.get("target", {}).get("id", "")
        if source_id in node_ids and target_id in node_ids:
            G.add_edge(source_id, target_id)

    # Ultra-high resolution setup
    plt.figure(figsize=(60, 48), dpi=900)  # 3x original size in each dimension
    pos = nx.spring_layout(G, k=6, iterations=200)  # Increased spacing parameters

    # Draw nodes with increased visibility
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color="black",
        node_size=1,  # Increased node size
        arrows=False,
        edge_color="gray",
        width=0.2,  # Thinner edges for better clarity
        alpha=0.4,  # Slightly increased opacity
    )

    plt.axis("off")
    plt.tight_layout()

    # Save ultra-high-res PNG
    plt.savefig("evals/output/knowledge_graph.png", dpi=900, bbox_inches="tight")
    plt.show()


# Load and visualize the knowledge graph
with open(latest_kg_path) as f:
    kg_data = json.load(f)
    visualize_knowledge_graph(kg_data)

# %% ##################  ISOLATED NODES ###################################################


def find_isolated_nodes(kg_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Find nodes that have no relationships in the knowledge graph.

    Args:
        kg_data: Dictionary containing nodes and relationships data

    Returns:
        List of complete node objects that have no relationships
    """
    # Get all nodes that participate in relationships
    connected_nodes = set()
    for rel in kg_data["relationships"]:
        connected_nodes.add(rel.get("source", {}).get("id", ""))
        connected_nodes.add(rel.get("target", {}).get("id", ""))

    # Find complete node objects that don't appear in any relationships
    isolated_nodes = [
        node for node in kg_data["nodes"] if node.get("id", "") not in connected_nodes
    ]

    return isolated_nodes


# Load and analyze the knowledge graph
with open(latest_kg_path) as f:
    kg_data = json.load(f)
    isolated_nodes = find_isolated_nodes(kg_data)

print(f"Number of isolated nodes: {len(isolated_nodes)}")
print("\nIsolated nodes found. Saving to JSON file...")

# Save isolated nodes to JSON file
with open(isolated_nodes_path, "w") as f:
    json.dump({"isolated_nodes": isolated_nodes}, f, indent=2)

print(f"Saved {len(isolated_nodes)} isolated nodes to {isolated_nodes_path}")

# %% ##################  ANALYZE ISOLATED NODES ###################################################

with open(isolated_nodes_path, "r") as f:
    isolated_nodes = json.load(f)["isolated_nodes"]

    # Get page content lengths for isolated nodes
    page_content_lengths = [
        len(node["properties"]["page_content"].split()) for node in isolated_nodes
    ]
    print(f"\nPage content length stats:")
    print(
        f"Average length: {sum(page_content_lengths) / len(page_content_lengths):.1f}"
    )
    print(f"Min length: {min(page_content_lengths)}")
    print(f"Max length: {max(page_content_lengths)}")

    # Count frequency of each source file
    source_counts = {}
    for node in isolated_nodes:
        metadata = node["properties"]["document_metadata"]
        source_file = metadata["source_file"]
        key = f"{node['type']}__{source_file}"
        source_counts[key] = source_counts.get(key, 0) + 1

    print("\nSource file counts:")
    for source, count in sorted(
        source_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{source}: {count}")

# %% ##################  DUPLICATE NODES ###################################################


def find_duplicate_id_nodes(kg_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Find nodes with duplicate IDs in the knowledge graph.

    Args:
        kg_data: Dictionary containing nodes and relationships data

    Returns:
        List of node objects that have duplicate IDs, including all instances
    """
    id_counts = {}
    id_to_nodes = {}

    # First pass: count IDs and collect nodes
    for node in kg_data["nodes"]:
        node_id = node["id"]
        id_counts[node_id] = id_counts.get(node_id, 0) + 1
        id_to_nodes.setdefault(node_id, []).append(node)

    # Collect all nodes that have duplicate IDs
    duplicate_nodes = []
    for node_id, count in id_counts.items():
        if count > 1:
            duplicate_nodes.extend(id_to_nodes[node_id])

    # Sort duplicate nodes by ID
    duplicate_nodes.sort(key=lambda x: x["id"])
    return duplicate_nodes


# Load and analyze the knowledge graph
with open(latest_kg_path) as f:
    kg_data = json.load(f)
    duplicate_nodes = find_duplicate_id_nodes(kg_data)

print(f"Number of duplicate ID nodes: {len(duplicate_nodes)}")
print("\nDuplicate ID nodes found. Saving to JSON file...")

# Save duplicate ID nodes to JSON file
with open(duplicate_nodes_path, "w") as f:
    json.dump({"duplicate_nodes": duplicate_nodes}, f, indent=2)

# %% ##################  COUNT SOURCE FILES ###################################################

dupe_source_counts = {}
for node in duplicate_nodes:
    metadata = node["properties"]["document_metadata"]
    source_file = metadata["source_file"]
    key = f"{node['type']}__{source_file}"
    dupe_source_counts[key] = dupe_source_counts.get(key, 0) + 1

print("\nSource file counts:")
for source, count in sorted(
    dupe_source_counts.items(), key=lambda x: x[1], reverse=True
):
    print(f"{source}: {count}")
