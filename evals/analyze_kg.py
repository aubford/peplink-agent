# %% ##########################################
from ragas.testset.graph import (
    NodeType,
    KnowledgeGraph,
    Node,
    Relationship,
    UUIDEncoder,
)
from ragas.utils import num_tokens_from_string
import pandas as pd
import json
import numpy as np
import typing as t
import random
import os
from pathlib import Path

# Get the directory of the current file
current_dir = Path(__file__).parent
output_dir = current_dir / "output"

# Update paths to be relative to the current file
latest_kg_path = str(output_dir / "kg_output_LATEST.json")
latest_nodes_path = str(output_dir / "__nodes_LATEST.json")
isolated_nodes_path = str(output_dir / "__isolated_nodes.json")
duplicate_nodes_path = str(output_dir / "__duplicate_nodes.json")

"""
Learnings from analysis of KG output:
- Nodes w/ page_content less than 500 tokens are added as nodes to the KG but they are otherwise completely ignored by the algorithm. They have no relationships or computed properties.
- Other nodes have headlines, summary and summary_embedding properties added to them and spawn chunk nodes. They also have relationships.
"""

kg = KnowledgeGraph.load(latest_kg_path)
kg_relationships = kg.relationships
kg_nodes = kg.nodes
kg_nodes = kg.nodes


def meta_prop(node: dict[str, dict[str, t.Any]], prop: str) -> t.Any:
    return node["properties"]["document_metadata"][prop]


def clean_node_properties(node: dict[str, t.Any]) -> dict[str, t.Any]:
    """Clean node properties by removing summary embeddings and null/NaN values.

    Args:
        node: Node dictionary containing properties to clean

    Returns:
        Cleaned node dictionary with filtered properties
    """
    cleaned_node = node.copy()

    # Remove summary_embedding if it exists
    if "summary_embedding" in cleaned_node.get("properties", {}):
        del cleaned_node["properties"]["summary_embedding"]

    if "document_metadata" in cleaned_node["properties"]:
        metadata = cleaned_node["properties"]["document_metadata"]
        # Remove null and NaN values
        cleaned_metadata = {
            k: v
            for k, v in metadata.items()
            if v is not None and (not isinstance(v, float) or not pd.isna(v))
        }
        cleaned_node["properties"]["document_metadata"] = cleaned_metadata

    return cleaned_node


def clean_and_write_nodes(
    nodes: list[dict[str, t.Any]], output_path: str, is_rel: bool
) -> None:
    """Clean node list and write to JSON file.

    Args:
        nodes: List of node dictionaries to clean and save
        output_path: Path to save the JSON file
        is_rel: Whether the nodes are relationships
    """
    if is_rel:
        cleaned_nodes = [
            {
                "source": clean_node_properties(node["source"]),
                "target": clean_node_properties(node["target"]),
                **node,
            }
            for node in nodes
        ]
    else:
        cleaned_nodes = [clean_node_properties(node) for node in nodes]

    with open(output_path, "w") as f:
        json.dump({"cleaned_nodes": cleaned_nodes}, f, indent=2)

    print(f"Saved {len(cleaned_nodes)} cleaned nodes to {output_path}")


def get_single_node_and_its_relationships(
    kg: KnowledgeGraph, id: str | None = None
) -> tuple[Node, list[tuple[Relationship, str, str]]]:
    """Get a single node and its relationships from the knowledge graph.

    Args:
        kg: KnowledgeGraph object

    Returns:
        Tuple containing (node, relationships_with_content) where:
        - node: A randomly selected node object
        - relationships_with_content: List of tuples containing (relationship, source_content, target_content)
          where source_content and target_content are the page_content of the source and target nodes
    """
    # Find the node with the specified ID
    nodes: list[Node] = kg.nodes
    if id:
        print(f"Finding node with ID: {id}")
        node = next((node for node in nodes if str(node.id) == str(id)), None)
        if node is None:
            raise ValueError(f"Node with ID {id} not found")
    else:
        node = random.choice(nodes)

    # Find all relationships involving this node
    relationships = [
        rel
        for rel in kg.relationships
        if str(rel.source.id) == str(node.id) or str(rel.target.id) == str(node.id)
    ]

    # Add page_content for source and target nodes
    relationships_with_content = []
    for rel in relationships:
        source_content = rel.source.properties.get(
            "page_content", "No page content available"
        )
        target_content = rel.target.properties.get(
            "page_content", "No page content available"
        )
        relationships_with_content.append((rel, source_content, target_content))

    return node, relationships_with_content


# %% ################## SIMPLE KG ACCESS ########################
kg_55 = KnowledgeGraph.load(
    f"{output_dir}/kg_output__n_55__sample_with_custom_transforms__03_05_14_56.json"
)
kg_1000 = KnowledgeGraph.load(
    f"{output_dir}/kg_output__n_1112__sample_with_custom_transforms__03_04_22_21.json"
)
print(f"55: {len(kg_55.nodes)}")
print(f"1000: {len(kg_1000.nodes)}")
print(f"55: {len(kg_55.relationships)}")
print(f"1000: {len(kg_1000.relationships)}")

# %% ################## SAMPLE SINGLE NODE AND ITS RELATIONSHIPS ########################
# Get a random node and its relationships and print to json file
sn_kg = KnowledgeGraph.load(f"{output_dir}/__strict_kg.json")
# sn_kg = KnowledgeGraph.load(f"{output_dir}/kg_output__n_1112__sample_with_custom_transforms__03_04_22_21.json")
node, relationships_with_content = get_single_node_and_its_relationships(
    sn_kg, id="dd380401-a68c-429d-b322-5b92c2e7e69e"
)

# Prepare data for JSON serialization
relationships_data = []
for rel, source_content, target_content in relationships_with_content:
    rel_data = rel.model_dump()
    rel_data["source_content"] = source_content
    rel_data["target_content"] = target_content
    relationships_data.append(rel_data)

with open(f"{output_dir}/__single_node_and_relationships.json", "w") as f:
    node_dump = node.model_dump()
    del node_dump["properties"]["summary_embedding"]
    node_dump["properties"]["page_content"] = node_dump["properties"]["page_content"][
        :100
    ]
    json.dump(
        {"node": node_dump, "relationships": relationships_data},
        f,
        cls=UUIDEncoder,
        indent=2,
    )


# %% ############### WEED OUT LOW-QUALITY RELATIONSHIPS ########################

similarity_threshold = 0.91
overlap_threshold = 0.15
passing_sim_relationships = [
    rel
    for rel in kg_relationships
    if "summary_similarity" not in rel.properties
    or 0.99 > rel.properties["summary_similarity"] > similarity_threshold
]
passing_overlap_relationships = [
    rel
    for rel in passing_sim_relationships
    if "entities_overlap_score" not in rel.properties
    or rel.properties["entities_overlap_score"] > overlap_threshold
    and len(rel.properties["overlapped_items"]) > 2
]
# Prune any nodes that have no relationships
connected_nodes = [rel.source for rel in passing_overlap_relationships] + [
    rel.target for rel in passing_overlap_relationships
]
non_isolated_nodes = [node for node in kg_nodes if node in connected_nodes]

print(f"kg_relationships: {len(kg_relationships)}")
print(f"after sim filter: {len(passing_sim_relationships)}")
print(f"after overlap filter: {len(passing_overlap_relationships)}")
print(f"start nodes: {len(kg_nodes)}")
print(f"filtered nodes: {len(non_isolated_nodes)}")

filtered_kg = KnowledgeGraph(
    nodes=non_isolated_nodes, relationships=passing_overlap_relationships
)
filtered_kg.save(f"{output_dir}/__strict_kg.json")

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

# Count docs and chunks
doc_count = sum(1 for node in nodes_list if node.get("type") == NodeType.DOCUMENT.value)
chunk_count = sum(1 for node in nodes_list if node.get("type") == NodeType.CHUNK.value)

# Count those with multiple properties
doc_multi_props = sum(
    1
    for node in nodes_list
    if node.get("type") == NodeType.DOCUMENT.value
    and len(node.get("properties", {})) > 2
)
chunk_multi_props = sum(
    1
    for node in nodes_list
    if node.get("type") == NodeType.CHUNK.value and len(node.get("properties", {})) > 2
)

print(f"Documents: {doc_count} (with multiple props: {doc_multi_props})")
print(f"Chunks: {chunk_count} (with multiple props: {chunk_multi_props})")


# %% ##################  ANALYZE DOCUMENT NODES ###################################################

with open(latest_nodes_path) as f:
    nodes_list = json.load(f)

for node_type in [NodeType.DOCUMENT, NodeType.CHUNK]:
    # Get all nodes of current type
    nodes = [node for node in nodes_list if node.get("type") == node_type.value]

    # Get unique property key sets and collect page_content token counts
    prop_key_sets = set()
    token_counts = {}
    for node in nodes:
        prop_keys = frozenset(node.get("properties", {}).keys())
        prop_key_sets.add(prop_keys)

        # Track token counts for this key combination
        if prop_keys not in token_counts:
            token_counts[prop_keys] = []
        page_content = node.get("properties", {}).get("page_content", "")
        token_counts[prop_keys].append(num_tokens_from_string(page_content))

    # Print each unique combination with count and token count stats
    print(f"\n{node_type.value} property key combinations:")
    for prop_keys in sorted(prop_key_sets, key=lambda x: len(x)):
        count = sum(
            1
            for node in nodes
            if frozenset(node.get("properties", {}).keys()) == prop_keys
        )
        tokens = np.array(token_counts[prop_keys])

        print(
            f"\nKeys ({count} nodes, tokens avg: {tokens.mean():.0f}, "
            f"min: {tokens.min()}, max: {tokens.max()}):"
        )
        for key in sorted(prop_keys):
            print(f"  - {key}")


# %% ################## GET ISOLATED NODES ###################################################

print(
    "-------------------------------- ISOLATED NODES --------------------------------"
)


def find_isolated_nodes(kg_data: dict[str, t.Any]) -> list[dict[str, t.Any]]:
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
        source_file = meta_prop(node, "source_file")
        key = f"{node['type']}__{source_file}"
        source_counts[key] = source_counts.get(key, 0) + 1

    print("\nSource file counts:")
    for source, count in sorted(
        source_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{source}: {count}")

# %% ##################  DUPLICATE NODES ###################################################


def find_duplicate_id_nodes(kg_data: dict[str, t.Any]) -> list[dict[str, t.Any]]:
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
    duplicate_nodes.sort(key=lambda x: x["id"])  # type: ignore
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

# %% ##################  COUNT SOURCE FILES of DUPLICATE NODES ###################################################

dupe_source_counts = {}
for node in duplicate_nodes:
    source_file = meta_prop(node, "source_file")
    key = f"{node['type']}__{source_file}"
    dupe_source_counts[key] = dupe_source_counts.get(key, 0) + 1

print("\nSource file counts:")
for source, count in sorted(
    dupe_source_counts.items(), key=lambda x: x[1], reverse=True
):
    print(f"{source}: {count}")


# %% ##################  GET RELATIONSHIPS FOR DUPLICATE NODES ###################################################


def find_relationships_for_nodes(
    kg_data: dict[str, t.Any], node_ids: set[str]
) -> list[dict[str, t.Any]]:
    """Find all relationships where either source or target is in the given node IDs.

    Args:
        kg_data: Dictionary containing nodes and relationships data
        node_ids: Set of node IDs to find relationships for

    Returns:
        List of relationship objects that involve the specified nodes
    """
    related = []
    for rel in kg_data["relationships"]:
        source_id = rel.get("source", {}).get("id", "")
        target_id = rel.get("target", {}).get("id", "")
        if source_id in node_ids or target_id in node_ids:
            related.append(rel)
    return related


duplicate_ids = {duplicate_nodes[0]["id"]}
duplicate_relationships = find_relationships_for_nodes(kg_data, duplicate_ids)

print(
    f"\nNumber of relationships involving duplicate nodes: {len(duplicate_relationships)}"
)

clean_and_write_nodes(
    duplicate_relationships, "evals/output/__duplicate_relationships.json", is_rel=True
)


# %% ##################  GET DUPLICATE YOUTUBE NODES ###################################################
youtube_dupes = [
    node for node in duplicate_nodes if "youtube" in meta_prop(node, "source_file")
]
print(f"\nNumber of duplicate YouTube nodes: {len(youtube_dupes)}")

# Save duplicate YouTube nodes to JSON file
youtube_nodes_path = "evals/output/__duplicate_youtube_nodes.json"

# Replace the YouTube dupes writing code with:
clean_and_write_nodes(youtube_dupes, youtube_nodes_path, is_rel=False)


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

    dpi = 600
    # Ultra-high resolution setup
    plt.figure(figsize=(60, 48), dpi=dpi)  # 3x original size in each dimension
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
    plt.savefig("evals/output/knowledge_graph.png", dpi=dpi, bbox_inches="tight")
    plt.show()


# Load and visualize the knowledge graph
with open(latest_kg_path) as f:
    kg_data = json.load(f)
    visualize_knowledge_graph(kg_data)
