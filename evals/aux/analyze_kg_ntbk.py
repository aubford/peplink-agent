# %% TOP
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
from pathlib import Path
from evals.analytics_utils import (
    extract_nodes_to_file,
    extract_relationships_to_file,
    meta_prop,
    gen_nodes_parquet,
)

# Get the directory of the current file
evals_dir = Path(__file__).parent.parent
output_dir = evals_dir / "output"

target_kg_path = output_dir / "kg_output__LATEST.json"
latest_nodes_path = output_dir / "__nodes_LATEST.json"
latest_relationships_path = output_dir / "__relationships_LATEST.json"
isolated_nodes_path = output_dir / "__isolated_nodes.json"
duplicate_nodes_path = output_dir / "__duplicate_nodes.json"


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


def sample_one_document_node_and_one_chunk_and_relationships(
    kg: KnowledgeGraph,
) -> None:
    # Find the node with the specified ID
    document_nodes = [node for node in kg.nodes if node.type == NodeType.DOCUMENT.value]
    chunk_nodes = [node for node in kg.nodes if node.type == NodeType.CHUNK.value]
    nodes = [random.choice(document_nodes), random.choice(chunk_nodes)]

    # Find all relationships involving this node
    relationships = [
        rel for rel in kg.relationships if rel.source in nodes or rel.target in nodes
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

    relationships_data = []
    for rel in relationships:
        source_content = rel.source.properties.get(
            "page_content", "**** No page content available ****"
        )
        target_content = rel.target.properties.get(
            "page_content", "**** No page content available ****"
        )
        rel_data = rel.model_dump()
        rel_data["source_content"] = source_content
        rel_data["target_content"] = target_content
        relationships_data.append(rel_data)

    for node in nodes:
        node.properties["summary_embedding"] = node.properties["summary_embedding"][:5]
    with open(f"{output_dir}/__nodes_and_relationships_example.json", "w") as f:
        json.dump(
            {
                "nodes": [node.model_dump() for node in nodes],
                "relationships": relationships_data,
            },
            f,
            cls=UUIDEncoder,
            indent=2,
        )


def truncate_embedding_lists(node: dict[str, t.Any]) -> dict[str, t.Any]:
    """Truncate all vector/embedding lists to 2 elements recursively.

    Args:
        node: Node dictionary containing properties to process
    """
    for key, value in node.get("properties", {}).items():
        if (
            isinstance(value, list)
            and len(value) > 2
            and all(isinstance(x, (int, float)) for x in value[:5])
        ):
            node["properties"][key] = value[:2]

        # Process nested document_metadata properties
        if key == "document_metadata" and isinstance(value, dict):
            for meta_key, meta_value in value.items():
                if (
                    isinstance(meta_value, list)
                    and len(meta_value) > 2
                    and all(isinstance(x, (int, float)) for x in meta_value[:5])
                ):
                    node["properties"]["document_metadata"][meta_key] = meta_value[:2]

    return node


def get_node_metadata(node: dict[str, t.Any]) -> dict[str, t.Any]:
    return truncate_embedding_lists(node)["properties"]["document_metadata"]


# %%

################## CREATE NODES-ONLY FILE ######################################################################################################################################

# Extract nodes from latest KG
extract_nodes_to_file(target_kg_path, latest_nodes_path)

# %%

kg = KnowledgeGraph.load(target_kg_path)
# %%
gen_nodes_parquet(kg, output_dir / "__nodes_LATEST.parquet")


# %% ################## CREATE RELATIONSHIPS-ONLY FILE ######################################################################################################################################

# Extract relationships from latest KG
extract_relationships_to_file(target_kg_path, latest_relationships_path)

# %% ################## GET ALL NODES THAT HAVE MORE THAN ONE NON-SIBLING RELATIONSHIP ######################################################################################################################################


def get_nodes_with_multiple_relationships(
    relationships: list[Relationship],
) -> None:
    """Find all nodes with more than one relationship and return their counts.

    Args:
        relationships: List of relationship dictionaries

    Returns:
        Dictionary with node ids as keys and count of relationships as values
    """
    node_relationship_counts = {}

    # Count relationships for each node
    for rel in relationships:
        print(rel.type)
        if "sibling" in rel.type:
            continue
        source_id = rel.source.id
        target_id = rel.target.id

        # Count source node
        if source_id:
            node_relationship_counts[source_id] = (
                node_relationship_counts.get(source_id, 0) + 1
            )

        # Count target node
        if target_id:
            node_relationship_counts[target_id] = (
                node_relationship_counts.get(target_id, 0) + 1
            )

    # Filter to only nodes with multiple relationships
    multi_rel_nodes = {
        node_id: count
        for node_id, count in node_relationship_counts.items()
        if count > 1
    }

    print(f"Number of nodes with multiple relationships: {len(multi_rel_nodes)}")
    # Print each node with multiple relationships, sorted by count in descending order
    sorted_nodes = sorted(multi_rel_nodes.items(), key=lambda x: x[1], reverse=True)
    for node_id, count in sorted_nodes:
        print(f"{node_id}: {count}")


kg = KnowledgeGraph.load(target_kg_path)
relationships_list = kg.relationships

get_nodes_with_multiple_relationships(relationships_list)

# %% ################## COUNT TRANSFORMED NODES ######################################################################################################################################

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


# %% ##################  ANALYZE DOCUMENT NODES ######################################################################################################################################

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
        connected_nodes.add(rel["source"])
        connected_nodes.add(rel["target"])

    # Find complete node objects that don't appear in any relationships
    isolated_nodes = [
        node for node in kg_data["nodes"] if node["id"] not in connected_nodes
    ]

    return isolated_nodes


# Load and analyze the knowledge graph
with open(target_kg_path) as f:
    kg_data = json.load(f)
    isolated_nodes = find_isolated_nodes(kg_data)

print(f"Number of isolated nodes: {len(isolated_nodes)}")
print("\nIsolated nodes found. Saving to JSON file...")

# Save isolated nodes to JSON file
with open(isolated_nodes_path, "w") as f:
    json.dump({"isolated_nodes": isolated_nodes}, f, indent=2)

print(f"Saved {len(isolated_nodes)} isolated nodes to {isolated_nodes_path}")

# %% ##################  ANALYZE ISOLATED NODES ######################################################################################################################################

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

# %% #################  VISUALIZER ######################################################################################################################################
from pyvis.network import Network

net = Network("1800px", "1200px")


def visualize_knowledge_graph() -> None:
    kg_data = KnowledgeGraph.load(target_kg_path)
    # Add nodes to the network
    net.add_nodes([str(node.id) for node in kg_data.nodes])
    # Add edges using node IDs (as strings) instead of Node objects
    for r in kg_data.relationships:
        net.add_edge(str(r.source.id), str(r.target.id))
    # net.toggle_physics(False)
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=1000,
        spring_strength=0.08,
        damping=0.4,
        overlap=1,
    )
    net.show_buttons(filter_=["physics"])
    net.show("visualizer.html", notebook=False)


visualize_knowledge_graph()
