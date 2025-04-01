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

# Get the directory of the current file
current_dir = Path(__file__).parent
output_dir = current_dir / "output"

# Update paths to be relative to the current file
latest_kg_path = str(
    output_dir / "kg_output__n_200__subsample_with_four_transforms__04_01_15_39.json"
)
latest_nodes_path = str(output_dir / "__nodes_LATEST.json")
isolated_nodes_path = str(output_dir / "__isolated_nodes.json")
duplicate_nodes_path = str(output_dir / "__duplicate_nodes.json")
latest_relationships_path = str(output_dir / "__relationships_LATEST.json")
"""
Learnings from analysis of KG output:
- Nodes w/ page_content less than 500 tokens are added as nodes to the KG but they are otherwise completely ignored by the algorithm. They have no relationships or computed properties.
- Other nodes have headlines, summary and summary_embedding properties added to them and spawn chunk nodes. They also have relationships.
"""


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


################## CREATE NODES-ONLY FILE ######################################################################################################################################
def extract_nodes_to_file(input_path: str, output_path: str) -> None:
    """Extract nodes from knowledge graph JSON and save to new file."""
    with open(input_path) as f:
        kg_data = json.load(f)
        nodes = kg_data["nodes"]

    # Truncate all vector/embedding lists to 2 elements
    for node in nodes:
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
                        node["properties"]["document_metadata"][meta_key] = meta_value[
                            :2
                        ]

    with open(output_path, "w") as f:
        json.dump(nodes, f, indent=2)


# Extract nodes from latest KG
extract_nodes_to_file(latest_kg_path, latest_nodes_path)


################## CREATE RELATIONSHIPS-ONLY FILE ######################################################################################################################################
def extract_relationships_to_file(input_path: str, output_path: str) -> None:
    """Extract relationships from knowledge graph JSON and save to new file."""
    with open(input_path) as f:
        kg_data = json.load(f)
        relationships = kg_data["relationships"]
        nodes = kg_data["nodes"]

    # Create a dictionary mapping node IDs to nodes for faster lookup
    node_map = {node["id"]: node for node in nodes}

    transformed_relationships = []
    for rel in relationships:
        source_node = node_map[rel["source"]]
        target_node = node_map[rel["target"]]
        transformed_relationships.append(
            {
                "TYPE": next(
                    (
                        key.upper().replace('_', ' ')
                        for key, value in rel["properties"].items()
                        if isinstance(value, float)
                    )
                ),
                "id": rel["id"],
                "source_id": rel["source"],
                "target_id": rel["target"],
                "source_page_content": source_node["properties"]["page_content"],
                "target_page_content": target_node["properties"]["page_content"],
                "source_title": source_node["properties"]["document_metadata"]["title"],
                "target_title": target_node["properties"]["document_metadata"]["title"],
                "source_themes": source_node["properties"]["themes"],
                "target_themes": target_node["properties"]["themes"],
                "source_entities": source_node["properties"]["entities"],
                "target_entities": target_node["properties"]["entities"],
                "properties": rel["properties"],
            }
        )

    with open(output_path, "w") as f:
        json.dump(transformed_relationships, f, indent=2)


# Extract relationships from latest KG
extract_relationships_to_file(latest_kg_path, latest_relationships_path)


# %% ################## SAMPLE NODES AND ITS RELATIONSHIPS ######################################################################################################################################
# Get a random node and its relationships and print to json file
sn_kg = KnowledgeGraph.load(f"{output_dir}/__strict_kg.json")
# sn_kg = KnowledgeGraph.load(f"{output_dir}/kg_output__n_1112__sample_with_custom_transforms__03_04_22_21.json")
node, relationships_with_content = get_single_node_and_its_relationships(sn_kg)

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

sample_one_document_node_and_one_chunk_and_relationships(sn_kg)

# %% ############### WEED OUT LOW-QUALITY RELATIONSHIPS ######################################################################################################################################
kg = KnowledgeGraph.load(latest_kg_path)
kg_relationships = kg.relationships
kg_nodes = kg.nodes

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
with open(latest_kg_path) as f:
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

# %% ##################  DUPLICATE NODES ######################################################################################################################################


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

# %% ##################  COUNT SOURCE FILES of DUPLICATE NODES ######################################################################################################################################

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


# %% ##################  GET RELATIONSHIPS FOR DUPLICATE NODES ######################################################################################################################################


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


# %% #################  VISUALIZER ######################################################################################################################################
from pyvis.network import Network

net = Network("1800px", "1200px")


def visualize_knowledge_graph() -> None:
    kg_data = KnowledgeGraph.load(latest_kg_path)
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
