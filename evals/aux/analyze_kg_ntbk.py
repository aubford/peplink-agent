# %% TOP
from ragas.testset.graph import (
    KnowledgeGraph,
    Node,
    Relationship,
)
import json
import typing as t
import random
from evals.evals_utils import (
    output_dir,
    dict_meta_prop,
    kg_json_path,
)

isolated_nodes_path = output_dir / "__isolated_nodes.json"


def get_single_node_and_its_relationships(
    kg: KnowledgeGraph, node_id: str | None = None
) -> tuple[Node, list[tuple[Relationship, str, str]]]:
    """Get a single node and its relationships from the knowledge graph.
    Returns:
        Tuple containing (node, relationships_with_content) where:
        - node: A randomly selected node object
        - relationships_with_content: List of tuples containing (relationship, source_content, target_content)
          where source_content and target_content are the page_content of the source and target nodes
    """
    # Find the node with the specified ID
    nodes: list[Node] = kg.nodes
    if node_id:
        print(f"Finding node with ID: {node_id}")
        node = next((node for node in nodes if str(node.id) == str(node_id)), None)
        if node is None:
            raise ValueError(f"Node with ID {node_id} not found")
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


kg = KnowledgeGraph.load(kg_json_path)
relationships_list = kg.relationships

get_nodes_with_multiple_relationships(relationships_list)


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
with open(kg_json_path) as f:
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
        source_file = dict_meta_prop(node, "source_file")
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
    kg_data = KnowledgeGraph.load(kg_json_path)
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
