import json
import typing as t
from pathlib import Path

# Get the directory of the current file
current_dir = Path(__file__).parent
output_dir = current_dir / "output"

target_kg_path = output_dir / "kg_output__LATEST.json"
latest_nodes_path = output_dir / "__nodes_LATEST.json"
latest_relationships_path = output_dir / "__relationships_LATEST.json"
isolated_nodes_path = output_dir / "__isolated_nodes.json"
duplicate_nodes_path = output_dir / "__duplicate_nodes.json"


def meta_prop(node: dict[str, dict[str, t.Any]], prop: str) -> t.Any:
    return node["properties"]["document_metadata"].get(prop, None)


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


def extract_nodes_to_file(input_path: Path, output_path: Path) -> None:
    """Extract nodes from knowledge graph JSON and save to new file."""
    print(f"Extracting nodes from {input_path} to {output_path}")
    with open(input_path) as f:
        kg_data = json.load(f)
        nodes = kg_data["nodes"]

    # Truncate all vector/embedding lists to 2 elements
    for node in nodes:
        truncate_embedding_lists(node)

    with open(output_path, "w") as f:
        json.dump(nodes, f, indent=2)


def extract_relationships_to_file(input_path: Path, output_path: Path) -> None:
    """Extract relationships from knowledge graph JSON and save to new file."""
    print(f"Extracting relationships from {input_path} to {output_path}")
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
        display_type = next(
            (
                key.upper().replace('_', ' ')
                for key, value in rel["properties"].items()
                if isinstance(value, float)
            )
        )

        if "sibling" in rel["type"]:
            continue

        base_dict = {
            "TYPE": display_type,
            "source_page_content": source_node["properties"]["page_content"],
            "target_page_content": target_node["properties"]["page_content"],
            "source_doc_type": source_node["properties"]["document_metadata"]["type"],
            "target_doc_type": target_node["properties"]["document_metadata"]["type"],
            "source_parent": meta_prop(source_node, "post_id")
            or meta_prop(source_node, "parent_doc_id"),
            "target_parent": meta_prop(target_node, "post_id")
            or meta_prop(target_node, "parent_doc_id"),
            "source_title": meta_prop(source_node, "title"),
            "target_title": meta_prop(target_node, "title"),
            # "id": rel["id"],
            # "source_id": rel["source"],
            # "target_id": rel["target"],
        }

        if display_type == "ENTITIES OVERLAP SCORE":
            transformed_relationships.append(
                {
                    **base_dict,
                    "score": rel["properties"]["entities_overlap_score"],
                    "source_entities": source_node["properties"]["entities"],
                    "target_entities": target_node["properties"]["entities"],
                    "overlapped_items": rel["properties"]["overlapped_items"],
                    "num_noisy_items": rel["properties"]["num_noisy_items"],
                }
            )
        elif display_type == "HTML OVERLAP SCORE":
            transformed_relationships.append(
                {
                    **base_dict,
                    "score": rel["properties"]["html_overlap_score"],
                    "source_entities": source_node["properties"]["entities"],
                    "target_entities": target_node["properties"]["entities"],
                    "overlapped_items": rel["properties"]["overlapped_items"],
                    "num_noisy_items": rel["properties"]["num_noisy_items"],
                }
            )
        elif display_type == "THEMES OVERLAP SCORE":
            transformed_relationships.append(
                {
                    **base_dict,
                    "score": rel["properties"]["themes_overlap_score"],
                    "source_themes": source_node["properties"]["themes"],
                    "target_themes": target_node["properties"]["themes"],
                    "overlapped_items": rel["properties"]["overlapped_items"],
                    "num_noisy_items": rel["properties"]["num_noisy_items"],
                }
            )
        elif display_type == "TITLE SIMILARITY":
            transformed_relationships.append(
                {
                    **base_dict,
                    "score": rel["properties"]["title_similarity"],
                }
            )
        elif display_type == "SUMMARY SIMILARITY":
            transformed_relationships.append(
                {
                    **base_dict,
                    "score": rel["properties"]["summary_similarity"],
                    "source_summary": source_node["properties"]["technical_summary"],
                    "target_summary": target_node["properties"]["technical_summary"],
                }
            )
        elif display_type == "MULTI":
            transformed_relationships.append(
                {
                    **base_dict,
                    **rel["properties"],
                }
            )

    with open(output_path, "w") as f:
        json.dump(transformed_relationships, f, indent=2)
    with open(str(output_path) + "__RAW.json", "w") as f:
        json.dump(relationships, f, indent=2)
