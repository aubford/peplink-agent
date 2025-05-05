import json
import typing as t
from pathlib import Path
import pandas as pd
from ragas.testset.graph import KnowledgeGraph
from util.util_main import to_serialized_parquet
from ragas.testset.graph import Node

output_dir = Path(__file__).parent / "output"
kg_json_path = output_dir / "knowledge_graph.json"
output_nodes_path = output_dir / "__nodes.parquet"
output_relationships_path = output_dir / "__relationships.parquet"
kg_input_data_path = output_dir / "kg_input_data.parquet"


def node_meta(node: Node) -> dict[str, t.Any]:
    return node.properties["document_metadata"]


def dict_meta_prop(node: dict[str, dict[str, t.Any]], prop: str) -> t.Any:
    return node["properties"]["document_metadata"].get(prop, None)


def gen_nodes_parquet(kg: KnowledgeGraph, path: Path) -> None:
    data = []
    for node in kg.nodes:
        properties = node.properties
        metadata = properties.pop("document_metadata")
        properties = {**properties, **metadata}
        properties = {k: v for k, v in properties.items() if "embed" not in k}
        data.append(
            {
                "node_id": str(node.id),
                **properties,
            }
        )
    df = pd.DataFrame(data)
    to_serialized_parquet(df, path)


def gen_relationships_parquet(kg: KnowledgeGraph) -> None:
    data = []
    for rel in kg.relationships:
        # Base relationship data
        rel_data = {
            "source_id": str(rel.source.id),
            "target_id": str(rel.target.id),
            "relationship_type": rel.type,
            "bidirectional": rel.bidirectional,
        }
        rel_data.update(rel.properties)
        data.append(rel_data)

    # Create DataFrame from collected data
    df = pd.DataFrame(data)
    # Convert overlapped_items lists to JSON strings if present
    df["overlapped_items"] = df["overlapped_items"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else x
    )
    df.to_parquet(output_relationships_path)


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
                key.upper().replace("_", " ")
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
            "source_parent": dict_meta_prop(source_node, "post_id")
            or dict_meta_prop(source_node, "parent_doc_id"),
            "target_parent": dict_meta_prop(target_node, "post_id")
            or dict_meta_prop(target_node, "parent_doc_id"),
            "source_title": dict_meta_prop(source_node, "title"),
            "target_title": dict_meta_prop(target_node, "title"),
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
        else:
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
