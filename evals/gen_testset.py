from ragas.testset.graph import KnowledgeGraph, Relationship
from ragas.testset.persona import Persona
from pathlib import Path
from dotenv import load_dotenv
from evals.analytics_utils import extract_relationships_to_file

load_dotenv()

network_persona = Persona(
    name="Technical Network Engineer",
    role_description="A professional specializing in network configurations and troubleshooting, particularly with experience in WAN setups and integrating satellite internet systems like Starlink with networking hardware such as Peplink Balance and Juniper SRX. This individual actively participates in technical forums to share knowledge, solve problems, and seek advice on complex network-related issues.",
)
full_time_rv_persona = Persona(
    name="Full-Time RV Owner",
    role_description="A full-time RV owner who is interested in using Pepwave devices to connect to the internet while traveling.",
)


LLM_MODEL = "gpt-4o"
TESTSET_SIZE = 50
output_dir = Path(__file__).parent / "output"
kg_path = output_dir / "kg_output__LATEST.json"
target_kg_path = output_dir / "kg_output__MERGED.json"
relationships_path = output_dir / "__relationships_MERGED.json"


def merge_kg_relationships(kg: KnowledgeGraph):
    """
    Merge relationships that have the same source-target pairs.
    Consider bidirectionality by merging A->B with both A->B and B->A relations.
    Skip relationships with 'sibling' in their type.
    """
    # Get all relationships
    relationships: list[Relationship] = kg.relationships
    original_count = len(relationships)

    # Separate sibling and non-sibling relationships
    sibling_relationships = [rel for rel in relationships if "sibling" in rel.type]
    non_sibling_relationships = [
        rel for rel in relationships if "sibling" not in rel.type
    ]

    # Create a dict to group relationships by source-target pairs
    relationship_groups = {}

    # First, identify all relationships with the same source-target pairs
    for rel in non_sibling_relationships:
        # Create a normalized key for source-target pairs
        # Sort to handle bidirectionality (A->B and B->A will have the same key)
        source, target = rel.source, rel.target
        # Sort by string representation of IDs to avoid comparing Node objects directly
        key = tuple(sorted([str(source.id), str(target.id)]))

        if key not in relationship_groups:
            relationship_groups[key] = []
        relationship_groups[key].append(rel)

    # Create new merged relationships
    merged_relationships = []
    for key, rels in relationship_groups.items():
        if len(rels) == 1:
            # No merging needed if there's only one relationship
            merged_relationships.append(rels[0])
        else:
            # Merge relationships
            base_rel = rels[0]

            # Create a new relationship with merged properties
            merged_properties = {}
            for rel in rels:
                merged_properties.update(rel.properties)

            # Create new relationship based on the first one
            merged_rel = Relationship(
                source=base_rel.source,
                target=base_rel.target,
                type="multi",
                bidirectional=base_rel.bidirectional,
                properties=merged_properties,
            )

            merged_relationships.append(merged_rel)

    # Combine merged non-sibling relationships with unchanged sibling relationships
    final_relationships = merged_relationships + sibling_relationships

    # Update the knowledge graph
    kg.relationships = final_relationships

    # Print statistics
    final_count = len(final_relationships)
    merged_count = original_count - final_count
    print(f"Original relationship count: {original_count}")
    print(f"Merged relationship count: {final_count}")
    print(f"Number of relationships merged: {merged_count}")
    print(f"Reduction percentage: {(merged_count/original_count)*100:.2f}%")
    print(f"Sibling relationships preserved: {len(sibling_relationships)}")

    return kg


kg = KnowledgeGraph.load(kg_path)
kg = merge_kg_relationships(kg)
kg.save(target_kg_path)
extract_relationships_to_file(target_kg_path, relationships_path)

# df = dataset.to_pandas()
# current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
# to_serialized_parquet(
#     df, Path.joinpath(evals_dir, f"ragas_testset_{current_date}.parquet")
# )
# dataset.upload()
