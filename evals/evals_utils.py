from ragas.testset.graph import Node
import typing as t

def node_meta(node: Node) -> dict[str, t.Any]:
    return node.properties["document_metadata"]
