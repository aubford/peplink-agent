from util.nlp import TokenizedDoc, get_duplicates

def test_get_duplicates():
    # Create test documents with known duplicates
    docs = [
        TokenizedDoc("doc1", ["hello", "world", "test"]),
        TokenizedDoc("doc2", ["hello", "world", "test", "extra"]),  # Superset of doc1
        TokenizedDoc("doc3", ["completely", "different", "text"]),
        TokenizedDoc("doc4", ["hello", "world", "different", "test"]),  # Similar to doc1/2
    ]

    # Create candidate pairs to test - using list instead of set
    candidate_pairs = [
        (docs[0], docs[1]),  # doc1 should be marked as duplicate (subset)
        (docs[0], docs[3]),  # Similar but not duplicate
        (docs[2], docs[3]),  # Different enough to not be duplicate
    ]

    duplicates = get_duplicates(candidate_pairs)

    assert duplicates == {"doc1"}  # doc1 should be marked as duplicate since it's a subset of doc2
    assert "doc3" not in duplicates  # doc3 is unique
    assert "doc4" not in duplicates  # doc4 is different enough
