from util.nlp import (
    TokenizedDoc,
    get_duplicate_candidates_minhash_precision,
)
from util.deduplication_pipeline import DeduplicationPipeline


def test_get_duplicate_candidates_minhash_precision():
    # Create test documents with known similarities
    # doc1 and doc2: Jaccard = 0.75 (3/4 shared tokens)
    # doc1 and doc4: Jaccard = 0.2 (1/5 shared tokens)
    # doc3 completely different
    docs = [
        TokenizedDoc("doc1", ["hello", "world", "test"]),
        TokenizedDoc("doc2", ["hello", "world", "test", "python"]),  # Jaccard with doc1 = 3/4 = 0.75
        TokenizedDoc("doc3", ["completely", "different", "text", "here"]),  # No overlap
        TokenizedDoc("doc4", ["hello", "different", "code", "example", "here"]),  # Jaccard with doc1 = 1/7 ≈ 0.14
    ]

    # Test with threshold=0.8 - should catch no pairs since highest Jaccard is 0.75
    candidates_strict = get_duplicate_candidates_minhash_precision(docs, threshold=0.8)
    assert len(candidates_strict) == 0  # No pairs should exceed 0.8 threshold

    # Test with threshold=0.7 - should catch doc1-doc2 pair (Jaccard = 0.75)
    candidates_medium = get_duplicate_candidates_minhash_precision(docs, threshold=0.7)
    assert len(candidates_medium) == 1
    assert (docs[0], docs[1]) in candidates_medium

    # Test with threshold=0.1 - should catch doc1-doc4 pair (Jaccard ≈ 0.14) but not doc3 pairs
    candidates_loose = get_duplicate_candidates_minhash_precision(docs, threshold=0.1)
    assert len(candidates_loose) >= 2
    assert any(docs[0] in pair for pair in candidates_loose)  # doc1 should be in some pairs
    assert not any(docs[2] in pair for pair in candidates_loose)  # doc3 should not be in any pairs


pipeline = DeduplicationPipeline("test")

# %%

# %%

alphabet_list = list("abcdefghijklmnopqrstuvwxyz")


def test_filter_exact_duplicates_minhash():
    # Create test documents with known duplicates
    docs = [
        TokenizedDoc("doc1", alphabet_list),
        TokenizedDoc("doc2", alphabet_list[1:]),
        TokenizedDoc("doc2", alphabet_list[2:]),
        TokenizedDoc("doc2", alphabet_list[5:]),
        TokenizedDoc("doc3", ["completely", "different", "text"]),
    ]

    filtered_docs = pipeline._filter_exact_duplicates_minhash(docs)


def test_confirm_duplicates():
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

    duplicates = pipeline._confirm_duplicates(candidate_pairs)

    assert duplicates == {"doc1"}  # doc1 should be marked as duplicate since it's a subset of doc2
    assert "doc3" not in duplicates  # doc3 is unique
    assert "doc4" not in duplicates  # doc4 is different enough
