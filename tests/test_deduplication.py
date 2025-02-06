# %%
from util.nlp import TokenizedDoc
from util.deduplication_pipeline import DeduplicationPipeline
import pandas as pd

pipeline = DeduplicationPipeline("test")

documents = [
    # 0
    {
        "id": "doc_a",
        "page_content": "Once upon a time long ago in a galaxy far far away, there was a cat and a bat and a rat and a gnat. They all lived happily ever after. The end.",
    },
    # 1
    {
        "id": "doc_b_shifted_evenly",
        "page_content": "Long ago in a galaxy far far away, there was a cat and a bat and a rat and a gnat. They all lived happily ever after. The end.",
    },
    # 2
    {
        "id": "doc_like_b_not_a",
        "page_content": "Long ago in a galaxy far far away, there was a cat and a bat and a rat and a gnat. They all lived happily ever after. The end. Oh and dad.",
    },
    # 3
    {
        "id": "doc_c_shifted",
        "page_content": "A time long ago in a galaxy far far away,there was a cat and a bat and a rat and a gnat. They all lived happily ever after. The end.",
    },
    # 4
    {
        "id": "doc_bb_shifted_evenly",
        "page_content": "Long ago in a galaxy far far away, there was a cat and a bat and a rat and a gnat. They all lived happily ever after. The end.",
    },
    # 5
    {
        "id": "doc_aa",
        "page_content": "Once upon a time long ago in a galaxy far far away, there was a cat and a bat and a rat and a gnat. They all lived happily ever after. The end.",
    },
    # 6
    {
        "id": "doc_d",
        "page_content": "Once upon a time long ago in a galaxy far far away, there was a cat and a bat and a rat and a quokka and a gnat. They all lived happily ever after. The end.",
    },
    # 7
    {
        "id": "doc_e",
        "page_content": "This is a completely different story. It definitely has nothing to do with cats or bats or rats or gnats or even quokkas.",
    },
]


def assert_ids_equal(docs: list[TokenizedDoc], expected_ids: list[str]):
    ids_list = [doc.doc_id for doc in docs]
    print(f"\n\n**Doc Ids: {ids_list}\n\n")
    assert set(ids_list) == set(expected_ids)


def test_filter_exact_duplicates_minhash():
    tokenized_docs = pipeline._tokenize_documents(pd.DataFrame(documents))
    print("-" * 20)
    for doc in tokenized_docs:
        print(doc.tokens)
    print("-" * 20)

    filtered_docs = pipeline.filter_exact_duplicates_minhash(tokenized_docs, threshold=0.80, min_unique_tokens=1)
    assert_ids_equal(filtered_docs, ["doc_a", "doc_e"])


# %%


def test_confirm_duplicates():
    tokenized_docs = pipeline._tokenize_documents(pd.DataFrame(documents))
    candidates = pipeline.get_duplicate_candidates_simple_precision(tokenized_docs)
    duplicates = pipeline._confirm_duplicates(candidates)
    assert set(duplicates) == {"doc_a", "doc_d", "doc_e"}
