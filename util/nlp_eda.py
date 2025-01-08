# %% ########################################################################
import importlib
from itertools import chain
from util.nlp import *
from util.viz import (
    plot_list_length_dist,
    get_word_counts,
    plot_item_frequency,
)
import time
from typing import List
from util.nlp import TokenizedDoc
from transform.web.web_transform import WebTransform
from transform.youtube.youtube_transform import YouTubeTransform
import pandas as pd


web_dfs = WebTransform.get_parquet_dfs()
youtube_dfs = YouTubeTransform.get_parquet_dfs()

df = pd.concat(youtube_dfs)

################################################################
############ PIPELINE ##########################################
################################################################

print(f"\nTotal docs: {len(df)}")

start = time.time()
tokenized_docs = []
for _, row in df.iterrows():
    doc = TokenizedDoc(doc_id=str(row["id"]), text=row["page_content"])
    tokenized_docs.append(doc)
token_time = time.time() - start
print(f"\nTokenized: {token_time:.2f}s")

start = time.time()
filtered_docs = filter_exact_duplicates_minhash(tokenized_docs, threshold=0.95)
filter_time = time.time() - start
print(f"Filter complete: {filter_time:.2f}s")


start = time.time()
duplicate_candidates = get_duplicate_candidates_simple_precision(filtered_docs, threshold=0.75, chunk_size=2)
candidate_time = time.time() - start
print(f"Candidates complete: {candidate_time:.2f}s")


import util.nlp

importlib.reload(util.nlp)
from util.nlp import (
    get_duplicate_candidates_simple_precision,
    get_duplicate_candidates_minhash_precision,
    confirm_duplicates,
    filter_exact_duplicates_minhash,
)

start = time.time()
duplicate_doc_ids = get_duplicates(duplicate_candidates)
dedupe_time = time.time() - start
print(f"\nDeduplication complete: {dedupe_time:.2f}s")

filtered_docs_ids_deduped = [
    doc.doc_id for doc in filtered_docs if doc.doc_id not in duplicate_doc_ids
]
print(f"\nFiltered docs after deduplication: {len(filtered_docs_ids_deduped)}")


df_deduped = df[df["id"].isin(filtered_docs_ids_deduped)]
print(f"Rows after deduplication: {len(df_deduped)}")


# %% ########################################################################
############ VIZ ###############################################
#############################################################################


tokenized_corpus = [doc.tokens for doc in filtered_docs]


def get_intersection_stats(tokensA, tokensB):
    print("-" * 100)
    print(f"Intersection: {len(set(tokensA) & set(tokensB))}")
    print(f"TextA: {len(tokensA)} -> set: {len(set(tokensA))}")
    print(f"TextB: {len(tokensB)} -> set: {len(set(tokensB))}")
    get_duplicate_candidates_simple_precision(
        [tokensA, tokensB],
        report="print",
    )
