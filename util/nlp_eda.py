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

df = pd.concat(web_dfs)

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
duplicate_candidates = get_duplicate_candidates_simple_precision(filtered_docs, threshold=0.95)
candidate_time = time.time() - start
print(f"Candidates complete: {candidate_time:.2f}s")


# %%

import util.nlp

importlib.reload(util.nlp)
from util.nlp import (
    get_duplicate_candidates_simple_precision,
    get_duplicate_candidates_minhash_precision,
    get_duplicates,
    filter_exact_duplicates_minhash,
)

start = time.time()
duplicate_doc_ids = get_duplicates(duplicate_candidates)
dedupe_time = time.time() - start
print(f"\nDeduplication complete: {dedupe_time:.2f}s")

# %%

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


def get_intersection_stats(idx1, idx2):
    text1 = tokenized_corpus[idx1]
    text2 = tokenized_corpus[idx2]
    print("-" * 100)
    print(f"Intersection: {len(set(text1) & set(text2))}")
    print(f"Text[{idx1}]: {len(text1)} -> set: {len(set(text1))}")
    print(f"Text[{idx2}]: {len(text2)} -> set: {len(set(text2))}")
    get_duplicate_candidates_minhash_precision(
        [tokenized_corpus[idx1], tokenized_corpus[idx2]],
        report="print",
    )
    get_duplicate_candidates_simple_precision(
        [tokenized_corpus[idx1], tokenized_corpus[idx2]],
        report="print",
    )
