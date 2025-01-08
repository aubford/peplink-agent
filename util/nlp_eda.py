# %% ########################################################################
%load_ext autoreload
%autoreload 2

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
from util.nlp import TokenizedDoc, tokenize_documents
from transform.web.web_transform import WebTransform
from transform.youtube.youtube_transform import YouTubeTransform
import pandas as pd
from config import RotatingFileLogWriter

logger = RotatingFileLogWriter("nlp")

web_dfs = WebTransform.get_parquet_dfs()
youtube_dfs = YouTubeTransform.get_parquet_dfs()

df = pd.concat(youtube_dfs)

################################################################
############ PIPELINE ##########################################
################################################################

logger.print_header(f"\nTotal docs: {len(df)}")

logger.print_header("Removing duplicate IDs")
initial_len = len(df)
df = df.drop_duplicates(subset=['id']).set_index("id", drop=False, verify_integrity=True)
logger.print(f"Removed {initial_len - len(df)} duplicate IDs")
logger.print(f"Total docs: {len(df)}")

tokenized_docs = tokenize_documents(df)
filtered_docs = filter_exact_duplicates_minhash(tokenized_docs, threshold=0.95)
duplicate_candidates = get_duplicate_candidates_simple_precision(filtered_docs, threshold=0.75, chunk_size=2)
duplicate_doc_ids = confirm_duplicates(duplicate_candidates)
filtered_docs_ids_deduped = [
    doc.doc_id for doc in filtered_docs if doc.doc_id not in duplicate_doc_ids
]
logger.print(f"\nFiltered doc ids after deduplication: {len(filtered_docs_ids_deduped)}")

df_deduped = df[df["id"].isin(filtered_docs_ids_deduped)]
logger.print(f"Rows after deduplication: {len(df_deduped)}")

#%%
# ... existing code ...

# Check for duplicate IDs in original df
print("Duplicate ID counts in original df:")
print(df["id"].value_counts().head())

# Fix: Add drop_duplicates to remove duplicate rows with same ID
df_deduped = df[df["id"].isin(filtered_docs_ids_deduped)].drop_duplicates(subset=["id"])
print(f"Rows after deduplication with drop_duplicates: {len(df_deduped)}")
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
