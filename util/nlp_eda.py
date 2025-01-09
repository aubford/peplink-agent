# %% ########################################################################

# %load_ext autoreload
# %autoreload 2

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

df = pd.concat(web_dfs)

def dedupe_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.print_header("Removing duplicate IDs")
    initial_len = len(df)
    logger.print(f"\nTotal docs: {len(df)}")
    df = df.drop_duplicates(subset=['id']).set_index("id", drop=False, verify_integrity=True)
    logger.print(f"*Removed {initial_len - len(df)} duplicate IDs")
    logger.print(f"Total docs: {len(df)}")
    return df

################################################################
############ PIPELINE ##########################################
################################################################


df = dedupe_df(df)
tokenized_docs = tokenize_documents(df)
filtered_docs = filter_exact_duplicates_minhash(tokenized_docs, threshold=0.95)
duplicate_candidates = get_duplicate_candidates_simple_precision(filtered_docs, threshold=0.7, ngram=1)
duplicate_doc_ids = confirm_duplicates(duplicate_candidates)

filtered_docs_ids_deduped = [
    doc.doc_id for doc in filtered_docs if doc.doc_id not in duplicate_doc_ids
]
logger.print(f"Filtered doc ids after deduplication: {len(filtered_docs_ids_deduped)}")

df_deduped = df[df["id"].isin(filtered_docs_ids_deduped)]
logger.print(f"Rows after deduplication: {len(df_deduped)}")


# %% ########################################################################
############ VIZ ############################################################
#############################################################################


tokenized_corpus = [doc.tokens for doc in filtered_docs]


def get_intersection_stats(tokens_a, tokens_b):
    print("-" * 100)
    print(f"Intersection: {len(set(tokens_a) & set(tokens_b))}")
    print(f"TextA: {len(tokens_a)} -> set: {len(set(tokens_a))}")
    print(f"TextB: {len(tokens_b)} -> set: {len(set(tokens_b))}")
    get_duplicate_candidates_simple_precision(
        [tokens_a, tokens_b],
        report="print",
    )


########## NOTES ############################################################

# YouTube
# Threshold=.8, ng=1 =>           candidates   /      dupes
# Threshold=.5, ng=2 =>           candidates   /      dupes
# Threshold=.6, ng=2 =>   25      candidates   /  23  dupes
# Threshold=.6, ng=3 =>   7       candidates   /  6   dupes
#
#
# Web
# Threshold=.7, ng=1  =>  12423   candidates   /  14  dupes
# Threshold=.8, ng=1  =>  3029    candidates   /  14  dupes
# Threshold=.6, ng=2  =>  7061    candidates   /  13  dupes
# Threshold=.7, ng=2  =>  1762    candidates   /  9   dupes
# Threshold=.7, ng=3  =>  1228    candidates   /  5   dupes
# Threshold=.8, ng=2  =>  234     candidates   /  3   dupes
# Threshold=.85,ng=3  =>  1       candidates   /  0   dupes
#
#
#
