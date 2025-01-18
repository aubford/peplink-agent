# %% ########################################################################
# %load_ext autoreload
# %autoreload 2

from util.nlp import *
from util.deduplication_pipeline import DeduplicationPipeline
from transform.web.web_transform import WebTransform
from transform.youtube.youtube_transform import YouTubeTransform
import pandas as pd
from config import RotatingFileLogWriter

logger = RotatingFileLogWriter("nlp")

web_dfs = WebTransform.get_parquet_dfs()
web_df = pd.concat(web_dfs)
youtube_dfs = YouTubeTransform.get_parquet_dfs()
youtube_df = pd.concat(youtube_dfs)

pipeline = DeduplicationPipeline("web")

##############################################################################
############ WEB #############################################################
##############################################################################

ng_1 = pipeline.run(web_df, precision_threshold=0.8, precision_ngram=1)
ng_2 = pipeline.run(web_df, precision_threshold=0.5, precision_ngram=2)

# %%

# Find rows in ng_1 that are not in ng_2
ng_1_not_in_ng_2 = ng_1[~ng_1["id"].isin(ng_2["id"])]
print(f"\nRows in ng_1 but not ng_2: {len(ng_1_not_in_ng_2)}")
print(f"Total rows ng_1: {len(ng_1)}")
print(f"Total rows ng_2: {len(ng_2)}")

# %%
import json

# Output discrepancies to jsonl file
with open("discrepancy_content.jsonl", "w") as f:
    for content in ng_1_not_in_ng_2["page_content"]:
        f.write(f'{{"content": {json.dumps(content)}}}\n')


# %% ########################################################################
############ VIZ ############################################################
#############################################################################


# tokenized_corpus = [doc.tokens for doc in filtered_docs]


def get_intersection_stats(tokens_a, tokens_b):
    print("-" * 100)
    print(f"Intersection: {len(set(tokens_a) & set(tokens_b))}")
    print(f"TextA: {len(tokens_a)} -> set: {len(set(tokens_a))}")
    print(f"TextB: {len(tokens_b)} -> set: {len(set(tokens_b))}")
    pipeline.get_duplicate_candidates_simple_precision(
        [tokens_a, tokens_b],
        report="print",
    )


########## NOTES ############################################################

# YouTube
# Threshold=.6, ng=1  =>   16778   candidates   /  54    dupes
# Threshold=.75,ng=1  =>   2140    candidates   /  54   dupes
# Thr=.2,ng=2,cdt=95 =>    91      candidates   /  45   dupes
# Threshold=.2, ng=2  =>   91      candidates   /  48   dupes
# Threshold=.5, ng=2  =>   34      candidates   /  31   dupes
# Threshold=.6, ng=2  =>   25      candidates   /  23   dupes
# Threshold=.6, ng=3  =>   7       candidates   /  6    dupes
#
#
# Web
# Threshold=.7, ng=1  =>  12423   candidates   /  16  dupes  3.5 hours
# Threshold=.8, ng=1  =>  3029    candidates   /  14  dupes
# Threshold=.6, ng=2  =>  7061    candidates   /  13  dupes
# Threshold=.7, ng=2  =>  1762    candidates   /  9   dupes
# Threshold=.7, ng=3  =>  1228    candidates   /  5   dupes
# Threshold=.8, ng=2  =>  234     candidates   /  3   dupes
# Threshold=.85,ng=3  =>  1       candidates   /  0   dupes
#
#
#
