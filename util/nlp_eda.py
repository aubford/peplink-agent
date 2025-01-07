# %% ########################################################################
import importlib
from itertools import chain
from util.nlp import *
from util.viz import (
    plot_content_length_dist,
    plot_list_length_dist,
    get_word_counts,
    plot_item_frequency,
)
import time
from extract.web.web_extractor import WebExtractor
from extract.reddit.reddit_extractor import RedditExtractor


web_dfs = WebExtractor.get_rawfile_dataframes()
reddit_dfs = RedditExtractor.get_rawfile_dataframes()


for idx, item in enumerate(web_dfs):
    print(f"{idx}: {item[0]}")

demo_texts = []
for df_tuple in web_dfs + reddit_dfs:
    _, df = df_tuple  # Unpack name and dataframe
    demo_texts.extend(df["page_content"].tolist())


# %% ########################################################################
############ TOKENIZE ###################################################
#############################################################################

nltk_tokenized_corpus = [nltk_get_tokens(text, chunk_size=2) for text in demo_texts]
nltk_tokenized_corpus_encoded = [
    [token.encode("utf8") for token in text] for text in nltk_tokenized_corpus
]


# %% ########################################################################
############ VIZ TOKENIZE ###################################################
#############################################################################
from itertools import chain
import util

importlib.reload(util.viz)
from util.viz import plot_item_frequency

# plot_list_length_dist(nltk_tokenized_corpus)
# for text in nltk_tokenized_corpus:
#     get_word_counts(text)
# flattened_tokens = list(chain.from_iterable(nltk_tokenized_corpus))
# plot_item_frequency(flattened_tokens)

# %% ########################################################################
############ EDA TOKENIZE NLTK ##############################################
#############################################################################
def get_intersection_stats(idx1, idx2):
    text1 = nltk_tokenized_corpus[idx1]
    text2 = nltk_tokenized_corpus[idx2]
    print("-" * 100)
    print(f"Intersection: {len(set(text1) & set(text2))}")
    print(f"Text[{idx1}]: {len(text1)} -> set: {len(set(text1))}")
    print(f"Text[{idx2}]: {len(text2)} -> set: {len(set(text2))}")
    get_duplicate_candidates_minhash_precision(
        [nltk_tokenized_corpus_encoded[idx1], nltk_tokenized_corpus_encoded[idx2]],
        report="print",
    )
    get_duplicate_candidates_simple_precision(
        [nltk_tokenized_corpus[idx1], nltk_tokenized_corpus[idx2]],
        report="print",
    )


get_intersection_stats(723, 721)
get_intersection_stats(25, 419)


# %% ########################################################################
############ TEST ###########################################################
#############################################################################
import importlib
import util.nlp

importlib.reload(util.nlp)

from util.nlp import (
    get_duplicate_candidates_minhash_precision,
    get_duplicate_candidates_simple_precision,
    get_duplicates,
)

# start = time.time()
# simple_result = get_duplicate_candidates_simple_precision(nltk_tokenized_corpus)
# time1 = time.time() - start
# print(f"Simple Precision: {time1:.2f}s")
# print(simple_result)
# print(f"len: {len(simple_result)}")

start = time.time()
minhash_result = get_duplicate_candidates_minhash_precision(
    nltk_tokenized_corpus_encoded
)
time3 = time.time() - start
print(f"Minhash precision: {time3:.2f}s")
print(minhash_result)
print(f"len: {len(minhash_result)}")


# %%

potential_duplicates = [demo_texts[i] for i in minhash_result]
print(len(potential_duplicates))

start = time.time()
result4 = get_duplicates(potential_duplicates)
time4 = time.time() - start
print(f"Rapidfuzz: {time4:.2f}s")
print(result4)
print(f"len: {len(result4)}")
