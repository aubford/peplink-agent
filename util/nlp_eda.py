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
from extract.web.web_extractor import WebExtractor
from extract.reddit.reddit_extractor import RedditExtractor
from typing import List


web_dfs = WebExtractor.get_rawfile_dataframes()

for idx, item in enumerate(web_dfs):
    print(f"{idx}: {item[0]}")

demo_texts = []
for df_tuple in web_dfs:
    _, df = df_tuple  # Unpack name and dataframe
    demo_texts.extend(df["page_content"].tolist())


# %% ########################################################################
############ TOKENIZE ###################################################
#############################################################################

time_start = time.time()
nltk_tokenized_corpus = [nltk_get_tokens(text, chunk_size=2) for text in demo_texts]
nltk_tokenized_corpus_encoded = [
    [token.encode("utf8") for token in text] for text in nltk_tokenized_corpus
]
time_end = time.time()
print(f"Tokenization time: {time_end - time_start:.2f}s")


# %% ########################################################################
############ VIZ TOKENIZE ###################################################
#############################################################################
import util

importlib.reload(util.viz)
from util.viz import plot_item_frequency, plot_list_length_dist


plot_list_length_dist(nltk_tokenized_corpus)
for text in nltk_tokenized_corpus:
    get_word_counts(text)
flattened_tokens = list(chain.from_iterable(nltk_tokenized_corpus))
plot_item_frequency(flattened_tokens)


# %% ########################################################################
############ EDA INTERSECTION ###############################################
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


# %% ########################################################################
############ MAIN ###########################################################
#############################################################################
import importlib
import util.nlp

importlib.reload(util.nlp)

from util.nlp import (
    filter_exact_duplicates_minhash,
    get_duplicate_candidates_minhash_precision,
    get_duplicate_candidates_simple_precision,
    get_duplicates,
)

# start = time.time()
# simple_result = get_duplicate_candidates_simple_precision(
#     nltk_tokenized_corpus, threshold=0.9
# )
# time1 = time.time() - start
# print(f"Simple Precision: {time1:.2f}s")
# print(simple_result)
# print(f"len: {len(simple_result)}")

start = time.time()
minhash_result = filter_exact_duplicates_minhash(
    nltk_tokenized_corpus_encoded, threshold=0.97
)
time3 = time.time() - start
print(f"Minhash exact time: {time3:.2f}s")
# print(minhash_result)
print(
    f"Minhash exact duplicates to remove: {len(minhash_result)}"
)
print(len(nltk_tokenized_corpus))

# start = time.time()
# minhash_result = get_duplicate_candidates_minhash_precision(
#     nltk_tokenized_corpus_encoded,
#     threshold=0.9,
#     report="plot",
# )
# time3 = time.time() - start
# print(f"Minhash precision: {time3:.2f}s")
# print(minhash_result)
# %% ########################################################################
############ VIZ ###########################################################
#############################################################################

# print(minhash_result)

for i, result in enumerate(minhash_result):
    if len(result) > 1:
        print(f"\nindex[{i}]: {result}")

#%%

for i in minhash_result[1340]:
    for idx, j in enumerate(minhash_result):
        if i in j and len(j) < 5:
            print(f"num[{i}]: {j}")


# for i in range(len(minhash_result)):
#     get_intersection_stats(i, i + 1)


# %% ########################################################################
############ DEDUPE #########################################################
#############################################################################
potential_duplicates = [demo_texts[i] for i in minhash_result]

start = time.time()
result4 = get_duplicates(potential_duplicates)
time4 = time.time() - start
print(f"Rapidfuzz: {time4:.2f}s")
print(result4)
print(f"len: {len(result4)}")

# #%%
# def clean_incremental_groups(minhash_result: List[list]) -> List[list]:
#     """Remove intermediate groups that are subsets of later groups."""
#     cleaned_groups = []

#     for i, current_group in enumerate(minhash_result):
#         # Convert lists to sets for comparison
#         current_set = set(current_group)
#         # Check if this group is a subset of any later group
#         is_intermediate = any(
#             current_set.issubset(set(later_group)) and current_set != set(later_group)
#             for later_group in minhash_result[i+1:]
#         )

#         if not is_intermediate:
#             cleaned_groups.append(current_group)

#     return cleaned_groups

# def analyze_overlapping_groups(minhash_result):
#     # First clean up intermediate groups
#     cleaned_results = clean_incremental_groups(minhash_result)
#     print(f"Reduced from {len(minhash_result)} to {len(cleaned_results)} groups")

#     # Rest of the analysis remains the same
#     number_to_groups = {}
#     for group_idx, group in enumerate(cleaned_results):
#         if len(group) > 1:
#             for num in group:
#                 if num not in number_to_groups:
#                     number_to_groups[num] = []
#                 number_to_groups[num].append(group_idx)

#     overlapping_numbers = {
#         num: groups for num, groups in number_to_groups.items()
#         if len(groups) > 1
#     }


#     if overlapping_numbers:
#         print("\nNumbers appearing in multiple groups:")
#         for num, group_indices in overlapping_numbers.items():
#             print(f"\nNumber {num} appears in {len(group_indices)} groups:")
#             for group_idx in group_indices:
#                 print(f"  Group {group_idx}: {cleaned_results[group_idx]}")
#     else:
#         print("\nNo numbers appear in multiple groups")

#     return overlapping_numbers

# # Call the analysis function
# overlapping = analyze_overlapping_groups(minhash_result)
# print(f"\nTotal numbers appearing in multiple groups: {len(overlapping)}")

# cleaned_results = clean_incremental_groups(minhash_result)
# print(cleaned_results)
