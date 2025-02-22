# %%

from pathlib import Path
from transform.youtube.youtube_transform import YouTubeTransform
import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np

pd.set_option("display.max_columns", None)

# Initialize transformer
artifacts = YouTubeTransform.get_artifacts()
df = pd.concat(artifacts)


# %% ########################### EXPLORE CORRELATION BETWEEN DURATION AND VIEW COUNT ######################################
df["duration_seconds"] = df["duration"].dt.total_seconds()

# Calculate correlation between duration and view_count
correlation = df["duration_seconds"].corr(df["view_count"])
print(f"Correlation between duration and view count: {correlation:.3f}")

long_videos = df[df["duration"] >= pd.Timedelta(minutes=30)]
display(long_videos[["duration", "view_count"]].describe())
# Sort by duration in descending order and display top videos
display(long_videos.sort_values("duration", ascending=False).head(50))


# %% ########################### SEARCH FOR DISFLUENCIES IN ARTIFACTS ######################################
from collections import Counter
from util.nlp import DEFAULT_DISFLUENCIES

# Search for each disfluency in the page content
any_found = False
for disfluency in DEFAULT_DISFLUENCIES:
    # Count occurrences in each document's page content
    mask = df["page_content"].str.contains(rf"\b{disfluency}\b", case=False, regex=True)
    count = mask.sum()
    if count > 0:
        print(f"Found '{disfluency}': {count} occurrences")
        any_found = True

if not any_found:
    print("No disfluencies found in the corpus")


# %% ########################### SKL TEXT ANOMALY DETECTION ######################################

# TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numbers that ML models can understand
# It does this by:
# 1. Breaking text into individual words (terms)
# 2. Calculating how important each word is in each document
# 3. Creating a matrix where each row is a document and each column is a word
vectorizer = TfidfVectorizer(
    # Only keep the 1000 most common words as features
    # This prevents memory issues since YouTube transcripts contain many unique words
    # But we only want to analyze the most meaningful/common ones
    max_features=1000,
    # Convert accented characters to regular ones
    # e.g. "café" becomes "cafe"
    # This ensures words are counted the same regardless of accents
    strip_accents="unicode",
    # Convert all text to lowercase
    # This ensures "Word" and "word" are counted as the same term
    lowercase=True,
    # L2 normalization makes documents comparable regardless of length:
    # 1. Square each term's importance score
    # 2. Sum all squares
    # 3. Divide each score by square root of sum
    # This ensures a 10-minute and 2-hour transcript can be compared fairly
    norm="l2",
)

# Convert all transcripts into TF-IDF features
# Result is a sparse matrix where:
# - Each row represents one YouTube transcript
# - Each column represents one of the 1000 most common words
# - Each cell contains the importance score of that word in that transcript
text_features = vectorizer.fit_transform(df["page_content"].fillna(""))

# Calculate basic statistical features about each transcript
char_length = df["page_content"].str.len()  # Total number of characters
word_count = df["page_content"].str.split().str.len()  # Total number of words
avg_word_length = char_length / word_count  # Average length of words
# First find all non-word characters
non_word_chars = df["page_content"].str.findall(r"\W").str.len()
# calculate ratio of non-word characters to word count
non_word_char_to_word_count = non_word_chars / word_count

# Combine statistical features into a single matrix
# Stack the features side-by-side into columns
statistical_features = np.column_stack([avg_word_length, non_word_char_to_word_count])

# StandardScaler makes features comparable by:
# 1. Subtracting the mean (centering around 0)
# 2. Dividing by standard deviation (similar scale)
statistical_features = StandardScaler().fit_transform(statistical_features)

# Increase the weight of non_word_char_to_word_count (second column) by multiplying it by 3
statistical_features[:, 1] *= 3

# Convert TF-IDF matrix to dense array and combine with statistical features
text_features_dense = text_features.toarray()
combined_features = np.hstack([statistical_features, text_features_dense])

# Isolation Forest detects anomalies by:
# 1. Randomly selecting a feature (e.g. word count)
# 2. Randomly picking a split point
# 3. Repeating until each sample is isolated
# Anomalies are isolated in fewer splits than normal samples
iso_forest = IsolationForest(
    random_state=42,  # Set random seed for reproducibility
)
# Returns 1 for normal samples, -1 for anomalies
anomalies = iso_forest.fit_predict(statistical_features)

# Get indices of anomalous transcripts
anomaly_indices = np.where(anomalies == -1)[0]
print(f"\nFound {len(anomaly_indices)} anomalies")

# %%
# Print details about the first 5 anomalous transcripts
print("\nSample of anomalous texts:")
for idx in anomaly_indices:
    video = df.iloc[idx]
    print(f"\n--- Text sample for ID {video['id']} ---")
    print(f"Word count: {len(video['page_content'].split())}")
    print("Text:")
    print(video["page_content"])


# Results: We get no anomalies when including the text features but statistical features on their own return some strange looking transcripts.
# On further inspection, they all seem to be OK, just some weird behavior from the youtube transcriber.

# %%
test_string = "s,;39j- \n="
import re

# Find all non-word characters
non_word_chars = re.findall(r"\W", test_string)
print(f"Non-word characters found: {non_word_chars}")
print(f"Count: {len(non_word_chars)}")
