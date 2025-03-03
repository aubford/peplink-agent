# %%
from transform.youtube.youtube_transform import YouTubeTransform
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from textstat import flesch_kincaid_grade
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from util.nlp import DEFAULT_DISFLUENCIES, get_keywords

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

# calculate basic statistical features about each transcript
char_length = df["page_content"].str.len()  # Total number of characters
word_count = df["page_content"].str.split().str.len()  # Total number of words
avg_word_length = char_length / word_count  # Average length of words
# find all non-word characters
non_word_chars = df["page_content"].str.findall(r"\W").str.len()
# ratio of non-word characters to word count
non_word_char_to_word_count = non_word_chars / word_count
# detect anomalies in text complexity
readability_score = df["page_content"].apply(flesch_kincaid_grade)
# other features
stopwords = df["page_content"].apply(
    lambda x: get_keywords(x, list(ENGLISH_STOP_WORDS))
)
stopword_ratio = stopwords.str.len() / word_count
sentence_length_var = df["page_content"].apply(
    lambda x: (
        np.var([len(word_tokenize(sent)) for sent in sent_tokenize(x)]) if x else 0
    )
)


# %% ########################### APPLY ANOMALY DETECTION ######################################

scaler = StandardScaler()
scaled_features = np.column_stack(
    [
        scaler.fit_transform(avg_word_length.to_frame()),
        scaler.fit_transform(non_word_char_to_word_count.to_frame()),
        scaler.fit_transform(readability_score.to_frame()),
        scaler.fit_transform(stopword_ratio.to_frame()),
        scaler.fit_transform(sentence_length_var.to_frame()),
    ]
)

# Isolation Forest detects anomalies by:
# 1. Randomly selecting a feature (e.g. word count)
# 2. Randomly picking a split point
# 3. Repeating until each sample is isolated
# Anomalies are isolated in fewer splits than normal samples
iso_forest = IsolationForest()
# Returns 1 for normal samples, -1 for anomalies
anomalies = iso_forest.fit_predict(scaled_features)

# Get indices of anomalous transcripts
anomaly_indices = np.where(anomalies == -1)[0]
print(f"\nFound {len(anomaly_indices)} anomalies")

# %%

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = df["page_content"].apply(lambda x: model.encode(x))

# Scale and combine features
embeddings_array = np.vstack(embedding.values)

embedding_iso_forest = IsolationForest()
# Returns 1 for normal samples, -1 for anomalies
anomalies = embedding_iso_forest.fit_predict(embeddings_array)

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
non_word_chars_ser = re.findall(r"\W", test_string)
print(f"Non-word characters found: {non_word_chars_ser}")
print(f"Count: {len(non_word_chars_ser)}")
