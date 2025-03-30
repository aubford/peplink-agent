# %%
import pandas as pd
from ragas.utils import num_tokens_from_string
from load.reddit_general.reddit_general_load import RedditGeneralLoad
from load.reddit.reddit_load import RedditLoad
from load.html.html_load import HtmlLoad
from load.youtube.youtube_load import YoutubeLoad
from load.mongo.mongo_load import MongoLoad


def get_dataset_df(sample_individual: bool = False) -> pd.DataFrame:
    dfs = [
        HtmlLoad.get_artifact(select_merged=True),
        # MongoLoad.get_artifact(select_merged=True).sample(frac=0.2),
        RedditLoad.get_artifact(select_merged=True),
        RedditGeneralLoad.get_artifact(select_merged=True),
        YoutubeLoad.get_artifact(select_merged=True),
    ]

    if sample_individual:
        dfs = [df.sample(frac=0.2) for df in dfs]

    return pd.concat(dfs, ignore_index=True)


def get_slim_dataset_df() -> pd.DataFrame:
    dfs = [
        HtmlLoad.get_artifact(select_merged=True).sample(3),
        MongoLoad.get_artifact(select_merged=True).sample(3),
        RedditLoad.get_artifact(select_merged=True).sample(3),
        YoutubeLoad.get_artifact(select_merged=True).sample(3),
    ]

    return pd.concat(dfs, ignore_index=True)


# %% #####################################  ANALYZE DF #########################

dataset_df = get_dataset_df()

print(f"Total number of rows in dataset: {len(dataset_df)}")
# Check for duplicate content and IDs
duplicate_content = dataset_df[
    dataset_df.duplicated(subset=["page_content"], keep=False)
]
duplicate_ids = dataset_df[dataset_df.duplicated(subset=["id"], keep=False)]

print(f"\nDuplicate content rows: {len(duplicate_content)}")
print(f"Duplicate ID rows: {len(duplicate_ids)}")

# %% #####################################  GET INDIVIDUAL DFS #########################

mongo_df = MongoLoad.get_artifact(select_merged=True)
youtube_df = YoutubeLoad.get_artifact(select_merged=True)
reddit_df = RedditLoad.get_artifact(select_merged=True)
reddit_general_df = RedditGeneralLoad.get_artifact(select_merged=True)
html_df = HtmlLoad.get_artifact(select_merged=True)


# %% #####################################  YOUTUBE TOKENS #####################################
total_youtube_tokens = (
    youtube_df["page_content"].apply(lambda x: num_tokens_from_string(x)).sum()
)
print(f"Total youtube tokens: {total_youtube_tokens}")

# %% ##################################### FILTER MONGO #####################################

# Count rows where page_content has significantly more words than topic_content
print(f"Total mongo docs: {len(mongo_df)}")
long_mongo_docs = mongo_df[
    (
        mongo_df["page_content"].str.split().str.len()
        - mongo_df["topic_content"].str.split().str.len()
    )
    > 130
]
print(f"Number of documents with >min comment/reply words: {len(long_mongo_docs)}")
print(long_mongo_docs["topic_category_name"].value_counts())
print(f"\n\nFinal docs: {len(long_mongo_docs)}")

# %% ##################################### SAVE SAMPLE DATAFRAME #####################################

dataset_df_sample = get_slim_dataset_df()
# Calculate token count of all page_content in the dataset
total_tokens = 0
for content in dataset_df_sample["page_content"]:
    tokens = num_tokens_from_string(content)
    total_tokens += tokens

print(f"Total tokens in dataset page_content: {total_tokens}")
print(f"Type value counts: {dataset_df_sample['type'].value_counts()}")


# Save the sample dataframe to a parquet file
dataset_df_sample.to_parquet("evals/sample_df.parquet", index=False)
