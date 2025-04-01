# %%
import pandas as pd
from ragas.utils import num_tokens_from_string
from load.reddit_general.reddit_general_load import RedditGeneralLoad
from load.reddit.reddit_load import RedditLoad
from load.html.html_load import HtmlLoad
from load.youtube.youtube_load import YoutubeLoad
from load.mongo.mongo_load import MongoLoad
from util.util_main import to_serialized_parquet
from pathlib import Path


def get_dataset_df(sample_individual: bool = False) -> pd.DataFrame:
    dfs = [
        HtmlLoad.get_artifact(),
        MongoLoad.get_artifact(),
        RedditLoad.get_artifact(),
        RedditGeneralLoad.get_artifact(),
        YoutubeLoad.get_artifact(),
    ]

    if sample_individual:
        dfs = [df.sample(frac=0.2) for df in dfs]

    return pd.concat(dfs, ignore_index=True)


def get_slim_dataset_df(sample_size: int = 15) -> pd.DataFrame:
    dfs = [
        HtmlLoad.get_artifact().sample(sample_size),
        MongoLoad.get_artifact().sample(sample_size),
        RedditLoad.get_artifact().sample(sample_size),
        RedditGeneralLoad.get_artifact().sample(sample_size),
        YoutubeLoad.get_artifact().sample(sample_size),
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


# %% ##################################### SAVE SAMPLE DATAFRAME #####################################

dataset_df_sample = get_slim_dataset_df(40)
# Calculate token count of all page_content in the dataset
total_tokens = 0
for content in dataset_df_sample["page_content"]:
    tokens = num_tokens_from_string(content)
    total_tokens += tokens

print(f"Total tokens in dataset page_content: {total_tokens}")
print(f"Type value counts: {dataset_df_sample['type'].value_counts()}")


# Save the sample dataframe to a parquet file
to_serialized_parquet(dataset_df_sample, Path("evals/sample_df.parquet"))
