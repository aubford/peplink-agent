# %%
import pandas as pd
import json
from load.reddit_general.reddit_general_load import RedditGeneralLoad
from load.reddit.reddit_load import RedditLoad
from load.html.html_load import HtmlLoad
from load.youtube.youtube_load import YoutubeLoad
from load.mongo.mongo_load import MongoLoad
from util.util_main import to_serialized_parquet
from pathlib import Path
from rapidfuzz import distance


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


STOP_ENTITIES = ["pepwave", "peplink"]


def skip_stop_entity(entity: str, threshold: float = 0.9) -> bool:
    should_skip = any(
        1 - distance.JaroWinkler.distance(entity, stop_entity) > threshold
        for stop_entity in STOP_ENTITIES
    )
    if should_skip:
        print(f"Skipping stop entity: {entity}")
    return should_skip


def normalize_entities_and_themes(
    df: pd.DataFrame, similarity_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Normalize the entities column by merging nearly identical elements
    using JARO_WINKLER distance comparison.

    Args:
        df: DataFrame with entities column
        similarity_threshold: Threshold for considering two strings similar (default: 0.95)

    Returns:
        DataFrame with normalized entities
    """
    df = df.copy()
    df['entities'] = df['entities'].apply(
        lambda x: json.loads(x) if pd.notna(x) else []
    )
    df['entities'] = df['entities'].apply(lambda x: [str(item).lower() for item in x])
    df['entities'] = df['entities'].apply(
        lambda x: [item for item in x if not skip_stop_entity(item, 0.9)]
    )

    def normalize_list(elements) -> tuple[list, list]:
        # Track all merges to report later
        merges = []

        # Step 1: Group similar items
        similarity_groups = []
        processed = set()

        # Create groups of similar items
        for i, item in enumerate(elements):
            if i in processed or not item:
                continue

            # Start a new group with this item
            group = [item]
            processed.add(i)

            # Find similar items
            for j, other_item in enumerate(elements[i + 1 :], i + 1):
                if j in processed or not other_item:
                    continue

                # Calculate similarity using JARO_WINKLER
                try:
                    similarity = 1 - distance.JaroWinkler.distance(
                        str(item).lower(), str(other_item).lower()
                    )

                    if similarity >= similarity_threshold:
                        group.append(other_item)
                        processed.add(j)
                except Exception as e:
                    print(f"Error comparing {item} and {other_item}: {str(e)}")
                    continue

            similarity_groups.append(group)

        # Step 2: Merge each group, keeping the shortest lowercase version
        normalized = []
        for group in similarity_groups:
            if len(group) == 1:
                normalized.append(group[0])
            else:
                # Find item with shortest lowercase form
                try:
                    shortest = min(group, key=lambda x: len(str(x).lower()))

                    # Record the merge
                    if len(group) > 1:
                        merges.append(
                            {
                                "merged_items": sorted(group, key=lambda x: str(x)),
                                "into": shortest,
                                "similarity_threshold": similarity_threshold,
                            }
                        )
                except Exception as e:
                    print(f"Error finding shortest item in {group}: {str(e)}")
                    shortest = group[0]  # Fall back to first item

                normalized.append(shortest)

        return normalized, merges

    all_merges = []
    normalized_entities = []

    # Apply normalization to entities column
    for idx, row_entities in enumerate(df['entities']):
        try:
            norm_entities, merges = normalize_list(row_entities)
            normalized_entities.append(norm_entities)
            all_merges.extend(merges)
        except Exception as e:
            print(f"Error normalizing entities in row {idx}: {str(e)}")
            print(f"Value: {row_entities}")
            normalized_entities.append(row_entities)  # Keep original on error

    df['entities'] = normalized_entities

    print(f"\n===== Entity Normalization Report =====")
    print(f"Using similarity threshold: {similarity_threshold}")
    print(f"Total merges performed: {len(all_merges)}")

    for i, merge in enumerate(all_merges, 1):
        items_str = ", ".join(f'"{item}"' for item in merge["merged_items"])
        print(f"  {i}. Merged [{items_str}] → \"{merge['into']}\"")

    print("===============================================\n")

    return df


dataset_df_sample = get_slim_dataset_df(40)

# Normalize entities and themes in the sample dataframe
dataset_df_sample = normalize_entities_and_themes(dataset_df_sample)

# Save the sample dataframe to a parquet file
to_serialized_parquet(dataset_df_sample, Path("evals/sample_df.parquet"))

print('done')


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
