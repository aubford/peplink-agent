# %%

from pathlib import Path
from transform.reddit.reddit_transform import RedditTransform
from transform.mongo.mongo_pepwave_transform import MongoPepwaveTransform
import pandas as pd
from IPython.display import display

pd.set_option("display.max_columns", None)


mongo_artifacts = MongoPepwaveTransform.get_artifacts()
df = mongo_artifacts[0]

df.info()


# %%

df.info()
display(df[df["like_count"].isna()])


def print_column_transformation(column_name):
    display(list(zip(df[column_name].unique(), df[column_name].unique()))[0:20])


display(df[["comment_count", "like_count", "view_count"]].describe())
# print(df['comment_count'].dtype)  # Should show Int64


# %%
## Exploring correlations ######################################
df["duration_seconds"] = df["duration"].dt.total_seconds()

# Calculate correlation between duration and view_count
correlation = df["duration_seconds"].corr(df["view_count"])
print(f"Correlation between duration and view count: {correlation:.3f}")

# %%

long_videos = df[df["duration"] >= pd.Timedelta(minutes=30)]
display(long_videos[["duration", "view_count"]].describe())
# Sort by duration in descending order and display top videos
display(long_videos.sort_values("duration", ascending=False).head(50))
