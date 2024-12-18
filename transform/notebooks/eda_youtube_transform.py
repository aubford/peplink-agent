# %%

from pathlib import Path
from transform.youtube_transform import YouTubeTransform
import pandas as pd
from IPython.display import display

pd.set_option('display.max_columns', None)

# Initialize transformer
transformer = YouTubeTransform()

raw_dir = Path("data") / "youtube" / "raw"
# file_path = Path("data/youtube/raw/youtube_video_item_MobileInternetResourceCenter__T_20241215_164803.jsonl")


dfs = []
for file_path in raw_dir.glob("*.jsonl"):
    dfs.append(transformer.transform_file(file_path))

df = pd.concat(dfs, ignore_index=True)


#%%

df.info()
display(df[df['like_count'].isna()])

def print_column_transformation(column_name):
    display(list(zip(df[column_name].unique(), dfc[column_name].unique()))[0:20])

display(df[['comment_count', 'like_count', 'view_count']].describe())
# print(df['comment_count'].dtype)  # Should show Int64


# %%
## Exploring correlations ######################################
df['duration_seconds'] = df['duration'].dt.total_seconds()

# Calculate correlation between duration and view_count
correlation = df['duration_seconds'].corr(df['view_count'])
print(f"Correlation between duration and view count: {correlation:.3f}")

#%%

long_videos = df[df['duration'] >= pd.Timedelta(minutes=30)]
display(long_videos[['duration', 'view_count']].describe())
# Sort by duration in descending order and display top videos
display(long_videos.sort_values('duration', ascending=False).head(50))
