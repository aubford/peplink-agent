# %%

from pathlib import Path
from transform.youtube.youtube_transform import YouTubeTransform
import pandas as pd
from IPython.display import display

pd.set_option("display.max_columns", None)

# Initialize transformer
transformer = YouTubeTransform()


raw_dir = Path("data") / "youtube" / "raw"

dfs = []
for file_path in raw_dir.glob("*.jsonl"):
    dfs.append(transformer.transform_file(file_path))

df = pd.concat(dfs, ignore_index=True)


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
