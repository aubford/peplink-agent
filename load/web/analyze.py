# %%
import pandas as pd
from pathlib import Path

staging__save_path = Path("../staging_save.parquet")
staging_path = Path("../staging.parquet")
df = pd.read_parquet(staging_path)
df_save = pd.read_parquet(staging__save_path)
unique_sources = df["source_file"].unique()
unique_sources_save = df_save["source_file"].unique()

print(df.shape)
print(df_save.shape)

# %%

# Find duplicated page content
duplicated_content = df[df["page_content"].duplicated(keep=False)]
print(f"Number of duplicated pages: {len(duplicated_content)}")
duplicated_content.sort_values("page_content").head()

# %%

from util.deduplication_pipeline import DeduplicationPipeline

dedup_pipeline = DeduplicationPipeline("web_staging_eda")

deduped = dedup_pipeline.run(
    duplicated_content, precision_threshold=0.8, precision_ngram=1
)


# %%
num_peplink_com = len(
    df[df["source_file"].str.contains("web_ldoc_peplink_com__T_20241228_213623")]
)
print(f"Number of peplink.com pages: {num_peplink_com}")
