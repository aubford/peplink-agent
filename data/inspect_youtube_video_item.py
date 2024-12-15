# %%
from IPython.display import display
import pandas as pd
import json
from pathlib import Path

DATA_ITEM = 'youtube'
raw_path = Path(__file__).parent / DATA_ITEM / 'raw'

# get the latest parquet file
raw_files = list(raw_path.glob('*.jsonl'))
latest_raw_file = max(raw_files, key=lambda x: x.stat().st_mtime)

# read the latest raw file
mainframe = pd.read_json(latest_raw_file, lines=True)

def get_column_json_info(df: pd.DataFrame, column_name: str) -> list[str]:
    # Display column information
    if column_name in df.columns:
        print(f"\n{column_name} Summary:")
        # Count non-null values
        non_null_count = df[column_name].notna().sum()
        print(f"Non-null entries: {non_null_count} ({(non_null_count / len(df)) * 100:.1f}%)")

    # Get non-null entries and convert to JSON strings
    non_null_values = df[df[column_name].notna()][column_name]
    unique_values = list({json.dumps(val, sort_keys=True) for val in non_null_values})

    print(f"\nUnique {column_name} values ({len(unique_values)}):")
    for value in unique_values:
        print(f"\n{value}")

    return unique_values

# %%


mainframe.iloc[1]


# %%
