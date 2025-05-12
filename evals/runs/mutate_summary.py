import pandas as pd

# Load the parquet file
df = pd.read_parquet('test_runs_summary.parquet')

# Print original column order for reference
print("Original columns:", list(df.columns))

# Reorder the columns to indexes [3, 1, 2, 0, 4, 5, 6]
columns = list(df.columns)
reordered_columns = [
    columns[3],
    columns[1],
    columns[2],
    columns[0],
    columns[4],
    columns[5],
    columns[6],
]

# Create a new dataframe with reordered columns
df_reordered = df[reordered_columns]

# Print reordered column order for confirmation
print("Reordered columns:", reordered_columns)

# Write the reordered dataframe back to parquet
df_reordered.to_parquet('test_runs_summary.parquet', index=False)

print("Column reordering complete. File saved.")
