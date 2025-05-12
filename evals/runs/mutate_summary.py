import pandas as pd


df = pd.read_parquet('./test_runs_summary.parquet')
df_reordered = df.loc[[1, 3, 0, 2, 4, 5]]
df_reordered.to_parquet('./test_runs_summary_out.parquet', index=False)
