import pandas as pd


df = pd.read_parquet('./test_runs_summary.parquet')
df_reordered = df.loc[[3, 1, 2, 0, 4, 5]].set_index('run_name')
df_reordered.to_parquet('./test_runs_summary_out.parquet')
