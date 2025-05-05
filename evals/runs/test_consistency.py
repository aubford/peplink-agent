import pandas as pd


dfa = pd.read_parquet(
    "langchain-pepwave/evals/runs/testrun_50_new_queries/testrun_50_new_queries__17_37.parquet"
)

dfb = pd.read_parquet("/evals/runs/test_eval_consistency/testrun_50_new_queries__18_09.parquet"
)


