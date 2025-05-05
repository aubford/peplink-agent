from util.document_utils import get_all_parquet_in_dir, load_parquet_files
import pandas as pd
from pathlib import Path
from typing import List, cast
import pingouin as pg

# List of test run directories
TEST_RUN_DIRS: list[str] = [
    "test_eval_consistency_new_queries",
    "test_eval_consistency_new_queries_B",
    "test_eval_consistency_old_queries",
    "test_eval_consistency_old_queries_B",
]

# Base path for the runs directory
RUNS_BASE_PATH = Path(__file__).parent

# Dictionary to store parquet file paths for each test run
test_run_parquet_files: dict[str, list[Path]] = {}

for run_dir in TEST_RUN_DIRS:
    dir_path = RUNS_BASE_PATH / run_dir
    parquet_files = get_all_parquet_in_dir(dir_path)
    test_run_parquet_files[run_dir] = parquet_files


def calculate_icc_for_column(dfs: List[pd.DataFrame], column: str) -> float:
    """
    Calculate the Intraclass Correlation Coefficient (ICC) for a given column across multiple LLM-as-a-judge runs
    using the same testset/testrun (including the same inference response/context results). This measures how
    consistent our RAGAS LLM-as-a-judge metric is across different runs. This measures stuff that happens after the
    we have the batch results from the initial OpenAI API inference call.

    Each DataFrame represents ratings from a single rater (ragas evaluation run), with rows as the subjects (samples).

    Args:
        dfs: List of DataFrames, each representing a rater. All must have the same number/order of rows (subjects).
        column: Name of the column to compute ICC for.

    Returns:
        ICC value (float) for the given column across raters.
    """
    long_data = []
    for rater_idx, df in enumerate(dfs):
        for subject_idx, value in enumerate(df[column].values):
            if pd.notna(value):
                long_data.append(
                    {
                        "targets": subject_idx,
                        "raters": rater_idx,
                        "scores": value,
                    }
                )
    long_df = pd.DataFrame(long_data)
    icc_result = pg.intraclass_corr(
        data=long_df,
        targets="targets",
        raters="raters",
        ratings="scores",
        nan_policy="omit",
    )
    icc2_row = icc_result[icc_result["Type"] == "ICC2"]
    if not icc2_row.empty:
        return icc2_row["ICC"].values[0]
    else:
        raise ValueError("ICC2 value not found in the result.")


def get_metric_columns(dfs: list[pd.DataFrame]) -> set[str]:
    """
    Identify metric columns as all columns of float dtype present in all DataFrames.
    Returns the intersection of float columns across all DataFrames.
    """
    float_cols_sets = [set(df.select_dtypes(include=["float"]).columns) for df in dfs]
    if not float_cols_sets:
        return set()
    return set.intersection(*float_cols_sets)


def missing_values_report(
    dfs: list[pd.DataFrame], parquet_files: list[Path], run_dir: str
) -> None:
    """
    Print a detailed report of missing values for all float columns in the provided DataFrames.
    Warn if missing values are found, but do not raise an error.
    """
    float_cols_sets = [set(df.select_dtypes(include=["float"]).columns) for df in dfs]
    float_cols = set.intersection(*float_cols_sets) if float_cols_sets else set()
    missing_values = []
    for rater_idx, df in enumerate(dfs):
        for col in float_cols:
            missing = df[col].isna()
            if missing.any():
                for row_idx in df.index[missing]:
                    missing_values.append(
                        {
                            "run_dir": run_dir,
                            "file": (
                                parquet_files[rater_idx].name
                                if rater_idx < len(parquet_files)
                                else f"rater_{rater_idx}"
                            ),
                            "rater_idx": rater_idx,
                            "row_idx": row_idx,
                            "column": col,
                        }
                    )
    if missing_values:
        print("WARNING: Missing values found:")
        for report in missing_values:
            print(
                f"RunDir: {report['run_dir']}, File: {report['file']}, RaterIdx: {report['rater_idx']}, RowIdx: {report['row_idx']}, Column: {report['column']}"
            )
        print(
            "Proceeding with ICC calculation using listwise deletion (nan_policy='omit')."
        )


def calculate_icc_for_all_metrics(dfs: list[pd.DataFrame]) -> dict[str, float]:
    """
    Calculate ICC for each metric column (float columns) across the provided DataFrames.
    Returns a dictionary mapping metric column names to their ICC values.
    """
    metric_columns = get_metric_columns(dfs)
    icc_results: dict[str, float] = {}
    for col in metric_columns:
        try:
            icc = calculate_icc_for_column(dfs, col)
            icc_results[col] = round(icc, 4)
        except Exception as e:
            print(f"Failed to calculate ICC for column '{col}': {e}")
    return icc_results


if __name__ == "__main__":
    results = []
    for run_dir, parquet_files in test_run_parquet_files.items():
        print(f"\n=== ICC Results for Test Run: {run_dir} ===")
        dfs = load_parquet_files(parquet_files)
        missing_values_report(dfs, parquet_files, run_dir)
        icc_results = calculate_icc_for_all_metrics(dfs)
        icc_results = {"run_dir": run_dir, **icc_results}
        results.append(icc_results)

    results_df = pd.DataFrame(results)

    # Add a row with the mean of each column
    mean_row = results_df.drop(columns=["run_dir"]).mean(numeric_only=True)
    mean_row["run_dir"] = "mean"
    results_df = pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)

    results_df.to_parquet("icc_results.parquet")
