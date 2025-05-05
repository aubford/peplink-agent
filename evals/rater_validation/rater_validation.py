from util.document_utils import get_all_parquet_in_dir, load_parquet_files
import pandas as pd
from pathlib import Path
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


def volatility_dispersion_ratio(df_runs: pd.DataFrame) -> float:
    """
    Computes the ratio of the std dev of per-sample volatility to inter-sample score variation.
    This is a measure of how much the instability of the judge varies across samples.
    It is measured relative to the variation in samples similarly to ICC.

    Parameters
    ----------
    df_runs : pd.DataFrame
        Rows = samples, columns = repeated scores from the judge

    Returns
    -------
    float
        Volatility dispersion ratio (VDR)
    """
    sample_sds = df_runs.std(axis=1)
    dispersion_of_volatility = sample_sds.std()
    inter_sample_sd = df_runs.mean(axis=1).std()

    return (
        dispersion_of_volatility / inter_sample_sd
        if inter_sample_sd > 0
        else float("inf")
    )


def calculate_icc_for_column(dfs: list[pd.DataFrame], column: str) -> float:
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
            assert pd.notna(value)
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
    )
    icc2_row = icc_result[icc_result["Type"] == "ICC2"]
    if not icc2_row.empty:
        return icc2_row["ICC"].values[0]
    else:
        raise ValueError("ICC2 value not found in the result.")


def missing_values_report(
    dfs: list[pd.DataFrame], parquet_files: list[Path], run_dir: str
) -> None:
    """
    Print a detailed report of missing values for all columns in the provided DataFrames.
    Warn if missing values are found, but do not raise an error.
    """
    if not dfs:
        return
    metric_cols = dfs[0].columns
    missing_values = []
    for rater_idx, df in enumerate(dfs):
        for col in metric_cols:
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
    Calculate ICC for each metric column (all columns) across the provided DataFrames.
    Returns a dictionary mapping metric column names to their ICC values.
    """
    if not dfs:
        return {}
    icc_results: dict[str, float] = {}
    for col in dfs[0].columns:
        try:
            icc = calculate_icc_for_column(dfs, col)
            icc_results[col] = round(icc, 4)
        except Exception as e:
            print(f"Failed to calculate ICC for column '{col}': {e}")
    return icc_results


if __name__ == "__main__":
    results = []
    vdr_results = []
    for run_dir, parquet_files in test_run_parquet_files.items():
        print(f"\n=== ICC Results for Test Run: {run_dir} ===")
        raw_dfs = [
            df[
                [
                    col
                    for col in df.columns
                    if (
                        "relevancy" in col
                        or "factual_correctness" in col
                        or "accuracy" in col
                    )
                ]
            ]
            for df in load_parquet_files(parquet_files)
        ]
        missing_values_report(raw_dfs, parquet_files, run_dir)
        # Impute missing values: fill NA in each column with the mean of that column
        dfs = [df.fillna(df.mean(axis=0), axis=0) for df in raw_dfs]
        icc_results = calculate_icc_for_all_metrics(dfs)
        icc_results = {"run_dir": run_dir, **icc_results}
        results.append(icc_results)

        # Volatility Dispersion Ratio calculation
        vdr_row: dict[str, float | str] = {"run_dir": run_dir}
        for col in dfs[0].columns:
            # Build a DataFrame where rows are samples, columns are raters (runs)
            col_matrix = pd.DataFrame({i: df[col] for i, df in enumerate(dfs)})
            vdr_row[col] = round(volatility_dispersion_ratio(col_matrix), 4)
        vdr_results.append(vdr_row)

    results_df = pd.DataFrame(results)
    vdr_results_df = pd.DataFrame(vdr_results)

    # Add a row with the mean of each column for ICC
    mean_row = results_df.drop(columns=["run_dir"]).mean(numeric_only=True)
    mean_row["run_dir"] = "mean"
    results_df = pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)

    # Add a row with the mean of each column for VDR
    vdr_mean_row = vdr_results_df.drop(columns=["run_dir"]).mean(numeric_only=True)
    vdr_mean_row_dict = vdr_mean_row.to_dict()
    vdr_mean_row_dict["run_dir"] = "mean"
    vdr_results_df = pd.concat(
        [vdr_results_df, pd.DataFrame([vdr_mean_row_dict])], ignore_index=True
    )

    results_df.to_parquet("icc_results.parquet")
    vdr_results_df.to_parquet("vdr_results.parquet")
