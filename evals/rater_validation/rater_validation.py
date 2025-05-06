from util.document_utils import get_all_parquet_in_dir, load_parquet_files
import pandas as pd
from pathlib import Path
import pingouin as pg
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations
from sklearn.linear_model import LinearRegression

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
    Computes the ratio of the std dev of per-sample volatility (std dev of std dev) to
    inter-sample score variation. This is a measure of how much the instability of the
    judge varies across samples. It is measured relative to the variation between samples
    similarly to ICC. Informs whether the judge is especially inconsistent on some queries
    while being stable on others.

    Apparently this isn't a standard metric but it produced interesting results...

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

    return dispersion_of_volatility / inter_sample_sd


def calculate_icc_for_column(dfs: list[pd.DataFrame], column: str) -> float:
    """
    Calculate the Intraclass Correlation Coefficient (ICC) for a given column across multiple LLM-as-a-judge runs
    using the same testset/testrun (including the same inference response/context results). This measures how
    consistent our RAGAS LLM-as-a-judge metric is across different evaluation runs performed on the same inference result.
    i.e. It measures stuff that happens after the we have the batch results from the first pass of RagasEval when we
    run RagInference on the testset.

    Args:
        dfs: List of DataFrames, each representing a single rater (ragas evaluation run), with rows as the subjects (samples).
            All must have the same number/order of rows (subjects).

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


def calculate_icc_for_all_metrics(
    dfs: list[pd.DataFrame], run_dir: str
) -> dict[str, float | str]:
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
    return {"run_dir": run_dir, **icc_results}


def get_metric_dfs(
    dfs: list[pd.DataFrame], parquet_files: list[Path], run_dir: str
) -> list[pd.DataFrame]:
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
        for df in dfs
    ]
    missing_values_report(raw_dfs, parquet_files, run_dir)
    # Impute missing values: fill NA in each column with the mean of that column
    return [df.fillna(df.mean(axis=0), axis=0) for df in raw_dfs]


def write_results_parquet(results: list[dict[str, float | str]], filename: str) -> None:
    """
    Appends a row to the DataFrame containing the mean of each column (excluding the index).
    The mean row will have the index 'mean'.
    """
    df = pd.DataFrame(results).set_index("run_dir")
    mean_row = df.mean()
    df_with_mean = pd.concat([df, pd.DataFrame([mean_row], index=["mean"])])
    df_with_mean.to_parquet(filename)


def create_faithfulness_parquet(all_dfs_by_run: dict[str, list[pd.DataFrame]]) -> None:
    """
    Simply concat the faithfulness columns across all runs for inspection. It's impractical
    to run metrics on these.
    """
    columns = {}
    for run_dir, df_list in all_dfs_by_run.items():
        for i, df in enumerate(df_list):
            col_label = f"{run_dir.replace("test_eval_consistency_", "")}_{i}"
            columns[col_label] = df["faithfulness"].reset_index(drop=True)

    df = pd.DataFrame(columns)
    df.to_parquet("faithfulness.parquet")


def get_column_run_means(
    all_dfs_by_run: dict[str, list[pd.DataFrame]], column: str
) -> pd.DataFrame:
    """
    Calculate the per-sample mean for a given column across all runs per run_dir.
    """
    response_relevancy_means = {}
    for run_dir, df_list in all_dfs_by_run.items():
        series = [df[column] for df in df_list]
        means = pd.concat(series, axis=1).mean(axis=1)
        response_relevancy_means[run_dir.replace("test_eval_consistency_", "")] = means
    df_response_relevancy = pd.DataFrame(response_relevancy_means)
    return df_response_relevancy


def print_regression_summary(
    a_deltas: np.ndarray, b_deltas: np.ndarray, column_a: str, column_b: str
) -> None:
    """
    Print a regression summary for the deltas between two columns: slope, intercept, and R^2.
    """
    a_deltas = a_deltas.reshape(-1, 1)
    b_deltas = b_deltas.reshape(-1, 1)

    reg = LinearRegression().fit(b_deltas, a_deltas)
    r2 = reg.score(b_deltas, a_deltas)
    slope = to_scalar(reg.coef_)
    intercept = to_scalar(reg.intercept_)
    print(
        f"Across-run regression summary: Δ{column_a} = {slope:.5f} * Δ{column_b} + {intercept:.5f} (R²={r2:.6f})\n"
    )


def column_correlation(
    all_dfs_by_run: dict[str, list[pd.DataFrame]], column_a: str, column_b: str
) -> None:
    """
    Calculate the Pearson correlation and p-value between all pairwise per-sample deltas of two columns
    across and within runs. Also prints a regression summary for the across-run deltas.
    """
    # Within-run correlation
    run_dir_corrs = {}
    for run_dir, df_list in all_dfs_by_run.items():
        corrs = []
        ps = []
        for df in df_list:
            corr, p = pearsonr(df[column_a], df[column_b])
            corrs.append(corr)
            ps.append(p)
        run_dir_corrs[run_dir] = (np.mean(corrs), np.mean(ps))
    print(
        f"\nWithin-run correlation between {column_a} and {column_b} (mean across raters):"
    )
    for run_dir, (mean_corr, mean_p) in run_dir_corrs.items():
        print(f"  {run_dir}: mean r={mean_corr:.6f}, mean p={mean_p:.6f}")
    # Across-run delta correlation
    a_means_df = get_column_run_means(all_dfs_by_run, column_a)
    b_means_df = get_column_run_means(all_dfs_by_run, column_b)

    a_means = a_means_df.values  # shape: (n_samples, n_runs)
    b_means = b_means_df.values

    n_samples, n_runs = a_means.shape
    a_deltas = []
    b_deltas = []

    for i in range(n_samples):
        for j, k in combinations(range(n_runs), 2):
            a_deltas.append(a_means[i, k] - a_means[i, j])
            b_deltas.append(b_means[i, k] - b_means[i, j])

    a_deltas = np.array(a_deltas)
    b_deltas = np.array(b_deltas)
    corr, p = pearsonr(a_deltas, b_deltas)
    print(
        f"\nAcross-run delta correlation between {column_a} and {column_b}: {corr:.5f} (p={p:.5f})"
    )
    print_regression_summary(a_deltas, b_deltas, column_a, column_b)


def to_scalar(x):
    return x.item() if hasattr(x, "item") else x


if __name__ == "__main__":
    results = []
    vdr_results = []
    faithfulness_cols = []
    # Load all dfs for all run_dirs before the loop
    all_dfs_by_run: dict[str, list[pd.DataFrame]] = {
        run_dir: load_parquet_files(parquet_files)
        for run_dir, parquet_files in test_run_parquet_files.items()
    }
    # check that metrics that are measuring the same thing actually move together
    # when run against different inference runs.
    column_correlation(all_dfs_by_run, "answer_relevancy", "answer_relevancy_diverse")
    column_correlation(
        all_dfs_by_run, "factual_correctness(mode=recall)", "nv_accuracy"
    )
    for run_dir, parquet_files in test_run_parquet_files.items():
        dfs = all_dfs_by_run[run_dir]
        metric_dfs = get_metric_dfs(dfs, parquet_files, run_dir)
        icc_results = calculate_icc_for_all_metrics(metric_dfs, run_dir)
        results.append(icc_results)

        # Volatility Dispersion Ratio calculation
        vdr_row: dict[str, float | str] = {"run_dir": run_dir}
        for col in metric_dfs[0].columns:
            # Build a DataFrame where rows are samples, columns are raters (runs)
            col_matrix = pd.concat([df[col] for df in metric_dfs], axis=1)
            vdr_row[col] = round(volatility_dispersion_ratio(col_matrix), 4)
        vdr_results.append(vdr_row)

    create_faithfulness_parquet(all_dfs_by_run)
    write_results_parquet(results, "icc_results.parquet")
    write_results_parquet(vdr_results, "vdr_results.parquet")
