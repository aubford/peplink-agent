# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_content_length_dist(df_column, title=None, bins=500, figsize=(10, 6)):
    """
    Plot distribution of string lengths in a pandas DataFrame text column with mean and median lines.

    Args:
        df_column: pandas.Series or pandas.DataFrame[str] - Text column to analyze
        title: str - Plot title (defaults to column name if None)
        bins: int - Number of histogram bins
        figsize: tuple - Figure dimensions (width, height)
    """
    # Convert to series if DataFrame column and get lengths
    series = df_column if isinstance(df_column, pd.Series) else df_column.squeeze()
    length_series = series.str.len()
    word_series = series.str.split().str.len()

    # Create histogram using pandas
    ax = length_series.plot(kind="hist", bins=bins, figsize=figsize, edgecolor="black")
    ax.set_xscale('log')  # Set x-axis to log scale

    # Set title (use series name if title not provided)
    title = title or f"Distribution of {series.name} Lengths"
    stats_text = (
        f"\nContent lengths: {length_series.min():,} to {length_series.max():,} chars"
        f"\nWord counts: {word_series.min():,} to {word_series.max():,} words"
    )
    ax.set_title(title + stats_text)
    ax.set_xlabel("Length (characters)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    # Add mean and median lines
    ax.axvline(
        length_series.mean(),
        color="red",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {length_series.mean():,.0f}",
    )
    ax.axvline(
        length_series.median(),
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"Median: {length_series.median():,.0f}",
    )
    ax.legend()

    plt.show()


def plot_list_length_dist(
    data: list[str] | list[list],
    title: str = None,
    bins: int = 500,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot distribution of lengths from a list of strings or nested lists.

    Args:
        data: List of strings or nested lists to analyze
        title: Plot title (defaults to "Distribution of List Lengths")
        bins: Number of histogram bins
        figsize: Figure dimensions (width, height)
    """
    # Convert to pandas series for consistent analysis
    series = pd.Series(data)
    length_series = series.apply(len)

    # Create histogram
    ax = length_series.plot(kind="hist", bins=bins, figsize=figsize, edgecolor="black")
    ax.set_xscale('log')  # Set x-axis to log scale

    # Set title and stats
    title = title or "Distribution of List Lengths"
    stats_text = f"\nLengths: {length_series.min():,} to {length_series.max():,}"
    ax.set_title(title + stats_text)
    ax.set_xlabel("Length")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    # Add mean and median lines
    ax.axvline(
        length_series.mean(),
        color="red",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {length_series.mean():,.0f}",
    )
    ax.axvline(
        length_series.median(),
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"Median: {length_series.median():,.0f}",
    )
    ax.legend()

    plt.show()


def plot_item_frequency(items: list, top_n: int = 300, title: str = None) -> None:
    """
    Plot frequency distribution of items in a list.

    Args:
        items: List of hashable items to analyze
        top_n: Number of most frequent items to display (default 300)
        title: Plot title (defaults to "Item Frequency Distribution")
    """
    # Create frequency series directly from items
    frequencies = pd.Series(items).value_counts()

    # Adjust figure height based on number of items
    height = max(6, top_n * 0.12)
    figsize = (12, height)

    # Plot horizontal bar chart
    ax = frequencies.head(top_n).plot(kind="barh", figsize=figsize)

    title = title or "Item Frequency Distribution"
    ax.set_title(f"{title}\n(Top {top_n} items)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Item")
    ax.grid(True, alpha=0.3)

    # Add frequency values at end of each bar
    for i, v in enumerate(frequencies.head(top_n)):
        ax.text(v, i, f" {v:,}", va="center")

    plt.tight_layout()
    plt.show()


def get_word_counts(text: list[str], verbose: bool = False) -> dict[str, int]:
    """
    Get word frequency counts from a tokenized text.

    Args:
        text: List of tokens
        top_n: Optional number of top frequency words to return. Returns all if None.

    Returns:
        Dictionary mapping words to their counts, sorted by frequency
    """
    print("Word Counts" + "-" * 100)
    print(f"TOTAL WORDS: {len(text)}")
    word_counts = {}
    for token in text:
        word_counts[token] = word_counts.get(token, 0) + 1
    sorted_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
    if verbose:
        for word, count in sorted_counts.items():
            print(f"{word}: {count}")
    print(f"UNIQUE WORDS: {len(word_counts)}")


def plot_number_dist(
    numbers: list[float],
    title: str = "Distribution Plot",
    bins: int = 10000,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Creates a histogram of numerical values with statistics.

    Args:
        numbers: List of numbers to analyze
        title: Plot title (default: "Distribution Plot")
        bins: Number of histogram bins (default: 50)
        figsize: Figure dimensions (width, height) (default: (10, 6))
    """
    # Convert to numpy array for efficient computation
    data = np.array(numbers)

    # Create histogram
    ax = plt.figure(figsize=figsize).gca()
    ax.hist(data, bins=bins, edgecolor="black", alpha=0.7)

    # Set y-axis to log scale
    ax.set_yscale("log")

    # Add mean and median lines
    mean_val = data.mean()
    median_val = np.median(data)
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Mean: {mean_val:.3f}",
    )
    ax.axvline(
        median_val,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"Median: {median_val:.3f}",
    )

    # Set labels and title
    stats_text = f"\nRange: {data.min():.3f} to {data.max():.3f}"
    ax.set_title(title + stats_text)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
