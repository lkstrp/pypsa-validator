"""Read benchmark data and generate plots comparing execution time and memory peak."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import get_env_var

DIR_ARTIFACTS: Path = Path(
    get_env_var("DIR_ARTIFACTS", Path(get_env_var("HOME")) / "artifacts")
)


def read_benchmark_dir(directory: Path) -> pd.DataFrame:
    """
    Read benchmark data from a directory.

    Parameters
    ----------
    directory : Path
        Directory containing benchmark data for a single run

    Returns
    -------
    pd.DataFrame
        DataFrame containing benchmark data for a single run

    """
    data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == ".DS_Store":
                continue
            filepath = os.path.join(root, file)
            df = pd.read_csv(filepath, sep="\t")
            df["name"] = "/".join(os.path.relpath(filepath, directory).split("/")[2:])
            data.append(df)
    return pd.concat(data, ignore_index=True)


def read_benchmarks(directory: Path) -> pd.DataFrame:
    """
    Read benchmark data from the main and feature branches.

    Parameters
    ----------
    directory : Path
        Directory containing benchmark data

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing benchmark data from the main and feature branches

    """
    dir_main = DIR_ARTIFACTS / "benchmarks/main/benchmarks/"
    dir_feat = DIR_ARTIFACTS / "benchmarks/feature/benchmarks/"

    df_main = read_benchmark_dir(dir_main)
    df_feat = read_benchmark_dir(dir_feat)

    # Add a column to identify the run
    df_main["run"] = "main"
    df_feat["run"] = "feature"

    # Combine the dataframes
    df = pd.concat([df_main, df_feat], ignore_index=True)

    return df


def create_bar_chart_comparison(
    df: pd.DataFrame,
    x_column: str,
    title: str,
    xlabel: str,
    filename: str,
    ignore_stacked_plot: bool = False,
) -> str:
    """
    Create a horizontal bar chart comparing execution time or memory peak.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing benchmark data
    x_column : str
        Column to use for comparison
    title : str
        Title of the plot
    xlabel : str
        Label for the x-axis
    filename : str
        Output filename
    ignore_stacked_plot : bool, optional (default=False)
        If True, do not include the stacked bar plot subplot

    Returns
    -------
    None

    """
    if ignore_stacked_plot:
        fig, ax1 = plt.subplots(figsize=(8, max(6, len(df["name"].unique()) * 0.4)))
    else:
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(10, max(8, len(df["name"].unique()) * 0.4)),
            gridspec_kw={"width_ratios": [3, 1]},
        )

    unique_jobs = df["name"].unique()
    color_palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(df)))
    job_colors = dict(zip(unique_jobs, color_palette))

    # Horizontal bar plot
    job_positions = {job: i for i, job in enumerate(unique_jobs)}
    bar_width = 0.35

    for i, run in enumerate(df["run"].unique()):
        df_run = df[df["run"] == run]
        positions = [
            job_positions[job] + (i - 0.5) * bar_width for job in df_run["name"]
        ]
        ax1.barh(
            positions,
            df_run[x_column],
            bar_width,
            color=[job_colors[job] for job in df_run["name"]],
            edgecolor="black",
            linewidth=0.8,
            label=run,
        )

    ax1.set_yticks([job_positions[job] for job in unique_jobs])
    ax1.set_yticklabels(unique_jobs)
    ax1.tick_params(
        axis="y", which="major", labelsize=7
    )  # Adjust the size (8) as needed

    ax1.set_title(f"{title} - Detailed Comparison")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Benchmark")
    if not ignore_stacked_plot:
        # Two single vertical bars
        totals = df.groupby("run")[x_column].sum()
        bar_width = 0.8
        index = np.arange(len(totals))

        for i, run in enumerate(totals.index):
            bottom = 0
            for job in unique_jobs:
                value = df[(df["run"] == run) & (df["name"] == job)][x_column].values
                if len(value) > 0:
                    ax2.bar(
                        i,
                        value,
                        bar_width,
                        bottom=bottom,
                        color=job_colors[job],
                        edgecolor="black",
                        linewidth=0.8,
                    )
                    bottom += value[0]

            ax2.set_title(f"{title} - Total")
            ax2.set_ylabel(xlabel)
            ax2.set_xlabel("Run")
            ax2.set_xticks(index)
            ax2.set_xticklabels(totals.index)

            # Remove all spines
            for spine in ax2.spines.values():
                spine.set_visible(False)

            # Add total value labels on top of the stacked bars
            for i, v in enumerate(totals.values):
                ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    return filename


def create_scatter_memory(df: pd.DataFrame, filename: str) -> str:
    """
    Create a scatter plot of max_rss vs max_uss.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing benchmark data
    filename : str
        Output filename

    Returns
    -------
    None

    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="max_rss", y="max_uss", hue="run", style="run", data=df)
    plt.title("Memory Usage: max_rss vs max_uss")
    plt.xlabel("Maximum Resident Set Size (MB)")
    plt.ylabel("Maximum Unique Set Size (MB)")

    # Add explanatory note
    note = (
        "RSS (Resident Set Size): Total memory allocated to the process, including "
        "shared libraries.\n"
        "USS (Unique Set Size): Memory unique to the process, excluding shared "
        "libraries."
    )
    plt.text(
        0.05,
        -0.15,
        note,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        filename, bbox_inches="tight"
    )  # Added bbox_inches='tight' to include the text box
    plt.close()

    return filename


def main():
    """Read benchmark data and generate plots."""
    # Read data from main and feature run
    df = read_benchmarks(DIR_ARTIFACTS / "benchmarks")

    plots = []
    # 1. Execution time comparison plots
    file_name = create_bar_chart_comparison(
        df, "s", "Execution Time", "Time (seconds)", "execution_time.png"
    )
    plots.append(file_name)

    # 2. Memory peak comparison plots
    file_name = create_bar_chart_comparison(
        df,
        "max_rss",
        "Memory Peak",
        "Max RSS (MB)",
        "memory_peak.png",
        ignore_stacked_plot=True,
    )
    plots.append(file_name)

    # 3. Scatter plot of max_rss vs max_uss
    file_name = create_scatter_memory(df, "memory_scatter.png")
    plots.append(file_name)

    plots_string = " ".join(plots)

    print(plots_string)


if __name__ == "__main__":
    main()
