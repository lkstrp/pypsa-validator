"""Text components to generate validator report."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import min_max_normalized_mae, normalized_root_mean_square_error
from utils import get_env_var


@dataclass
class CommentData:
    """Class to store data for comment generation."""

    github_repository: str = get_env_var("GITHUB_REPOSITORY")
    github_run_id: str = get_env_var("GITHUB_RUN_ID")
    github_base_ref: str = get_env_var("GITHUB_BASE_REF")
    github_head_ref: str = get_env_var("GITHUB_HEAD_REF")
    hash_main: str = get_env_var("HASH_MAIN")
    hash_feature: str = get_env_var("HASH_FEATURE")
    ahead_count: str = get_env_var("AHEAD_COUNT")
    behind_count: str = get_env_var("BEHIND_COUNT")
    git_diff_config: str = get_env_var("GIT_DIFF_CONFIG", default="")
    # For bodys
    dir_artifacts: Path = Path(
        get_env_var("DIR_ARTIFACTS", Path(get_env_var("HOME")) / "artifacts")
    )
    # For RunSuccessfull body
    plots_hash: str = get_env_var("PLOTS_HASH", "")
    plots_string: str = get_env_var("PLOTS", "")

    _sucessfull_run = None

    def __init__(self):
        """Initialize comment data class."""
        self.plots_base_url = (
            f"https://raw.githubusercontent.com/lkstrp/"
            f"pypsa-validator/{self.plots_hash}/_validation-images/"
        )

    def errors(self, branch_type: str) -> list:
        """Return errors for branch type."""
        if branch_type not in ["main", "feature"]:
            msg = "Branch type must be 'main' or 'feature'."
            raise ValueError(msg)

        logs = list(
            Path(f"{self.dir_artifacts}/logs/{branch_type}/.snakemake/log/").glob("*")
        )
        if len(logs) > 1:
            msg = (
                "Expected exactly one log fiie in snakemake/log directory "
                f"({branch_type} branch). Found {len(logs)}."
            )
            raise ValueError(msg)
        elif len(logs) == 0:
            inpt_erros = ['no_logs_found']
            return inpt_erros

        with logs[0].open() as file:
            log = file.read()

        rule_errors = re.findall(r"(?<=\nError in rule )(.*?)(?=:\n)", log)
        inpt_errors = re.findall(
            r"(?<=\nMissingInputException: Missing input files for rule )(.*?)(?=:\n)",
            log,
        )
        inpt_errors = list(set(inpt_errors))

        return rule_errors + inpt_errors

    @property
    def sucessfull_run(self) -> bool:
        """
        Check if run was successfull via errors in logs.

        Returns
        -------
        bool: True if run was successfull, False otherwise.

        """
        if self._sucessfull_run is None:
            self._sucessfull_run = not bool(
                self.errors("main") + self.errors("feature")
            )
        return self._sucessfull_run


def get_deviation_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Calculate deviation dataframe between two dataframes."""
    nrmse_series = df1.apply(
        lambda row: normalized_root_mean_square_error(
            row.values,
            df2.loc[row.name].values,
            normalization="min-max",
        ),
        axis=1,
    )
    pearson_series = df1.apply(
        lambda row: np.corrcoef(row.values, df2.loc[row.name].values)[0, 1],
        axis=1,
    ).fillna(0)

    if df1.empty:
        return pd.DataFrame(columns=["NRMSE", "Pearson"])
    else:
        deviation_df = pd.DataFrame(
            {"NRMSE": nrmse_series, "Pearson": pearson_series}
        ).sort_values(by="NRMSE", ascending=False)

        return deviation_df


def create_details_block(summary: str, content: str) -> str:
    """Wrap content in a details block (if content is not empty)."""
    if content:
        return (
            f"<details>\n"
            f"    <summary>{summary}</summary>\n"
            f"{content}"
            f"</details>\n"
            f"\n"
            f"\n"
        )
    else:
        return ""


class RunSuccessfullComponent(CommentData):
    """Class to generate successfull run component."""

    def __init__(self):
        """Initialize successfull run component."""
        super().__init__()
        self.dir_main = [
            file
            for file in (self.dir_artifacts / "results/main/results").iterdir()
            if file.is_dir()
        ]
        if len(self.dir_main) != 1:
            msg = "Expected exactly one directory (prefix) in 'results/main/results'."
            raise ValueError(msg)
        self.dir_main = self.dir_main[0]

        self.dir_feature = [
            file
            for file in (self.dir_artifacts / "results/feature/results").iterdir()
            if file.is_dir()
        ]
        if len(self.dir_feature) != 1:
            msg = (
                "Expected exactly one directory (prefix) in 'results/feature/results'."
            )
            raise ValueError(msg)
        self.dir_feature = self.dir_feature[0]

        self.plots_list = [
            plot
            for plot in self.plots_string.split(" ")
            if plot.split(".")[-1] in ["png", "jpg", "jpeg", "svg"]
        ]

        self._variables_deviation_df = None

    # Status strings for file comparison table
    STATUS_FILE_MISSING = " :warning: Missing"
    STATUS_EQUAL = ":white_check_mark: Equal"
    STATUS_TYPE_MISMATCH = ":warning: Type mismatch"
    STATUS_NAN_MISMATCH = ":warning: NaN mismatch"
    STATUS_INF_MISMATCH = ":warning: Inf mismatch"
    STATUS_CHANGED_NUMERIC = ":warning:Changed"
    STATUS_CHANGED_NON_NUMERIC = ":warning: Changed (non-numeric data)"
    STATUS_ALMOST_EQUAL = ":white_check_mark: Almost equal"
    STATUS_NEW = ":warning: New"

    VARIABLES_FILE = "KN2045_Bal_v4/ariadne/exported_variables_full.xlsx"
    VARIABLES_THRESHOLD = 2

    @property
    def variables_deviation_df(self):
        """Get the deviation dataframe for variables."""
        if self._variables_deviation_df is not None:
            return self._variables_deviation_df
        vars1 = pd.read_excel(self.dir_main / self.VARIABLES_FILE)
        vars2 = pd.read_excel(self.dir_feature / self.VARIABLES_FILE)
        vars1 = vars1.set_index("Variable").loc[
            :, [col for col in vars1.columns if str(col).replace(".", "", 1).isdigit()]
        ]
        vars2 = vars2.set_index("Variable").loc[
            :, [col for col in vars2.columns if str(col).replace(".", "", 1).isdigit()]
        ]

        assert vars1.index.equals(vars2.index)

        deviation_df = get_deviation_df(vars1, vars2)

        # Filter for threshold
        deviation_df = deviation_df.loc[
            deviation_df["NRMSE"] > self.VARIABLES_THRESHOLD
        ]

        self._variables_deviation_df = deviation_df
        return self._variables_deviation_df

    @property
    def variables_plot_strings(self):
        """Return list of variable plot strings."""
        plots = (
            self.variables_deviation_df.index.to_series()
            .apply(lambda x: re.sub(r"[ |/]", "-", x))
            .apply(lambda x: "ariadne_comparison/" + x + ".png")
            .to_list()
        )
        return plots

    @property
    def variables_comparison(self) -> str:
        """Return variables comparison table."""
        if (
            not (self.dir_main / self.VARIABLES_FILE).exists()
            or not (self.dir_feature / self.VARIABLES_FILE).exists()
        ):
            return ""

        df = self.variables_deviation_df.map(lambda x: f"{x:.3f}")
        df.index.name = None

        return (
            f"{df.to_html(escape=False)}\n"
            f"\n"
            f"NRMSE: Normalized (min-max) Root Mean Square Error\n"
            f"Pearson: Pearson correlation coefficient\n"
            f"Threshold: NRMSE > {self.VARIABLES_THRESHOLD}\n"
            f"Only variables reaching the threshold are shown. Find the equivalent "
            f"plot for all of them below.\n\n"
        )

    @property
    def changed_variables_plots(self) -> str:
        """Return plots for variables that have changed significantly."""
        if (
            not (self.dir_main / self.VARIABLES_FILE).exists()
            or not (self.dir_feature / self.VARIABLES_FILE).exists()
        ):
            return ""

        rows: list = []
        for plot in self.variables_plot_strings:
            url_a = self.plots_base_url + "main/" + plot
            url_b = self.plots_base_url + "feature/" + plot
            rows.append(
                [
                    f'<img src="{url_a}" alt="Image not available">',
                    f'<img src="{url_b}" alt="Image not available">',
                ]
            )

        df = pd.DataFrame(
            rows,
            columns=pd.Index(["Main branch", "Feature branch"]),
        )
        return df.to_html(escape=False, index=False) + "\n"

    @property
    def plots_table(self) -> str:
        """Plots comparison table."""
        rows: list = []
        for plot in self.plots_list:
            url_a = self.plots_base_url + "main/" + plot
            url_b = self.plots_base_url + "feature/" + plot
            rows.append(
                [
                    f'<img src="{url_a}" alt="Image not available">',
                    f'<img src="{url_b}" alt="Image not available">',
                ]
            )

        df = pd.DataFrame(
            rows,
            columns=pd.Index(["Main branch", "Feature branch"]),
            index=self.plots_list,
        )
        return df.to_html(escape=False, index=False) + "\n"

    def _format_csvs_dir_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format csvs dir dataframes."""
        if "planning_horizon" in df.columns:
            df = df.T.reset_index()
        # Relevant header row
        header_row_index = df.index[df.iloc[:, 0] == "planning_horizon"].tolist()[0]
        # Set header
        df.columns = df.iloc[header_row_index]
        # Fill nan values in header
        df.columns = pd.Index(
            [
                f"col{i+1}" if pd.isna(col) or col == "" or col is None else col
                for i, col in enumerate(df.columns)
            ]
        )
        # Remove all rows before header
        df = df.iloc[header_row_index + 1 :]

        # Make non-numeric values the index
        non_numeric = df.apply(
            lambda col: pd.to_numeric(col, errors="coerce").isna().all()  # type: ignore
        )

        if non_numeric.any():
            df = df.set_index(df.columns[non_numeric].to_list())  # type: ignore
        else:
            df = df.set_index("planning_horizon")

        # Make the rest numeric
        df = df.apply(pd.to_numeric)

        df.index.name = None
        df.columns.name = None

        return df

    @property
    def files_table(self) -> str:
        """Files comparison table."""
        rows = {}

        # Loop through all files in main dir
        for root, _, files in os.walk(self.dir_main):
            for file in files:
                path_in_main = Path(root) / file
                relative_path = os.path.relpath(path_in_main, self.dir_main)
                index_str = "/".join(str(relative_path).split("/")[1:])
                path_in_feature = self.dir_feature / relative_path

                if path_in_main.parent.name == "csvs" and path_in_main.suffix == ".csv":
                    df1 = pd.read_csv(path_in_main)
                else:
                    continue

                if not path_in_feature.exists():
                    rows[file] = [index_str, self.STATUS_FILE_MISSING, "", ""]
                    continue
                else:
                    df2 = pd.read_csv(path_in_feature)

                df1 = self._format_csvs_dir_files(df1)
                df2 = self._format_csvs_dir_files(df2)

                if df1.equals(df2):
                    rows[file] = [index_str, self.STATUS_EQUAL, "", ""]

                # Numeric type mismatch
                elif df1.apply(pd.to_numeric, errors="coerce").equals(
                    df2.apply(pd.to_numeric, errors="coerce")
                ):
                    rows[file] = [index_str, self.STATUS_TYPE_MISMATCH, "", ""]

                # Nan mismatch
                elif not df1.isna().equals(df2.isna()):
                    rows[file] = [index_str, self.STATUS_NAN_MISMATCH, "", ""]

                # Inf mismatch
                elif not df1.isin([np.inf, -np.inf]).equals(
                    df2.isin([np.inf, -np.inf])
                ):
                    rows[file] = [index_str, self.STATUS_INF_MISMATCH, "", ""]
                # Changed
                else:
                    # Get numeric mask
                    numeric_mask = ~np.isnan(
                        df1.apply(pd.to_numeric, errors="coerce").to_numpy()
                    )
                    assert (
                        numeric_mask
                        == ~np.isnan(
                            df2.apply(pd.to_numeric, errors="coerce").to_numpy()
                        )
                    ).all()

                    # Check for changes in descriptive data
                    df1_des = df1.copy()
                    df2_des = df2.copy()
                    df1_des.loc[numeric_mask] = np.nan
                    df2_des.loc[numeric_mask] = np.nan

                    # Check for changes in numeric data
                    arr1_num = pd.to_numeric(df1.to_numpy()[numeric_mask])
                    arr2_num = pd.to_numeric(df2.to_numpy()[numeric_mask])

                    nrmse = normalized_root_mean_square_error(
                        arr1_num, arr2_num, normalization="min-max"
                    )
                    mae_n = min_max_normalized_mae(arr1_num, arr2_num)

                    if not df1_des.equals(df2_des):
                        status = self.STATUS_CHANGED_NON_NUMERIC
                    elif nrmse > 2 and mae_n > 0.05:
                        status = self.STATUS_CHANGED_NUMERIC
                    else:
                        status = self.STATUS_ALMOST_EQUAL

                    rows[file] = [
                        index_str,
                        status,
                        f"{nrmse:.3f}",
                        f"{mae_n:.2f}",
                    ]

        # Loop through all files in feature dir to check for new files
        for root, _, files in os.walk(self.dir_feature):
            for file in files:
                path_in_feature = Path(root) / file
                relative_path = os.path.relpath(path_in_feature, self.dir_feature)
                index_str = "../" + "/".join(str(relative_path).split("/")[1:])
                path_in_main = self.dir_main / relative_path

                if (
                    path_in_feature.parent.name == "csvs"
                    and path_in_feature.suffix == ".csv"
                ):
                    df1 = pd.read_csv(path_in_main)
                else:
                    continue

                if not path_in_main.exists():
                    rows[file] = [index_str, self.STATUS_NEW, "", ""]

        # Combine and sort the results
        df = pd.DataFrame(rows, index=["Path", "Status", "NRMSE", "MAE (norm)"]).T

        status_order = {
            self.STATUS_CHANGED_NUMERIC: 0,
            self.STATUS_CHANGED_NON_NUMERIC: 1,
            self.STATUS_TYPE_MISMATCH: 2,
            self.STATUS_NAN_MISMATCH: 3,
            self.STATUS_INF_MISMATCH: 4,
            self.STATUS_FILE_MISSING: 5,
            self.STATUS_NEW: 6,
            self.STATUS_ALMOST_EQUAL: 7,
            self.STATUS_EQUAL: 8,
        }
        df = df.sort_values(by="Status", key=lambda x: x.map(status_order))
        df = df.set_index("Path")
        df.index.name = None

        return (
            f"{df.to_html(escape=False)}\n"
            f"\n"
            f"NRMSE: Normalized (min-max) Root Mean Square Error\n"
            f"MAE (norm): Mean Absolute Error on normalized Data (min-max\n"
            f"Status Threshold: MAE (norm) > 0.05 and NRMSE > 2\n"
        )

    @property
    def body(self) -> str:
        """Body text for successfull run."""
        if self.variables_comparison and self.changed_variables_plots:
            if self.variables_deviation_df.empty:
                variables_txt = (
                    "**Ariadne Variables**\n"
                    "No significant changes in variables detected. :white_check_mark:\n"
                    "\n\n"
                )
            else:
                variables_txt = (
                    f"**Ariadne Variables**\n"
                    f"{create_details_block('Comparison', self.variables_comparison)}"
                    f"{create_details_block('Plots', self.changed_variables_plots)}"
                )
        elif self.variables_comparison or self.changed_variables_plots:
            raise ValueError(
                "Both variables_comparison and changed_variables_plots must be set or "
                "unset."
            )
        else:
            variables_txt = ""
        return (
            f"{variables_txt}"
            f"**General**\n"
            f"{create_details_block('Plots comparison', self.plots_table)}"
            f"{create_details_block('Files comparison', self.files_table)}"
        )

    def __call__(self) -> str:
        """Return text for successfull run component."""
        return self.body


class RunFailedComponent(CommentData):
    """Class to generate failed run component."""

    def __init__(self):
        """Initialize failed run component."""
        super().__init__()

    def body(self) -> str:
        """Body text for failed run."""
        main_errors = self.errors("main")
        feature_errors = self.errors("feature")

        main_status = (
            "passed! :white_check_mark:"
            if not main_errors
            else f"failed in: `{'`, `'.join(main_errors)}`"
        )
        feature_status = (
            "passed! :white_check_mark:"
            if not feature_errors
            else f"failed in: `{'`, `'.join(feature_errors)}`"
        )

        return (
            f"<details open>\n"
            f"    <summary>:exclamation: Run failed!</summary>\n\n"
            f"_Download 'logs' artifact to see more details._\n"
            f"- `{self.github_base_ref}` {main_status}\n"
            f"- `{self.github_head_ref}` {feature_status}\n"
            f"</details>\n"
            f"\n"
        )

    def __call__(self) -> str:
        """Return text for failed run component."""
        return self.body()


class ModelMetricsComponent(CommentData):
    """Class to generate model metrics component."""

    def __init__(self):
        """Initialize model metrics component."""
        super().__init__()

    @property
    def benchmark_plots(self) -> str:
        """Benchmark plots."""
        "execution_time.png", "memory_peak.png", "memory_scatter.png"
        return (
            f'<img src="{self.plots_base_url}benchmarks/execution_time.png" '
            'alt="Image not available">\n'
            f'<img src="{self.plots_base_url}benchmarks/memory_peak.png" '
            'alt="Image not available">\n'
            f'<img src="{self.plots_base_url}benchmarks/memory_scatter.png" '
            'alt="Image not available">\n'
        )

    def body(self) -> str:
        """Body text for Model Metrics."""
        return (
            f"**Model Metrics**\n"
            f"{create_details_block('Benchmarks', self.benchmark_plots)}\n"
        )

    def __call__(self) -> str:
        """Return text for model metrics component."""
        return self.body()