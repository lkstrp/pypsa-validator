"""
Draft comment for pypsa-validator GitHub PRs.

Script can be called via command line or imported as a module.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import argparse


def min_max_normalized_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculate the min-max normalized Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : array-like
        True values

    y_pred : array-like
        Predicted values

    Returns
    -------
    float: Min-max normalized MAE

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ignore -inf and inf values in y_true
    y_true = y_true[np.isfinite(y_true)]
    y_pred = y_pred[np.isfinite(y_pred)]

    # Calculate the absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Check if all errors are the same
    if np.all(abs_errors == abs_errors[0]):
        return 0  # Return 0 if all errors are identical to avoid division by zero

    # Min-max normalization
    min_error = np.min(abs_errors)
    max_error = np.max(abs_errors)

    normalized_errors = (abs_errors - min_error) / (max_error - min_error)

    return np.mean(normalized_errors)


def mean_absolute_percentage_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    epsilon: float = 1e-5,
    aggregate: bool = True,
    ignore_inf=True,
) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    epsilon : float, optional (default=1e-10)
        Small value to avoid division by zero

    Returns
    -------
    float: MAPE

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ignore -inf and inf values
    if ignore_inf:
        y_true = y_true[np.isfinite(y_true)]
        y_pred = y_pred[np.isfinite(y_pred)]

    # Avoid division by zero
    y_true = y_true + epsilon
    y_pred = y_pred + epsilon

    # Calculate the absolute percentage errors
    mape = np.abs((y_true - y_pred) / y_true)

    if aggregate:
        mape = np.mean(mape)

    return mape


def create_numeric_mask(arr: ArrayLike) -> np.ndarray:
    """
    Create a mask where True indicates numeric and finite values.

    Parameters
    ----------
    arr : array-like
        Input array

    Returns
    -------
    np.ndarray: Numeric mask where True indicates numeric and finite sort_values

    """
    arr = np.array(arr)
    return np.vectorize(lambda x: isinstance(x, (int, float)) and np.isfinite(x))(arr)


def get_env_var(var_name: str, default: Any = None) -> Any:
    """Get environment variable or raise an error if not set and no default provided."""
    value = os.getenv(var_name, default)
    if value == "" and default is None:
        msg = f"The environment variable '{var_name}' is not set."
        raise OSError(msg)
    if str(value).lower() in ["true", "false"]:
        return str(value).lower() == "true"
    return value


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

    def errors(self, branch_type: str) -> list:
        """Return errors for branch type."""
        if branch_type not in ["main", "feature"]:
            msg = "Branch type must be 'main' or 'feature'."
            raise ValueError(msg)

        logs = list(
            Path(f"{self.dir_artifacts}/logs/{branch_type}/.snakemake/log/").glob("*")
        )
        if len(logs) != 1:
            msg = (
                f"Expected exactly one log file in {branch_type}.snakemake/log "
                "directory."
            )
            raise ValueError(msg)

        with logs[0].open() as file:
            log = file.read()

        pattern = r"(?<=\nError in rule )(.*?)(?=:\n)"

        return re.findall(pattern, log)

    @property
    def sucessfull_run(self) -> bool:
        """Check if run was successfull via errors in logs.

        Returns
        -------
        bool: True if run was successfull, False otherwise.
        """
        if self._sucessfull_run is None:
            self._sucessfull_run = not bool(
                self.errors("main") + self.errors("feature")
            )
        return self._sucessfull_run


class RunSuccessfull(CommentData):
    """Class to generate successfull run component."""

    def __init__(self):
        """Initialize class."""
        self.dir_main = [
            file
            for file in (self.dir_artifacts / "results (main branch)").iterdir()
            if file.is_dir()
        ]
        if len(self.dir_main) != 1:
            msg = "Expected exactly one directory in 'results (main branch)'."
            raise ValueError(msg)
        self.dir_main = self.dir_main[0]

        self.dir_feature = [
            file
            for file in (self.dir_artifacts / "results (feature branch)").iterdir()
            if file.is_dir()
        ]
        if len(self.dir_feature) != 1:
            msg = "Expected exactly one directory in 'results (feature branch)'."
            raise ValueError(msg)
        self.dir_feature = self.dir_feature[0]

        self.plots_list = [
            plot
            for plot in self.plots_string.split(" ")
            if plot.split(".")[-1] in ["png", "jpg", "jpeg", "svg"]
        ]

        self._variables_deviation_ds = None

        self.plots_base_url = f"https://raw.githubusercontent.com/lkstrp/pypsa-validator/{self.plots_hash}/_validation-images/"

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
    VARIABLES_THRESHOLD = 5

    @property
    def variables_deviation_ds(self):
        if self._variables_deviation_ds is not None:
            return self._variables_deviation_ds
        vars1 = pd.read_excel(self.dir_main / self.VARIABLES_FILE)
        vars2 = pd.read_excel(self.dir_feature / self.VARIABLES_FILE)
        vars1 = vars1.set_index("Variable").loc[
            :, [col for col in vars1.columns if str(col).replace(".", "", 1).isdigit()]
        ]
        vars2 = vars2.set_index("Variable").loc[
            :, [col for col in vars2.columns if str(col).replace(".", "", 1).isdigit()]
        ]

        assert vars1.index.equals(vars2.index)

        deviation = mean_absolute_percentage_error(
            vars1, vars2, ignore_inf=False, aggregate=False
        )
        deviation = pd.Series(
            np.round(deviation, 2).mean(axis=1), index=vars1.index
        ).sort_values(ascending=False)

        # Filter for threshold
        deviation = deviation.loc[deviation > self.VARIABLES_THRESHOLD]

        self._variables_deviation_ds = deviation
        return self._variables_deviation_ds

    @property
    def variables_plot_strings(self):
        plots = (
            self.variables_deviation_ds.index.to_series()
            .apply(lambda x: re.sub(r"[ |]", "-", x))
            .apply(lambda x: "ariadne_comparison/" + x + ".png")
            .to_list()
        )
        return plots

    @property
    def variables_comparison(self) -> str:
        if (
            not (self.dir_main / self.VARIABLES_FILE).exists()
            or not (self.dir_feature / self.VARIABLES_FILE).exists()
        ):
            return ""

        df = self.variables_deviation_ds.apply(lambda x: f"{x:.2f}%")
        df = pd.DataFrame(df, columns=["MAPE"])
        df.index.name = None

        return (
            f"{df.to_html(escape=False)}\n"
            f"\n"
            f"MAPE: Mean Absolute Percentage Error\n"
            f"Threshold: MAPE > 5%\n"
            f"Only variables reaching the threshold are shown. Find the equivalent "
            f"plot for all of them below.\n\n"
        )

    @property
    def changed_variables_plots(self) -> str:
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
                    f'<img src="{url_a}" alt="Error in loading image.">',
                    f'<img src="{url_b}" alt="Error in loading image.">',
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
                    f'<img src="{url_a}" alt="Image not found in results">',
                    f'<img src="{url_b}" alt="Image not found in results">',
                ]
            )

        df = pd.DataFrame(
            rows,
            columns=pd.Index(["Main branch", "Feature branch"]),
            index=self.plots_list,
        )
        return df.to_html(escape=False, index=False) + "\n"

    def _read_to_dataframe(self, path: Path) -> pd.DataFrame:
        """Read a file to a dataframe."""
        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        else:
            return None

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

                if not path_in_feature.exists():
                    rows[file] = [index_str, "", self.STATUS_FILE_MISSING, "", ""]
                    continue

                df1 = self._read_to_dataframe(path_in_main)
                if df1 is None:
                    continue
                df2 = self._read_to_dataframe(path_in_feature)

                if df1.equals(df2):
                    rows[file] = [index_str, "", self.STATUS_EQUAL, "", ""]

                # Numeric type mismatch
                elif df1.apply(pd.to_numeric, errors="coerce").equals(
                    df2.apply(pd.to_numeric, errors="coerce")
                ):
                    rows[file] = [index_str, "", self.STATUS_TYPE_MISMATCH, "", ""]

                # Nan mismatch
                elif not df1.isna().equals(df2.isna()):
                    rows[file] = [index_str, "", self.STATUS_NAN_MISMATCH, "", ""]

                # Inf mismatch
                elif not df1.isin([np.inf, -np.inf]).equals(
                    df2.isin([np.inf, -np.inf])
                ):
                    rows[file] = [index_str, "", self.STATUS_INF_MISMATCH, "", ""]
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
                    df1_des.loc[~numeric_mask] = np.nan
                    df2_des.loc[~numeric_mask] = np.nan

                    # Check for changes in numeric data
                    arr1_num = pd.to_numeric(df1.to_numpy()[numeric_mask])
                    arr2_num = pd.to_numeric(df2.to_numpy()[numeric_mask])

                    nmae = min_max_normalized_mae(arr1_num, arr2_num)
                    mape = mean_absolute_percentage_error(arr1_num, arr2_num)

                    if not df1_des.equals(df2_des):
                        status = self.STATUS_CHANGED_NON_NUMERIC
                    elif nmae > 0.05 and mape > 0.05:
                        status = self.STATUS_CHANGED_NUMERIC
                    else:
                        status = self.STATUS_ALMOST_EQUAL

                    rows[file] = [
                        index_str,
                        f"{numeric_mask.mean():.1%}",
                        status,
                        f"{nmae:.2f}",
                        f"{mape*100:.1f}%" if mape < 1 else f"{mape*100:.2e}%",
                    ]

        # Loop through all files in feature dir to check for new files
        for root, _, files in os.walk(self.dir_feature):
            for file in files:
                path_in_feature = Path(root) / file
                relative_path = os.path.relpath(path_in_feature, self.dir_feature)
                index_str = "../" + "/".join(str(relative_path).split("/")[1:])

                df = self._read_to_dataframe(path_in_feature)
                if df is None:
                    continue

                if not path_in_feature.exists():
                    rows[file] = [index_str, "", self.STATUS_NEW, "", ""]

        # Combine and sort the results
        df = pd.DataFrame(rows, index=["Path", "Numeric", "Status", "NMAE", "MAPE"]).T

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
            f"MAPE: Mean Absolute Percentage Error\n"
            f"NMAE: Mean Absolute Error on Min-Max Normalized Data\n"
            f"Status Threshold: NMAE > 0.05 and MAPE > 5%\n\n"
        )

    @property
    def body(self) -> str:
        """Body text for successfull run."""

        def create_details_block(summary: str, content: str) -> str:
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

        return (
            f"{create_details_block('Variables comparison', self.variables_comparison)}"
            f"{create_details_block('Variables changed plots', self.changed_variables_plots)}"
            f"{create_details_block('General Plots comparison', self.plots_table)}"
            f"{create_details_block('General Files comparison', self.files_table)}"
        )

    def __call__(self) -> str:
        """Return text for successfull run component."""
        return self.body


class RunFailed(CommentData):
    """Class to generate failed run component."""

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


class Comment(CommentData):
    """Class to generate pypsa validator comment for GitHub PRs."""

    @property
    def header(self) -> str:
        """
        Header text.

        Contains the title, identifier, and short description.
        """
        return (
            f""
            f"<!-- _val-bot-id-keyword_ -->\n"
            f"## Validator Report\n"
            f"I am the Validator. Download all artifacts [here](https://github.com/"
            f"{self.github_repository}/actions/runs/{self.github_run_id}).\n"
            f"I'll be back and edit this comment for each new commit.\n\n"
        )

    @property
    def config_diff(self) -> str:
        """
        Config diff text.

        Only use when there are changes in the config.
        """
        return (
            f"<details>\n"
            f"    <summary>:warning: Config changes detected!</summary>\n"
            f"\n"
            f"Results may differ due to these changes:\n"
            f"```diff\n"
            f"{self.git_diff_config}\n"
            f"```\n"
            f"</details>\n\n"
        )

    @property
    def subtext(self) -> str:
        """Subtext for the comment."""
        if self.hash_feature:
            hash_feature = (
                f"([{self.hash_feature[:7]}](https://github.com/"
                f"{self.github_repository}/commits/{self.hash_feature})) "
            )
        if self.hash_main:
            hash_main = (
                f"([{self.hash_main[:7]}](https://github.com/"
                f"{self.github_repository}/commits/{self.hash_main}))"
            )
        time = (
            pd.Timestamp.now()
            .tz_localize("UTC")
            .tz_convert("Europe/Berlin")
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        return (
            f"Comparing `{self.github_head_ref}` {hash_feature}with "
            f"`{self.github_base_ref}` {hash_main}.\n"
            f"Branch is {self.ahead_count} commits ahead and {self.behind_count} "
            f"commits behind.\n"
            f"Last updated on `{time}`."
        )

    def needed_plots(self):
        if self.sucessfull_run:
            body_sucessfull = RunSuccessfull()
            plots_string = " ".join(body_sucessfull.variables_plot_strings)
            return plots_string
        else:
            ""

    def __repr__(self) -> str:
        """Return full formatted comment."""
        if self.sucessfull_run:
            body_sucessfull = RunSuccessfull()

            return (
                f"{self.header}"
                f"{self.config_diff if self.git_diff_config else ''}"
                f"{body_sucessfull()}"
                f"{self.subtext}"
            )

        else:
            body_failed = RunFailed()

            return (
                f"{self.header}"
                f"{body_failed()}"
                f"{self.config_diff if self.git_diff_config else ''}"
                f"{self.subtext}"
            )


def main():
    parser = argparse.ArgumentParser(description="Process some comments.")
    parser.add_argument(
        "command", nargs="?", default="", help='Command to run, e.g., "plots".'
    )
    args = parser.parse_args()

    comment = Comment()

    if args.command == "plots":
        print(comment.needed_plots())

    else:
        print(comment)  # noqa T201


if __name__ == "__main__":
    main()
