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
    y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-5
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

    # Ignore -inf and inf values in y_true
    y_true = y_true[np.isfinite(y_true)]
    y_pred = y_pred[np.isfinite(y_pred)]

    # Avoid division by zero
    y_true = y_true + epsilon
    y_pred = y_pred + epsilon

    return np.mean(np.abs((y_true - y_pred) / y_true))


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


def get_env_var(var_name: str, default: Any = None) -> str:
    """Get environment variable or raise an error if not set and no default provided."""
    value = os.getenv(var_name, default)
    if value == "" and default is None:
        msg = f"The environment variable '{var_name}' is not set."
        raise OSError(msg)
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
    dir_artifacts: Path = Path(get_env_var("HOME")) / "artifacts"
    # For RunSuccessfull body
    plots_hash: str = get_env_var("PLOTS_HASH")
    plots_string: str = get_env_var("PLOTS")

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
        self.dir_main = self.dir_artifacts / "results (main branch)"
        self.dir_feature = self.dir_artifacts / "results (feature branch)"
        self.plots_list = [
            plot.split("/")[-1]
            for plot in self.plots_string.split(" ")
            if plot.split(".")[-1] in ["png", "jpg", "jpeg", "svg"]
        ]

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

    @property
    def plots_table(self) -> str:
        """Plots comparison table."""
        base_url = f"https://raw.githubusercontent.com/lkstrp/pypsa-validator/{self.plots_hash}/_validation-images/"

        rows: list = []
        for plot in self.plots_list:
            url_a = base_url + "main/" + plot
            url_b = base_url + "feature/" + plot
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

    @property
    def files_table(self) -> str:
        """Files comparison table."""
        rows = {}

        # Loop through all files in main dir
        for root, _, files in os.walk(self.dir_main):
            for file in files:
                if file.endswith(".csv"):
                    path_in_main = Path(root) / file
                    relative_path = os.path.relpath(path_in_main, self.dir_main)
                    index_str = "../" + "/".join(str(relative_path).split("/")[1:])
                    path_in_feature = self.dir_feature / relative_path

                    if not path_in_feature.exists():
                        rows[file] = [index_str, "", self.STATUS_FILE_MISSING, "", ""]
                        continue

                    df1 = pd.read_csv(path_in_main)
                    df2 = pd.read_csv(path_in_feature)

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
                if file.endswith(".csv"):
                    path_in_feature = Path(root) / file
                    relative_path = os.path.relpath(path_in_feature, self.dir_feature)
                    index_str = "../" + "/".join(str(relative_path).split("/")[1:])

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
            f"Status Thresholds: NMAE > 0.05 and MAPE > 5%\n\n"
        )

    @property
    def body(self) -> str:
        """Body text for successfull run."""
        return (
            f"<details>\n"
            f"    <summary>Result plots comparison</summary>\n"
            f"{self.plots_table}"
            f"</details>\n"
            f"\n"
            f"\n"
            f"<details>\n"
            f"    <summary>Result files comparison</summary>\n"
            f"{self.files_table}"
            f"</details>\n"
            f"\n"
            f"\n"
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
            else f"`failed in: `{'`, `'.join(main_errors)}`"
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


if __name__ == "__main__":
    comment = Comment()

    print(comment)  # noqa T201
