"""
Draft comment for pypsa-validator GitHub PRs.

Script can be called via command line or imported as a module.
"""

import os

# from pathlib import Path
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

    # Calculate the mean of normalized errors
    min_max_normalized_mae = np.mean(normalized_errors)

    return min_max_normalized_mae


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
    numeric_mask = np.vectorize(
        lambda x: isinstance(x, (int, float)) and np.isfinite(x)
    )(arr)
    return numeric_mask


class RunSuccessfull:
    def __init__(self, kwargs):
        self.dir_main = kwargs.get("dir_main", "")
        self.dir_feature = kwargs.get("dir_feature", "")
        self.plots_hash = kwargs.get("plots_hash", "")
        self.plots = [
            plot.split("/")[-1]
            for plot in kwargs.get("plots", [])
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
        for plot in self.plots:
            url_a = base_url + "main/" + plot
            url_b = base_url + "feature/" + plot
            rows.append(
                [
                    f'<img src="{url_a}" alt="Image not found">',
                    f'<img src="{url_b}" alt="Image not found">',
                ]
            )

        df = pd.DataFrame(
            rows, columns=pd.Index(["Main branch", "Feature branch"]), index=self.plots
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
                    path_in_main = os.path.join(root, file)
                    relative_path = os.path.relpath(path_in_main, self.dir_main)
                    index_str = "../" + "/".join(str(relative_path).split("/")[1:])
                    path_in_feature = os.path.join(self.dir_feature, relative_path)

                    if not os.path.exists(path_in_feature):
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
                    path_in_feature = os.path.join(root, file)
                    relative_path = os.path.relpath(path_in_feature, self.dir_feature)
                    index_str = "../" + "/".join(str(relative_path).split("/")[1:])

                    if not os.path.exists(path_in_feature):
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

        final_text = (
            f"{df.to_html(escape=False)}\n"
            f"\n"
            f"MAPE: Mean Absolute Percentage Error\n"
            f"NMAE: Mean Absolute Error on Min-Max Normalized Data\n"
            f"Status Thresholds: NMAE > 0.05 and MAPE > 5%\n\n"
        )
        return final_text

    @property
    def body_sucessfull(self) -> str:
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


class RunFailed:
    def __init__(self, kwargs):
        self.errors_main = kwargs.get("errors_main", [])
        self.errors_feature = kwargs.get("errors_feature", [])

    def body_failed(self) -> str:
        """Body text for failed run."""
        return (
            f":warning: Run failed!\n"
            f"\n"
            f"#### Main branch\n"
            f"```diff\n"
            f"{self.errors_main}\n"
            f"```\n"
            f"\n"
            f"#### Feature branch\n"
            f"```diff\n"
            f"{self.errors_feature}\n"
            f"```\n"
            f"\n"
        )


def get_env_var(var_name, default=None):
    value = os.getenv(var_name, default)
    if value is None or value == "":
        raise EnvironmentError(f"The environment variable '{var_name}' is not set.")
    return value


class Comment(RunSuccessfull, RunFailed):
    """Class to generate pypsa validator comment for GitHub PRs."""

    def __init__(
        self,
        # Header and subtext
        repo: str = get_env_var("REPO"),
        artifact_url: str = get_env_var("ARTIFACT_URL"),
        branch_name_main: str = get_env_var("BRANCH_NAME_MAIN"),
        branch_name_feature: str = get_env_var("BRANCH_NAME_FEATURE"),
        hash_main: str = get_env_var("HASH_MAIN"),
        hash_feature: str = get_env_var("HASH_FEATURE"),
        ahead_count: str = get_env_var("AHEAD_COUNT"),
        behind_count: str = get_env_var("BEHIND_COUNT"),
        git_diff_config: str = get_env_var("GIT_DIFF_CONFIG"),
        # For bodys
        dir_main: str = get_env_var("DIR_MAIN"),
        dir_feature: str = get_env_var("DIR_FEATURE"),
        dir_logs: str = get_env_var("DIR_LOGS"),
        # For RunSuccessfull body
        plots_hash: str = "",
        plots: list = [],
    ):
        """Initialize the Comment object."""
        self.repo = repo
        self.artifact_url = artifact_url
        self.branch_name_main = branch_name_main
        self.branch_name_feature = branch_name_feature
        self.hash_main = hash_main
        self.hash_feature = hash_feature
        self.ahead_count = ahead_count
        self.behind_count = behind_count
        self.git_diff_config = git_diff_config
        if self.sucessfull_run:
            RunSuccessfull.__init__(self, locals())
        else:
            RunFailed.__init__(self, locals())

    @property
    def sucessfull_run(self) -> bool:
        return True

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
            f"I am the Validator. Download all artifacts [here]({self.artifact_url}).\n"
            f"I'll be back and edit this comment for each new commit.\n\n"
            # f"**Config**\nprefix: `{self.config_prefix}`\n\n"
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
                f"{self.repo}/commits/{self.hash_feature})) "
            )
        if self.hash_main:
            hash_main = (
                f"([{self.hash_main[:7]}](https://github.com/"
                f"{self.repo}/commits/{self.hash_main}))"
            )
        time = (
            pd.Timestamp.now()
            .tz_localize("UTC")
            .tz_convert("Europe/Berlin")
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        return (
            f"Comparing `{self.branch_name_feature}` {hash_feature}with "
            f"{self.branch_name_main} {hash_main}.\n"
            f"Branch is {self.ahead_count} commits ahead and {self.behind_count} "
            f"commits behind.\n"
            f"Last updated on `{time}`."
        )

    def __repr__(self) -> str:
        """Return full formatted comment."""
        return (
            f"{self.header}"
            f"{self.body_failed if not self.sucessfull_run else ''}"
            f"{self.config_diff if self.git_diff_config else ''}"
            f"{self.body_sucessfull if self.sucessfull_run else ''}"
            f"{self.subtext}"
        )


if __name__ == "__main__":
    comment = Comment()

    print(comment)
