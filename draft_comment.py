"""
Draft comment for pypsa-validator GitHub PRs.

Script can be called via command line or imported as a module.
"""

import argparse
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


data = np.array([1, 2, "a", 3, np.nan, np.inf, -np.inf, 4.5, "3.14"], dtype=object)


class Comment:
    """Class to generate pypsa validator comment for GitHub PRs."""

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

    def __init__(
        self,
        repo: str,
        artifact_url: str,
        branch_name_base: str,
        branch_name_feature: str,
        config_prefix: str,
        git_diff_config: str,
        hash_base: str,
        hash_feature: str,
        dir_base: str,
        dir_feature: str,
        plots_hash: str,
        plots: list,
    ):
        """Initialize the Comment object."""
        self.repo = repo
        self.artifact_url = artifact_url
        self.branch_name_feature = branch_name_feature
        self.branch_name_base = branch_name_base
        self.config_prefix = config_prefix
        self.git_diff_config = git_diff_config
        self.hash_base = hash_base
        self.hash_feature = hash_feature
        self.dir_base = dir_base
        self.dir_feature = dir_feature
        self.plots_hash = plots_hash
        self.plots = [
            plot.split("/")[-1]
            for plot in plots
            if plot.split(".")[-1] in ["png", "jpg", "jpeg", "svg"]
        ]

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
            f"**Config**\nprefix: `{self.config_prefix}`\n\n"
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
            f"Results may differ due to these changes\n"
            f"```diff\n"
            f"{self.git_diff_config}\n"
            f"```\n"
            f"</details>\n\n"
        )

    @property
    def plots_table(self) -> str:
        """Plots comparison table."""
        base_url = f"https://raw.githubusercontent.com/lkstrp/pypsa-validator/{self.plots_hash}/_validation-images/"

        rows: list = []
        for plot in self.plots:
            url_a = base_url + "base/" + plot
            url_b = base_url + "feature/" + plot
            rows.append(
                [
                    f'<img src="{url_a}" alt="Image not found">',
                    f'<img src="{url_b}" alt="Image not found">',
                ]
            )

        df = pd.DataFrame(
            rows, columns=pd.Index(["Base branch", "Feature branch"]), index=self.plots
        )
        return df.to_html(escape=False) + "\n"

    @property
    def files_table(self) -> str:
        """Files comparison table."""
        rows = {}

        # Loop through all files in base dir
        for root, _, files in os.walk(self.dir_base):
            for file in files:
                if file.endswith(".csv"):
                    path_in_a = os.path.join(root, file)
                    relative_path = os.path.relpath(path_in_a, self.dir_base)
                    path_str = str(relative_path).replace(
                        f"{self.config_prefix}/", "../"
                    )
                    path_in_b = os.path.join(self.dir_feature, relative_path)

                    if not os.path.exists(path_in_b):
                        rows[file] = [path_str, "", self.STATUS_FILE_MISSING, "", ""]
                        continue

                    df1 = pd.read_csv(path_in_a)
                    df2 = pd.read_csv(path_in_b)

                    if df1.equals(df2):
                        rows[file] = [path_str, "", self.STATUS_EQUAL, "", ""]

                    # Numeric type mismatch
                    elif df1.apply(pd.to_numeric, errors="coerce").equals(
                        df2.apply(pd.to_numeric, errors="coerce")
                    ):
                        rows[file] = [path_str, "", self.STATUS_TYPE_MISMATCH, "", ""]

                    # Nan mismatch
                    elif not df1.isna().equals(df2.isna()):
                        rows[file] = [path_str, "", self.STATUS_NAN_MISMATCH, "", ""]

                    # Inf mismatch
                    elif not df1.isin([np.inf, -np.inf]).equals(
                        df2.isin([np.inf, -np.inf])
                    ):
                        rows[file] = [path_str, "", self.STATUS_INF_MISMATCH, "", ""]
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
                            path_str,
                            f"{numeric_mask.mean():.1%}",
                            status,
                            f"{nmae:.2f}",
                            f"{mape*100:.1f}%" if mape < 1 else f"{mape*100:.2e}%",
                        ]

        # Loop through all files in feature dir to check for new files
        for root, _, files in os.walk(self.dir_feature):
            for file in files:
                if file.endswith(".csv"):
                    path_in_b = os.path.join(root, file)
                    relative_path = os.path.relpath(path_in_b, self.dir_feature)
                    path_str = str(relative_path).replace(
                        f"{self.config_prefix}/", "../"
                    )
                    path_in_a = os.path.join(self.dir_base, relative_path)

                    if not os.path.exists(path_in_a):
                        rows[file] = [path_str, "", self.STATUS_NEW, "", ""]

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
    def subtext(self) -> str:
        """Subtext for the comment."""
        if self.hash_feature:
            hash_feature = (
                f"([{self.hash_feature[:7]}](https://github.com/"
                "{self.repo}/commits/{self.hash_feature})) "
            )
        if self.hash_base:
            hash_base = (
                f"([{self.hash_base[:7]}](https://github.com/"
                "{self.repo}/commits/{self.hash_base}))"
            )
        time = (
            pd.Timestamp.now()
            .tz_localize("UTC")
            .tz_convert("Europe/Berlin")
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        return (
            f"Comparing {self.branch_name_feature} {hash_feature}with "
            f"{self.branch_name_base} {hash_base}.\n"
            f"Last updated on {time}."
        )

    def __repr__(self) -> str:
        """Return full formatted comment."""
        return (
            f"{self.header}"
            f"{self.config_diff if self.git_diff_config else ''}"
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
            f"{self.subtext}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str)
    parser.add_argument("--artifact_url", type=str)
    parser.add_argument("--branch_name_base", type=str)
    parser.add_argument("--branch_name_feature", type=str)
    parser.add_argument("--hash_base", type=str, default="")
    parser.add_argument("--hash_feature", type=str, default="")
    parser.add_argument("--config_prefix", type=str)
    parser.add_argument("--dir_base", type=str)
    parser.add_argument("--dir_feature", type=str)
    parser.add_argument("--plots_hash", type=str)
    parser.add_argument("--plots", nargs="*", type=str)

    args = parser.parse_args()

    comment = Comment(
        repo=args.repo,
        artifact_url=args.artifact_url,
        branch_name_base=args.branch_name_base,
        branch_name_feature=args.branch_name_feature,
        hash_base=args.hash_base,
        hash_feature=args.hash_feature,
        config_prefix=args.config_prefix,
        git_diff_config=os.getenv("GIT_DIFF_CONFIG", ""),
        dir_base=args.dir_base,
        dir_feature=args.dir_feature,
        plots_hash=args.plots_hash,
        plots=args.plots,
    )

    print(comment)
