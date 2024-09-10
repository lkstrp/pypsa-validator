"""
Draft comment for pypsa-validator GitHub PRs.

Script can be called via command line or imported as a module.
"""

import argparse

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from comment_components import (
    CommentData,
    ModelMetricsComponent,
    RunFailedComponent,
    RunSuccessfullComponent,
)


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


class Comment(CommentData):
    """Class to generate pypsa validator comment for GitHub PRs."""

    def __init__(self) -> None:
        """Initialize comment class. It will put all text components together."""
        super().__init__()

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

    def dynamic_plots(self) -> str:
        """
        Return a list of dynamic results plots needed for the comment.

        Returns
        -------
            str: Space separated list of dynamic plots.

        """
        if self.sucessfull_run:
            body_sucessfull = RunSuccessfullComponent()
            plots_string = " ".join(body_sucessfull.variables_plot_strings)
            return plots_string
        else:
            return ""

    def __repr__(self) -> str:
        """Return full formatted comment."""
        body_benchmarks = ModelMetricsComponent()
        if self.sucessfull_run:
            body_sucessfull = RunSuccessfullComponent()

            return (
                f"{self.header}"
                f"{self.config_diff if self.git_diff_config else ''}"
                f"{body_sucessfull()}"
                f"{body_benchmarks()}"
                f"{self.subtext}"
            )

        else:
            body_failed = RunFailedComponent()

            return (
                f"{self.header}"
                f"{body_failed()}"
                f"{self.config_diff if self.git_diff_config else ''}"
                f"{body_benchmarks()}"
                f"{self.subtext}"
            )


def main():
    """
    Run draft comment script.

    Command line interface for the draft comment script. Use no arguments to print the
    comment, or use the "plots" argument to print the dynamic plots which will be needed
    for the comment.
    """
    parser = argparse.ArgumentParser(description="Process some comments.")
    parser.add_argument(
        "command", nargs="?", default="", help='Command to run, e.g., "plots".'
    )
    args = parser.parse_args()

    comment = Comment()

    if args.command == "plots":
        print(comment.dynamic_plots())

    else:
        print(comment)  # noqa T201


if __name__ == "__main__":
    main()
