"""
Module to retrieve repository information.

Provides:
    - BRANCH_FEAT: Feature branch name.
    - BRANCH_MAIN: Main branch name.
    - DIFF_CONFIG: Difference between feature and main branch for the snakemake config
      file.
    - AHEAD_COUNT: Number of commits ahead of the main branch.
    - BEHIND_COUNT: Number of commits behind the main branch.
"""

import git

from reporter.meta.config import HASH_FEAT, HASH_MAIN, PATH_CONFIG
from reporter.meta.paths import PATH_REPO

repo = git.Repo(PATH_REPO)

try:
    repo.git.fetch("--all")
except git.exc.GitCommandError:
    print("Failed to fetch all branches.")

try:
    BRANCH_FEAT = repo.commit(HASH_FEAT).name_rev.split(" ")[-1].split("/")[-1]
    HASH_FEAT_SHORT = HASH_FEAT[:7]
except git.exc.GitCommandError:
    msg = f"Failed to checkout feature branch with hash {HASH_FEAT}."
    raise ValueError(msg)

try:
    BRANCH_MAIN = repo.commit(HASH_MAIN).name_rev.split(" ")[-1].split("/")[-1]
    HASH_MAIN_SHORT = HASH_MAIN[:7]
except git.exc.GitCommandError:
    msg = f"Failed to checkout main branch with hash {HASH_MAIN}."
    raise ValueError(msg)


DIFF_CONFIG = repo.git.diff(HASH_MAIN, HASH_FEAT, PATH_CONFIG)
AHEAD_COUNT = len(
    list(
        repo.iter_commits(
            f"{HASH_FEAT}...{HASH_MAIN}",
            left_only=True,
        )
    )
)
BEHIND_COUNT = len(
    list(
        repo.iter_commits(
            f"{HASH_FEAT}...{HASH_MAIN}",
            right_only=True,
        )
    )
)
