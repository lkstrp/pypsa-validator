import pandas as pd

from reporter.meta.config import DOMAIN, OWNER, REPO
from reporter.meta.repo import (
    AHEAD_COUNT,
    BEHIND_COUNT,
    BRANCH_FEAT,
    BRANCH_MAIN,
    HASH_FEAT,
    HASH_FEAT_SHORT,
    HASH_MAIN,
    HASH_MAIN_SHORT,
)


def text() -> str:
    """Subtext for the comment."""
    hash_feature = (
        f"([{HASH_FEAT_SHORT}]({DOMAIN}/" f"{OWNER}/{REPO}/commits/{HASH_FEAT})) "
    )
    hash_main = (
        f"([{HASH_MAIN_SHORT}]({DOMAIN}/" f"{OWNER}/{REPO}/commits/{HASH_MAIN}))"
    )
    time = (
        pd.Timestamp.now()
        .tz_localize("UTC")
        .tz_convert("Europe/Berlin")
        .strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    return (
        f"Comparing `{BRANCH_FEAT}` {hash_feature}with "
        f"`{BRANCH_MAIN}` {hash_main}.\n"
        f"Branch is {AHEAD_COUNT} commits ahead and {BEHIND_COUNT} "
        f"commits behind.\n"
        f"Last updated on `{time}`."
    )
