from reporter.meta.repo import DIFF_CONFIG


def text() -> str:
    """
    Config diff text.

    Only use when there are changes in the config.
    """
    if not DIFF_CONFIG:
        return ""
    else:
        return (
            f"<details>\n"
            f"    <summary>:warning: Config changes detected!</summary>\n"
            f"\n"
            f"Results may differ due to these changes:\n"
            f"```diff\n"
            f"{DIFF_CONFIG}\n"
            f"```\n"
            f"</details>\n\n"
        )
