from reporter.meta.config import URL_RUNNER
from reporter.utils import REPORT_IDENTIFIER


def text() -> str:
    """
    Header text.

    Contains the title, identifier, and short description.
    """
    return (
        f""
        f"{REPORT_IDENTIFIER}\n"
        f"## Validator Report\n"
        f"I am the Validator. Download all artifacts [here]({URL_RUNNER}).\n"
        f"I'll be back and edit this comment for each new commit.\n\n"
    )
