from typing import Any


def create_details_block(summary: str, content: Any) -> str:
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
