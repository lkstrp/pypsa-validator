import os
from typing import Any

URL_IMAGE_PLACEHOLDER = "<URL_IMAGE_PLACEHOLDER:{}>"
REPORT_IDENTIFIER = "<!-- _val-bot-identifier_ -->"


def read_env_var(var_name: str, default: Any = None, optional: bool = False) -> Any:
    """Get environment variable or raise an error if not set and no default provided."""
    value = os.getenv(var_name, default)

    # Check if set
    if not optional and value is None:
        msg = f"The environment variable '{var_name}' is not set."
        raise OSError(msg)

    # Type conversions
    if str(value).lower() in ["true", "false"]:
        value = str(value).lower() == "true"

    return value
