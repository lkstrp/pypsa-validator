from typing import Any
import os


def get_env_var(var_name: str, default: Any = None) -> Any:
    """Get environment variable or raise an error if not set and no default provided."""
    value = os.getenv(var_name, default)
    if value == "" and default is None:
        msg = f"The environment variable '{var_name}' is not set."
        raise OSError(msg)
    if str(value).lower() in ["true", "false"]:
        return str(value).lower() == "true"
    return value
