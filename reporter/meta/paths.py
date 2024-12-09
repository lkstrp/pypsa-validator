from pathlib import Path

# Mounted Piaths
PATH_ARTIFACTS = Path("/artifacts")
PATH_REPO = Path("/repo")

# Do some checks
if not PATH_ARTIFACTS.exists():
    msg = f"Path '{PATH_ARTIFACTS}' must be mounted."
    raise OSError(msg)
needed_subdir = [
    "feat/benchmarks",
    "feat/results",
    "feat/logs",
    "feat/.snakemake/log",
    "main/benchmarks",
    "main/results",
    "main/logs",
    "main/.snakemake/log",
]
if not all((PATH_ARTIFACTS / subdir).exists() for subdir in needed_subdir):
    msg = f"Path '{PATH_ARTIFACTS}' must contain subdirectories: {needed_subdir}"
    raise OSError(msg)

if not PATH_REPO.exists():
    msg = f"Path '{PATH_REPO}' must be mounted."
    raise OSError(msg)
