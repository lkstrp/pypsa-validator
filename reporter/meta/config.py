from reporter.meta.paths import PATH_ARTIFACTS, PATH_REPO
from reporter.utils import read_env_var

# Project
PLATFORM = read_env_var("PLATFORM", "gitlab")
DOMAIN = read_env_var("DOMAIN", "https://gitlab.com")
OWNER = read_env_var("OWNER")
REPO = read_env_var("REPO")
URL_RUNNER = read_env_var("URL_RUNNER", "google.com")

# Comparing branches
HASH_MAIN = read_env_var("HASH_MAIN")
HASH_FEAT = read_env_var("HASH_FEAT")

# Configuration
PLOTS = read_env_var("PLOTS")


prefixes = [
    prefix
    for prefix in (PATH_ARTIFACTS / "main" / "results").iterdir()
    if prefix.is_dir()
]

PREFIX = prefixes[0].name # TODO: Handle different prefixes

PATH_CONFIG = PATH_REPO / read_env_var("PATH_CONFIG")
