import re

import pandas as pd

from reporter.components.utils import create_details_block
from reporter.meta.config import PLOTS, PREFIX
from reporter.utils import URL_IMAGE_PLACEHOLDER


def plots_table() -> str:
    """Plots comparison table."""
    plots_list = re.split(r"\s+", PLOTS)
    plots_list = [x for x in plots_list if x]
    rows: list = []
    for plot in plots_list:
        if not plot:
            continue
        url_a = URL_IMAGE_PLACEHOLDER.format(f"main/results/{PREFIX}/{plot}")
        url_b = URL_IMAGE_PLACEHOLDER.format(f"feat/results/{PREFIX}/{plot}")
        rows.append(
            [
                f'<img src="{url_a}" alt="Image not available">',
                f'<img src="{url_b}" alt="Image not available">',
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=pd.Index(["Main branch", "Feature branch"]),
        index=plots_list,
    )
    return df.to_html(escape=False, index=False) + "\n"


def text() -> str:
    """Body text for general component."""
    return f"**General**\n" f"{create_details_block('Plots comparison', plots_table())}"
