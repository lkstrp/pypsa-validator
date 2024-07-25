import os
import pandas as pd
import argparse


def get_plot_comparison_table_html(commit_id, plots):
    base_url = f"https://raw.githubusercontent.com/lkstrp/pypsa-validator/{commit_id}/dir-validation-images/"

    rows = []
    for plot in plots:
        url_a = base_url + "base/" + plot
        url_b = base_url + "feature/" + plot
        rows.append(
            [
                f'<img src="{url_a}" alt="Image not found">',
                f'<img src="{url_b}" alt="Image not found">',
            ]
        )

    df = pd.DataFrame(rows, columns=["Base branch", "Feature branch"], index=plots)
    return df.to_html(escape=False)


def get_csv_comparison_table_html(dir_base, dir_feature):
    status_not_equal = "Changed :warning:"
    status_missing = "Missing :warning:"
    status_new = "New :warning:"

    not_equal = []
    missing = []
    new = []
    equal = []

    # Get list of all csv files in directory a
    for root, dirs, files in os.walk(dir_base):
        for file in files:
            if file.endswith(".csv"):
                path_in_a = os.path.join(root, file)
                relative_path = os.path.relpath(path_in_a, dir_base)
                path_in_b = os.path.join(dir_feature, relative_path)

                if not os.path.exists(path_in_b):
                    missing.append(
                        f"<tr><td>{relative_path}</td><td>{status_missing}</td></tr>\n"
                    )
                    continue

                df1 = pd.read_csv(path_in_a)
                df2 = pd.read_csv(path_in_b)

                try:
                    pd.testing.assert_frame_equal(df1, df2)
                    equal.append(f"<tr><td>{relative_path}</td><td>Equal</td></tr>\n")
                except AssertionError:
                    not_equal.append(
                        f"<tr><td>{relative_path}</td><td>{status_not_equal}</td></tr>\n"
                    )

    # Check for new files in directory b
    for root, dirs, files in os.walk(dir_feature):
        for file in files:
            if file.endswith(".csv"):
                path_in_b = os.path.join(root, file)
                relative_path = os.path.relpath(path_in_b, dir_feature)
                path_in_a = os.path.join(dir_base, relative_path)

                if not os.path.exists(path_in_a):
                    new.append(
                        f"<tr><td>{relative_path}</td><td>{status_new}</td></tr>\n"
                    )

    # Combine and sort the results
    result_table = (
        "<table><thead><tr><th>File Path</th><th>Status</th></tr></thead><tbody>\n"
        + "".join(not_equal + missing + new + equal)
        + "</tbody></table>"
    )

    n_files = len(not_equal) + len(missing) + len(new) + len(equal)
    parts = [
        f"{len(not_equal)}/{n_files} files have changed" if not_equal else "",
        f"{len(missing)}/{n_files} files are missing" if missing else "",
        f"{len(new)}/{n_files} files are new" if new else "",
    ]
    parts = [part for part in parts if part]  # Remove empty strings
    if parts:
        equal_count = n_files - sum(len(x) for x in [not_equal, missing, new])
        if equal_count > 0:
            parts.append("and the rest are equal")
    header_line = ", ".join(parts) + "." if parts else "All files are equal."

    final_text = f"<p>{header_line}</p>\n" + result_table
    return final_text


def get_subtext(repo, feature_branch, base_branch, feature_commit="", base_commit=""):
    if feature_commit:
        feature_commit = f"([{feature_commit[:7]}](https://github.com/{repo}/commits/{feature_commit})) "
    if base_commit:
        base_commit = (
            f" ([{base_commit[:7]}](https://github.com/{repo}/commits/{base_commit})) "
        )

    return f"Comparing {feature_branch}{feature_commit} with {base_branch}{base_commit}.\n\
Last updated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}."


def combine_texts(artifact_url, plot_table, comparison_table, subtext):
    return f"\
## Validator Report\n\
I am the Validator. Download all artifacts [here]({artifact_url}).\n\
I'll be back and edit this comment for every commit. \n\n\
<details>\n\
    <summary>Result plots comparison</summary>\n\
{plot_table}\n\
</details>\n\
\n\
\n\
<details>\n\
    <summary>Result files comparison</summary>\n\
{comparison_table}\n\
</details>\n\
\n\
{subtext}\
"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="PyPSA/pypsa-ariadne")
    parser.add_argument("--dir_base", type=str, default="results (base)")
    parser.add_argument("--dir_feature", type=str, default="results (feature)")
    parser.add_argument(
        "--plots_commit_id",
        type=str,
        default="53cae58b007e4ff119155ff41b1bc1d5c36c63db",
    )
    parser.add_argument("--plots", nargs="*", type=str, default=["capacity.png"])
    parser.add_argument(
        "--artifact_url", nargs="*", type=str, default="https://www.google.com"
    )
    parser.add_argument("--feature_branch_name", type=str, default="lkstrp/dev")
    parser.add_argument("--base_branch_name", type=str, default="main")
    parser.add_argument("--feature_commit", type=str, default="")
    parser.add_argument("--base_commit", type=str, default="")

    args = parser.parse_args()

    final_text = combine_texts(
        artifact_url=args.artifact_url,
        plot_table=get_plot_comparison_table_html(args.plots_commit_id, args.plots),
        comparison_table=get_csv_comparison_table_html(args.dir_base, args.dir_feature),
        subtext=get_subtext(
            repo=args.repo,
            feature_branch=args.feature_branch_name,
            base_branch=args.base_branch_name,
            feature_commit=args.feature_commit,
            base_commit=args.base_commit,
        ),
    )
    print(final_text)
