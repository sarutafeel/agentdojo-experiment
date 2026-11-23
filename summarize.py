"""
Summarize benchmark results by suite, model, and defense.
Show benign utility and the results for tool-output and tool-description conditions.
"""

import sys

import pandas as pd


def main(path: str, markdown: bool) -> None:
    df = pd.read_csv(path)
    # keep latest result for each condition and task pair
    keys = ["suite", "model", "defense", "attack_location", "template", "target_tool", "user_task", "injection_task"]
    df = df.fillna("").sort_values("timestamp").drop_duplicates(keys, keep="last")

    rows = []
    # build one summary row for each suite, model, defense
    for (suite, model, defense), g in df.groupby(["suite", "model", "defense"]):
        row = {"suite": suite, "model": model, "defense": defense}
        benign = g[g.attack_location == "none"]
        row["benign utility %"] = round(100 * benign.utility.mean(), 1) if len(benign) else None
        for loc, label in [("data", "data plane"), ("metadata", "metadata plane")]:
            a = g[g.attack_location == loc]
            if len(a):
                row[f"utility under attack ({label}) %"] = round(100 * a.utility.mean(), 1)
                row[f"attack success ({label}) %"] = round(100 * a.attack_success.astype(float).mean(), 1)
                row[f"n pairs ({label})"] = len(a)
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_markdown(index=False) if markdown else table.to_string(index=False))


if __name__ == "__main__":
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(paths[0] if paths else "results/results.csv", "--markdown" in sys.argv)
