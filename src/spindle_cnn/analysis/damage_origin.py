"""Document damage-origin confounding and secondary-protocol feasibility."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "results" / "damage_origin_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "DAMAGE_ORIGIN_ANALYSIS.md",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    by_class = (
        frame.drop_duplicates(["fold", "bearing_code", "class_name", "damage_origin"])
        .groupby(["class_name", "damage_origin"])["bearing_code"]
        .nunique()
        .reset_index(name="bearing_codes")
    )
    by_split = (
        frame.groupby(["split", "damage_origin"])
        .agg(
            windows=("window_count", "sum"),
            bearing_conditions=("bearing_code", "nunique"),
        )
        .reset_index()
    )
    heldout = frame.loc[frame["split"].eq("test")].drop_duplicates(
        ["bearing_code", "class_name", "damage_origin"]
    )
    lines = [
        "# Damage-Origin Analysis",
        "",
        "## Finding",
        "",
        "The primary validation-to-test gap coincides with joint changes in physical "
        "bearing identity, operating condition, and damage origin. The current "
        "four-class protocol cannot attribute the gap to one of these factors.",
        "",
        "All held-out fault bearings have real (accelerated-lifetime) damage. "
        "Artificial damage appears only in the fitting set for outer- and inner-ring "
        "bearings. Both fitting multiple-damage bearings are real-damage bearings. "
        "The held-out healthy bearing is not subject to a damage-origin change.",
        "",
        "## Held-out origin",
        "",
        "| Bearing | Class | Origin |",
        "| --- | --- | --- |",
    ]
    for _, row in heldout.sort_values("bearing_code").iterrows():
        lines.append(
            f"| {row['bearing_code']} | {row['class_name']} | {row['damage_origin']} |"
        )
    lines.extend(
        [
            "",
            "## Bearing codes by class and origin",
            "",
            "| Class | Origin | Bearing codes |",
            "| --- | --- | ---: |",
        ]
    )
    for _, row in by_class.iterrows():
        lines.append(
            f"| {row['class_name']} | {row['damage_origin']} | {int(row['bearing_codes'])} |"
        )
    lines.extend(
        [
            "",
            "## Window distribution",
            "",
            "| Split | Origin | Windows | Bearings represented |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for _, row in by_split.iterrows():
        lines.append(
            f"| {row['split']} | {row['damage_origin']} | "
            f"{int(row['windows'])} | {int(row['bearing_conditions'])} |"
        )
    lines.extend(
        [
            "",
            "## Feasibility of damage-origin-only controls",
            "",
            "| Proposed control | Feasible with selected 16 codes? | Reason |",
            "| --- | --- | --- |",
            "| Artificial to artificial, four classes | No | No artificial multiple-damage bearing is available. |",
            "| Lifetime to lifetime, four classes | Partial only | Four classes exist, but single-location lifetime groups contain only one training bearing each. |",
            "| Artificial to lifetime, four classes | No | All held-out fault bearings are lifetime-damaged and no artificial multiple-damage source exists. |",
            "| Artificial to lifetime, three classes | Possible but strongly reduced | It can compare healthy, outer, and inner classes after removing multiple damage, but source diversity is low. |",
            "",
            "A one-origin holdout would also change bearing identity and condition. "
            "It would not create a clean causal estimate. The study therefore retains "
            "the joint-shift interpretation and reports the origin distribution as a "
            "confounding audit rather than converting it into a single-factor claim.",
            "",
            "## Result provenance",
            "",
            "- `results/damage_origin_manifest.csv`",
            "- `results/damage_origin_distribution.csv`",
            "- `reports/label_audit.csv`",
            "- `data/manifests/paderborn_selected_bearings.csv`",
            "",
        ]
    )
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
