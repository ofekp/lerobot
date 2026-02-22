#!/usr/bin/env python3
"""Collect eval results from an experiment output folder into a CSV.

Usage:
    python scripts/collect_results.py output_korean
    python scripts/collect_results.py output_korean -o results.csv

Directory structure expected:
    <exp_folder>/<suite>/<camera>/<task_name>/<mode>/eval/eval_info.json

The script reads data["overall"]["pc_success"] from each eval_info.json.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def read_pc_success(eval_json: Path) -> float | None:
    """Read overall.pc_success from an eval_info.json file."""
    try:
        with open(eval_json) as f:
            data = json.load(f)
        return data["overall"]["pc_success"]
    except (KeyError, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[WARN] Skipping {eval_json}: {e}", file=sys.stderr)
        return None


def collect_results(exp_folder: Path) -> dict[str, list[dict]]:
    """Collect results grouped by suite, with rgb/rgbd as separate columns."""
    # key: (suite, camera, task) → {"rgb": score, "rgbd": score}
    scores: dict[tuple, dict] = defaultdict(dict)

    for eval_json in sorted(exp_folder.rglob("eval/eval_info.json")):
        # .../suite/camera/task_name/mode/eval/eval_info.json
        mode_dir = eval_json.parent.parent          # mode (rgb / rgbd)
        task_dir = mode_dir.parent                  # task_name
        camera_dir = task_dir.parent                # camera
        suite_dir = camera_dir.parent               # suite

        mode = mode_dir.name
        task_name = task_dir.name
        camera = camera_dir.name
        suite = suite_dir.name

        pc_success = read_pc_success(eval_json)
        if pc_success is not None:
            scores[(suite, camera, task_name)][mode] = pc_success

    # Group rows by suite
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for (suite, camera, task_name), mode_scores in sorted(scores.items()):
        by_suite[suite].append({
            "camera": camera,
            "task": task_name,
            "rgb": mode_scores.get("rgb", ""),
            "rgbd": mode_scores.get("rgbd", ""),
        })

    return dict(by_suite)


def main():
    parser = argparse.ArgumentParser(description="Collect LIBERO eval results into a CSV.")
    parser.add_argument("exp_folder", type=Path, help="Experiment output folder (e.g. output_korean)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output CSV path (default: <exp_folder>_results.csv)")
    args = parser.parse_args()

    if not args.exp_folder.is_dir():
        print(f"[ERROR] Not a directory: {args.exp_folder}", file=sys.stderr)
        sys.exit(1)

    by_suite = collect_results(args.exp_folder)

    if not by_suite:
        print("[WARN] No eval results found.", file=sys.stderr)
        sys.exit(0)

    out_path = args.output or Path(f"{args.exp_folder.name}_results.csv")
    fields = ["suite", "camera", "task", "rgb", "rgbd"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for suite in sorted(by_suite):
            rows = by_suite[suite]
            for row in rows:
                writer.writerow({"suite": suite, **row})
            # Blank separator row between suites
            writer.writerow({field: "" for field in fields})

    # Also print a readable summary to stdout
    for suite in sorted(by_suite):
        rows = by_suite[suite]
        print(f"\n{'=' * 70}")
        print(f"  {suite}")
        print(f"{'=' * 70}")
        print(f"  {'Camera':<25} {'Task':<45} {'RGB':>6} {'RGBD':>6}")
        print(f"  {'-'*25} {'-'*45} {'-'*6} {'-'*6}")
        for row in rows:
            rgb_str = f"{row['rgb']:.1f}" if row["rgb"] != "" else "N/A"
            rgbd_str = f"{row['rgbd']:.1f}" if row["rgbd"] != "" else "N/A"
            print(f"  {row['camera']:<25} {row['task']:<45} {rgb_str:>6} {rgbd_str:>6}")
        print()

    total = sum(len(rows) for rows in by_suite.values())
    print(f"[INFO] Wrote {total} tasks to {out_path}")


if __name__ == "__main__":
    main()
