#!/usr/bin/env python3
"""Validate a recorded launch plan and summarize correct CUDA measurements."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


KEY_FIELDS = ("kernel_id", "size", "block_x", "block_y", "round")
TRIAL_FIELDS = KEY_FIELDS + ("status", "kernel_ms", "error")
MEASUREMENT_FIELDS = KEY_FIELDS[:-1] + (
    "repeats_ok", "repeats_failed", "median_ms", "mad_ms", "relative_mad", "status")
ACTION_FIELDS = KEY_FIELDS[:-1] + (
    "median_ms", "relative_mad", "baseline_block_x", "baseline_block_y",
    "reward_log_vs_default", "regret")


def trial_key(row):
    """Accept only explicit, nonempty IDs and integral launch dimensions."""
    kernel = row.get("kernel_id")
    if not isinstance(kernel, str) or not kernel.strip():
        raise ValueError("kernel_id must be a nonempty string")
    values = []
    for field in KEY_FIELDS[1:]:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("{} must be an integer".format(field))
        try:
            value = int(value)
        except ValueError:
            raise ValueError("{} must be an integer".format(field)) from None
        if value < (0 if field == "round" else 1):
            raise ValueError("{} is out of range".format(field))
        values.append(value)
    return (kernel, *values)


def load_plan(path, errors):
    try:
        plan = json.loads(path.read_text())
        if not isinstance(plan, list) or not plan:
            raise ValueError("expected a nonempty JSON list")
    except (OSError, ValueError) as exc:
        errors.append("planned.json: {}".format(exc))
        return Counter()
    keys = Counter()
    for index, row in enumerate(plan):
        try:
            if not isinstance(row, dict):
                raise ValueError("expected an object")
            keys[trial_key(row)] += 1
        except ValueError as exc:
            errors.append("planned.json item {}: {}".format(index, exc))
    return keys


def load_trials(path, errors):
    rows = []
    try:
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or not set(TRIAL_FIELDS).issubset(fields):
                raise ValueError("missing or duplicate required CSV columns")
            for line, row in enumerate(reader, 2):
                try:
                    if None in row or any(row[field] is None for field in TRIAL_FIELDS):
                        raise ValueError("malformed or truncated CSV row")
                    key = trial_key(row)
                except ValueError as exc:
                    errors.append("trials.csv row {}: {}".format(line, exc))
                    continue
                status = row["status"]
                milliseconds = None
                if status == "ok":
                    try:
                        milliseconds = float(row["kernel_ms"])
                        if not math.isfinite(milliseconds) or milliseconds <= 0:
                            raise ValueError()
                    except ValueError:
                        errors.append("trials.csv row {}: ok requires finite positive kernel_ms".format(line))
                        milliseconds = None
                elif not status.strip():
                    errors.append("trials.csv row {}: empty status".format(line))
                rows.append((key, milliseconds))
    except (OSError, ValueError, csv.Error) as exc:
        errors.append("trials.csv: {}".format(exc))
    return rows


def write_csv(path, fields, rows):
    # Replace the named output itself, rather than following an existing symlink.
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise ValueError("Refusing symlink output: {}".format(temporary))
    with temporary.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text(path, contents):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise ValueError("Refusing symlink output: {}".format(temporary))
    temporary.write_text(contents)
    temporary.replace(path)


def markdown(summary):
    lines = ["# LS-CAT pilot collection summary", "", "Status: **{}**".format(summary["status"]), "",
             "COMPLETE means the recorded plan has exactly one successful, correct, finite-positive timing "
             "for every planned trial. It does not establish statistical sufficiency or model generalization.", "",
             "Planned trials: {}; recorded trials: {}; valid timings: {}; failed records: {}.".format(
                 summary["planned_trials"], summary["observed_trials"], summary["ok_trials"], summary["failed_trials"]),
             "Missing trials: {}; extra trials: {}.".format(summary["missing_trials"], summary["extra_trials"]),
             "Action groups: {} full, {} failed, {} incomplete.".format(
                 summary["full_groups"], summary["failed_groups"], summary["incomplete_groups"]), "",
             "Only action groups with every planned round present exactly once and valid enter action_table.csv. "
             "The default is the fully valid action with the largest block_x * block_y; ties prefer larger "
             "block_x, then block_y. The tested oracle is the minimum median over fully valid actions. "
             "reward_log_vs_default = log(default_median_ms / action_median_ms); "
             "regret = action_median_ms / best_median_ms - 1. These are this pilot's definitions, "
             "not a reproduction of the original paper's score.", "",
             "relative_mad = median(abs(trial_ms - median_ms)) / median_ms; it is a ratio, not a percentage.", "",
             "These comparisons describe the recorded kernels and sizes. "
             "The same measurements choose the tested best action and estimate its speedup."]
    for context in summary["contexts"]:
        label = json.dumps(context["kernel_id"], ensure_ascii=True)
        lines.extend(["", "- {} size {}: {} fully valid actions; default ({}, {}), tested best ({}, {}); "
                      "default / tested-best median = {:.6g}x.".format(
                          label, context["size"], context["valid_actions"], context["default_block_x"],
                          context["default_block_y"], context["best_block_x"], context["best_block_y"],
                          context["default_to_best_speedup"])])
    if summary["validation_errors"]:
        lines.extend(["", "Validation errors:", ""])
        lines.extend("- " + error.replace("\n", " ") for error in summary["validation_errors"])
    return "\n".join(lines) + "\n"


def analyze(run_dir):
    """Write reports, preserve failed groups, and return a COMPLETE/INCOMPLETE summary."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError("Run directory does not exist: {}".format(run_dir))
    # A stale marker must never survive a failed reanalysis.
    for marker in ("COMPLETE", "INCOMPLETE"):
        (run_dir / marker).unlink(missing_ok=True)
    errors = []
    expected = load_plan(run_dir / "planned.json", errors)
    rows = load_trials(run_dir / "trials.csv", errors)
    observed = Counter(key for key, _ in rows)
    missing, extra = sum((expected - observed).values()), sum((observed - expected).values())
    duplicate_planned = sum(count > 1 for count in expected.values())
    duplicate_observed = sum(count > 1 for count in observed.values())
    if duplicate_planned:
        errors.append("planned.json has {} duplicate trial keys".format(duplicate_planned))
    if duplicate_observed:
        errors.append("trials.csv has {} duplicate trial keys".format(duplicate_observed))
    if missing or extra:
        errors.append("Trial keys differ from plan: {} missing, {} extra".format(missing, extra))
    grouped = defaultdict(list)
    planned_groups = defaultdict(Counter)
    for key, milliseconds in rows:
        grouped[key[:-1]].append((key, milliseconds))
    for key, count in expected.items():
        planned_groups[key[:-1]][key] = count
    measurements = []
    for group in sorted(set(grouped) | set(planned_groups)):
        trials = grouped[group]
        values = [milliseconds for _, milliseconds in trials if milliseconds is not None]
        failures = len(trials) - len(values)
        group_actual = Counter(key for key, _ in trials)
        group_plan = planned_groups[group]
        full = (bool(group_plan) and group_plan == group_actual and
                all(count == 1 for count in group_plan.values()) and not failures)
        median = statistics.median(values) if values else None
        mad = statistics.median(abs(value - median) for value in values) if values else None
        measurements.append(dict(zip(KEY_FIELDS[:-1], group),
                                 repeats_ok=len(values), repeats_failed=failures,
                                 median_ms=median, mad_ms=mad,
                                 relative_mad=mad / median if median is not None else None,
                                 status="full" if full else "failed" if failures else "incomplete"))
    valid = defaultdict(list)
    for row in measurements:
        if row["status"] == "full":
            valid[(row["kernel_id"], row["size"])].append(row)
    contexts, actions = [], []
    # shortcut: this curated pilot only describes collected contexts; add held-out kernel groups before learning claims.
    for (kernel, size), options in sorted(valid.items()):
        default = max(options, key=lambda row: (row["block_x"] * row["block_y"], row["block_x"], row["block_y"]))
        best = min(options, key=lambda row: (row["median_ms"], row["block_x"], row["block_y"]))
        contexts.append(dict(kernel_id=kernel, size=size, valid_actions=len(options),
                             default_block_x=default["block_x"], default_block_y=default["block_y"],
                             best_block_x=best["block_x"], best_block_y=best["block_y"],
                             default_ms=default["median_ms"], best_ms=best["median_ms"],
                             default_to_best_speedup=default["median_ms"] / best["median_ms"]))
        for row in options:
            actions.append({field: row[field] for field in KEY_FIELDS[:-1]} | dict(
                median_ms=row["median_ms"], relative_mad=row["relative_mad"],
                baseline_block_x=default["block_x"], baseline_block_y=default["block_y"],
                reward_log_vs_default=math.log(default["median_ms"] / row["median_ms"]),
                regret=row["median_ms"] / best["median_ms"] - 1.0))
    failed = sum(milliseconds is None for _, milliseconds in rows)
    complete = bool(expected) and not errors and not failed
    counts = Counter(row["status"] for row in measurements)
    summary = dict(status="COMPLETE" if complete else "INCOMPLETE",
                   planned_trials=sum(expected.values()), observed_trials=len(rows),
                   ok_trials=len(rows) - failed, failed_trials=failed,
                   missing_trials=missing, extra_trials=extra,
                   duplicate_planned_keys=duplicate_planned, duplicate_observed_keys=duplicate_observed,
                   full_groups=counts["full"], failed_groups=counts["failed"], incomplete_groups=counts["incomplete"],
                   contexts=contexts, validation_errors=errors)
    write_csv(run_dir / "measurements.csv", MEASUREMENT_FIELDS, measurements)
    write_csv(run_dir / "action_table.csv", ACTION_FIELDS, actions)
    write_text(run_dir / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_text(run_dir / "summary.md", markdown(summary))
    write_text(run_dir / summary["status"], "{} planned trials; {} recorded trials; {} valid timings\n".format(
        summary["planned_trials"], summary["observed_trials"], summary["ok_trials"]))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = analyze(args.run_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, "Analysis failed: {}\n".format(exc))
    print("{}: {} valid timings; see {}".format(summary["status"], summary["ok_trials"], args.run_dir / "summary.md"))
    return 0 if summary["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
