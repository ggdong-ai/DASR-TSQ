"""
28_2_compare_parallel_determinism.py
====================================
Compare Step 28 v2 serial and parallel engineering runs.

The comparison ignores elapsed time and cache-accounting fields, but requires
identical P0, parent/child trajectories, final populations, oracle budgets,
termination reasons, and generation-level scientific metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VARIANTS = [
    "baseline_random_no_tsq",
    "tsq_only_random_tsq10",
    "dasr_only_listwise_no_tsq",
    "dasr_tsq_listwise_tsq10",
    "absolute_only_no_tsq",
    "absolute_tsq_tsq10",
]
TOL = 1e-12


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compare_numeric_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_columns: list[str],
    ignored_columns: set[str],
) -> list[str]:
    errors: list[str] = []
    left = left.sort_values(key_columns).reset_index(drop=True)
    right = right.sort_values(key_columns).reset_index(drop=True)
    if len(left) != len(right):
        return [f"row_count_mismatch:{len(left)}!={len(right)}"]

    common = [
        column for column in left.columns
        if column in right.columns and column not in ignored_columns
    ]
    if set(key_columns) - set(common):
        errors.append("key_columns_missing")
        return errors

    for column in common:
        a = left[column]
        b = right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = a.to_numpy(dtype=np.float64)
            bv = b.to_numpy(dtype=np.float64)
            if not np.allclose(av, bv, atol=TOL, rtol=0.0, equal_nan=True):
                errors.append(f"numeric_mismatch:{column}")
        else:
            aa = a.fillna("<NA>").astype(str).to_numpy()
            bb = b.fillna("<NA>").astype(str).to_numpy()
            if not np.array_equal(aa, bb):
                errors.append(f"text_mismatch:{column}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Step 28 v2 n_jobs=1 and n_jobs>1 engineering outputs."
    )
    parser.add_argument("--serial_dir", type=Path, required=True)
    parser.add_argument("--parallel_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    serial_dir = args.serial_dir.resolve()
    parallel_dir = args.parallel_dir.resolve()
    critical_errors: list[str] = []
    details: dict[str, Any] = {}

    serial_p0 = pd.read_csv(serial_dir / "selected_p0.csv")
    parallel_p0 = pd.read_csv(parallel_dir / "selected_p0.csv")
    if frame_hash(serial_p0) != frame_hash(parallel_p0):
        critical_errors.append("selected_p0_differs")

    local_ignore = {
        "elapsed_seconds",
        "worker_model_evaluations",
        "worker_cache_hits",
    }
    history_ignore = {
        "local_model_evaluations_generation",
        "local_cache_hits_generation",
        "outer_model_evaluations_generation",
        "outer_cache_hits_generation",
        "model_evaluations_executed_generation",
        "model_evaluations_executed_cumulative",
        "unique_structures_scored_globally",
    }

    for variant in VARIANTS:
        variant_errors: list[str] = []
        left_dir = serial_dir / variant
        right_dir = parallel_dir / variant
        for path in (left_dir, right_dir):
            if not path.exists():
                variant_errors.append(f"missing_directory:{path}")

        if not variant_errors:
            left_initial = pd.read_csv(left_dir / "population_initial.csv")
            right_initial = pd.read_csv(right_dir / "population_initial.csv")
            variant_errors.extend(
                f"initial:{item}"
                for item in compare_numeric_frames(
                    left_initial,
                    right_initial,
                    key_columns=["smiles"],
                    ignored_columns=set(),
                )
            )

            left_local = pd.read_csv(left_dir / "local_mcts_summary.csv")
            right_local = pd.read_csv(right_dir / "local_mcts_summary.csv")
            variant_errors.extend(
                f"local:{item}"
                for item in compare_numeric_frames(
                    left_local,
                    right_local,
                    key_columns=["generation", "child_index"],
                    ignored_columns=local_ignore,
                )
            )

            left_history = pd.read_csv(left_dir / "history.csv")
            right_history = pd.read_csv(right_dir / "history.csv")
            variant_errors.extend(
                f"history:{item}"
                for item in compare_numeric_frames(
                    left_history,
                    right_history,
                    key_columns=["generation"],
                    ignored_columns=history_ignore,
                )
            )

            left_final = pd.read_csv(left_dir / "final_population.csv")
            right_final = pd.read_csv(right_dir / "final_population.csv")
            variant_errors.extend(
                f"final:{item}"
                for item in compare_numeric_frames(
                    left_final,
                    right_final,
                    key_columns=["smiles"],
                    ignored_columns=set(),
                )
            )

        details[variant] = {
            "passed": not variant_errors,
            "errors": variant_errors,
        }
        critical_errors.extend(f"{variant}:{error}" for error in variant_errors)

    report = {
        "passed": not critical_errors,
        "critical_errors": critical_errors,
        "serial_dir": str(serial_dir),
        "parallel_dir": str(parallel_dir),
        "tolerance": TOL,
        "ignored_fields": {
            "local": sorted(local_ignore),
            "history": sorted(history_ignore),
        },
        "variant_details": details,
        "interpretation": (
            "A pass establishes deterministic scientific outputs across the tested "
            "worker counts. Timing and process-local cache-accounting fields are "
            "excluded because worker assignment can legitimately change them."
        ),
    }
    write_json(args.output_json.resolve(), report)
    print(f"Parallel determinism passed: {report['passed']}")
    print(f"Report: {args.output_json.resolve()}")
    if critical_errors:
        raise RuntimeError("Parallel determinism failed: " + "; ".join(critical_errors[:20]))


if __name__ == "__main__":
    main()
