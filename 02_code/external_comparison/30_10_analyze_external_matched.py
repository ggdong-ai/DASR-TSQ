from __future__ import annotations

"""Final common analysis for the Step-28-matched external benchmark v3.1."""

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import importlib.util

THIS_DIR = Path(__file__).resolve().parent
COMMON_PATH = THIS_DIR / "30_0_external_matched_common.py"


def load_common():
    name = "external_matched_common_v31_analysis"
    spec = importlib.util.spec_from_file_location(name, COMMON_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(COMMON_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = load_common()
METHOD_ORDER = ["dasr_tsq", "graphga", "mars", "mnopt"]
DISPLAY = {
    "dasr_tsq": "DASR-TSQ",
    "graphga": "GraphGA",
    "mars": "MARS",
    "mnopt": "MNopt (paper-faithful reimplementation)",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_population_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    smi_col = C.detect_smiles_column(out)
    if "canonical_smiles" not in out.columns:
        canonical = [C.canonicalize_smiles(s) for s in out[smi_col].astype(str)]
        if any(x is None for x in canonical):
            raise RuntimeError("Population contains invalid SMILES.")
        out["canonical_smiles"] = [str(x) for x in canonical]
    if "dual_min_pic50" not in out.columns:
        out["dual_min_pic50"] = np.minimum(
            out["pred_gsk3b_pic50"].to_numpy(float),
            out["pred_jnk3_pic50"].to_numpy(float),
        )
    if "scaffold" not in out.columns:
        out["scaffold"] = [C.generic_scaffold(s) for s in out["canonical_smiles"].astype(str)]
    if "sa_reward" not in out.columns and "sa_score" in out.columns:
        out["sa_reward"] = np.clip((10.0 - out["sa_score"].to_numpy(float)) / 9.0, 0.0, 1.0)
    required = {
        "canonical_smiles", "final_reward", "pred_gsk3b_pic50", "pred_jnk3_pic50",
        "dual_min_pic50", "qed", "sa_score", "sa_reward", "scaffold",
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise RuntimeError(f"Population missing required columns: {missing}")
    if len(out) != 200:
        raise RuntimeError(f"Population size={len(out)}, expected 200.")
    return out.reset_index(drop=True)


def external_population_from_snapshot(snapshot: dict[str, Any], archive: pd.DataFrame) -> pd.DataFrame:
    indexed = archive.set_index("canonical_smiles", drop=False)
    rows = []
    for smi in snapshot["canonical_smiles"]:
        if smi not in indexed.index:
            raise RuntimeError(f"Snapshot molecule missing from archive: {smi}")
        row = indexed.loc[smi]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rows.append(row.to_dict())
    return ensure_population_columns(pd.DataFrame(rows))


def top10_reward(pop: pd.DataFrame) -> float:
    values = np.sort(pop["final_reward"].to_numpy(float))[::-1]
    return float(np.mean(values[:10]))


def collapse_snapshots(snapshots: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    if not snapshots:
        raise RuntimeError("Empty population snapshot list.")
    ordered = sorted(snapshots, key=lambda s: (int(s["requests"]), int(s.get("update", -999))))
    by_request: dict[int, dict[str, Any]] = {}
    for snapshot in ordered:
        request = int(snapshot["requests"])
        if request < 200 or request > cap:
            raise RuntimeError(f"Snapshot request {request} outside [200,{cap}].")
        if len(snapshot["canonical_smiles"]) != 200:
            raise RuntimeError("Snapshot population size is not 200.")
        by_request[request] = snapshot
    collapsed = [by_request[x] for x in sorted(by_request)]
    if int(collapsed[0]["requests"]) != 200:
        raise RuntimeError("Trajectory must begin with P0 at request 200.")
    return collapsed


def stepwise_auc(requests: list[int], values: list[float], cap: int) -> float:
    if len(requests) != len(values) or not requests:
        raise ValueError("Invalid trajectory for AUC.")
    if requests[0] != 200:
        raise RuntimeError("Resource AUC must start at request 200.")
    area = 0.0
    for i, x0 in enumerate(requests):
        x1 = requests[i + 1] if i + 1 < len(requests) else cap
        if x1 < x0:
            raise RuntimeError("Trajectory requests are not monotone.")
        area += float(values[i]) * float(x1 - x0)
    span = cap - 200
    return float(area / span) if span > 0 else float(values[-1])


def checkpoint_requests(cap: int) -> list[tuple[str, int]]:
    return [
        ("25%", int(round(200 + 0.25 * (cap - 200)))),
        ("50%", int(round(200 + 0.50 * (cap - 200)))),
        ("75%", int(round(200 + 0.75 * (cap - 200)))),
        ("100%", int(cap)),
    ]


def state_at_or_before(snapshots: list[dict[str, Any]], request: int) -> dict[str, Any]:
    eligible = [s for s in snapshots if int(s["requests"]) <= int(request)]
    if not eligible:
        raise RuntimeError(f"No population state available at request {request}.")
    return eligible[-1]


def load_dasr_seed(protocol: dict[str, Any], seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    rec = protocol["matched_request_caps"][str(seed)]
    task = Path(rec["step28_task_dir"]).resolve()
    task_marker_path = task / "TASK_COMPLETED.json"
    validation_path = task / "integration_validation.json"
    if C.sha256_file(task_marker_path) != rec["task_completed_sha256"]:
        raise RuntimeError(f"Step-28 seed {seed} TASK_COMPLETED changed after v3.1 freeze.")
    if C.sha256_file(validation_path) != rec["integration_validation_sha256"]:
        raise RuntimeError(f"Step-28 seed {seed} integration_validation changed after v3.1 freeze.")
    task_marker = C.load_json(task_marker_path)
    file_records = dict(task_marker.get("files", {}))

    history = pd.read_csv(task / "history.csv").sort_values("generation").reset_index(drop=True)
    if C.sha256_file(task / "history.csv") != rec["history_sha256"]:
        raise RuntimeError(f"Step-28 seed {seed} history changed after v3.1 freeze.")
    initial_path = task / "population_initial.csv"
    if C.sha256_file(initial_path) != rec["population_initial_sha256"]:
        raise RuntimeError(f"Step-28 seed {seed} population_initial changed after v3.1 freeze.")
    initial = ensure_population_columns(pd.read_csv(initial_path))
    p0_can = initial["canonical_smiles"].astype(str).tolist()
    protocol_p0 = pd.read_csv(C.DEFAULT_PROTOCOL_DIR / "formal_p0.csv")
    p0_col = C.detect_smiles_column(protocol_p0)
    protocol_p0_can = [C.canonicalize_smiles(x) for x in protocol_p0[p0_col].astype(str)]
    if p0_can != [str(x) for x in protocol_p0_can]:
        raise RuntimeError(f"Step-28 seed {seed} initial population order differs from frozen P0.")

    snapshots: list[dict[str, Any]] = [
        {"stage": "p0", "update": -1, "requests": 200, "population": initial}
    ]
    for generation in range(50):
        path = task / f"population_gen{generation:03d}.csv"
        if not path.exists() and generation == 49:
            path = task / "final_population.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        file_record = file_records.get(path.name)
        if not file_record or C.sha256_file(path) != str(file_record.get("sha256", "")):
            raise RuntimeError(f"Step-28 seed {seed} population hash mismatch: {path.name}")
        pop = ensure_population_columns(pd.read_csv(path))
        request = int(
            history.loc[history["generation"].astype(int) == generation, "total_oracle_requests_cumulative"].iloc[0]
        )
        snapshots.append(
            {"stage": "generation_update", "update": generation, "requests": request, "population": pop}
        )
    final = snapshots[-1]["population"]
    final_source = task / "final_population.csv"
    if not final_source.exists():
        final_source = task / "population_gen049.csv"
    if C.sha256_file(final_source) != rec["final_population_sha256"]:
        raise RuntimeError(f"Step-28 seed {seed} final population changed after v3.1 freeze.")
    cap = int(rec["request_cap"])
    if int(snapshots[-1]["requests"]) != cap:
        raise RuntimeError(f"Step-28 seed {seed} final trajectory request != matched cap.")
    audit = {
        "request_cap": cap,
        "actual_requests": cap,
        "request_utilization": 1.0,
        "counters": {
            "requests": cap,
            "request_budget": cap,
            "model_evaluations": int(rec["step28_model_evaluations_cumulative"]),
            "cache_hits": int(cap - int(rec["step28_model_evaluations_cumulative"])),
            "invalid_requests": math.nan,
            "unique_valid_evaluations": int(rec["step28_unique_structures_scored_globally"]),
        },
        "native_early_stop_before_request_cap": False,
        "request_cap_boundary_stop": False,
    }
    return final, snapshots, audit


def load_external_seed(protocol: dict[str, Any], marker: dict[str, Any], method: str, seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    task = C.DEFAULT_RESULT_ROOT / method / f"seed_{seed}"
    if not C.validate_completed_task(task, method, seed, marker["protocol_sha256"]):
        raise FileNotFoundError(f"Missing completed {method} seed {seed}: {task}")
    completed = C.load_json(task / "TASK_COMPLETED.json")
    archive = pd.read_csv(task / "archive_unique.csv.gz")
    snapshots_raw = C.read_json_gz(task / "population_snapshots.json.gz")
    cap = int(protocol["matched_request_caps"][str(seed)]["request_cap"])
    snapshots_raw = collapse_snapshots(snapshots_raw, cap)
    snapshots: list[dict[str, Any]] = []
    for raw in snapshots_raw:
        snapshots.append({**raw, "population": external_population_from_snapshot(raw, archive)})
    final = ensure_population_columns(pd.read_csv(task / "final_population.csv"))
    if final["canonical_smiles"].astype(str).tolist() != snapshots[-1]["population"]["canonical_smiles"].astype(str).tolist():
        raise RuntimeError(f"{method} seed {seed} final_population.csv differs from last completed snapshot.")
    return final, snapshots, completed, archive


def summarize_methods(run_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "final_top10_mean_final_reward", "final_mean_final_reward", "final_max_final_reward",
        "final_hypervolume_4d", "final_tier1_count", "final_unique_scaffolds",
        "final_pareto_front_size", "final_max_scaffold_occupancy",
        "normalized_auc_top10_mean_final_reward", "request_utilization",
    ]
    rows = []
    for method in METHOD_ORDER:
        sub = run_metrics[run_metrics["method"] == method]
        if len(sub) != 5:
            raise RuntimeError(f"Expected 5 runs for {method}, got {len(sub)}")
        row: dict[str, Any] = {"method": method, "display_name": DISPLAY[method], "n_seeds": 5}
        for metric in metric_cols:
            values = sub[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(run_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("final_top10_mean_final_reward", 1),
        ("final_hypervolume_4d", 1),
        ("final_tier1_count", 1),
        ("final_unique_scaffolds", 1),
        ("final_mean_final_reward", 1),
        ("final_pareto_front_size", 1),
        ("final_max_scaffold_occupancy", -1),
        ("normalized_auc_top10_mean_final_reward", 1),
    ]
    rows = []
    for method in ("graphga", "mars", "mnopt"):
        for metric, favorable_sign in metrics:
            pivot = run_metrics.pivot(index="seed", columns="method", values=metric).sort_index()
            diff = pivot[method].to_numpy(float) - pivot["dasr_tsq"].to_numpy(float)
            low, high = C.bootstrap_ci(diff, n_resamples=20000, seed=42000 + len(rows))
            sd = float(np.std(diff, ddof=1))
            favorable = diff * favorable_sign
            rows.append(
                {
                    "comparator": method,
                    "comparator_display": DISPLAY[method],
                    "reference": "dasr_tsq",
                    "metric": metric,
                    "difference_definition": "comparator_minus_DASR-TSQ",
                    "favorable_sign": favorable_sign,
                    "n_pairs": 5,
                    "mean_difference": float(np.mean(diff)),
                    "sd_difference": sd,
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "exact_signflip_p_two_sided": C.exact_signflip_p(diff),
                    "paired_effect_size_dz": float(np.mean(diff) / sd) if sd > 0 else math.nan,
                    "n_favorable_seeds": int(np.sum(favorable > 0)),
                    "n_tied_seeds": int(np.sum(np.isclose(favorable, 0.0, atol=1e-12))),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", type=Path, default=C.DEFAULT_PROTOCOL_DIR)
    ap.add_argument("--result_root", type=Path, default=C.DEFAULT_RESULT_ROOT)
    args = ap.parse_args()

    # The package uses fixed standard locations to make all hashes and BATs unambiguous.
    if args.protocol_dir.resolve() != C.DEFAULT_PROTOCOL_DIR.resolve():
        raise RuntimeError("v3.1 analysis currently requires the frozen default protocol directory.")
    if args.result_root.resolve() != C.DEFAULT_RESULT_ROOT.resolve():
        raise RuntimeError("v3.1 analysis currently requires the default result root.")

    protocol, marker = C.load_protocol(args.protocol_dir)
    for filename in ("30_0_external_matched_common.py", "30_10_analyze_external_matched.py"):
        C.verify_active_code_hash(protocol, filename)

    analysis_dir = C.DEFAULT_RESULT_ROOT / "analysis"
    if analysis_dir.exists() and any(analysis_dir.iterdir()):
        existing = analysis_dir / "EXTERNAL_ANALYSIS_COMPLETED.json"
        if existing.exists():
            old = C.load_json(existing)
            if old.get("completed") is True and old.get("protocol_sha256") == marker["protocol_sha256"]:
                print("EXTERNAL MATCHED ANALYSIS: ALREADY COMPLETE")
                return
        raise FileExistsError(f"Nonempty analysis directory exists: {analysis_dir}")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    for seed in C.FORMAL_SEEDS:
        cap = C.request_cap_for_seed(protocol, seed)
        for method in METHOD_ORDER:
            if method == "dasr_tsq":
                final, snapshots, completed = load_dasr_seed(protocol, seed)
            else:
                final, snapshots, completed, _archive = load_external_seed(protocol, marker, method, seed)

            final_metrics = C.population_metrics(final, compute_hypervolume=True)
            requests = [int(s["requests"]) for s in snapshots]
            top10_values = [top10_reward(s["population"]) for s in snapshots]
            auc = stepwise_auc(requests, top10_values, cap)

            for snapshot, value in zip(snapshots, top10_values):
                trajectory_rows.append(
                    {
                        "method": method,
                        "display_name": DISPLAY[method],
                        "seed": seed,
                        "request_cap": cap,
                        "requests": int(snapshot["requests"]),
                        "request_fraction": (int(snapshot["requests"]) - 200) / (cap - 200),
                        "stage": snapshot.get("stage"),
                        "update": snapshot.get("update"),
                        "top10_mean_final_reward": value,
                    }
                )

            for label, checkpoint in checkpoint_requests(cap):
                state = state_at_or_before(snapshots, checkpoint)
                metrics = C.population_metrics(state["population"], compute_hypervolume=True)
                checkpoint_rows.append(
                    {
                        "method": method,
                        "display_name": DISPLAY[method],
                        "seed": seed,
                        "checkpoint": label,
                        "checkpoint_requests": checkpoint,
                        "state_requests": int(state["requests"]),
                        "carried_forward": int(state["requests"]) < checkpoint,
                        **metrics,
                    }
                )

            counters = dict(completed.get("counters", {}))
            row = {
                "method": method,
                "display_name": DISPLAY[method],
                "seed": seed,
                "request_cap": cap,
                "actual_requests": int(completed["actual_requests"]),
                "request_utilization": float(completed["request_utilization"]),
                "native_early_stop_before_request_cap": bool(completed.get("native_early_stop_before_request_cap", False)),
                "request_cap_boundary_stop": bool(completed.get("request_cap_boundary_stop", False)),
                "model_evaluations": counters.get("model_evaluations", math.nan),
                "cache_hits": counters.get("cache_hits", math.nan),
                "invalid_requests": counters.get("invalid_requests", math.nan),
                "unique_valid_evaluations": counters.get("unique_valid_evaluations", math.nan),
                "normalized_auc_top10_mean_final_reward": auc,
            }
            row.update({f"final_{key}": value for key, value in final_metrics.items()})
            run_rows.append(row)

    run_df = pd.DataFrame(run_rows).sort_values(["method", "seed"]).reset_index(drop=True)
    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(["method", "seed", "requests"]).reset_index(drop=True)
    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values(["method", "seed", "checkpoint_requests"]).reset_index(drop=True)
    summary_df = summarize_methods(run_df)
    effects_df = paired_effects(run_df)

    run_path = analysis_dir / "external_run_metrics.csv"
    traj_path = analysis_dir / "external_population_trajectory.csv.gz"
    cp_path = analysis_dir / "external_resource_checkpoints.csv"
    summary_path = analysis_dir / "external_summary_mean_sd.csv"
    effects_path = analysis_dir / "external_paired_effects_vs_dasr.csv"
    run_df.to_csv(run_path, index=False)
    trajectory_df.to_csv(traj_path, index=False, compression="gzip")
    checkpoint_df.to_csv(cp_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    effects_df.to_csv(effects_path, index=False)

    completed = {
        "completed": True,
        "created_utc": utc_now(),
        "protocol_sha256": marker["protocol_sha256"],
        "methods": METHOD_ORDER,
        "seeds": list(C.FORMAL_SEEDS),
        "n_runs": int(len(run_df)),
        "primary_resource": "raw oracle requests matched per seed to completed DASR-TSQ Step 28",
        "primary_population_object": "last completed native population of 200 molecules",
        "resource_auc_rule": "piecewise-constant last-completed-population trajectory from request 200; final state carried to cap after native early stop",
        "small_n_boundary": "Five paired seeds; no conventional exact two-sided p<0.05 claim.",
        "files": {
            run_path.name: C.sha256_file(run_path),
            traj_path.name: C.sha256_file(traj_path),
            cp_path.name: C.sha256_file(cp_path),
            summary_path.name: C.sha256_file(summary_path),
            effects_path.name: C.sha256_file(effects_path),
        },
        "performance_used_for_tuning": False,
    }
    C.write_json(analysis_dir / "EXTERNAL_ANALYSIS_COMPLETED.json", completed)

    print("=" * 88)
    print("EXTERNAL MATCHED STEP-28 V3.1 ANALYSIS: PASS")
    print("=" * 88)
    print("Runs analyzed:", len(run_df), "(4 methods x 5 paired seeds)")
    print("Primary resource: raw oracle requests")
    print("Primary object: last completed native population, N=200")
    print("Output:", analysis_dir)
    print("No post-formal tuning is permitted.")


if __name__ == "__main__":
    main()
