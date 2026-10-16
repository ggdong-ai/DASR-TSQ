from __future__ import annotations

"""Aggregate audited REINVENT4 formal seeds 42-46.

This script is analysis-only.  It requires each seed to have passed the v2
formal audit and creates the official 5-seed REINVENT4 summary used for later
cross-method comparison.  It does not call the oracle and does not modify any
per-seed raw or audited artifact.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\pycharm\project\tsq_mcts_rebuild").resolve()
FORMAL_ROOT = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_reinvent4_formal"
)
PROTOCOL_DIR = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_reinvent4_protocol"
)
PROTOCOL_JSON = PROTOCOL_DIR / "REINVENT4_FORMAL_PROTOCOL.json"

OUT_JSON = FORMAL_ROOT / "REINVENT4_5SEED_SUMMARY.json"
OUT_CSV = FORMAL_ROOT / "REINVENT4_5SEED_PER_SEED_METRICS.csv"
OUT_STATS_CSV = FORMAL_ROOT / "REINVENT4_5SEED_MEAN_SD.csv"
OUT_TASK = FORMAL_ROOT / "REINVENT4_5SEED_AGGREGATE_TASK_COMPLETED.json"

EXPECTED_PROTOCOL_SHA = "592929f79e673058fe3d308519df9378dea28c268e28778a3836e84928ff9379"
SEEDS = [42, 43, 44, 45, 46]

METRIC_KEYS = [
    "mean_final_reward",
    "top10_mean_final_reward",
    "max_final_reward",
    "mean_dual_min_pic50",
    "top10_mean_dual_min_pic50",
    "tier1_count",
    "tier1_rate",
    "unique_scaffolds",
    "max_scaffold_occupancy",
    "pareto_front_size",
    "hypervolume_4d",
    "no_scaffold_count",
]
RESOURCE_KEYS = [
    "requests",
    "model_evaluations",
    "cache_hits",
    "unique_valid_evaluations",
    "method_scoring_requests_excluding_p0",
]
FINAL_GENERATOR_KEYS = [
    "valid_slots",
    "duplicate_slots",
    "invalid_slots",
    "unique_noninvalid_project_canonical",
]


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sample_sd": float(arr.std(ddof=1)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> None:
    require(PROTOCOL_JSON.exists(), f"Missing formal protocol: {PROTOCOL_JSON}")
    protocol = load_json(PROTOCOL_JSON)
    require(protocol.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA,
            "Formal protocol SHA mismatch")
    require(protocol.get("formal_seeds") == SEEDS, "Formal seed list mismatch")
    require(protocol.get("performance_used_for_tuning") is False,
            "Protocol says performance was used for tuning")

    rows = []
    provenance = {}

    for seed in SEEDS:
        d = FORMAL_ROOT / f"seed_{seed}"
        task_path = d / "TASK_COMPLETED.json"
        metrics_path = d / "final_metrics.json"
        audit_path = d / "formal_audit_report.json"
        pop_path = d / "final_population.csv"

        for p in (task_path, metrics_path, audit_path, pop_path):
            require(p.exists(), f"seed {seed}: missing audited artifact {p}")

        task = load_json(task_path)
        fm = load_json(metrics_path)
        audit = load_json(audit_path)

        require(task.get("completed") is True and task.get("audited") is True,
                f"seed {seed}: TASK_COMPLETED is not audited PASS")
        require(task.get("audit_version") == "v2",
                f"seed {seed}: expected v2 audit")
        require(task.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA,
                f"seed {seed}: TASK_COMPLETED protocol SHA mismatch")
        require(task.get("scientific_protocol_changed_by_audit_fix") is False,
                f"seed {seed}: audit reports scientific protocol change")
        require(audit.get("passed") is True,
                f"seed {seed}: formal_audit_report is not PASS")
        require(fm.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA,
                f"seed {seed}: final_metrics protocol SHA mismatch")
        require(int(fm.get("seed")) == seed,
                f"seed {seed}: final_metrics seed field mismatch")
        require(fm.get("final_population_policy") == "last native 200 Mol2Mol output slots",
                f"seed {seed}: final population policy mismatch")
        require(fm.get("posthoc_evaluation_counts_toward_optimization_budget") is False,
                f"seed {seed}: post-hoc evaluation budget policy mismatch")

        r = {"seed": seed}
        for key in METRIC_KEYS:
            require(key in fm["metrics"], f"seed {seed}: missing metric {key}")
            r[key] = fm["metrics"][key]
        for key in RESOURCE_KEYS:
            require(key in fm["oracle_accounting"], f"seed {seed}: missing resource {key}")
            r[key] = fm["oracle_accounting"][key]
        for key in FINAL_GENERATOR_KEYS:
            require(key in fm["generator_final_step"], f"seed {seed}: missing final generator key {key}")
            r[key] = fm["generator_final_step"][key]

        r["canonical_drift_count"] = fm["canonicalization_diagnostic"][
            "n_cross_environment_canonical_string_drift"
        ]
        r["canonical_compared"] = fm["canonicalization_diagnostic"][
            "n_unique_batch_input_smiles_compared"
        ]
        r["max_abs_final_score_diff"] = fm["cross_environment_final_score_check"][
            "max_abs_diff_valid_component"
        ]
        r["final_valid_rate"] = float(r["valid_slots"]) / 200.0
        r["final_unique_scaffold_rate"] = float(r["unique_scaffolds"]) / 200.0
        r["final_max_occupancy_rate"] = float(r["max_scaffold_occupancy"]) / 200.0
        rows.append(r)

        provenance[str(seed)] = {
            "task_completed_sha256": sha256_file(task_path),
            "final_metrics_sha256": sha256_file(metrics_path),
            "formal_audit_report_sha256": sha256_file(audit_path),
            "final_population_sha256": sha256_file(pop_path),
            "actual_requests": int(task["actual_requests"]),
            "unique_evaluations": int(task["unique_evaluations"]),
            "cache_hits": int(task["cache_hits"]),
        }

    # Per-seed CSV
    columns = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [k for k in columns if k != "seed"]
    summary_stats = {key: stats([float(r[key]) for r in rows]) for key in numeric_keys}

    with OUT_STATS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "sample_sd", "min", "max", "n"])
        for key in numeric_keys:
            s = summary_stats[key]
            writer.writerow([key, s["mean"], s["sample_sd"], s["min"], s["max"], s["n"]])

    result = {
        "completed": True,
        "method": "REINVENT4-Mol2Mol",
        "formal_protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "seeds": SEEDS,
        "n_seeds": 5,
        "aggregation": "arithmetic mean and sample SD across formal seeds 42-46",
        "final_population_policy": "last native 200 Mol2Mol output slots per seed",
        "per_seed": rows,
        "summary_stats": summary_stats,
        "provenance": provenance,
    }
    write_json_atomic(OUT_JSON, result)

    task = {
        "completed": True,
        "method": "REINVENT4-Mol2Mol",
        "formal_protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "seeds": SEEDS,
        "n_seeds": 5,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "performance_used_for_tuning": False,
        "summary_json_sha256": sha256_file(OUT_JSON),
        "per_seed_csv_sha256": sha256_file(OUT_CSV),
        "mean_sd_csv_sha256": sha256_file(OUT_STATS_CSV),
        "seed_task_completed_sha256": {
            str(seed): provenance[str(seed)]["task_completed_sha256"] for seed in SEEDS
        },
    }
    write_json_atomic(OUT_TASK, task)

    print("=" * 92)
    print("REINVENT4 FORMAL 5-SEED AGGREGATION PASS")
    print("=" * 92)
    print("seeds                 :", SEEDS)
    for key in (
        "mean_final_reward",
        "top10_mean_final_reward",
        "tier1_rate",
        "unique_scaffolds",
        "max_scaffold_occupancy",
        "hypervolume_4d",
        "requests",
        "model_evaluations",
    ):
        s = summary_stats[key]
        print(f"{key:24s}: {s['mean']:.12g} +/- {s['sample_sd']:.12g}")
    print("summary JSON          :", OUT_JSON)
    print("per-seed CSV          :", OUT_CSV)
    print("mean/SD CSV           :", OUT_STATS_CSV)
    print("aggregate TASK        :", OUT_TASK)
    print("reinvent4_5seed_complete = True")


if __name__ == "__main__":
    main()
