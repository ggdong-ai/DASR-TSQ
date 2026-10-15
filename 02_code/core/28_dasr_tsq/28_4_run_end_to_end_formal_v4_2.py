"""
28_4_run_end_to_end_formal_v4_2.py
=============================
Resumable, fail-fast Step 28 amended-v4.2 formal end-to-end runner.
Step 28 修订版 v4.2 正式端到端运行器。

The scientific search engine is unchanged. The amendment only aligns the
formal task audit with the already implemented local-MCTS termination logic:
101 is a maximum number of unique oracle calls, and a search may terminate
early only when its local chemical search space is exhausted or stagnant.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL_DIR = PROJECT_ROOT / "06_paper" / "experiment_records" / "dasr_step28_formal_protocol_amended_v4_2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "05_results" / "dasr_tsq_end_to_end" / "step28_formal_v4_2"
DEFAULT_SMOKE_DIR = PROJECT_ROOT / "05_results" / "dasr_tsq_end_to_end" / "step28_formal_runner_smoke_v4_2"

PROTOCOL_FILE = "formal_protocol.json"
TASKS_FILE = "formal_tasks.csv"
P0_FILE = "formal_p0.csv"
FREEZE_MARKER = "FORMAL_PROTOCOL_FROZEN.json"
TASK_MARKER = "TASK_COMPLETED.json"
FORMAL_MARKER = "FORMAL_RUNS_COMPLETED.json"
SMOKE_MARKER = "FORMAL_RUNNER_SMOKE.json"

ALLOWED_TERMINATION_REASONS = {
    "oracle_budget_reached",
    "search_space_exhausted_or_stagnant",
}
EARLY_TERMINATION_REASONS = {"search_space_exhausted_or_stagnant"}


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


integration = _load("step28_v2_formal_integration", _THIS_DIR / "28_0_dasr_tsq_evolution_v2.py")
DASRTSQEvolutionConfig = integration.DASRTSQEvolutionConfig
DASRTSQNSGA2Evolution = integration.DASRTSQNSGA2Evolution
STEP28_VARIANTS = integration.STEP28_VARIANTS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".__tmp__.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def dataframe_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    normalized = frame[columns].copy().sort_values(columns).reset_index(drop=True)
    payload = normalized.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_protocol(protocol_dir: Path, verify_current_code: bool = True) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, str]:
    protocol_dir = protocol_dir.resolve()
    protocol_path = protocol_dir / PROTOCOL_FILE
    marker_path = protocol_dir / FREEZE_MARKER
    tasks_path = protocol_dir / TASKS_FILE
    p0_path = protocol_dir / P0_FILE
    for path in (protocol_path, marker_path, tasks_path, p0_path):
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = load_json(protocol_path)
    marker = load_json(marker_path)
    observed = sha256_file(protocol_path)
    expected = str(marker["protocol_sha256"])
    if observed != expected:
        raise RuntimeError("Frozen Step 28 protocol hash mismatch.")
    tasks = pd.read_csv(tasks_path).sort_values("scheduler_order").reset_index(drop=True)
    p0 = pd.read_csv(p0_path)
    if len(tasks) != int(protocol["design"]["n_tasks"]):
        raise RuntimeError("Task manifest count differs from protocol.")
    if tasks["task_id"].nunique() != len(tasks):
        raise RuntimeError("Duplicate task_id in formal manifest.")
    if len(p0) != 200 or p0["smiles"].nunique() != 200:
        raise RuntimeError("Frozen formal P0 must contain 200 unique molecules.")
    p0_record = protocol.get("inputs", {}).get("frozen_formal_p0", {})
    expected_p0_sha = str(p0_record.get("sha256", ""))
    expected_p0_content_hash = str(p0_record.get("smiles_content_hash", ""))
    if not expected_p0_sha or sha256_file(p0_path) != expected_p0_sha:
        raise RuntimeError("Frozen formal_p0.csv hash differs from amended-v4.2 protocol.")
    if not expected_p0_content_hash or dataframe_hash(p0, ["smiles"]) != expected_p0_content_hash:
        raise RuntimeError("Frozen formal P0 molecule content differs from amended-v4.2 protocol.")
    if verify_current_code:
        hashes = protocol["inputs"]["code_hashes"]
        for name, expected_hash in hashes.items():
            if name.startswith("28_5_"):
                path = PROJECT_ROOT / "02_code" / "analysis" / name
            else:
                path = _THIS_DIR / name
            if not path.exists() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"Current formal code differs from frozen protocol: {name}")
        for label, record in protocol["inputs"].get("upstream_hash_checks", {}).items():
            path = Path(record["path"])
            expected_hash = str(record["expected_sha256"])
            if not path.exists() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"Current frozen upstream input differs from protocol: {label}")
    return protocol, tasks, p0, expected


def task_dir(output_dir: Path, task_id: str) -> Path:
    return output_dir / "tasks" / task_id


def validate_completed_task(path: Path, task_id: str, verify_hashes: bool = False) -> tuple[bool, str]:
    marker_path = path / TASK_MARKER
    if not marker_path.exists():
        return False, "completion_marker_missing"
    try:
        marker = load_json(marker_path)
    except Exception as exc:
        return False, f"invalid_marker:{exc}"
    if marker.get("task_id") != task_id or not bool(marker.get("completed", False)):
        return False, "completion_marker_identity_mismatch"
    for relative, record in marker.get("files", {}).items():
        file_path = path / relative
        if not file_path.exists():
            return False, f"missing:{relative}"
        if verify_hashes and sha256_file(file_path) != record.get("sha256"):
            return False, f"hash_mismatch:{relative}"
    return True, "ok"


def quarantine(path: Path) -> None:
    if not path.exists():
        return
    destination = path.with_name(path.name + f".__incomplete__.{int(time.time())}")
    os.replace(path, destination)


def build_config(task: dict[str, Any], output_dir: Path, freeze_dir: Path, n_jobs: int,
                 pop_size: int, generations: int, budget: int) -> Any:
    variant = str(task["variant"])
    definition = STEP28_VARIANTS[variant]
    return DASRTSQEvolutionConfig(
        pop_size=pop_size,
        generations=generations,
        offspring_multiplier=1.0,
        tournament_k=5,
        max_per_scaffold=(10 if definition["tsq_enabled"] else pop_size),
        random_seed=int(task["seed"]),
        output_dir=str(output_dir),
        experiment_name="step28_formal_end_to_end",
        run_id=str(task["task_id"]),
        n_jobs=n_jobs,
        variant_name=variant,
        allocation_policy=str(definition["allocation_policy"]),
        tsq_enabled=bool(definition["tsq_enabled"]),
        tsq_quota=10,
        local_oracle_budget=budget,
        local_max_depth=5,
        raw_candidate_pool_size=32,
        oracle_child_budget=8,
        top_ranked_count=6,
        random_exploration_count=2,
        max_tree_policy_steps=max(3000, budget * 20),
        max_stagnant_steps=100,
        freeze_dir=str(freeze_dir),
        device="cpu",
    )


def audit_run(run_dir: Path, task: dict[str, Any], p0: pd.DataFrame,
              pop_size: int, generations: int, budget: int) -> dict[str, Any]:
    required = [
        "history.csv", "final_population.csv", "population_initial.csv",
        "local_mcts_summary.csv", "local_mcts_errors.csv",
        "integration_validation.json", "run_metadata.json",
    ]
    errors: list[str] = []
    for name in required:
        if not (run_dir / name).exists():
            errors.append(f"missing:{name}")
    if errors:
        return {"passed": False, "critical_errors": errors}
    history = pd.read_csv(run_dir / "history.csv")
    final = pd.read_csv(run_dir / "final_population.csv")
    initial = pd.read_csv(run_dir / "population_initial.csv")
    local = pd.read_csv(run_dir / "local_mcts_summary.csv")
    local_errors = pd.read_csv(run_dir / "local_mcts_errors.csv")
    validation = load_json(run_dir / "integration_validation.json")
    expected_searches = pop_size * generations
    if len(history) != generations or list(history["generation"].astype(int)) != list(range(generations)):
        errors.append("history_generation_mismatch")
    if len(final) != pop_size or final["smiles"].nunique() != pop_size:
        errors.append("final_population_mismatch")
    if len(initial) != pop_size or initial["smiles"].nunique() != pop_size:
        errors.append("initial_population_mismatch")
    if dataframe_hash(initial, ["smiles"]) != dataframe_hash(p0, ["smiles"]):
        errors.append("initial_P0_differs_from_frozen_P0")
    if len(local) != expected_searches:
        errors.append("local_search_count_mismatch")
    if not local_errors.empty:
        errors.append("local_mcts_errors_present")
    if not bool(validation.get("passed", False)):
        errors.append("integration_validation_failed")
    if not local.empty:
        if not bool((local["status"] == "ok").all()):
            errors.append("local_status_error")

        requests = pd.to_numeric(local["local_oracle_requests"], errors="coerce")
        reasons = local["termination_reason"].astype(str)
        if bool(requests.isna().any()):
            errors.append("local_budget_non_numeric")
        else:
            requests = requests.astype(int)
            if bool((requests < 1).any()):
                errors.append("local_budget_below_one")
            if bool((requests > budget).any()):
                errors.append("local_budget_exceeded")

            unknown_reason = ~reasons.isin(ALLOWED_TERMINATION_REASONS)
            if bool(unknown_reason.any()):
                errors.append("unexpected_termination_reason")

            full_budget_mismatch = (requests == budget) & (reasons != "oracle_budget_reached")
            early_stop_mismatch = (requests < budget) & (~reasons.isin(EARLY_TERMINATION_REASONS))
            if bool((full_budget_mismatch | early_stop_mismatch).any()):
                errors.append("budget_termination_inconsistent")

        if int(local["n_prior_gate_open_events"].sum()) != 0:
            errors.append("legacy_prior_activated")
        if int(local["max_raw_candidate_count"].max()) > 32:
            errors.append("K32_exceeded")
        if int(local["max_selected_candidate_count"].max()) > 8:
            errors.append("B8_exceeded")
        if int(local["policy_count_violation_events"].sum()) != 0:
            errors.append("policy_count_violation")
        if int(local["nonfinite_score_events"].sum()) != 0:
            errors.append("nonfinite_ranker_score")
    if int(history.get("scaffold_failed_count", pd.Series([0])).sum()) != 0:
        errors.append("scaffold_assignment_failed")
    if int(history.get("invalid_scaffold_token_count", pd.Series([0])).sum()) != 0:
        errors.append("invalid_scaffold_token")
    for generation in range(generations):
        path = run_dir / f"population_gen{generation:03d}.csv"
        if not path.exists() or len(pd.read_csv(path)) != pop_size:
            errors.append(f"population_checkpoint_invalid:{generation}")
    return {
        "passed": len(errors) == 0,
        "critical_errors": errors,
        "task_id": task["task_id"],
        "variant": task["variant"],
        "seed": int(task["seed"]),
        "expected_local_searches": expected_searches,
        "observed_local_searches": len(local),
        "budget_semantics": "maximum unique oracle calls per local MCTS, including the root",
        "exact_local_budget_rate": float(np.mean(local["local_oracle_requests"] == budget)) if len(local) else 0.0,
        "early_termination_count": int(np.sum(local["local_oracle_requests"] < budget)) if len(local) else 0,
        "early_termination_rate": float(np.mean(local["local_oracle_requests"] < budget)) if len(local) else 0.0,
        "minimum_local_oracle_requests": int(local["local_oracle_requests"].min()) if len(local) else 0,
        "maximum_local_oracle_requests": int(local["local_oracle_requests"].max()) if len(local) else 0,
        "termination_reason_counts": (
            {str(k): int(v) for k, v in local["termination_reason"].value_counts().to_dict().items()}
            if len(local) else {}
        ),
        "oracle_request_count_distribution": (
            {str(int(k)): int(v) for k, v in local["local_oracle_requests"].value_counts().sort_index().to_dict().items()}
            if len(local) else {}
        ),
        "full_K32_rate": float(local["full_k_fraction"].mean()) if len(local) else math.nan,
        "more_than_B8_rate": float(local["more_than_b_fraction"].mean()) if len(local) else math.nan,
    }


def execute_task(task: dict[str, Any], protocol: dict[str, Any], p0: pd.DataFrame,
                 output_dir: Path, protocol_sha: str, *, smoke: bool = False) -> dict[str, Any]:
    task_id = str(task["task_id"])
    final_dir = task_dir(output_dir, task_id)
    ok, _ = validate_completed_task(final_dir, task_id, verify_hashes=False)
    if ok:
        return {"task_id": task_id, "status": "skipped_complete"}
    if final_dir.exists():
        quarantine(final_dir)
    staging_root = output_dir / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"{task_id}.__building__.{os.getpid()}"
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)

    design = protocol["design"]
    if smoke:
        pop_size, generations, budget, n_jobs = 8, 1, 15, 2
        source = p0.iloc[:pop_size].copy()
    else:
        pop_size = int(design["population"])
        generations = int(design["generations"])
        budget = int(design["local_oracle_budget"])
        n_jobs = int(design["n_jobs_per_run"])
        source = p0.copy()
    config = build_config(task, staging, Path(protocol["inputs"]["frozen_ranker_dir"]), n_jobs,
                          pop_size, generations, budget)
    start = time.time()
    engine = DASRTSQNSGA2Evolution(source["smiles"].astype(str).tolist(), config)
    engine.run()
    audit = audit_run(staging, task, source, pop_size, generations, budget)
    audit["elapsed_seconds"] = float(time.time() - start)
    write_json_atomic(staging / "task_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"Task audit failed: {task_id}: {audit['critical_errors']}")
    files: dict[str, Any] = {}
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(staging)).replace("\\", "/")
            files[relative] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json_atomic(staging / TASK_MARKER, {
        "completed": True, "task_id": task_id, "variant": task["variant"],
        "seed": int(task["seed"]), "protocol_sha256": protocol_sha,
        "budget_semantics": "maximum unique oracle calls; legitimate stagnation termination is allowed",
        "completed_utc": utc_now(), "files": files,
    })
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final_dir)
    return {"task_id": task_id, "status": "completed", "elapsed_seconds": audit["elapsed_seconds"]}


def scan(output_dir: Path, tasks: pd.DataFrame, verify_hashes: bool = False) -> dict[str, Any]:
    completed, pending, reasons = [], [], {}
    for task in tasks.to_dict(orient="records"):
        task_id = str(task["task_id"])
        ok, reason = validate_completed_task(task_dir(output_dir, task_id), task_id, verify_hashes)
        if ok: completed.append(task_id)
        else:
            pending.append(task_id)
            if task_dir(output_dir, task_id).exists(): reasons[task_id] = reason
    return {"n_total": len(tasks), "n_completed": len(completed), "n_pending": len(pending),
            "completed": completed, "pending": pending, "reasons": reasons}


def require_smoke(smoke_dir: Path) -> None:
    path = smoke_dir / SMOKE_MARKER
    if not path.exists():
        raise RuntimeError("Formal runner smoke marker is missing.")
    report = load_json(path)
    if not bool(report.get("passed", False)) or not bool(report.get("formal_authorized", False)):
        raise RuntimeError("Run the formal runner smoke twice; resume/skip verification is not complete.")


def run_smoke(args: argparse.Namespace) -> None:
    protocol, _, p0, protocol_sha = verify_protocol(args.protocol_dir)
    output = args.smoke_dir.resolve()
    if args.reset_runner_smoke and output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    smoke_tasks = pd.DataFrame([
        {"task_id": "runner_smoke_baseline_seed424242", "variant": "baseline_random_no_tsq", "seed": 424242},
        {"task_id": "runner_smoke_listwise_seed424242", "variant": "dasr_only_listwise_no_tsq", "seed": 424242},
    ])
    before = scan(output, smoke_tasks)
    results = []
    for task in smoke_tasks.to_dict(orient="records"):
        results.append(execute_task(task, protocol, p0, output, protocol_sha, smoke=True))
    after = scan(output, smoke_tasks, verify_hashes=True)
    passed = after["n_pending"] == 0
    resume_test_passed = before["n_completed"] == len(smoke_tasks) and all(r["status"] == "skipped_complete" for r in results)
    write_json_atomic(output / SMOKE_MARKER, {
        "passed": passed, "completed_tasks": after["n_completed"], "expected_tasks": len(smoke_tasks),
        "newly_completed": sum(r["status"] == "completed" for r in results),
        "resume_test_passed": resume_test_passed,
        "formal_authorized": bool(passed and resume_test_passed),
        "instruction": "Run the identical command a second time; newly_completed should be 0.",
    })
    print(f"Runner smoke passed: {passed}; resume_test_passed={resume_test_passed}; completed={after['n_completed']}/{len(smoke_tasks)}")
    print("Run the identical command a second time to verify skip/resume behavior.")
    if not passed: raise RuntimeError("Formal runner smoke failed.")


def run_formal(args: argparse.Namespace) -> None:
    protocol, tasks, p0, protocol_sha = verify_protocol(args.protocol_dir)
    require_smoke(args.smoke_dir.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / FORMAL_MARKER).exists():
        raise FileExistsError("Formal completion marker exists; do not rerun or overwrite.")
    status = scan(output, tasks)
    pending_set = set(status["pending"])
    selected = [t for t in tasks.to_dict(orient="records") if str(t["task_id"]) in pending_set]
    if args.task_id:
        selected = [t for t in selected if str(t["task_id"]) == args.task_id]
        if not selected and args.task_id not in status["completed"]:
            raise ValueError(f"Unknown task_id: {args.task_id}")
    if args.max_tasks is not None:
        selected = selected[:args.max_tasks]
    failures_dir = output / "failures"
    failures_dir.mkdir(exist_ok=True)
    newly, failed = 0, 0
    first_failure: RuntimeError | None = None
    for index, task in enumerate(selected, start=1):
        task_id = str(task["task_id"])
        print(
            f"[{index}/{len(selected)}] Running formal task {task_id}",
            flush=True,
        )
        try:
            result = execute_task(
                task,
                protocol,
                p0,
                output,
                protocol_sha,
                smoke=False,
            )
            if result["status"] == "completed":
                newly += 1
        except Exception as exc:
            failed += 1
            failure_path = failures_dir / f"{task_id}.{int(time.time())}.json"
            write_json_atomic(
                failure_path,
                {
                    "task_id": task_id,
                    "time_utc": utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"FAILED {task_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            print(f"Failure record: {failure_path}", flush=True)
            first_failure = RuntimeError(
                f"Formal task failed and the runner stopped: {task_id}"
            )
            break

    final_status = scan(output, tasks)
    write_json_atomic(output / "formal_progress.json", {
        "updated_utc": utc_now(), "newly_completed": newly, "failed_this_invocation": failed, **final_status,
    })
    print(
        f"Completed total: {final_status['n_completed']}/{final_status['n_total']}; "
        f"pending={final_status['n_pending']}",
        flush=True,
    )
    if first_failure is not None:
        raise first_failure

    if final_status["n_pending"] == 0:
        verified = scan(output, tasks, verify_hashes=True)
        if verified["n_pending"] != 0:
            raise RuntimeError("Full task hash verification failed.")
        write_json_atomic(output / FORMAL_MARKER, {
            "completed": True, "completed_utc": utc_now(), "protocol_sha256": protocol_sha,
            "n_tasks": len(tasks), "n_variants": tasks["variant"].nunique(), "n_seeds": tasks["seed"].nunique(),
            "population": int(protocol["design"]["population"]), "generations": int(protocol["design"]["generations"]),
            "local_oracle_budget": int(protocol["design"]["local_oracle_budget"]),
            "budget_semantics": protocol["design"].get(
                "local_oracle_budget_semantics",
                "maximum unique oracle calls per local MCTS, including root",
            ),
            "next_action": "Run 28_5_analyze_end_to_end_formal_v4_1.py",
        })
        print("ALL FORMAL TASKS COMPLETED AND HASH VERIFIED.")


def print_status(args: argparse.Namespace) -> None:
    _, tasks, _, _ = verify_protocol(args.protocol_dir, verify_current_code=False)
    status = scan(args.output_dir.resolve(), tasks, verify_hashes=args.verify_existing_hashes)
    print(json.dumps(status, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 28 amended-v4.2 formal end-to-end runner.")
    parser.add_argument("--stage", choices=["runner_smoke", "formal", "status"], required=True)
    parser.add_argument("--protocol_dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke_dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--max_tasks", type=int, default=1)
    parser.add_argument("--task_id", type=str, default=None)
    parser.add_argument("--verify_existing_hashes", action="store_true")
    parser.add_argument("--reset_runner_smoke", action="store_true")
    args = parser.parse_args()
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("--max_tasks must be >=1")
    if args.stage == "runner_smoke": run_smoke(args)
    elif args.stage == "formal": run_formal(args)
    else: print_status(args)


if __name__ == "__main__":
    main()
