"""
27_4_run_dasr_mcts_formal.py
================================
Parallel and resumable formal runner for frozen DASR-MCTS online evaluation.
冻结 DASR-MCTS 在线正式实验的并行、断点续跑运行器。

Scientific design frozen before formal execution
-------------------------------------------------
- 200 fixed P0 roots
- 7 frozen online variants
- seeds 42--46
- raw candidate pool K=32
- pre-oracle budget B=8
- primary method: Listwise Top-6 + Random-2
- 101 unique oracle calls per root/variant/seed search, including the root
- legacy P_success disabled for every variant

The script has four stages:
1. plan          Create an immutable protocol/task manifest; no oracle is called.
2. runner_smoke  Test multiprocessing, atomic outputs, and resume behavior.
3. formal        Execute/resume the complete 7000-task formal experiment.
4. status        Inspect progress without running tasks.

Completed task directories are never overwritten. A task is considered complete only
when TASK_COMPLETED.json exists and its output hashes are valid.
"""

from __future__ import annotations

# Avoid nested BLAS/OpenMP oversubscription when several worker processes are used.
import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import platform
import random
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase


# ============================================================
# 0. Paths and immutable formal settings
# ============================================================

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "02_code" / "analysis"

DEFAULT_P0_CSV = (
    PROJECT_ROOT
    / "04_runs"
    / "tsq_nsga2"
    / "P0_reference"
    / "P0_fixed_cold_start.csv"
)
DEFAULT_FREEZE_DIR = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "dasr_p0_safe_v2_test_lock"
)
DEFAULT_STEP26_FROZEN_DIR = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "dasr_p0_safe_v2_test_frozen"
)
DEFAULT_PILOT_ANALYSIS_DIR = (
    PROJECT_ROOT
    / "05_results"
    / "dasr_mcts_online"
    / "step27"
    / "pilot_analysis"
)
DEFAULT_PROTOCOL_DIR = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "dasr_step27_formal_protocol_frozen"
)
DEFAULT_FORMAL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "05_results"
    / "dasr_mcts_online"
    / "step27"
    / "formal"
)
DEFAULT_RUNNER_SMOKE_DIR = (
    PROJECT_ROOT
    / "05_results"
    / "dasr_mcts_online"
    / "step27"
    / "formal_runner_smoke"
)

FROZEN_ROOT_COUNT = 200
FROZEN_SEEDS = (42, 43, 44, 45, 46)
FROZEN_VARIANTS = (
    "random_8",
    "action_only_6plus2",
    "absolute_child_6plus2",
    "pointwise_delta_6plus2",
    "pairwise_ranker_6plus2",
    "listwise_ranker_6plus2",
    "listwise_ranker_top8",
)
PRIMARY_VARIANT = "listwise_ranker_6plus2"
RAW_CANDIDATE_POOL_SIZE = 32
ORACLE_CHILD_BUDGET = 8
TOP_RANKED_COUNT = 6
RANDOM_EXPLORATION_COUNT = 2
FORMAL_ORACLE_BUDGET = 101
FORMAL_MAX_DEPTH = 5
FORMAL_MAX_TREE_POLICY_STEPS = 3000
FORMAL_MAX_STAGNANT_STEPS = 100
EXPLORATION_CONSTANT = math.sqrt(2.0)
SCHEDULER_SEED = 27142
EPS = 1e-12

REQUIRED_MODEL_FILES = (
    "action_only.json",
    "absolute_child.pt",
    "pointwise_delta.pt",
    "pairwise_ranker.pt",
    "listwise_ranker.pt",
)

PROTOCOL_FILENAME = "formal_protocol.json"
ROOTS_FILENAME = "formal_roots.csv"
TASKS_FILENAME = "formal_tasks.csv"
FREEZE_MARKER_FILENAME = "FORMAL_PROTOCOL_FROZEN.json"
RUNNER_SMOKE_MARKER = "formal_runner_smoke_validation.json"
FORMAL_COMPLETED_MARKER = "FORMAL_RUNS_COMPLETED.json"

# Worker globals are initialized once in every spawned process.
_WORKER_DASR: Any = None
_WORKER_RANKER_BANK: Any = None
_WORKER_FITNESS: Any = None
_WORKER_MUTATOR: Any = None
_WORKER_SETTINGS: dict[str, Any] = {}


# ============================================================
# 1. Generic utilities
# ============================================================


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".__tmp__.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_smiles(smiles: Any) -> str | None:
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def copy_file_verified(source: Path, destination: Path) -> dict[str, Any]:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"Copy hash mismatch: {source} -> {destination}")
    return {
        "source": str(source),
        "frozen_copy": str(destination),
        "sha256": destination_hash,
        "size_bytes": destination.stat().st_size,
    }


def stable_scheduler_order(task_ids: Iterable[str], seed: int) -> list[str]:
    task_ids = list(task_ids)
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    return task_ids


# ============================================================
# 2. Protocol construction and verification
# ============================================================


def validate_upstream_scientific_state(
    *,
    freeze_dir: Path,
    step26_frozen_dir: Path,
    pilot_analysis_dir: Path,
) -> dict[str, Any]:
    freeze_dir = freeze_dir.resolve()
    step26_frozen_dir = step26_frozen_dir.resolve()
    pilot_analysis_dir = pilot_analysis_dir.resolve()

    missing_models = [name for name in REQUIRED_MODEL_FILES if not (freeze_dir / name).exists()]
    if missing_models:
        raise FileNotFoundError(f"Missing frozen ranker artifacts: {missing_models}")

    validation_decision_path = freeze_dir / "validation_model_selection_decision.json"
    if not validation_decision_path.exists():
        raise FileNotFoundError(validation_decision_path)
    validation_decision = load_json(validation_decision_path)
    if not bool(validation_decision.get("validation_candidate_selected", False)):
        raise RuntimeError("Frozen validation candidate was not selected.")
    p0_ready = bool(
        validation_decision.get("p0_online_deployment_ready", False)
        or validation_decision.get("online_deployment_ready", False)
    )
    if not p0_ready:
        raise RuntimeError("Frozen validation decision is not P0 deployment-ready.")

    step26_completed_path = step26_frozen_dir / "ONE_TIME_TEST_COMPLETED.json"
    if not step26_completed_path.exists():
        raise FileNotFoundError(step26_completed_path)
    step26_completed = load_json(step26_completed_path)
    if not bool(step26_completed.get("completed", False)):
        raise RuntimeError("Step 26 frozen record does not have completed=true.")

    pilot_protocol_path = pilot_analysis_dir / "pilot_protocol_audit.json"
    pilot_readiness_path = pilot_analysis_dir / "pilot_readiness_decision.json"
    if not pilot_protocol_path.exists() or not pilot_readiness_path.exists():
        raise FileNotFoundError(
            "Step 27_3 pilot audit/readiness files are missing. Run and freeze Step 27_3 first."
        )
    pilot_protocol = load_json(pilot_protocol_path)
    pilot_readiness = load_json(pilot_readiness_path)
    if not bool(pilot_protocol.get("passed", False)):
        raise RuntimeError("Step 27_3 pilot protocol audit did not pass.")
    if not bool(pilot_readiness.get("ready_for_formal_runner_development", False)):
        raise RuntimeError("Step 27_3 did not authorize formal runner development.")
    if str(pilot_readiness.get("primary_variant_remains_frozen")) != PRIMARY_VARIANT:
        raise RuntimeError("Pilot readiness record changed the frozen primary variant.")

    return {
        "validation_candidate_selected": True,
        "p0_online_deployment_ready": True,
        "step26_one_time_test_completed": True,
        "step26_confirmatory_replication_passed": bool(
            step26_completed.get("confirmatory_replication_passed", False)
        ),
        "pilot_protocol_audit_passed": True,
        "ready_for_formal_runner_development": True,
        "primary_variant_remains_frozen": PRIMARY_VARIANT,
        "scientific_boundary": (
            "Step 26 did not establish universal superiority against every learned comparator. "
            "The formal online experiment therefore tests a separate fixed-oracle-budget search "
            "efficiency hypothesis and must report learned-comparator results without post-test reselection."
        ),
        "upstream_files": {
            "validation_decision": {
                "path": str(validation_decision_path),
                "sha256": sha256_file(validation_decision_path),
            },
            "step26_completion": {
                "path": str(step26_completed_path),
                "sha256": sha256_file(step26_completed_path),
            },
            "pilot_protocol_audit": {
                "path": str(pilot_protocol_path),
                "sha256": sha256_file(pilot_protocol_path),
            },
            "pilot_readiness_decision": {
                "path": str(pilot_readiness_path),
                "sha256": sha256_file(pilot_readiness_path),
            },
        },
    }


def load_all_formal_roots(p0_csv: Path, smiles_column: str) -> pd.DataFrame:
    p0_csv = p0_csv.resolve()
    if not p0_csv.exists():
        raise FileNotFoundError(p0_csv)
    frame = pd.read_csv(p0_csv)
    if smiles_column not in frame.columns:
        raise ValueError(
            f"P0 column {smiles_column!r} not found. Available columns={list(frame.columns)}"
        )

    records: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    invalid_rows: list[int] = []
    duplicate_rows: list[dict[str, Any]] = []
    for source_row_index, raw in enumerate(frame[smiles_column].tolist()):
        canonical = canonicalize_smiles(raw)
        if canonical is None:
            invalid_rows.append(source_row_index)
            continue
        if canonical in seen:
            duplicate_rows.append(
                {
                    "source_row_index": source_row_index,
                    "duplicate_of_source_row_index": seen[canonical],
                    "canonical_smiles": canonical,
                }
            )
            continue
        seen[canonical] = source_row_index
        records.append(
            {
                "root_index": len(records),
                "source_row_index": source_row_index,
                "root_smiles": canonical,
            }
        )

    roots = pd.DataFrame(records)
    if invalid_rows:
        raise ValueError(f"P0 contains invalid SMILES rows: {invalid_rows[:20]}")
    if duplicate_rows:
        raise ValueError(f"P0 contains duplicate canonical molecules: {duplicate_rows[:5]}")
    if len(roots) != FROZEN_ROOT_COUNT:
        raise ValueError(
            f"Formal protocol requires exactly {FROZEN_ROOT_COUNT} unique P0 roots; "
            f"observed {len(roots)}. Do not sample or replace roots after pilot."
        )
    return roots


def build_formal_task_manifest(roots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    task_index = 0
    for root in roots.itertuples(index=False):
        for seed in FROZEN_SEEDS:
            search_seed = int(seed) + int(root.root_index) * 1009
            for variant in FROZEN_VARIANTS:
                task_id = f"formal_{variant}_root{int(root.root_index):03d}_seed{int(seed)}"
                rows.append(
                    {
                        "task_index": task_index,
                        "task_id": task_id,
                        "root_index": int(root.root_index),
                        "source_row_index": int(root.source_row_index),
                        "root_smiles": str(root.root_smiles),
                        "seed": int(seed),
                        "variant": str(variant),
                        "search_random_seed": search_seed,
                    }
                )
                task_index += 1
    manifest = pd.DataFrame(rows)
    expected = FROZEN_ROOT_COUNT * len(FROZEN_SEEDS) * len(FROZEN_VARIANTS)
    if len(manifest) != expected:
        raise RuntimeError(f"Task manifest size mismatch: {len(manifest)} != {expected}")
    if manifest["task_id"].duplicated().any():
        raise RuntimeError("Duplicate task IDs were generated.")
    return manifest


def analysis_plan_payload() -> dict[str, Any]:
    """Pre-specified Step 27_5 analysis plan frozen before formal outcomes exist."""
    return {
        "statistical_unit": (
            "P0 root. The five seeds are averaged within each root before paired inference, "
            "preventing pseudoreplication of repeated stochastic runs."
        ),
        "primary_variant": PRIMARY_VARIANT,
        "co_primary_endpoints": [
            "auc_best_reward",
            "auc_top10_mean_reward",
        ],
        "core_baseline_comparators": [
            "random_8",
            "action_only_6plus2",
        ],
        "core_family_hypotheses": 4,
        "core_pass_rule": (
            "For each co-primary endpoint against each core baseline, the paired root-level "
            "95% bootstrap CI lower bound must be >0 and the two-sided paired sign-flip "
            "permutation p-value must remain <0.05 after Holm correction across the four tests."
        ),
        "learned_comparator_family": [
            "absolute_child_6plus2",
            "pointwise_delta_6plus2",
            "pairwise_ranker_6plus2",
        ],
        "learned_comparator_role": (
            "Secondary confirmatory family with Holm correction across six comparisons. "
            "Failure to beat every learned comparator does not invalidate the narrower core "
            "baseline claim and must be reported transparently."
        ),
        "exploration_ablation": {
            "comparator": "listwise_ranker_top8",
            "quality_noninferiority_metrics": {
                "auc_best_reward": 0.005,
                "auc_top10_mean_reward": 0.005,
                "final_top10_mean_reward": 0.005,
            },
            "quality_noninferiority_rule": (
                "For primary-minus-Top8, the 95% CI lower bound must exceed the negative "
                "pre-frozen margin for every quality metric."
            ),
            "diversity_metric": "final_unique_scaffolds",
            "diversity_superiority_rule": (
                "The paired 95% CI lower bound for primary-minus-Top8 unique scaffolds must be >0."
            ),
            "status": "Margins selected from technical/pilot evidence and frozen before formal outcomes.",
        },
        "secondary_descriptive_endpoints": [
            "final_best_reward",
            "final_top10_mean_reward",
            "final_unique_scaffolds",
            "tier1_hits",
            "best_reward_improvement",
            "top10_reward_improvement",
            "elapsed_seconds",
            "checkpoint trajectories at oracle calls 1,10,25,50,75,101",
            "fixed reward-threshold hit rates and time-to-threshold",
        ],
        "tier1_definition": {
            "dual_min_pic50": ">=6.0",
            "qed": ">=0.50",
            "sa_score": "<=6.0",
        },
        "fixed_reward_thresholds": [0.45, 0.50, 0.55],
        "root_relative_improvement_thresholds": [0.025, 0.050],
        "bootstrap_resamples": 20000,
        "permutation_resamples": 20000,
        "confidence_level": 0.95,
        "multiplicity_method": "Holm step-down family-wise error control",
        "alpha": 0.05,
        "decision_boundaries": {
            "formal_online_primary_claim_supported": (
                "Protocol integrity passes and all four core-baseline co-primary tests pass."
            ),
            "exploration_tradeoff_supported": (
                "All three quality noninferiority checks and scaffold-diversity superiority pass."
            ),
            "universal_learned_method_superiority": (
                "Not assumed. It may be claimed only if all six secondary learned-comparator "
                "tests pass after Holm correction."
            ),
        },
    }


def create_frozen_protocol(args: argparse.Namespace) -> None:
    protocol_dir = args.protocol_dir.resolve()
    if protocol_dir.exists():
        raise FileExistsError(
            f"Protocol directory already exists and must not be overwritten: {protocol_dir}"
        )

    p0_csv = args.p0_csv.resolve()
    freeze_dir = args.freeze_dir.resolve()
    step26_frozen_dir = args.step26_frozen_dir.resolve()
    pilot_analysis_dir = args.pilot_analysis_dir.resolve()

    upstream = validate_upstream_scientific_state(
        freeze_dir=freeze_dir,
        step26_frozen_dir=step26_frozen_dir,
        pilot_analysis_dir=pilot_analysis_dir,
    )
    roots = load_all_formal_roots(p0_csv, args.p0_smiles_column)
    tasks = build_formal_task_manifest(roots)

    required_code = {
        "27_0_frozen_sibling_rankers.py": _THIS_DIR / "27_0_frozen_sibling_rankers.py",
        "27_1_dasr_mcts.py": _THIS_DIR / "27_1_dasr_mcts.py",
        "27_4_run_dasr_mcts_formal.py": Path(__file__).resolve(),
        "27_5_analyze_dasr_mcts_formal.py": ANALYSIS_DIR / "27_5_analyze_dasr_mcts_formal.py",
    }
    for path in required_code.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Required formal code is missing. Install both Step 27_4 and 27_5 before plan: {path}"
            )

    temporary_dir = protocol_dir.with_name(protocol_dir.name + f".__building__.{os.getpid()}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)

    try:
        roots_path = temporary_dir / ROOTS_FILENAME
        tasks_path = temporary_dir / TASKS_FILENAME
        roots.to_csv(roots_path, index=False)
        tasks.to_csv(tasks_path, index=False)

        frozen_code: dict[str, Any] = {}
        code_dir = temporary_dir / "code"
        for name, source in required_code.items():
            frozen_code[name] = copy_file_verified(source, code_dir / name)

        frozen_upstream: dict[str, Any] = {}
        upstream_dir = temporary_dir / "upstream_records"
        for logical_name, record in upstream["upstream_files"].items():
            source = Path(record["path"])
            frozen_upstream[logical_name] = copy_file_verified(
                source,
                upstream_dir / source.name,
            )

        model_hashes = {
            name: {
                "path": str((freeze_dir / name).resolve()),
                "sha256": sha256_file(freeze_dir / name),
                "size_bytes": (freeze_dir / name).stat().st_size,
            }
            for name in REQUIRED_MODEL_FILES
        }

        protocol = {
            "protocol_version": "dasr_step27_formal_v1",
            "created_at_utc": utc_now_iso(),
            "scientific_status": "FROZEN BEFORE FORMAL ONLINE OUTCOMES",
            "project_root": str(PROJECT_ROOT),
            "formal_design": {
                "n_roots": FROZEN_ROOT_COUNT,
                "seeds": list(FROZEN_SEEDS),
                "variants": list(FROZEN_VARIANTS),
                "primary_variant": PRIMARY_VARIANT,
                "raw_candidate_pool_size": RAW_CANDIDATE_POOL_SIZE,
                "oracle_child_budget": ORACLE_CHILD_BUDGET,
                "top_ranked_count": TOP_RANKED_COUNT,
                "random_exploration_count": RANDOM_EXPLORATION_COUNT,
                "oracle_budget_including_root": FORMAL_ORACLE_BUDGET,
                "max_depth": FORMAL_MAX_DEPTH,
                "max_tree_policy_steps": FORMAL_MAX_TREE_POLICY_STEPS,
                "max_stagnant_steps": FORMAL_MAX_STAGNANT_STEPS,
                "exploration_constant": EXPLORATION_CONSTANT,
                "legacy_p_success": "disabled for all variants",
                "n_tasks": len(tasks),
                "maximum_oracle_requests": len(tasks) * FORMAL_ORACLE_BUDGET,
                "scheduler_seed": SCHEDULER_SEED,
            },
            "analysis_plan": analysis_plan_payload(),
            "upstream_state": upstream,
            "inputs": {
                "p0_csv": {
                    "path": str(p0_csv),
                    "sha256": sha256_file(p0_csv),
                },
                "freeze_dir": str(freeze_dir),
                "step26_frozen_dir": str(step26_frozen_dir),
                "pilot_analysis_dir": str(pilot_analysis_dir),
                "model_artifacts": model_hashes,
            },
            "frozen_files": {
                "roots_csv": {
                    "path": ROOTS_FILENAME,
                    "sha256": sha256_file(roots_path),
                },
                "tasks_csv": {
                    "path": TASKS_FILENAME,
                    "sha256": sha256_file(tasks_path),
                },
                "code": frozen_code,
                "upstream_records": frozen_upstream,
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "rdkit": rdBase.rdkitVersion,
            },
            "immutability_policy": (
                "Do not change roots, seeds, variants, model files, budgets, endpoints, margins, "
                "or multiplicity rules after this marker is created. Technical failures may be "
                "retried with the identical task specification."
            ),
        }

        protocol_path = temporary_dir / PROTOCOL_FILENAME
        write_json_atomic(protocol_path, protocol)
        protocol_hash = sha256_file(protocol_path)
        marker = {
            "frozen": True,
            "created_at_utc": utc_now_iso(),
            "protocol_sha256": protocol_hash,
            "roots_sha256": sha256_file(roots_path),
            "tasks_sha256": sha256_file(tasks_path),
            "n_roots": len(roots),
            "n_tasks": len(tasks),
            "important_instruction": (
                "Run formal_runner_smoke next. Do not edit files in this directory."
            ),
        }
        write_json_atomic(temporary_dir / FREEZE_MARKER_FILENAME, marker)

        os.replace(temporary_dir, protocol_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print("=" * 96)
    print("Step 27_4 formal protocol frozen successfully")
    print(f"Protocol: {protocol_dir}")
    print(f"Roots: {FROZEN_ROOT_COUNT}")
    print(f"Tasks: {len(tasks)}")
    print(f"Maximum oracle requests: {len(tasks) * FORMAL_ORACLE_BUDGET:,}")
    print("Next mandatory stage: --stage runner_smoke")
    print("=" * 96)


def verify_frozen_protocol(protocol_dir: Path, verify_current_inputs: bool = True) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    protocol_dir = protocol_dir.resolve()
    marker_path = protocol_dir / FREEZE_MARKER_FILENAME
    protocol_path = protocol_dir / PROTOCOL_FILENAME
    roots_path = protocol_dir / ROOTS_FILENAME
    tasks_path = protocol_dir / TASKS_FILENAME
    for path in (marker_path, protocol_path, roots_path, tasks_path):
        if not path.exists():
            raise FileNotFoundError(path)

    marker = load_json(marker_path)
    protocol = load_json(protocol_path)
    if not bool(marker.get("frozen", False)):
        raise RuntimeError("Formal protocol marker does not have frozen=true.")
    if sha256_file(protocol_path) != marker.get("protocol_sha256"):
        raise RuntimeError("Formal protocol JSON hash mismatch.")
    if sha256_file(roots_path) != marker.get("roots_sha256"):
        raise RuntimeError("Formal roots CSV hash mismatch.")
    if sha256_file(tasks_path) != marker.get("tasks_sha256"):
        raise RuntimeError("Formal task manifest hash mismatch.")

    roots = pd.read_csv(roots_path)
    tasks = pd.read_csv(tasks_path)
    design = protocol["formal_design"]
    if len(roots) != int(design["n_roots"]):
        raise RuntimeError("Frozen root count mismatch.")
    if len(tasks) != int(design["n_tasks"]):
        raise RuntimeError("Frozen task count mismatch.")
    if tuple(design["seeds"]) != FROZEN_SEEDS:
        raise RuntimeError("Frozen seeds differ from Step 27 v1 protocol.")
    if tuple(design["variants"]) != FROZEN_VARIANTS:
        raise RuntimeError("Frozen variants differ from Step 27 v1 protocol.")
    if str(design["primary_variant"]) != PRIMARY_VARIANT:
        raise RuntimeError("Frozen primary variant mismatch.")

    if verify_current_inputs:
        p0_record = protocol["inputs"]["p0_csv"]
        if sha256_file(Path(p0_record["path"])) != p0_record["sha256"]:
            raise RuntimeError("Current P0 CSV hash differs from frozen protocol.")
        for name, record in protocol["inputs"]["model_artifacts"].items():
            if sha256_file(Path(record["path"])) != record["sha256"]:
                raise RuntimeError(f"Frozen model artifact hash changed: {name}")
        for name, record in protocol["frozen_files"]["code"].items():
            current_source = Path(record["source"])
            if sha256_file(current_source) != record["sha256"]:
                raise RuntimeError(
                    f"Current formal code differs from the frozen copy: {name}. "
                    "Run the exact frozen code; do not edit after planning."
                )

    return protocol, roots, tasks


# ============================================================
# 3. Task paths and completion validation
# ============================================================


def task_output_dir(output_dir: Path, task: dict[str, Any]) -> Path:
    return (
        output_dir
        / "tasks"
        / f"seed{int(task['seed'])}"
        / str(task["variant"])
        / f"root{int(task['root_index']):03d}"
    )


def validate_task_completion(final_dir: Path, expected_task_id: str, verify_hashes: bool = True) -> tuple[bool, str]:
    marker_path = final_dir / "TASK_COMPLETED.json"
    if not marker_path.exists():
        return False, "missing_completion_marker"
    try:
        marker = load_json(marker_path)
    except Exception as exc:
        return False, f"invalid_completion_marker:{type(exc).__name__}"
    if not bool(marker.get("completed", False)):
        return False, "completed_flag_false"
    if str(marker.get("task_id")) != expected_task_id:
        return False, "task_id_mismatch"
    required_outputs = marker.get("output_files", {})
    if not isinstance(required_outputs, dict) or not required_outputs:
        return False, "missing_output_hashes"
    for name, expected_hash in required_outputs.items():
        path = final_dir / name
        if not path.exists():
            return False, f"missing_output:{name}"
        if verify_hashes and sha256_file(path) != expected_hash:
            return False, f"output_hash_mismatch:{name}"
    return True, "ok"


def quarantine_incomplete_directory(final_dir: Path) -> Path | None:
    if not final_dir.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine = final_dir.with_name(final_dir.name + f".__incomplete__.{stamp}.{os.getpid()}")
    os.replace(final_dir, quarantine)
    return quarantine


# ============================================================
# 4. Worker process
# ============================================================


def _dynamic_load(module_name: str, path: Path) -> Any:
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def worker_initializer(
    step27_module_path: str,
    freeze_dir: str,
    device: str,
    settings: dict[str, Any],
) -> None:
    global _WORKER_DASR, _WORKER_RANKER_BANK, _WORKER_FITNESS, _WORKER_MUTATOR, _WORKER_SETTINGS

    # Import torch/RDKit models inside the worker, not in the parent process.
    module_name = f"dasr_mcts_formal_worker_{os.getpid()}"
    _WORKER_DASR = _dynamic_load(module_name, Path(step27_module_path))
    _WORKER_RANKER_BANK = _WORKER_DASR.FrozenSiblingRankerBank(
        freeze_dir=Path(freeze_dir),
        device=device,
    )
    _WORKER_FITNESS = _WORKER_DASR.FrozenFitnessAdapter()
    _WORKER_MUTATOR = _WORKER_DASR.MoleculeMutator()
    _WORKER_SETTINGS = dict(settings)


def clear_worker_caches() -> None:
    """Bound per-process memory while retaining loaded model objects."""
    if _WORKER_RANKER_BANK is not None:
        if hasattr(_WORKER_RANKER_BANK, "fp_cache"):
            _WORKER_RANKER_BANK.fp_cache.clear()
        if hasattr(_WORKER_RANKER_BANK, "score_cache"):
            _WORKER_RANKER_BANK.score_cache.clear()
    if _WORKER_FITNESS is not None and hasattr(_WORKER_FITNESS, "_cache"):
        _WORKER_FITNESS._cache.clear()


def dataframe_nonfinite_count(frame: pd.DataFrame, columns: list[str]) -> int:
    available = [column for column in columns if column in frame.columns]
    if not available or frame.empty:
        return 0
    values = frame[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return int((~np.isfinite(values)).sum())


def build_task_audit(
    *,
    summary: dict[str, Any],
    trajectory: pd.DataFrame,
    candidates: pd.DataFrame,
    nodes: pd.DataFrame,
    selections: pd.DataFrame,
    task: dict[str, Any],
) -> dict[str, Any]:
    variant = str(task["variant"])
    budget = int(_WORKER_SETTINGS["oracle_budget"])
    raw_k = int(_WORKER_SETTINGS["raw_candidate_pool_size"])
    child_b = int(_WORKER_SETTINGS["oracle_child_budget"])
    top_n = int(_WORKER_SETTINGS["top_ranked_count"])
    random_n = int(_WORKER_SETTINGS["random_exploration_count"])

    selection_rule_ok = True
    expected_selected = pd.Series(dtype=np.int64)
    if not candidates.empty:
        expected_selected = np.minimum(
            candidates["raw_candidate_count"].to_numpy(dtype=np.int64),
            child_b,
        )
        if variant == "random_8":
            selection_rule_ok = bool(
                (candidates["selected_top_count"].to_numpy(dtype=np.int64) == 0).all()
                and (
                    candidates["selected_random_count"].to_numpy(dtype=np.int64)
                    == expected_selected
                ).all()
            )
        elif variant == "listwise_ranker_top8":
            selection_rule_ok = bool(
                (
                    candidates["selected_top_count"].to_numpy(dtype=np.int64)
                    == expected_selected
                ).all()
                and (candidates["selected_random_count"].to_numpy(dtype=np.int64) == 0).all()
            )
        else:
            expected_top = np.minimum(expected_selected, top_n)
            expected_random = np.minimum(
                random_n,
                np.maximum(0, expected_selected - expected_top),
            )
            selection_rule_ok = bool(
                (
                    candidates["selected_top_count"].to_numpy(dtype=np.int64)
                    == expected_top
                ).all()
                and (
                    candidates["selected_random_count"].to_numpy(dtype=np.int64)
                    == expected_random
                ).all()
            )

    ranked = candidates[
        candidates.get(
            "ranker_name",
            pd.Series(index=candidates.index, dtype=object),
        ).notna()
    ].copy()
    score_spread_min = None
    ranker_scores_valid = True
    if not ranked.empty and {"score_min", "score_max"}.issubset(ranked.columns):
        ranked["score_spread"] = ranked["score_max"] - ranked["score_min"]
        informative = ranked[ranked["raw_candidate_count"] > 1]
        if not informative.empty:
            score_spread_min = float(informative["score_spread"].min())
            ranker_scores_valid = bool((informative["score_spread"] > EPS).all())

    checks = {
        "oracle_budget_exact": int(summary["oracle_calls_used"]) == budget,
        "oracle_budget_not_exceeded": int(summary["oracle_calls_used"]) <= budget,
        "terminated_by_budget": str(summary["termination_reason"]) == "oracle_budget_reached",
        "legacy_prior_disabled": int(summary["n_prior_gate_open_events"]) == 0,
        "trajectory_length_matches_oracle_calls": len(trajectory) == int(summary["oracle_calls_used"]),
        "trajectory_oracle_calls_contiguous": bool(
            np.array_equal(
                trajectory["oracle_call"].to_numpy(dtype=int),
                np.arange(1, len(trajectory) + 1, dtype=int),
            )
        ),
        "raw_candidate_budget_respected": bool(
            candidates.empty or (candidates["raw_candidate_count"] <= raw_k).all()
        ),
        "selected_candidate_budget_respected": bool(
            candidates.empty or (candidates["selected_candidate_count"] <= child_b).all()
        ),
        "selected_count_equals_available_budget": bool(
            candidates.empty
            or (
                candidates["selected_candidate_count"].to_numpy(dtype=np.int64)
                == expected_selected
            ).all()
        ),
        "variant_selection_rule_correct": selection_rule_ok,
        "ranker_scores_nonconstant_when_ranking_is_possible": bool(
            variant == "random_8" or ranker_scores_valid
        ),
        "scientific_numeric_fields_finite": (
            dataframe_nonfinite_count(
                trajectory,
                [
                    "oracle_call",
                    "final_reward",
                    "best_reward_so_far",
                    "top10_mean_reward_so_far",
                    "n_unique_generic_scaffolds",
                    "dual_min_pic50",
                    "qed",
                    "sa_score",
                ],
            )
            == 0
            and dataframe_nonfinite_count(
                candidates,
                [
                    "raw_candidate_count",
                    "selected_candidate_count",
                    "selected_top_count",
                    "selected_random_count",
                ],
            )
            == 0
        ),
    }
    critical_errors = [name for name, passed in checks.items() if not passed]
    return {
        "passed": len(critical_errors) == 0,
        "critical_errors": critical_errors,
        "checks": checks,
        "candidate_event_count": len(candidates),
        "candidate_events_full_K": int(
            0 if candidates.empty else (candidates["raw_candidate_count"] == raw_k).sum()
        ),
        "candidate_events_more_than_B": int(
            0 if candidates.empty else (candidates["raw_candidate_count"] > child_b).sum()
        ),
        "candidate_events_selected_B": int(
            0 if candidates.empty else (candidates["selected_candidate_count"] == child_b).sum()
        ),
        "max_raw_candidate_count": int(
            0 if candidates.empty else candidates["raw_candidate_count"].max()
        ),
        "max_selected_candidate_count": int(
            0 if candidates.empty else candidates["selected_candidate_count"].max()
        ),
        "minimum_ranker_score_spread": score_spread_min,
        "n_node_records": len(nodes),
        "n_uct_selection_records": len(selections),
    }


def execute_one_task(payload: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_DASR is None:
        raise RuntimeError("Worker initializer did not run.")

    task = dict(payload["task"])
    output_dir = Path(payload["output_dir"])
    protocol_hash = str(payload["protocol_hash"])
    final_dir = task_output_dir(output_dir, task)
    task_id = str(task["task_id"])

    completed, reason = validate_task_completion(final_dir, task_id, verify_hashes=False)
    if completed:
        return {
            "task_id": task_id,
            "status": "skipped_complete",
            "elapsed_seconds": 0.0,
            "output_dir": str(final_dir),
        }
    if final_dir.exists():
        quarantine_incomplete_directory(final_dir)

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = final_dir.with_name(
        final_dir.name + f".__running__.{os.getpid()}.{time.time_ns()}"
    )
    temp_dir.mkdir(parents=True, exist_ok=False)
    start = time.time()

    try:
        config = _WORKER_DASR.DASRMCTSConfig(
            max_depth=int(_WORKER_SETTINGS["max_depth"]),
            max_children_per_node=int(_WORKER_SETTINGS["oracle_child_budget"]),
            mcts_iterations=int(_WORKER_SETTINGS["max_tree_policy_steps"]),
            exploration_constant=float(_WORKER_SETTINGS["exploration_constant"]),
            p_call=0.0,
            gamma=0.0,
            random_seed=int(task["search_random_seed"]),
            experiment_name="dasr_mcts_online_step27_formal",
            run_id=task_id,
            raw_candidate_pool_size=int(_WORKER_SETTINGS["raw_candidate_pool_size"]),
            oracle_budget=int(_WORKER_SETTINGS["oracle_budget"]),
            candidate_policy=str(task["variant"]),
            top_ranked_count=int(_WORKER_SETTINGS["top_ranked_count"]),
            random_exploration_count=int(_WORKER_SETTINGS["random_exploration_count"]),
            max_tree_policy_steps=int(_WORKER_SETTINGS["max_tree_policy_steps"]),
            max_stagnant_steps=int(_WORKER_SETTINGS["max_stagnant_steps"]),
        )
        engine = _WORKER_DASR.DASRMCTS(
            config=config,
            ranker_bank=_WORKER_RANKER_BANK,
            fitness_adapter=_WORKER_FITNESS,
            mutator=_WORKER_MUTATOR,
        )
        result = engine.search(str(task["root_smiles"]))
        elapsed = time.time() - start
        trajectory = pd.DataFrame(result.oracle_trajectory)
        candidates = pd.DataFrame(result.candidate_selection_records)
        nodes = pd.DataFrame(result.node_records)
        selections = pd.DataFrame(result.selection_records)
        if trajectory.empty:
            raise RuntimeError("Task produced no oracle trajectory.")

        final_point = trajectory.iloc[-1]
        summary = {
            "task_id": task_id,
            "run_id": task_id,
            "mode": str(payload["mode"]),
            "variant": str(task["variant"]),
            "root_index": int(task["root_index"]),
            "source_row_index": int(task["source_row_index"]),
            "root_smiles": str(task["root_smiles"]),
            "seed": int(task["seed"]),
            "search_random_seed": int(task["search_random_seed"]),
            "elapsed_seconds": elapsed,
            "oracle_budget_requested": int(_WORKER_SETTINGS["oracle_budget"]),
            "oracle_calls_used": int(result.summary["oracle_calls_used"]),
            "tree_policy_steps": int(result.summary["tree_policy_steps"]),
            "termination_reason": str(result.summary["termination_reason"]),
            "n_nodes": int(result.summary["n_nodes"]),
            "best_smiles": str(result.summary["best_smiles"]),
            "best_final_reward": float(result.summary["best_final_reward"]),
            "initial_final_reward": float(trajectory.iloc[0]["final_reward"]),
            "final_top10_mean_reward": float(final_point["top10_mean_reward_so_far"]),
            "final_unique_scaffolds": int(final_point["n_unique_generic_scaffolds"]),
            "n_candidate_selection_events": int(
                result.summary["n_candidate_selection_events"]
            ),
            "n_prior_gate_open_events": int(
                result.summary["n_prior_gate_open_events"]
            ),
            "status": "ok",
            "protocol_sha256": protocol_hash,
        }
        audit = build_task_audit(
            summary=summary,
            trajectory=trajectory,
            candidates=candidates,
            nodes=nodes,
            selections=selections,
            task=task,
        )
        if not audit["passed"]:
            raise RuntimeError(
                "Task integrity audit failed: " + "; ".join(audit["critical_errors"])
            )
        summary["task_audit"] = audit

        summary_path = temp_dir / "run_summary.json"
        trajectory_path = temp_dir / "oracle_trajectory.csv"
        candidate_path = temp_dir / "candidate_selection_audit.csv"
        node_path = temp_dir / "node_records.csv"
        selection_path = temp_dir / "uct_selection_records.csv"
        write_json_atomic(summary_path, summary)
        trajectory.to_csv(trajectory_path, index=False)
        candidates.to_csv(candidate_path, index=False)
        nodes.to_csv(node_path, index=False)
        selections.to_csv(selection_path, index=False)

        output_files = {
            path.name: sha256_file(path)
            for path in (
                summary_path,
                trajectory_path,
                candidate_path,
                node_path,
                selection_path,
            )
        }
        completion = {
            "completed": True,
            "completed_at_utc": utc_now_iso(),
            "task_id": task_id,
            "protocol_sha256": protocol_hash,
            "output_files": output_files,
            "summary": {
                "oracle_calls_used": summary["oracle_calls_used"],
                "best_final_reward": summary["best_final_reward"],
                "final_top10_mean_reward": summary["final_top10_mean_reward"],
                "elapsed_seconds": elapsed,
            },
        }
        write_json_atomic(temp_dir / "TASK_COMPLETED.json", completion)
        os.replace(temp_dir, final_dir)
        return {
            "task_id": task_id,
            "status": "completed",
            "elapsed_seconds": elapsed,
            "output_dir": str(final_dir),
            "scientific_blinding": "performance metrics are stored only inside the task output",
        }
    except Exception as exc:
        elapsed = time.time() - start
        failure = {
            "task_id": task_id,
            "task": task,
            "failed_at_utc": utc_now_iso(),
            "elapsed_seconds": elapsed,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "worker_pid": os.getpid(),
        }
        failure_dir = output_dir / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / f"{task_id}__{time.time_ns()}.json"
        write_json_atomic(failure_path, failure)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {
            "task_id": task_id,
            "status": "failed",
            "elapsed_seconds": elapsed,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "failure_path": str(failure_path),
        }
    finally:
        clear_worker_caches()


# ============================================================
# 5. Parallel scheduler, progress, and smoke/formal validation
# ============================================================


def task_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: json_safe(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def scan_task_status(
    output_dir: Path,
    tasks: pd.DataFrame,
    verify_hashes: bool,
) -> dict[str, Any]:
    completed_ids: list[str] = []
    incomplete_ids: list[str] = []
    reasons: dict[str, str] = {}
    for task in task_records(tasks):
        final_dir = task_output_dir(output_dir, task)
        complete, reason = validate_task_completion(
            final_dir,
            str(task["task_id"]),
            verify_hashes=verify_hashes,
        )
        if complete:
            completed_ids.append(str(task["task_id"]))
        else:
            incomplete_ids.append(str(task["task_id"]))
            if final_dir.exists():
                reasons[str(task["task_id"])] = reason
    return {
        "completed_ids": completed_ids,
        "incomplete_ids": incomplete_ids,
        "incomplete_existing_reasons": reasons,
        "n_completed": len(completed_ids),
        "n_pending": len(incomplete_ids),
        "n_total": len(tasks),
    }


def write_progress(
    output_dir: Path,
    *,
    stage: str,
    total: int,
    completed_before: int,
    newly_completed: int,
    skipped: int,
    failed: int,
    pending_after: int,
    start_time: float,
    recent_results: list[dict[str, Any]],
) -> None:
    elapsed = time.time() - start_time
    processed = newly_completed + failed
    rate = processed / elapsed if elapsed > 0 else 0.0
    eta = pending_after / rate if rate > 0 else None
    payload = {
        "stage": stage,
        "updated_at_utc": utc_now_iso(),
        "total_tasks": total,
        "completed_before_this_invocation": completed_before,
        "newly_completed": newly_completed,
        "skipped_complete": skipped,
        "failed_this_invocation": failed,
        "pending_after_current_results": pending_after,
        "elapsed_seconds_this_invocation": elapsed,
        "tasks_per_second": rate,
        "estimated_remaining_seconds": eta,
        "recent_results": recent_results[-20:],
    }
    write_json_atomic(output_dir / "formal_progress.json", payload)


def run_task_batch(
    *,
    stage: str,
    output_dir: Path,
    tasks: pd.DataFrame,
    protocol: dict[str, Any],
    n_jobs: int,
    device: str,
    max_tasks: int | None,
    verify_existing_hashes: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256_file(Path(protocol["_protocol_path"]))

    scan = scan_task_status(output_dir, tasks, verify_hashes=verify_existing_hashes)
    completed_set = set(scan["completed_ids"])
    pending_frame = tasks[~tasks["task_id"].isin(completed_set)].copy()
    task_by_id = {str(row["task_id"]): row for row in task_records(pending_frame)}
    ordered_ids = stable_scheduler_order(task_by_id.keys(), SCHEDULER_SEED)
    if max_tasks is not None:
        ordered_ids = ordered_ids[: max(0, int(max_tasks))]

    settings = {
        "raw_candidate_pool_size": int(protocol["formal_design"]["raw_candidate_pool_size"]),
        "oracle_child_budget": int(protocol["formal_design"]["oracle_child_budget"]),
        "top_ranked_count": int(protocol["formal_design"]["top_ranked_count"]),
        "random_exploration_count": int(protocol["formal_design"]["random_exploration_count"]),
        "oracle_budget": int(protocol["formal_design"]["oracle_budget_including_root"]),
        "max_depth": int(protocol["formal_design"]["max_depth"]),
        "max_tree_policy_steps": int(protocol["formal_design"]["max_tree_policy_steps"]),
        "max_stagnant_steps": int(protocol["formal_design"]["max_stagnant_steps"]),
        "exploration_constant": float(protocol["formal_design"]["exploration_constant"]),
    }
    freeze_dir = Path(protocol["inputs"]["freeze_dir"])
    step27_module_path = _THIS_DIR / "27_1_dasr_mcts.py"

    if not ordered_ids:
        return {
            "total": len(tasks),
            "completed_before": scan["n_completed"],
            "newly_completed": 0,
            "skipped": scan["n_completed"],
            "failed": 0,
            "pending_after": scan["n_pending"],
            "results": [],
        }

    print("=" * 100)
    print(f"Step 27_4 stage: {stage}")
    print(f"Output: {output_dir}")
    print(f"Total manifest tasks: {len(tasks):,}")
    print(f"Already complete: {scan['n_completed']:,}")
    print(f"Scheduled now: {len(ordered_ids):,}")
    print(f"Workers: {n_jobs}")
    print(f"Device per worker: {device}")
    print("Completed tasks will be skipped on every resume.")
    print("=" * 100)

    start = time.time()
    results: list[dict[str, Any]] = []
    newly_completed = 0
    failed = 0
    submitted = 0
    finished = 0
    max_in_flight = max(n_jobs * 2, n_jobs)

    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=worker_initializer,
        initargs=(str(step27_module_path), str(freeze_dir), device, settings),
    ) as executor:
        futures: dict[Any, str] = {}

        def submit_next() -> bool:
            nonlocal submitted
            if submitted >= len(ordered_ids):
                return False
            task_id = ordered_ids[submitted]
            task = task_by_id[task_id]
            payload = {
                "task": task,
                "output_dir": str(output_dir),
                "protocol_hash": protocol_hash,
                "mode": stage,
            }
            future = executor.submit(execute_one_task, payload)
            futures[future] = task_id
            submitted += 1
            return True

        while len(futures) < max_in_flight and submit_next():
            pass

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "task_id": task_id,
                        "status": "failed_parent_future",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                results.append(result)
                finished += 1
                if result["status"] == "completed":
                    newly_completed += 1
                elif result["status"] == "skipped_complete":
                    pass
                else:
                    failed += 1

                elapsed = time.time() - start
                status_text = result["status"]
                print(
                    f"[{finished:5d}/{len(ordered_ids):5d}] {task_id} | "
                    f"{status_text} | {float(result.get('elapsed_seconds', 0.0)):.1f}s | "
                    f"total {elapsed / 60.0:.1f} min"
                )

                pending_after = max(
                    0,
                    len(tasks) - scan["n_completed"] - newly_completed,
                )
                if finished % max(1, n_jobs) == 0 or finished == len(ordered_ids):
                    write_progress(
                        output_dir,
                        stage=stage,
                        total=len(tasks),
                        completed_before=scan["n_completed"],
                        newly_completed=newly_completed,
                        skipped=scan["n_completed"],
                        failed=failed,
                        pending_after=pending_after,
                        start_time=start,
                        recent_results=results,
                    )
                submit_next()

    final_scan = scan_task_status(output_dir, tasks, verify_hashes=False)
    invocation = {
        "stage": stage,
        "started_at_utc": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
        "finished_at_utc": utc_now_iso(),
        "elapsed_seconds": time.time() - start,
        "total_manifest_tasks": len(tasks),
        "completed_before": scan["n_completed"],
        "scheduled_this_invocation": len(ordered_ids),
        "newly_completed": newly_completed,
        "failed_this_invocation": failed,
        "completed_after": final_scan["n_completed"],
        "pending_after": final_scan["n_pending"],
        "n_jobs": n_jobs,
        "device": device,
        "results": results,
    }
    invocations_dir = output_dir / "invocations"
    invocations_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        invocations_dir / f"{stage}_{time.time_ns()}.json",
        invocation,
    )
    return {
        "total": len(tasks),
        "completed_before": scan["n_completed"],
        "newly_completed": newly_completed,
        "skipped": scan["n_completed"],
        "failed": failed,
        "pending_after": final_scan["n_pending"],
        "results": results,
        "elapsed_seconds": invocation["elapsed_seconds"],
    }


def build_runner_smoke_tasks(roots: pd.DataFrame) -> pd.DataFrame:
    smoke_roots = roots.iloc[:2].copy()
    rows: list[dict[str, Any]] = []
    task_index = 0
    for root in smoke_roots.itertuples(index=False):
        for seed in (42, 43):
            for variant in FROZEN_VARIANTS:
                rows.append(
                    {
                        "task_index": task_index,
                        "task_id": f"runner_smoke_{variant}_root{int(root.root_index):03d}_seed{seed}",
                        "root_index": int(root.root_index),
                        "source_row_index": int(root.source_row_index),
                        "root_smiles": str(root.root_smiles),
                        "seed": int(seed),
                        "variant": variant,
                        "search_random_seed": int(seed) + int(root.root_index) * 1009,
                    }
                )
                task_index += 1
    return pd.DataFrame(rows)


def run_runner_smoke(args: argparse.Namespace) -> None:
    protocol, roots, _ = verify_frozen_protocol(args.protocol_dir)
    output_dir = args.runner_smoke_dir.resolve()
    if args.reset_runner_smoke and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # A technical runner smoke uses a smaller oracle budget but otherwise the same engine.
    smoke_protocol = copy.deepcopy(protocol)
    smoke_protocol["formal_design"]["oracle_budget_including_root"] = 31
    smoke_protocol["formal_design"]["max_tree_policy_steps"] = 1000
    smoke_protocol["_protocol_path"] = str((args.protocol_dir / PROTOCOL_FILENAME).resolve())
    smoke_tasks = build_runner_smoke_tasks(roots)
    result = run_task_batch(
        stage="runner_smoke",
        output_dir=output_dir,
        tasks=smoke_tasks,
        protocol=smoke_protocol,
        n_jobs=args.n_jobs,
        device=args.device,
        max_tasks=None,
        verify_existing_hashes=True,
    )

    scan = scan_task_status(output_dir, smoke_tasks, verify_hashes=True)
    summaries: list[dict[str, Any]] = []
    initial_rows: list[dict[str, Any]] = []
    critical_errors: list[str] = []
    for task in task_records(smoke_tasks):
        final_dir = task_output_dir(output_dir, task)
        ok, reason = validate_task_completion(final_dir, str(task["task_id"]), verify_hashes=True)
        if not ok:
            critical_errors.append(f"{task['task_id']}:{reason}")
            continue
        summary = load_json(final_dir / "run_summary.json")
        summaries.append(summary)
        trajectory = pd.read_csv(final_dir / "oracle_trajectory.csv", nrows=1)
        initial_rows.append(
            {
                "root_index": int(task["root_index"]),
                "seed": int(task["seed"]),
                "variant": str(task["variant"]),
                "initial_reward": float(trajectory.iloc[0]["final_reward"]),
            }
        )

    summary_df = pd.DataFrame(summaries)
    initial_df = pd.DataFrame(initial_rows)
    if not summary_df.empty:
        if not (summary_df["oracle_calls_used"] == 31).all():
            critical_errors.append("not_all_tasks_used_exact_smoke_oracle_budget")
        if not (summary_df["n_prior_gate_open_events"] == 0).all():
            critical_errors.append("legacy_prior_was_activated")
    if not initial_df.empty:
        spread = initial_df.groupby(["root_index", "seed"])["initial_reward"].agg(
            lambda x: float(np.max(x) - np.min(x))
        )
        if float(spread.max()) > EPS:
            critical_errors.append("same_root_seed_initial_reward_differs_across_variants")

    validation = {
        "passed": len(critical_errors) == 0 and scan["n_pending"] == 0,
        "critical_errors": critical_errors,
        "completed_tasks": scan["n_completed"],
        "expected_tasks": len(smoke_tasks),
        "pending_tasks": scan["n_pending"],
        "n_jobs": args.n_jobs,
        "oracle_budget": 31,
        "resume_test_instruction": (
            "Run the identical runner_smoke command a second time. It should report zero newly "
            "completed tasks and skip all completed task directories."
        ),
        "formal_authorized_after_this_marker": len(critical_errors) == 0 and scan["n_pending"] == 0,
        "result": result,
    }
    write_json_atomic(output_dir / RUNNER_SMOKE_MARKER, validation)
    print("=" * 96)
    print("Step 27_4 formal-runner smoke finished")
    print(f"Passed: {validation['passed']}")
    print(f"Completed: {scan['n_completed']}/{len(smoke_tasks)}")
    print(f"Output: {output_dir}")
    print("Run the identical command once more to verify resume/skip behavior.")
    print("Then inspect formal_runner_smoke_validation.json before --stage formal.")
    print("=" * 96)
    if not validation["passed"]:
        raise RuntimeError("Formal runner smoke failed: " + "; ".join(critical_errors))


def require_runner_smoke_passed(runner_smoke_dir: Path) -> dict[str, Any]:
    path = runner_smoke_dir.resolve() / RUNNER_SMOKE_MARKER
    if not path.exists():
        raise FileNotFoundError(
            f"Formal runner smoke marker missing: {path}. Run --stage runner_smoke first."
        )
    report = load_json(path)
    if not bool(report.get("passed", False)):
        raise RuntimeError("Formal runner smoke did not pass.")
    return report


def run_formal(args: argparse.Namespace) -> None:
    protocol, _, tasks = verify_frozen_protocol(args.protocol_dir)
    require_runner_smoke_passed(args.runner_smoke_dir)
    output_dir = args.formal_output_dir.resolve()
    completed_marker = output_dir / FORMAL_COMPLETED_MARKER
    if completed_marker.exists():
        raise FileExistsError(
            f"Formal completion marker already exists. Do not rerun or overwrite: {completed_marker}"
        )

    protocol["_protocol_path"] = str((args.protocol_dir / PROTOCOL_FILENAME).resolve())
    result = run_task_batch(
        stage="formal",
        output_dir=output_dir,
        tasks=tasks,
        protocol=protocol,
        n_jobs=args.n_jobs,
        device=args.device,
        max_tasks=args.max_tasks,
        verify_existing_hashes=args.verify_existing_hashes,
    )

    scan = scan_task_status(output_dir, tasks, verify_hashes=False)
    print("=" * 100)
    print("Step 27_4 formal invocation finished")
    print(f"Newly completed: {result['newly_completed']}")
    print(f"Failed this invocation: {result['failed']}")
    print(f"Completed total: {scan['n_completed']}/{len(tasks)}")
    print(f"Pending: {scan['n_pending']}")
    print(f"Elapsed this invocation: {result.get('elapsed_seconds', 0.0) / 3600.0:.2f} h")
    print(f"Output: {output_dir}")

    if scan["n_pending"] == 0:
        # A full hash verification is intentionally performed before the final marker.
        verified_scan = scan_task_status(output_dir, tasks, verify_hashes=True)
        if verified_scan["n_pending"] != 0:
            raise RuntimeError(
                "All task markers existed, but full output hash verification found incomplete tasks."
            )
        completion = {
            "completed": True,
            "completed_at_utc": utc_now_iso(),
            "scientific_status": "STEP 27 FORMAL ONLINE RUNS COMPLETED; ANALYSIS NOT YET RUN",
            "protocol_sha256": sha256_file(args.protocol_dir / PROTOCOL_FILENAME),
            "n_tasks": len(tasks),
            "n_roots": FROZEN_ROOT_COUNT,
            "n_seeds": len(FROZEN_SEEDS),
            "n_variants": len(FROZEN_VARIANTS),
            "oracle_budget_per_task": FORMAL_ORACLE_BUDGET,
            "maximum_oracle_requests": len(tasks) * FORMAL_ORACLE_BUDGET,
            "important_instruction": (
                "Freeze this output. Do not change methods or rerun completed tasks. "
                "Next run Step 27_5 exactly as frozen in the protocol directory."
            ),
        }
        write_json_atomic(completed_marker, completion)
        print("All formal task hashes verified.")
        print(f"Completion marker: {completed_marker}")
        print("Next mandatory step: 27_5_analyze_dasr_mcts_formal.py")
    else:
        print("Formal experiment is incomplete. Rerun the identical --stage formal command to resume.")
        print("Completed task directories will be skipped; failed/incomplete tasks will be retried.")
    print("=" * 100)

    if result["failed"] > 0:
        raise RuntimeError(
            f"{result['failed']} tasks failed in this invocation. Inspect failures/ and rerun identically."
        )


def print_status(args: argparse.Namespace) -> None:
    protocol, _, tasks = verify_frozen_protocol(args.protocol_dir, verify_current_inputs=False)
    output_dir = args.formal_output_dir.resolve()
    scan = scan_task_status(output_dir, tasks, verify_hashes=args.verify_existing_hashes)
    failure_files = list((output_dir / "failures").glob("*.json")) if (output_dir / "failures").exists() else []
    print("=" * 96)
    print("Step 27 formal status")
    print(f"Protocol version: {protocol.get('protocol_version')}")
    print(f"Completed: {scan['n_completed']:,}/{scan['n_total']:,}")
    print(f"Pending: {scan['n_pending']:,}")
    print(f"Historical failure records: {len(failure_files):,}")
    print(f"Formal completion marker: {(output_dir / FORMAL_COMPLETED_MARKER).exists()}")
    if scan["incomplete_existing_reasons"]:
        print("Incomplete existing directories (first 10):")
        for task_id, reason in list(scan["incomplete_existing_reasons"].items())[:10]:
            print(f"  {task_id}: {reason}")
    print("=" * 96)


# ============================================================
# 6. CLI
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze, smoke-test, run, and resume Step 27 formal DASR-MCTS tasks."
    )
    parser.add_argument(
        "--stage",
        choices=["plan", "runner_smoke", "formal", "status"],
        required=True,
    )
    parser.add_argument("--p0_csv", type=Path, default=DEFAULT_P0_CSV)
    parser.add_argument("--p0_smiles_column", type=str, default="smiles")
    parser.add_argument("--freeze_dir", type=Path, default=DEFAULT_FREEZE_DIR)
    parser.add_argument(
        "--step26_frozen_dir",
        type=Path,
        default=DEFAULT_STEP26_FROZEN_DIR,
    )
    parser.add_argument(
        "--pilot_analysis_dir",
        type=Path,
        default=DEFAULT_PILOT_ANALYSIS_DIR,
    )
    parser.add_argument("--protocol_dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument(
        "--formal_output_dir",
        type=Path,
        default=DEFAULT_FORMAL_OUTPUT_DIR,
    )
    parser.add_argument(
        "--runner_smoke_dir",
        type=Path,
        default=DEFAULT_RUNNER_SMOKE_DIR,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=max(1, min(5, os.cpu_count() or 1)),
        help=(
            "Worker processes. Start with 5 because each worker loads frozen RF oracles and "
            "all neural rankers. Increase only after checking memory and CPU utilization."
        ),
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Optional execution chunk size. It does not alter the frozen task manifest.",
    )
    parser.add_argument(
        "--verify_existing_hashes",
        action="store_true",
        help="Hash every previously completed task before scheduling; slower but stricter.",
    )
    parser.add_argument(
        "--reset_runner_smoke",
        action="store_true",
        help="Delete only the technical runner-smoke output and rerun it. Never affects formal outputs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.n_jobs < 1:
        raise ValueError("n_jobs must be >=1.")
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("max_tasks must be >=1 when supplied.")

    if args.stage == "plan":
        create_frozen_protocol(args)
    elif args.stage == "runner_smoke":
        run_runner_smoke(args)
    elif args.stage == "formal":
        run_formal(args)
    elif args.stage == "status":
        print_status(args)
    else:  # pragma: no cover
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
