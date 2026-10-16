from __future__ import annotations

"""REINVENT4 formal seed45 audit v2.

Why v2 exists
-------------
The original audit v1 re-canonicalized REINVENT4 reporter SMILES under the
project Python/RDKit environment and required that canonical string to be an
exact key in the REINVENT4-environment oracle archive.

The formal run uses a different RDKit patch version in the isolated REINVENT4
environment.  At least one valid generated molecule has two different valid
canonical SMILES representations across those two environments, while its
frozen-project score is unchanged.  Therefore exact cross-environment
canonical-string identity is too strong for generated molecules.

v2 preserves the already-frozen scientific protocol.  It changes only the
post-run audit/reconstruction rule:
  * optimization/oracle accounting is audited from the original broker files;
  * the final native 200 slots are evaluated post hoc by the authoritative
    project .venv FrozenProjectScorer;
  * reporter raw component rewards are independently checked against those
    post-hoc scores;
  * invalid slots remain zero and duplicate slots remain duplicated, exactly
    as frozen before formal performance was observed;
  * no post-hoc score is counted as an optimization oracle request.

This script NEVER reruns REINVENT4 and NEVER modifies the frozen formal
protocol, prior, plugin, raw log, raw CSV, broker archive, or broker batches.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\pycharm\project\tsq_mcts_rebuild").resolve()
CODE_DIR = PROJECT_ROOT / "02_code" / "external_comparison"

TASK_DIR = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_reinvent4_formal" / "seed_45"
)
PROTOCOL_DIR = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_reinvent4_protocol"
)
PREFLIGHT_DIR = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_reinvent4_preflight"
)

PROTOCOL_JSON = PROTOCOL_DIR / "REINVENT4_FORMAL_PROTOCOL.json"
FREEZE_MARKER = PROTOCOL_DIR / "REINVENT4_FORMAL_PROTOCOL_FROZEN.json"
P0_SMI = PREFLIGHT_DIR / "reinvent4_formal_p0.smi"

RUN_LOG = TASK_DIR / "reinvent4_formal.log"
RESOLVED_CONFIG = TASK_DIR / "resolved_config.json"
ORACLE_STATE = TASK_DIR / "oracle_state.json"
ORACLE_BATCHES = TASK_DIR / "oracle_batches.jsonl"
ORACLE_ARCHIVE = TASK_DIR / "oracle_archive.jsonl"
RAW_SUMMARY = TASK_DIR / "reinvent4_formal_1.csv"
CHECKPOINT = TASK_DIR / "reinvent4_formal.chkpt"

COMMON_V4 = CODE_DIR / "30_72_external_benchmark_common_v4.py"
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# v2 analysis outputs
FINAL_WORKER_JSON = TASK_DIR / "final_population_project_env_worker_v2.json"
ARCHIVE_CSV_GZ = TASK_DIR / "archive_unique.csv.gz"
FINAL_POPULATION = TASK_DIR / "final_population.csv"
FINAL_METRICS = TASK_DIR / "final_metrics.json"
TRAJECTORY_CSV = TASK_DIR / "trajectory_metrics.csv"
POPULATION_SNAPSHOTS = TASK_DIR / "population_snapshots.json.gz"
AUDIT_REPORT = TASK_DIR / "formal_audit_report.json"
TASK_COMPLETED = TASK_DIR / "TASK_COMPLETED.json"

EXPECTED_PROTOCOL_SHA256 = "592929f79e673058fe3d308519df9378dea28c268e28778a3836e84928ff9379"
EXPECTED_P0_SHA256 = "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"
EXPECTED_P0_SMI_SHA256 = "b0fcc657ebd306ac326699b0599bab57f5a91825396ffa3d8c6ca3b1ebb71fcd"
EXPECTED_SEED45_TOML_SHA256 = "431c2d7156997c9f7d4810121c4cb9fc25502e1101dbf8f66fd8b742e5a1af34"
EXPECTED_REQUEST_CAP = 1_040_200
EXPECTED_STEPS = 100
EXPECTED_SLOTS_PER_STEP = 200
P0_PRIME_REQUESTS = 200

STATE_NAMES = {0: "INVALID", 1: "VALID", 2: "DUPLICATE"}

# Reporter prints the "(raw)" field to four decimal places.
REPORTER_RAW_TOL = 1.1e-4
# The non-masked component column is printed with seven decimals.
REPORTER_COMPONENT_TOL = 6e-7


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
    return rows


def canonical_payload_sha(payload: dict) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_gzip_bytes_deterministic(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(payload)
    tmp.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_metadata() -> dict:
    out = {
        "python_executable": sys.executable,
        "python_version": sys.version,
    }
    for module_name in ("numpy", "pandas", "rdkit", "sklearn", "joblib"):
        try:
            module = __import__(module_name)
            out[f"{module_name}_version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            out[f"{module_name}_error"] = repr(exc)
    return out


def zero_invalid_record(slot_index: int, raw_row: pd.Series) -> dict:
    return {
        "slot_index": int(slot_index),
        "step": int(raw_row["step"]),
        "raw_smiles": str(raw_row["SMILES"]),
        "smiles_state": 0,
        "smiles_state_name": "INVALID",
        "input_smiles": "" if pd.isna(raw_row["Input_SMILES"]) else str(raw_row["Input_SMILES"]),
        "canonical_smiles": "",
        "scoring_status": "invalid_final_slot",
        "final_reward": 0.0,
        "pred_gsk3b_pic50": 0.0,
        "pred_jnk3_pic50": 0.0,
        "dual_min_pic50": 0.0,
        "qed": 0.0,
        "sa_score": 10.0,
        "sa_reward": 0.0,
        "scaffold": "no_scaffold",
        "tier1": False,
        "reinvent_score_after_masks": float(raw_row["Score"]),
        "reinvent_component_score": float(raw_row["Frozen project final reward"]),
        "reinvent_component_raw_rounded": float(raw_row["Frozen project final reward (raw)"]),
    }


def project_worker(out_json: Path) -> None:
    """Score only the final native 200 slots in the authoritative project env."""

    require(Path(sys.executable).resolve() == PROJECT_PYTHON.resolve(),
            f"Worker must use project Python {PROJECT_PYTHON}; got {sys.executable}")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    common = load_module("_reinvent4_audit_v2_common", COMMON_V4)
    scorer = common.FrozenProjectScorer()

    raw = pd.read_csv(RAW_SUMMARY)
    final = raw.loc[raw["step"].astype(int) == EXPECTED_STEPS].copy()
    require(len(final) == 200, "Final native REINVENT4 step must contain exactly 200 slots")

    output_rows: list[dict] = []
    component_diffs: list[float] = []
    raw_rounded_diffs: list[float] = []

    for slot_index, (_, row) in enumerate(final.iterrows(), 1):
        state = int(row["SMILES_state"])
        require(state in STATE_NAMES, f"Unknown final SMILES_state={state}")

        if state == 0:
            require(abs(float(row["Score"])) <= 1e-12, "Invalid final slot has nonzero REINVENT Score")
            require(
                abs(float(row["Frozen project final reward"])) <= 1e-12,
                "Invalid final slot has nonzero component score",
            )
            output_rows.append(zero_invalid_record(slot_index, row))
            continue

        raw_smiles = str(row["SMILES"])
        score = scorer.score_full(raw_smiles)
        rec = score.as_dict()

        observed_raw = float(row["Frozen project final reward (raw)"])
        raw_diff = abs(observed_raw - round(float(rec["final_reward"]), 4))
        raw_rounded_diffs.append(raw_diff)
        require(
            raw_diff <= REPORTER_RAW_TOL,
            "Project-env frozen score disagrees with REINVENT reporter raw reward "
            f"at final slot {slot_index}: {raw_smiles}; "
            f"reporter_raw={observed_raw}, project={rec['final_reward']}",
        )

        if state == 1:
            observed_component = float(row["Frozen project final reward"])
            comp_diff = abs(observed_component - float(rec["final_reward"]))
            component_diffs.append(comp_diff)
            require(
                comp_diff <= REPORTER_COMPONENT_TOL,
                "Project-env frozen score disagrees with non-masked REINVENT component "
                f"at final slot {slot_index}: {raw_smiles}; "
                f"reporter={observed_component}, project={rec['final_reward']}",
            )
        else:
            # DUPLICATE is masked by REINVENT after raw component evaluation.
            require(
                abs(float(row["Frozen project final reward"])) <= 1e-12,
                "Duplicate final slot should have masked component score 0",
            )

        output_rows.append(
            {
                "slot_index": int(slot_index),
                "step": int(row["step"]),
                "raw_smiles": raw_smiles,
                "smiles_state": state,
                "smiles_state_name": STATE_NAMES[state],
                "input_smiles": "" if pd.isna(row["Input_SMILES"]) else str(row["Input_SMILES"]),
                "canonical_smiles": str(rec["canonical_smiles"]),
                "scoring_status": str(rec["scoring_status"]),
                "final_reward": float(rec["final_reward"]),
                "pred_gsk3b_pic50": float(rec["pred_gsk3b_pic50"]),
                "pred_jnk3_pic50": float(rec["pred_jnk3_pic50"]),
                "dual_min_pic50": float(rec["dual_min_pic50"]),
                "qed": float(rec["qed"]),
                "sa_score": float(rec["sa_score"]),
                "sa_reward": float(rec["sa_reward"]),
                "scaffold": str(rec["scaffold"]),
                "tier1": bool(rec["tier1"]),
                "reinvent_score_after_masks": float(row["Score"]),
                "reinvent_component_score": float(row["Frozen project final reward"]),
                "reinvent_component_raw_rounded": observed_raw,
            }
        )

    population = pd.DataFrame(output_rows)
    metrics = common.population_metrics(population, compute_hypervolume=True)

    # Diagnose, but do not reject, cross-environment canonical-string drift.
    # The original broker batch rows are authoritative evidence of what the
    # REINVENT environment submitted/recorded.
    batch_rows = []
    for batch in load_jsonl(ORACLE_BATCHES):
        batch_rows.extend(batch.get("rows", []))

    canonical_drift_examples = []
    n_canonical_compared = 0
    n_canonical_drift = 0
    seen_raw = set()

    for row in batch_rows:
        raw_smiles = str(row.get("input_smiles", ""))
        if not raw_smiles or raw_smiles in seen_raw:
            continue
        seen_raw.add(raw_smiles)

        project_canonical = common.canonicalize_smiles(raw_smiles)
        reinvent_canonical = str(row.get("canonical_smiles", ""))
        if project_canonical is None or not reinvent_canonical:
            continue

        n_canonical_compared += 1
        if str(project_canonical) != reinvent_canonical:
            n_canonical_drift += 1
            if len(canonical_drift_examples) < 20:
                canonical_drift_examples.append(
                    {
                        "input_smiles": raw_smiles,
                        "project_env_canonical": str(project_canonical),
                        "reinvent_env_recorded_canonical": reinvent_canonical,
                        "recorded_final_reward": float(row.get("final_reward", float("nan"))),
                    }
                )

    payload = {
        "completed": True,
        "worker_purpose": "authoritative project-env final 200-slot evaluation",
        "runtime": runtime_metadata(),
        "n_final_slots": 200,
        "state_counts": {
            "VALID": int((population["smiles_state"] == 1).sum()),
            "DUPLICATE": int((population["smiles_state"] == 2).sum()),
            "INVALID": int((population["smiles_state"] == 0).sum()),
        },
        "score_consistency": {
            "n_noninvalid_slots_checked_against_reporter_raw": int(
                (population["smiles_state"] != 0).sum()
            ),
            "max_abs_diff_reporter_raw_rounded": (
                max(raw_rounded_diffs) if raw_rounded_diffs else 0.0
            ),
            "n_valid_slots_checked_against_component": int(
                (population["smiles_state"] == 1).sum()
            ),
            "max_abs_diff_valid_component": (
                max(component_diffs) if component_diffs else 0.0
            ),
        },
        "canonicalization_diagnostic": {
            "n_unique_batch_input_smiles_compared": n_canonical_compared,
            "n_cross_environment_canonical_string_drift": n_canonical_drift,
            "examples_first20": canonical_drift_examples,
            "interpretation": (
                "A canonical-string difference is diagnostic only. It is not a score "
                "failure when the same raw generated SMILES is numerically consistent "
                "under the frozen project scorer."
            ),
        },
        "metrics": metrics,
        "rows": output_rows,
    }
    write_json_atomic(out_json, payload)

    print("PROJECT-ENV FINAL POPULATION WORKER PASS")
    print("final slots          :", 200)
    print("valid/dup/invalid    :",
          payload["state_counts"]["VALID"],
          payload["state_counts"]["DUPLICATE"],
          payload["state_counts"]["INVALID"])
    print("max score diff valid :", payload["score_consistency"]["max_abs_diff_valid_component"])
    print("canonical drifts     :", n_canonical_drift, "/", n_canonical_compared)
    print("output               :", out_json)


def run_project_worker() -> dict:
    require(PROJECT_PYTHON.exists(), f"Project Python not found: {PROJECT_PYTHON}")
    cmd = [
        str(PROJECT_PYTHON),
        str(Path(__file__).resolve()),
        "--project-worker",
        "--worker-out",
        str(FINAL_WORKER_JSON),
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        raise RuntimeError(
            f"Authoritative project-env final scorer worker failed with code {proc.returncode}"
        )
    return load_json(FINAL_WORKER_JSON)


def trajectory_from_reporter(raw: pd.DataFrame, batches: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Build request-aligned reward trajectory without any new oracle calls.

    VALID rows use the full-precision component column.
    DUPLICATE rows use the reporter raw pre-mask value (4-decimal reporter field).
    INVALID rows are zero by frozen policy.
    """

    rows = []
    snapshots = []

    for step in range(1, EXPECTED_STEPS + 1):
        frame = raw.loc[raw["step"].astype(int) == step].copy()
        require(len(frame) == 200, f"Step {step} does not contain exactly 200 native slots")

        states = frame["SMILES_state"].astype(int).to_numpy()
        require(set(states.tolist()).issubset({0, 1, 2}), f"Unexpected state at step {step}")

        rewards = np.zeros(200, dtype=float)
        valid = states == 1
        duplicate = states == 2
        rewards[valid] = frame.loc[valid, "Frozen project final reward"].to_numpy(float)
        rewards[duplicate] = frame.loc[
            duplicate, "Frozen project final reward (raw)"
        ].to_numpy(float)

        sorted_rewards = np.sort(rewards)[::-1]
        batch = batches[step - 1]

        rec = {
            "step": step,
            "requests": int(batch["request_end"]),
            "method_batch_requests": int(batch["requested_count"]),
            "mean_final_reward": float(np.mean(sorted_rewards)),
            "top10_mean_final_reward": float(np.mean(sorted_rewards[:10])),
            "valid_slots": int(np.sum(valid)),
            "duplicate_slots": int(np.sum(duplicate)),
            "invalid_slots": int(np.sum(states == 0)),
        }
        rows.append(rec)

        snapshots.append(
            {
                "update": step,
                "requests": int(batch["request_end"]),
                "method_batch_requests": int(batch["requested_count"]),
                "population_size_slots": 200,
                "valid_slots": rec["valid_slots"],
                "duplicate_slots": rec["duplicate_slots"],
                "invalid_slots": rec["invalid_slots"],
                "raw_smiles": frame["SMILES"].astype(str).tolist(),
                "smiles_state": states.astype(int).tolist(),
                "final_reward_for_resource_trajectory": rewards.tolist(),
                "metrics": {
                    "mean_final_reward": rec["mean_final_reward"],
                    "top10_mean_final_reward": rec["top10_mean_final_reward"],
                },
            }
        )

    return pd.DataFrame(rows), snapshots


def main() -> None:
    required = [
        PROTOCOL_JSON, FREEZE_MARKER, P0_SMI, RUN_LOG, RESOLVED_CONFIG,
        ORACLE_STATE, ORACLE_BATCHES, ORACLE_ARCHIVE, RAW_SUMMARY, COMMON_V4,
    ]
    for path in required:
        require(path.exists(), f"Missing required file: {path}")

    require(
        not TASK_COMPLETED.exists(),
        "TASK_COMPLETED.json already exists. Do not replace an already completed formal audit.",
    )

    # ------------------------------------------------------------------
    # 1) Frozen protocol and raw formal-run identity
    # ------------------------------------------------------------------
    protocol = load_json(PROTOCOL_JSON)
    marker = load_json(FREEZE_MARKER)

    require(protocol.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "Protocol SHA field mismatch")
    protocol_core = dict(protocol)
    protocol_core.pop("protocol_sha256", None)
    require(
        canonical_payload_sha(protocol_core) == EXPECTED_PROTOCOL_SHA256,
        "Formal protocol canonical SHA verification failed",
    )
    require(marker.get("frozen") is True, "Formal protocol freeze marker is not affirmative")
    require(marker.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "Freeze marker protocol SHA mismatch")
    require(protocol.get("performance_used_for_tuning") is False, "Protocol says performance was used for tuning")
    FORMAL_TOML = CODE_DIR / "reinvent4_formal_seed45_v1.toml"
    require(FORMAL_TOML.exists(), f"Missing formal seed45 TOML: {FORMAL_TOML}")
    require(sha256_file(FORMAL_TOML) == EXPECTED_SEED45_TOML_SHA256, "Materialized seed45 TOML SHA mismatch")
    require(protocol["formal_parameters"]["max_steps"] == 100, "Frozen max_steps mismatch")
    require(protocol["formal_parameters"]["min_steps"] == 100, "Frozen min_steps mismatch")
    require(protocol["formal_parameters"]["randomize_smiles"] is False, "Frozen randomize_smiles mismatch")
    require(
        int(protocol["resource_accounting"]["seed_specific_nonbinding_ceiling_from_step28"]["45"])
        == EXPECTED_REQUEST_CAP,
        "Frozen seed45 request ceiling mismatch",
    )

    require(sha256_file(P0_SMI) == EXPECTED_P0_SMI_SHA256, "P0 .smi bytes changed")
    p0 = [line.strip() for line in P0_SMI.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(p0) == 200 and len(set(p0)) == 200, "Frozen P0 .smi is not 200 unique rows")

    cfg = load_json(RESOLVED_CONFIG)
    require(cfg.get("run_type") == "staged_learning", "Resolved run_type mismatch")
    require(cfg.get("device") == "cuda:0", "Resolved device mismatch")
    params = cfg["parameters"]
    require(params["sample_strategy"] == "multinomial", "Resolved sample_strategy mismatch")
    require(int(params["distance_threshold"]) == 100, "Resolved distance_threshold mismatch")
    require(int(params["batch_size"]) == 1, "Resolved batch_size mismatch")
    require(params["randomize_smiles"] is False, "Resolved randomize_smiles mismatch")
    require(params["isomeric_smiles"] is False, "Resolved isomeric_smiles mismatch")
    require(float(params["temperature"]) == 1.0, "Resolved temperature mismatch")

    require(len(cfg["stage"]) == 1, "Expected exactly one RL stage")
    stage = cfg["stage"][0]
    require(int(stage["min_steps"]) == 100 and int(stage["max_steps"]) == 100, "Resolved stage horizon mismatch")
    require(float(stage["max_score"]) == 1.0, "Resolved max_score mismatch")
    require(stage["termination"] == "simple", "Resolved termination mismatch")

    ep = stage["scoring"]["component"][0]["FrozenProjectReward"]["endpoint"][0]
    require(float(ep["weight"]) == 1.0, "FrozenProjectReward weight mismatch")
    require(int(ep["params"]["technical_request_cap"]) == EXPECTED_REQUEST_CAP, "Resolved request cap mismatch")
    require(cfg["learning_strategy"] == {"type": "dap", "sigma": 128, "rate": 0.0001}, "Resolved DAP config mismatch")
    require(cfg["diversity_filter"]["type"] == "IdenticalMurckoScaffold", "Resolved diversity filter mismatch")
    require(int(cfg["diversity_filter"]["bucket_size"]) == 25, "Resolved diversity bucket mismatch")

    log_text = RUN_LOG.read_text(encoding="utf-8", errors="replace")
    require("Set seed for all random generators to 45" in log_text, "Run log does not confirm seed 45")
    require("Using generator Mol2Mol" in log_text, "Run log does not confirm Mol2Mol")
    require("Creating scoring component FrozenProjectReward" in log_text, "Run log does not confirm FrozenProjectReward")
    require("Maximum number of steps of 100 reached in stage 1" in log_text, "Run did not reach frozen 100-step horizon")
    require("Finished REINVENT on 2026-09-20" in log_text, "Run log has no normal REINVENT finish marker")
    require("Traceback (most recent call last)" not in log_text, "Traceback found in formal run log")
    steps_in_log = [int(x) for x in re.findall(r"Step:\s+(\d+)", log_text)]
    require(steps_in_log == list(range(1, 101)), "Run log step sequence is not exactly 1..100")

    # ------------------------------------------------------------------
    # 2) Original optimization-oracle accounting
    # ------------------------------------------------------------------
    state = load_json(ORACLE_STATE)
    require(state.get("adapter") == "FrozenProjectReward", "Oracle adapter mismatch")
    require(state.get("method") == "reinvent4_mol2mol", "Oracle method mismatch")
    require(state.get("p0_sha256") == EXPECTED_P0_SHA256, "Oracle state P0 SHA mismatch")
    require(int(state.get("p0_size", -1)) == 200 and state.get("p0_primed") is True, "Oracle P0 priming mismatch")

    counters = state["counters"]
    require(int(counters["request_budget"]) == EXPECTED_REQUEST_CAP, "Oracle request cap mismatch")
    require(int(counters["requests"]) < EXPECTED_REQUEST_CAP, "Technical request cap was reached")
    require(state.get("technical_cap_reached") is False, "technical_cap_reached must be false")
    require(int(counters["invalid_requests"]) == 0, "Invalid requests unexpectedly reached FrozenProjectReward")
    require(
        int(counters["model_evaluations"]) + int(counters["cache_hits"])
        == int(counters["requests"]),
        "Oracle accounting does not close",
    )

    batches = load_jsonl(ORACLE_BATCHES)
    require(len(batches) == 100, "Expected exactly 100 method scoring batches")

    previous_end = P0_PRIME_REQUESTS
    sum_requested = 0
    for expected_id, batch in enumerate(batches, 1):
        require(int(batch["batch_id"]) == expected_id, f"Batch ID mismatch at {expected_id}")
        require(batch.get("phase") == "method", f"Unexpected phase in batch {expected_id}")
        require(batch.get("completed") is True, f"Batch {expected_id} incomplete")
        require(batch.get("rejected_before_scoring") is False, f"Batch {expected_id} was rejected")
        nreq = int(batch["requested_count"])
        nret = int(batch["returned_count"])
        require(nreq == nret == len(batch.get("rows", [])), f"Batch {expected_id} returned/requested mismatch")
        require(1 <= nreq <= 200, f"Batch {expected_id} request count outside 1..200")
        require(int(batch["request_start"]) == previous_end + 1, f"Batch {expected_id} request_start gap")
        require(int(batch["request_end"]) == int(batch["request_start"]) + nreq - 1, f"Batch {expected_id} request_end mismatch")
        require(
            int(batch["remaining_requests_after"]) == EXPECTED_REQUEST_CAP - int(batch["request_end"]),
            f"Batch {expected_id} remaining-request accounting mismatch",
        )
        previous_end = int(batch["request_end"])
        sum_requested += nreq

    require(
        P0_PRIME_REQUESTS + sum_requested == int(counters["requests"]) == previous_end,
        "P0 + batch requests do not equal final request counter",
    )
    require(int(state["n_scoring_batches"]) == 100, "Oracle state n_scoring_batches mismatch")

    archive_rows = load_jsonl(ORACLE_ARCHIVE)
    require(len(archive_rows) == int(state["n_archive_rows"]), "Archive row count/state mismatch")
    require(len(archive_rows) == int(counters["unique_valid_evaluations"]), "Archive/unique_valid_evaluations mismatch")
    require(len(archive_rows) == int(counters["model_evaluations"]), "Archive/model_evaluations mismatch")

    archive = pd.DataFrame(archive_rows)
    require(
        archive["unique_index"].astype(int).tolist() == list(range(1, len(archive) + 1)),
        "Archive unique_index is not consecutive",
    )
    require(archive["canonical_smiles"].nunique() == len(archive), "Archive canonical_smiles are not unique")
    require(
        archive.iloc[:200]["canonical_smiles"].astype(str).tolist() == p0,
        "Archive first 200 molecules are not exact ordered frozen P0",
    )
    require(
        archive.iloc[:200]["request_index_first_seen"].astype(int).tolist() == list(range(1, 201)),
        "P0 request indices are not exactly 1..200",
    )
    require(
        set(archive.iloc[:200]["phase_first_seen"].astype(str)) == {"initial_p0"},
        "First 200 archive rows are not marked initial_p0",
    )

    # ------------------------------------------------------------------
    # 3) Native REINVENT reporter integrity
    # ------------------------------------------------------------------
    raw = pd.read_csv(RAW_SUMMARY)
    expected_cols = {
        "Agent", "Prior", "Target", "Score", "SMILES", "SMILES_state",
        "Input_SMILES", "Scaffold", "Frozen project final reward",
        "Frozen project final reward (raw)", "step",
    }
    require(expected_cols.issubset(raw.columns), "REINVENT summary CSV columns changed")
    require(len(raw) == 100 * 200, "Summary CSV is not 100 x 200 rows")
    require(sorted(raw["step"].astype(int).unique().tolist()) == list(range(1, 101)), "Summary CSV steps are not 1..100")
    require((raw.groupby("step").size() == 200).all(), "At least one RL step does not contain exactly 200 slots")
    require(set(raw["SMILES_state"].astype(int).unique()).issubset({0, 1, 2}), "Unexpected SMILES_state value")

    # ------------------------------------------------------------------
    # 4) Authoritative project-env final evaluation
    # ------------------------------------------------------------------
    worker = run_project_worker()
    require(worker.get("completed") is True, "Project-env worker did not complete")
    require(int(worker["n_final_slots"]) == 200, "Project-env worker final size mismatch")

    final_population = pd.DataFrame(worker["rows"])
    require(len(final_population) == 200, "Final population reconstruction is not 200 slots")
    final_metrics = worker["metrics"]

    # ------------------------------------------------------------------
    # 5) Reward/resource trajectory from the original REINVENT reporter
    # ------------------------------------------------------------------
    trajectory, snapshots = trajectory_from_reporter(raw, batches)

    # Sanity: final reporter reward metrics should closely agree with the
    # authoritative project-env final metrics.  Duplicate raw reporter value is
    # rounded to four decimals, so a tiny difference is expected.
    traj_final = trajectory.iloc[-1]
    require(
        abs(float(traj_final["mean_final_reward"]) - float(final_metrics["mean_final_reward"]))
        <= 2e-6,
        "Final reporter mean reward disagrees with authoritative project-env final metric",
    )
    require(
        abs(float(traj_final["top10_mean_final_reward"]) - float(final_metrics["top10_mean_final_reward"]))
        <= 2e-6,
        "Final reporter Top-10 reward disagrees with authoritative project-env final metric",
    )

    # ------------------------------------------------------------------
    # 6) Deterministic analysis artifacts
    # ------------------------------------------------------------------
    archive_csv_bytes = archive.to_csv(index=False, lineterminator="\n").encode("utf-8")
    write_gzip_bytes_deterministic(ARCHIVE_CSV_GZ, archive_csv_bytes)

    write_text_atomic(
        FINAL_POPULATION,
        final_population.to_csv(index=False, lineterminator="\n"),
    )
    write_text_atomic(
        TRAJECTORY_CSV,
        trajectory.to_csv(index=False, lineterminator="\n"),
    )

    snapshots_bytes = (
        json.dumps(snapshots, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    write_gzip_bytes_deterministic(POPULATION_SNAPSHOTS, snapshots_bytes)

    state_counts = worker["state_counts"]
    generator_all = {
        "total_native_slots": int(len(raw)),
        "valid_slots": int((raw["SMILES_state"].astype(int) == 1).sum()),
        "duplicate_slots": int((raw["SMILES_state"].astype(int) == 2).sum()),
        "invalid_slots": int((raw["SMILES_state"].astype(int) == 0).sum()),
    }
    generator_final = {
        "population_size_slots": 200,
        "valid_slots": int(state_counts["VALID"]),
        "duplicate_slots": int(state_counts["DUPLICATE"]),
        "invalid_slots": int(state_counts["INVALID"]),
        "unique_noninvalid_project_canonical": int(
            final_population.loc[
                final_population["canonical_smiles"].astype(str) != "",
                "canonical_smiles",
            ].nunique()
        ),
    }

    metrics_payload = {
        "method": "REINVENT4-Mol2Mol",
        "seed": 45,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "formal_steps": 100,
        "final_population_policy": "last native 200 Mol2Mol output slots",
        "final_evaluation_environment": "authoritative project .venv FrozenProjectScorer",
        "posthoc_evaluation_counts_toward_optimization_budget": False,
        "metrics": final_metrics,
        "generator_all_steps": generator_all,
        "generator_final_step": generator_final,
        "oracle_accounting": {
            **{key: int(value) for key, value in counters.items()},
            "method_scoring_requests_excluding_p0": int(sum_requested),
            "p0_prime_requests": 200,
            "n_scoring_batches": len(batches),
            "technical_cap_reached": False,
        },
        "cross_environment_final_score_check": worker["score_consistency"],
        "canonicalization_diagnostic": worker["canonicalization_diagnostic"],
    }
    write_json_atomic(FINAL_METRICS, metrics_payload)

    audit_payload = {
        "passed": True,
        "audit_version": "reinvent4_formal_seed45_audit_v2",
        "method": "REINVENT4-Mol2Mol",
        "seed": 45,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "scientific_protocol_changed_by_v2": False,
        "v1_failure_classification": "post-run audit canonical-string identity assumption",
        "v2_fix": (
            "Do not require generated-molecule canonical strings produced by two "
            "different RDKit patch environments to be text-identical. The final native "
            "200 slots are instead evaluated in the authoritative project .venv with "
            "the unchanged frozen scorer, and their rewards are checked against the "
            "original REINVENT reporter."
        ),
        "checks": {
            "formal_protocol_frozen_and_hash_verified": True,
            "resolved_config_matches_frozen_protocol": True,
            "log_seed45_confirmed": True,
            "log_exact_steps_1_to_100": True,
            "normal_reinvent_finish": True,
            "p0_exact_ordered_first_200_archive_rows": True,
            "oracle_accounting_closed": True,
            "100_scoring_batches_complete_and_sequential": True,
            "archive_unique_and_consecutive": True,
            "raw_summary_exact_100x200_slots": True,
            "final_200_slots_rescored_in_authoritative_project_env": True,
            "final_scores_match_original_reinvent_reporter": True,
            "technical_request_cap_nonbinding": True,
        },
        "raw_input_sha256": {
            "reinvent4_formal.log": sha256_file(RUN_LOG),
            "resolved_config.json": sha256_file(RESOLVED_CONFIG),
            "oracle_state.json": sha256_file(ORACLE_STATE),
            "oracle_batches.jsonl": sha256_file(ORACLE_BATCHES),
            "oracle_archive.jsonl": sha256_file(ORACLE_ARCHIVE),
            "reinvent4_formal_1.csv": sha256_file(RAW_SUMMARY),
            "checkpoint": sha256_file(CHECKPOINT) if CHECKPOINT.exists() else None,
        },
        "project_env_worker_sha256": sha256_file(FINAL_WORKER_JSON),
        "score_consistency": worker["score_consistency"],
        "canonicalization_diagnostic": worker["canonicalization_diagnostic"],
        "final_metrics": final_metrics,
        "generator_final_step": generator_final,
        "oracle_accounting": metrics_payload["oracle_accounting"],
        "notes": [
            "No REINVENT4 optimization step was rerun.",
            "No frozen scientific parameter was changed.",
            "Post-hoc final rescoring is analysis-only and is not added to optimization request accounting.",
            "Duplicate final slots are retained; invalid final slots receive zero objectives and no scaffold, exactly as pre-performance frozen.",
            "The v1 audit failure was caused by cross-environment canonical-string drift, not by a missing optimization score or a failed formal run.",
        ],
    }
    write_json_atomic(AUDIT_REPORT, audit_payload)

    task_marker = {
        "completed": True,
        "audited": True,
        "audit_version": "v2",
        "method": "reinvent4_mol2mol",
        "seed": 45,
        "protocol_version": "reinvent4_mol2mol_formal_v1",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "performance_used_for_tuning": False,
        "scientific_protocol_changed_by_audit_fix": False,
        "formal_steps": 100,
        "actual_requests": int(counters["requests"]),
        "unique_evaluations": int(counters["unique_valid_evaluations"]),
        "cache_hits": int(counters["cache_hits"]),
        "technical_cap_reached": False,
        "archive_sha256": sha256_file(ARCHIVE_CSV_GZ),
        "final_population_sha256": sha256_file(FINAL_POPULATION),
        "population_snapshots_sha256": sha256_file(POPULATION_SNAPSHOTS),
        "trajectory_metrics_sha256": sha256_file(TRAJECTORY_CSV),
        "final_metrics_sha256": sha256_file(FINAL_METRICS),
        "audit_report_sha256": sha256_file(AUDIT_REPORT),
        "project_env_worker_sha256": sha256_file(FINAL_WORKER_JSON),
        "raw_log_sha256": sha256_file(RUN_LOG),
        "raw_resolved_config_sha256": sha256_file(RESOLVED_CONFIG),
        "raw_oracle_state_sha256": sha256_file(ORACLE_STATE),
        "raw_oracle_batches_sha256": sha256_file(ORACLE_BATCHES),
        "raw_oracle_archive_sha256": sha256_file(ORACLE_ARCHIVE),
        "raw_summary_csv_sha256": sha256_file(RAW_SUMMARY),
        "checkpoint_sha256": sha256_file(CHECKPOINT) if CHECKPOINT.exists() else None,
    }
    write_json_atomic(TASK_COMPLETED, task_marker)

    print("=" * 94)
    print("REINVENT4 FORMAL seed45 AUDIT v2 PASS")
    print("=" * 94)
    print("protocol SHA256        :", EXPECTED_PROTOCOL_SHA256)
    print("steps                   :", 100)
    print("native slots            :", len(raw))
    print("requests                :", counters["requests"])
    print("model evaluations       :", counters["model_evaluations"])
    print("cache hits              :", counters["cache_hits"])
    print("method requests         :", sum_requested)
    print("technical cap reached   :", False)
    print("final valid/dup/invalid :",
          state_counts["VALID"], state_counts["DUPLICATE"], state_counts["INVALID"])
    print("canonical drift count   :",
          worker["canonicalization_diagnostic"]["n_cross_environment_canonical_string_drift"],
          "/",
          worker["canonicalization_diagnostic"]["n_unique_batch_input_smiles_compared"])
    print("max final score diff    :",
          worker["score_consistency"]["max_abs_diff_valid_component"])
    print("final mean reward       :", f"{float(final_metrics['mean_final_reward']):.12f}")
    print("final Top-10 reward     :", f"{float(final_metrics['top10_mean_final_reward']):.12f}")
    print("final Tier-1            :", f"{int(final_metrics['tier1_count'])}/200")
    print("final unique scaffolds  :", int(final_metrics["unique_scaffolds"]))
    print("final max occupancy     :", int(final_metrics["max_scaffold_occupancy"]))
    print("final 4D HV             :", f"{float(final_metrics['hypervolume_4d']):.12f}")
    print("TASK_COMPLETED          :", TASK_COMPLETED)
    print("formal_seed45_audit_complete = True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-worker", action="store_true")
    parser.add_argument("--worker-out", type=Path)
    args = parser.parse_args()

    if args.project_worker:
        require(args.worker_out is not None, "--worker-out is required")
        project_worker(args.worker_out.resolve())
    else:
        main()
