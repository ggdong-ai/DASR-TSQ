"""
Create Step 28 amended-v4.1 protocol.

This amendment does not change the scientific P0 or any search setting. It
changes only the live P0 authority used by the fail-closed verifier:
- the byte-identical external source file P0_fixed_cold_start.csv is no longer
  present in the active project;
- the already frozen formal_p0.csv inside the v3/v4 protocol is retained as the
  authoritative formal P0;
- existing task population_initial.csv files are checked by a normalized
  SMILES-content hash against formal_p0.csv;
- the historical external P0 path and SHA256 remain recorded, but are not used
  as a live prerequisite.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "02_code" / "analysis"

DEFAULT_SOURCE_PROTOCOL_DIR = (
    PROJECT_ROOT / "06_paper" / "experiment_records" /
    "dasr_step28_formal_protocol_amended_v4"
)
DEFAULT_V3_OUTPUT_DIR = (
    PROJECT_ROOT / "05_results" / "dasr_tsq_end_to_end" / "step28_formal"
)
DEFAULT_TARGET_PROTOCOL_DIR = (
    PROJECT_ROOT / "06_paper" / "experiment_records" /
    "dasr_step28_formal_protocol_amended_v4_1"
)

CODE_FILES = (
    _THIS_DIR / "28_0_dasr_tsq_evolution_v2.py",
    _THIS_DIR / "28_1_run_dasr_tsq_smoke_pilot_v2.py",
    _THIS_DIR / "28_2_compare_parallel_determinism.py",
    _THIS_DIR / "28_3c_amend_protocol_v4_1_p0_authority.py",
    _THIS_DIR / "28_4_run_end_to_end_formal_v4_1.py",
    _THIS_DIR / "28_4b_migrate_v3_results_to_v4_1.py",
    ANALYSIS_DIR / "28_5_analyze_end_to_end_formal_v4_1.py",
)
UNCHANGED_ENGINE_FILES = (
    "28_0_dasr_tsq_evolution_v2.py",
    "28_1_run_dasr_tsq_smoke_pilot_v2.py",
    "28_2_compare_parallel_determinism.py",
)


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


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def smiles_content_hash(frame: pd.DataFrame) -> str:
    if "smiles" not in frame.columns:
        raise ValueError("Missing smiles column.")
    normalized = frame[["smiles"]].copy().sort_values(["smiles"]).reset_index(drop=True)
    payload = normalized.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def population_evidence(v3_output: Path, expected_hash: str) -> list[dict[str, Any]]:
    paths: list[Path] = []
    paths.extend(sorted((v3_output / "tasks").glob("*/population_initial.csv")))
    paths.extend(sorted((v3_output / ".staging").glob("*.__building__.*/population_initial.csv")))
    if not paths:
        raise FileNotFoundError("No existing Step 28 population_initial.csv evidence was found.")
    evidence: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_csv(path)
        observed = smiles_content_hash(frame)
        if len(frame) != 200 or frame["smiles"].nunique() != 200:
            raise RuntimeError(f"Invalid initial population: {path}")
        if observed != expected_hash:
            raise RuntimeError(f"Existing task P0 differs from frozen formal_p0.csv: {path}")
        evidence.append({
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "smiles_content_hash": observed,
            "n_rows": int(len(frame)),
            "n_unique_smiles": int(frame["smiles"].nunique()),
        })
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Create amended-v4.1 with frozen formal_p0.csv as P0 authority.")
    parser.add_argument("--source_protocol_dir", type=Path, default=DEFAULT_SOURCE_PROTOCOL_DIR)
    parser.add_argument("--v3_output_dir", type=Path, default=DEFAULT_V3_OUTPUT_DIR)
    parser.add_argument("--target_protocol_dir", type=Path, default=DEFAULT_TARGET_PROTOCOL_DIR)
    args = parser.parse_args()

    source_dir = args.source_protocol_dir.resolve()
    v3_output = args.v3_output_dir.resolve()
    target_dir = args.target_protocol_dir.resolve()
    if target_dir.exists():
        raise FileExistsError(f"Target protocol already exists; do not overwrite: {target_dir}")

    required = ["formal_protocol.json", "FORMAL_PROTOCOL_FROZEN.json", "formal_tasks.csv", "formal_p0.csv"]
    for name in required:
        if not (source_dir / name).exists():
            raise FileNotFoundError(source_dir / name)
    for path in CODE_FILES:
        if not path.exists():
            raise FileNotFoundError(path)

    source_marker = load_json(source_dir / "FORMAL_PROTOCOL_FROZEN.json")
    source_protocol_path = source_dir / "formal_protocol.json"
    source_sha = str(source_marker["protocol_sha256"])
    if sha256_file(source_protocol_path) != source_sha:
        raise RuntimeError("Source amended-v4 protocol hash mismatch.")
    source_protocol = load_json(source_protocol_path)
    if not str(source_protocol.get("protocol_version", "")).startswith("step28_end_to_end_formal_v4"):
        raise RuntimeError("Source protocol is not amended-v4.")

    tasks = pd.read_csv(source_dir / "formal_tasks.csv")
    p0 = pd.read_csv(source_dir / "formal_p0.csv")
    if len(tasks) != 30 or tasks["task_id"].nunique() != 30:
        raise RuntimeError("Task manifest must contain 30 unique tasks.")
    if len(p0) != 200 or p0["smiles"].nunique() != 200:
        raise RuntimeError("Frozen formal_p0.csv must contain 200 unique molecules.")
    p0_content = smiles_content_hash(p0)
    task_p0_evidence = population_evidence(v3_output, p0_content)

    current_hashes = {path.name: sha256_file(path) for path in CODE_FILES}
    source_hashes = source_protocol["inputs"]["code_hashes"]
    for name in UNCHANGED_ENGINE_FILES:
        if current_hashes[name] != source_hashes[name]:
            raise RuntimeError(f"Scientific engine changed unexpectedly: {name}")

    old_external = copy.deepcopy(source_protocol.get("inputs", {}).get("p0_csv", {}))
    old_upstream = copy.deepcopy(source_protocol.get("inputs", {}).get("upstream_hash_checks", {}))
    old_upstream_p0 = old_upstream.pop("p0_csv", None)
    historical = old_upstream_p0 or old_external
    historical_path = Path(str(historical.get("path", ""))) if historical.get("path") else None
    external_exists = bool(historical_path and historical_path.exists())
    external_current_hash = sha256_file(historical_path) if external_exists else None

    building = target_dir.with_name(target_dir.name + f".__building__.{os.getpid()}")
    if building.exists():
        shutil.rmtree(building)
    (building / "02_code").mkdir(parents=True)
    (building / "source_v4_protocol").mkdir(parents=True)
    shutil.copy2(source_dir / "formal_p0.csv", building / "formal_p0.csv")
    shutil.copy2(source_dir / "formal_tasks.csv", building / "formal_tasks.csv")
    for path in CODE_FILES:
        shutil.copy2(path, building / "02_code" / path.name)
    for name in ("formal_protocol.json", "FORMAL_PROTOCOL_FROZEN.json", "AMENDMENT_RECORD.json"):
        if (source_dir / name).exists():
            shutil.copy2(source_dir / name, building / "source_v4_protocol" / name)

    protocol = copy.deepcopy(source_protocol)
    protocol["protocol_version"] = "step28_end_to_end_formal_v4_1_p0_authority_2026-08-01"
    protocol["created_utc"] = utc_now()
    protocol["scientific_status"] = (
        "AMENDED P0 AUTHORITY AFTER LOSS OF THE EXTERNAL BYTE-IDENTICAL SOURCE FILE; "
        "FORMAL MOLECULE SET AND SCIENTIFIC DESIGN UNCHANGED"
    )
    protocol["inputs"].pop("p0_csv", None)
    protocol["inputs"]["upstream_hash_checks"] = old_upstream
    protocol["inputs"]["historical_external_p0_input"] = {
        **historical,
        "status": "historical provenance only; not a live prerequisite",
        "exists_at_amendment": external_exists,
        "current_sha256_if_present": external_current_hash,
    }
    protocol["inputs"]["frozen_formal_p0"] = {
        "relative_path": "formal_p0.csv",
        "size_bytes": (building / "formal_p0.csv").stat().st_size,
        "sha256": sha256_file(building / "formal_p0.csv"),
        "smiles_content_hash": p0_content,
        "n_rows": 200,
        "n_unique_smiles": 200,
        "authority": "sole live P0 input for all remaining Step 28 tasks",
    }
    protocol["inputs"]["code_hashes"] = current_hashes
    protocol["inputs"]["source_v4_protocol_sha256"] = source_sha
    protocol["inputs"]["source_v4_code_hashes"] = source_hashes

    p0_amendment = {
        "amendment_type": "P0 provenance authority correction",
        "amended_utc": utc_now(),
        "source_v4_protocol_sha256": source_sha,
        "historical_external_p0": historical,
        "external_file_exists_at_amendment": external_exists,
        "frozen_formal_p0_sha256": sha256_file(building / "formal_p0.csv"),
        "frozen_formal_p0_smiles_content_hash": p0_content,
        "existing_task_initial_population_evidence": task_p0_evidence,
        "reason": (
            "The historical source file P0_fixed_cold_start.csv is not present at its recorded path "
            "and no byte-identical copy is available in the active project. The formal protocol already "
            "contains formal_p0.csv, and every available formal task initial population has the same "
            "normalized set of 200 SMILES. formal_p0.csv is therefore promoted to the sole live P0 authority."
        ),
        "unchanged_scientific_elements": [
            "the exact 200 formal P0 molecule identities",
            "all 30 task identities, variants and seeds",
            "all frozen models and scientific engine files",
            "K=32, B=8, 6+2, TSQ10, population, generations and endpoints",
        ],
        "changed_elements": [
            "the external byte-level P0 source is retained only as historical provenance",
            "the frozen formal_p0.csv becomes the fail-closed live P0 authority",
        ],
        "performance_fields_read": False,
        "performance_based_decision": False,
    }
    protocol["p0_authority_amendment"] = p0_amendment
    protocol["prohibitions"] = list(protocol.get("prohibitions", [])) + [
        "Do not reconstruct or fabricate a file merely to reproduce the historical external P0 SHA256.",
        "Do not alter formal_p0.csv or any existing population_initial.csv.",
    ]

    write_json(building / "formal_protocol.json", protocol)
    protocol_sha = sha256_file(building / "formal_protocol.json")
    write_json(building / "P0_AUTHORITY_AMENDMENT_RECORD.json", p0_amendment)

    manifest: dict[str, Any] = {}
    for path in sorted(building.rglob("*")):
        if path.is_file():
            manifest[str(path.relative_to(building)).replace("\\", "/")] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    write_json(building / "FORMAL_PROTOCOL_FROZEN.json", {
        "frozen": True,
        "amended": True,
        "p0_authority_amended": True,
        "created_utc": utc_now(),
        "protocol_sha256": protocol_sha,
        "source_v4_protocol_sha256": source_sha,
        "n_tasks": 30,
        "files": manifest,
        "next_action": (
            "Run 28_4b_migrate_v3_results_to_v4_1.py --include_completed_staging, "
            "then run 28_4_run_end_to_end_formal_v4_1.py --stage status --verify_existing_hashes."
        ),
    })
    os.replace(building, target_dir)
    print("=" * 100)
    print("Step 28 amended-v4.1 P0-authority protocol created successfully")
    print(f"Source v4 protocol SHA256: {source_sha}")
    print(f"New v4.1 protocol SHA256: {protocol_sha}")
    print(f"Frozen P0 SMILES content hash: {p0_content}")
    print(f"Existing initial populations verified: {len(task_p0_evidence)}")
    print(f"Output: {target_dir}")
    print("No molecule, method, parameter, endpoint, or performance-based choice was changed.")
    print("=" * 100)


if __name__ == "__main__":
    main()
