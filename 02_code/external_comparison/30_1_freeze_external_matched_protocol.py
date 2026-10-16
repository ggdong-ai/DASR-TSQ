from __future__ import annotations

"""Freeze the final Step-28-matched external benchmark protocol (v3.1).

This is a zero-optimization preflight. It only verifies frozen scientific inputs,
reproduces P0 scoring, extracts per-seed raw request caps from completed Step 28,
and freezes the exact active external-benchmark code hashes.
"""

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

import importlib.util
import sys

THIS_DIR = Path(__file__).resolve().parent
COMMON_PATH = THIS_DIR / "30_0_external_matched_common.py"


def load_common():
    name = "external_matched_common_v31_freeze"
    spec = importlib.util.spec_from_file_location(name, COMMON_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(COMMON_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = load_common()

ACTIVE_PYTHON = [
    "30_0_external_matched_common.py",
    "30_1_freeze_external_matched_protocol.py",
    "30_2_run_graphga_matched.py",
    "30_3_mars_oracle_bridge.py",
    "30_4_run_mars_matched.py",
    "30_5_mnopt_source_snapshot.py",
    "30_6_medchem_source_snapshot.py",
    "30_7_mnopt_reimpl_core.py",
    "30_8_mnopt_oracle_server.py",
    "30_9_run_mnopt_matched.py",
    "30_10_analyze_external_matched.py",
]

ACTIVE_LAUNCHERS = [
    "30_11_FREEZE_PROTOCOL.bat",
    "30_12_GRAPHGA_5SEEDS.bat",
    "30_13_MARS_5SEEDS.bat",
    "30_14_MNOPT_5SEEDS.bat",
    "30_15_ANALYZE.bat",
    "30_16_RUN_ALL.bat",
]
ACTIVE_FILES = ACTIVE_PYTHON + ACTIVE_LAUNCHERS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".__tmp__")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def verify_graphga() -> dict:
    repo = (C.PROJECT_ROOT / "08_external" / "mol-opt").resolve()
    run_py = repo / "molopt" / "graph_ga" / "run.py"
    base_py = repo / "molopt" / "base.py"
    hparams = repo / "molopt" / "graph_ga" / "hparams_default.yaml"
    for path in (run_py, base_py, hparams):
        if not path.exists():
            raise FileNotFoundError(path)
    C.remove_python_cache_artifacts(repo)
    commit = C.git_commit(repo)
    if commit != C.EXPECTED_GRAPHGA_COMMIT:
        raise RuntimeError(f"GraphGA commit mismatch: {commit}")
    if C.git_is_dirty(repo):
        raise RuntimeError("GraphGA repository must be git-clean.")
    expected = {
        run_py: C.EXPECTED_GRAPHGA_RUN_SHA256,
        base_py: C.EXPECTED_GRAPHGA_BASE_SHA256,
        hparams: C.EXPECTED_GRAPHGA_HPARAMS_SHA256,
    }
    for path, sha in expected.items():
        observed = C.sha256_file(path)
        if observed != sha:
            raise RuntimeError(f"GraphGA frozen file changed: {path}\nexpected={sha}\nobserved={observed}")
    defaults = yaml.safe_load(hparams.read_text(encoding="utf-8")) or {}
    for key, value in {"population_size": 120, "offspring_size": 70, "mutation_rate": 0.067}.items():
        if key not in defaults or float(defaults[key]) != float(value):
            raise RuntimeError(f"Unexpected GraphGA default {key}: {defaults.get(key)}")
    return {
        "repo": str(repo),
        "commit": commit,
        "repo_clean": True,
        "run_py": str(run_py),
        "run_py_sha256": C.sha256_file(run_py),
        "base_py": str(base_py),
        "base_py_sha256": C.sha256_file(base_py),
        "hparams_default": str(hparams),
        "hparams_default_sha256": C.sha256_file(hparams),
        "official_defaults": {
            "population_size": int(defaults["population_size"]),
            "offspring_size": int(defaults["offspring_size"]),
            "mutation_rate": float(defaults["mutation_rate"]),
        },
        "benchmark_population_size": 200,
        "benchmark_population_size_reason": "accept exact common frozen P0=200 only",
        "patience": 5,
        "search_mechanics": "official GraphGA._optimize unchanged",
    }


def verify_mars() -> dict:
    repo = (C.PROJECT_ROOT / "08_external" / "MARS").resolve()
    sampler = repo / "sampler.py"
    main_py = repo / "main.py"
    if not sampler.exists() or not main_py.exists():
        raise FileNotFoundError(f"MARS source missing under {repo}")
    C.remove_python_cache_artifacts(repo)
    commit = C.git_commit(repo)
    if commit != C.EXPECTED_MARS_COMMIT:
        raise RuntimeError(f"MARS commit mismatch: {commit}")
    if C.git_is_dirty(repo):
        raise RuntimeError("MARS repository must be git-clean.")
    if C.sha256_file(sampler) != C.EXPECTED_MARS_SAMPLER_SHA256:
        raise RuntimeError("MARS sampler.py hash mismatch against the audited frozen source.")

    method_freeze = C.LEGACY_V2_PROTOCOL_DIR / "method_freezes" / "mars" / "MARS_METHOD_FROZEN.json"
    if not method_freeze.exists():
        raise FileNotFoundError(method_freeze)
    frozen = C.load_json(method_freeze)
    if frozen.get("frozen") is not True:
        raise RuntimeError("MARS method freeze is not affirmative.")
    source = frozen.get("mars_source", {})
    if source.get("commit") != commit:
        raise RuntimeError("MARS method-freeze commit differs from current frozen repo.")
    if source.get("sampler_py_sha256") != C.sha256_file(sampler):
        raise RuntimeError("MARS method-freeze sampler hash mismatch.")
    if source.get("main_py_sha256") != C.sha256_file(main_py):
        raise RuntimeError("MARS method-freeze main.py hash mismatch.")

    cfg = dict(frozen.get("mars_config", {}))
    expected_cfg = {
        "train": True,
        "vocab": "chembl",
        "vocab_size": 1000,
        "max_size": 40,
        "num_mols": 200,
        "num_step": 1000,
        "log_every": 1,
        "sampler": "sa",
        "proposal": "editor",
        "objectives": ["gsk3b", "jnk3", "qed", "sa"],
        "score_wght": [1.0, 1.0, 1.0, 1.0],
        "score_succ": [0.5, 0.5, 0.6, 0.67],
        "score_clip": [0.6, 0.6, 0.7, 0.7],
    }
    for key, expected in expected_cfg.items():
        if cfg.get(key) != expected:
            raise RuntimeError(f"Frozen MARS config changed at {key}: {cfg.get(key)!r} != {expected!r}")
    return {
        "repo": str(repo),
        "commit": commit,
        "repo_clean": True,
        "main_py_sha256": C.sha256_file(main_py),
        "sampler_py_sha256": C.sha256_file(sampler),
        "method_freeze": str(method_freeze.resolve()),
        "method_freeze_sha256": C.sha256_file(method_freeze),
        "mars_config": cfg,
        "native_population_capture_boundary": "Sampler.record(step, old_mols, old_dicts, acc_rates) after acceptance update",
        "native_property_scorers_used": False,
    }


def verify_mnopt() -> dict:
    method_dir = C.LEGACY_V2_PROTOCOL_DIR / "method_freezes" / "mnopt"
    method_freeze = method_dir / "MNOPT_METHOD_FROZEN.json"
    spec_path = method_dir / "MNOPT_REIMPLEMENTATION_SPEC.json"
    for path in (method_freeze, spec_path):
        if not path.exists():
            raise FileNotFoundError(path)
    frozen = C.load_json(method_freeze)
    if frozen.get("mnopt_method_frozen") is not True or frozen.get("formal_10k_method_ready") is not True:
        raise RuntimeError("MNopt pre-performance method freeze is not affirmative.")
    if frozen.get("implementation_identity") != "paper-faithful reimplementation":
        raise RuntimeError("Unexpected MNopt implementation identity.")
    if frozen.get("official_mnopt_commit") != C.EXPECTED_MNOPT_COMMIT:
        raise RuntimeError("MNopt frozen public commit mismatch.")

    core = THIS_DIR / "30_7_mnopt_reimpl_core.py"
    if C.sha256_file(core) != C.EXPECTED_MNOPT_CORE_SHA256:
        raise RuntimeError("MNopt reimplementation core differs from the pre-performance frozen core.")
    if frozen.get("core_sha256") != C.EXPECTED_MNOPT_CORE_SHA256:
        raise RuntimeError("MNopt method-freeze core hash mismatch.")

    snap = C.load_module("v31_mnopt_snapshot_check", THIS_DIR / "30_5_mnopt_source_snapshot.py")
    med = C.load_module("v31_medchem_snapshot_check", THIS_DIR / "30_6_medchem_source_snapshot.py")
    mnopt_dir = (C.PROJECT_ROOT / "08_external" / "MNopt_snapshot_44355e3").resolve()
    medchem_dir = (C.PROJECT_ROOT / "08_external" / "medchem_moves_snapshot").resolve()
    mnopt_prov = snap.verify_snapshot(mnopt_dir)
    if mnopt_prov.get("expected_public_commit") != C.EXPECTED_MNOPT_COMMIT:
        raise RuntimeError("MNopt source snapshot provenance commit mismatch.")
    medchem_prov = med.verify_snapshot(medchem_dir)

    # The 1.85 GB radius-3 DesignMove database is a scientifically material MNopt
    # input.  Its exact content was frozen in the pre-performance MNopt spec.
    # Re-verify the current workstation asset against that old immutable record
    # before freezing v3.1; do not inherit the old 10K budget policy itself.
    spec = C.load_json(spec_path)
    provenance = dict(spec.get("provenance", {}))
    db_path = (C.PROJECT_ROOT / "08_external" / "mnopt_assets" / "chemblDB3.sqlitdb").resolve()
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    expected_db_size = int(provenance.get("design_move_database_size_bytes", -1))
    expected_db_sha = str(provenance.get("design_move_database_sha256", ""))
    if expected_db_size <= 1_000_000_000 or len(expected_db_sha) != 64:
        raise RuntimeError("Frozen MNopt spec lacks a valid DesignMove database size/SHA256 record.")
    observed_db_size = int(db_path.stat().st_size)
    if observed_db_size != expected_db_size:
        raise RuntimeError(
            f"MNopt DesignMove database size changed: {observed_db_size} != {expected_db_size}"
        )
    with db_path.open("rb") as handle:
        if handle.read(16) != b"SQLite format 3\x00":
            raise RuntimeError("MNopt DesignMove database is not SQLite format 3.")
    observed_db_sha = C.sha256_file(db_path)
    if observed_db_sha != expected_db_sha:
        raise RuntimeError(
            "MNopt DesignMove database SHA256 changed after the pre-performance freeze."
        )

    return {
        "implementation_identity": "paper-faithful reimplementation",
        "public_commit_provenance": C.EXPECTED_MNOPT_COMMIT,
        "method_freeze": str(method_freeze.resolve()),
        "method_freeze_sha256": C.sha256_file(method_freeze),
        "spec": str(spec_path.resolve()),
        "spec_sha256": C.sha256_file(spec_path),
        "core_sha256": C.EXPECTED_MNOPT_CORE_SHA256,
        "mnopt_snapshot": str(mnopt_dir),
        "mnopt_snapshot_provenance_sha256": C.sha256_file(mnopt_dir / snap.PROVENANCE_NAME),
        "medchem_snapshot": str(medchem_dir),
        "medchem_snapshot_provenance_sha256": C.sha256_file(medchem_dir / med.PROVENANCE_NAME),
        "design_move_database": {
            "path": str(db_path),
            "size_bytes": observed_db_size,
            "sha256": observed_db_sha,
            "sqlite_magic_verified": True,
            "radius": 3,
        },
        "native_property_scorers_used": False,
        "performance_used_for_tuning": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step28_root", type=Path, default=None)
    ap.add_argument("--protocol_dir", type=Path, default=C.DEFAULT_PROTOCOL_DIR)
    args = ap.parse_args()

    out = args.protocol_dir.resolve()
    protocol_path = out / "external_matched_protocol.json"
    marker_path = out / "EXTERNAL_MATCHED_PROTOCOL_FROZEN.json"
    caps_path = out / "matched_resource_caps.csv"
    p0_out = out / "formal_p0.csv"
    p0_smi = out / "formal_p0.smi"
    scorer_details_out = out / "p0_scorer_reproduction_details.csv.gz"

    if protocol_path.exists() or marker_path.exists():
        if not (protocol_path.exists() and marker_path.exists()):
            raise RuntimeError("Partial v3.1 protocol freeze exists; inspect manually.")
        protocol = C.load_json(protocol_path)
        marker = C.load_json(marker_path)
        if C.sha256_file(protocol_path) != marker.get("protocol_sha256"):
            raise RuntimeError("Existing v3.1 protocol hash chain is broken.")
        if protocol.get("protocol_version") != C.PROTOCOL_VERSION:
            raise RuntimeError("Existing protocol directory belongs to a different protocol version.")
        # Verify active code has not changed since freeze.
        for filename in ACTIVE_FILES:
            expected = protocol.get("active_v31_code_sha256", {}).get(filename)
            path = THIS_DIR / filename
            if not expected or not path.exists() or C.sha256_file(path) != expected:
                raise RuntimeError(f"Active code differs from existing freeze: {filename}")
        print("=" * 88)
        print("EXTERNAL MATCHED STEP-28 V3.1 PROTOCOL: ALREADY FROZEN / HASH PASS")
        print("Protocol:", protocol_path)
        print("NO external optimization was run.")
        print("=" * 88)
        return

    for filename in ACTIVE_FILES:
        if not (THIS_DIR / filename).exists():
            raise FileNotFoundError(THIS_DIR / filename)

    scoring_files = C.audit_required_scoring_files()

    # Anchor the scorer binaries/source to the already-frozen pre-performance v2
    # record.  v3.1 supersedes only the RESOURCE semantics, never the scientific
    # evaluator.
    legacy_protocol_path = C.LEGACY_V2_PROTOCOL_DIR / "external_benchmark_protocol.json"
    if not legacy_protocol_path.exists():
        raise FileNotFoundError(legacy_protocol_path)
    legacy_protocol = C.load_json(legacy_protocol_path)
    legacy_scoring = dict(legacy_protocol.get("common_scoring_system", {}).get("files", {}))
    for name, current in scoring_files.items():
        prior = legacy_scoring.get(name)
        if not prior or str(prior.get("sha256", "")) != str(current["sha256"]):
            raise RuntimeError(
                f"Frozen scorer identity changed at {name}: "
                f"v2={None if not prior else prior.get('sha256')} current={current['sha256']}"
            )

    p0_path, p0_raw, p0_can = C.load_frozen_p0()
    task_dirs = C.find_step28_dasr_task_dirs(args.step28_root)
    caps = C.extract_step28_caps(task_dirs)
    scorer_summary, scorer_details = C.audit_current_scorer_against_step28_p0(p0_path, task_dirs)
    if not scorer_summary.get("passed"):
        raise RuntimeError(f"Frozen scorer failed Step-28 P0 reproduction: {scorer_summary}")
    if not scorer_summary.get("exact_ordered_p0_across_all_reference_seeds"):
        raise RuntimeError("Exact ordered P0 audit failed.")

    graphga = verify_graphga()
    mars = verify_mars()
    mnopt = verify_mnopt()

    # Cross-check against the old all-method gate for method identity only. Its old 10K
    # unique budget policy is deliberately NOT inherited.
    old_gate_path = C.LEGACY_V2_PROTOCOL_DIR / "ALL_METHODS_FROZEN.json"
    if not old_gate_path.exists():
        raise FileNotFoundError(old_gate_path)
    old_gate = C.load_json(old_gate_path)
    if C.sha256_file(legacy_protocol_path) != str(old_gate.get("protocol_sha256", "")):
        raise RuntimeError("Historical v2 protocol hash no longer matches ALL_METHODS_FROZEN gate.")
    if old_gate.get("p0_sha256") != C.FROZEN_P0_SHA256:
        raise RuntimeError("Historical all-method gate P0 hash mismatch.")
    if old_gate.get("all_methods_frozen") is not True:
        raise RuntimeError("Historical all-method freeze is not affirmative.")
    expected_methods = sorted(C.EXPECTED_METHODS)
    if sorted(C.METHODS) != expected_methods:
        raise RuntimeError(
            "Active v3.1 method identifiers were modified. "
            f"Expected={expected_methods} observed={sorted(C.METHODS)}"
        )
    if sorted(old_gate.get("methods", [])) != expected_methods:
        raise RuntimeError(
            "Historical all-method freeze method set mismatch. "
            f"Expected={expected_methods} observed={sorted(old_gate.get('methods', []))}"
        )

    if out.exists():
        if any(out.iterdir()):
            raise RuntimeError(f"Nonempty partial v3.1 protocol directory exists: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)
    copy_atomic(p0_path, p0_out)
    if C.sha256_file(p0_out) != C.FROZEN_P0_SHA256:
        raise RuntimeError("Copied v3.1 P0 hash mismatch.")
    p0_smi.write_text("\n".join(p0_raw) + "\n", encoding="utf-8")
    caps.to_csv(caps_path, index=False)
    scorer_details.to_csv(scorer_details_out, index=False, compression="gzip")

    cap_dict = {
        str(int(row.seed)): {
            "request_cap": int(row.request_cap),
            "step28_model_evaluations_cumulative": int(row.step28_model_evaluations_cumulative),
            "step28_unique_structures_scored_globally": int(row.step28_unique_structures_scored_globally),
            "step28_task_dir": str(row.step28_task_dir),
            "task_completed_sha256": str(row.task_completed_sha256),
            "integration_validation_sha256": str(row.integration_validation_sha256),
            "history_sha256": str(row.history_sha256),
            "population_initial_sha256": str(row.population_initial_sha256),
            "final_population_sha256": str(row.final_population_sha256),
        }
        for row in caps.itertuples(index=False)
    }

    active_hashes = {filename: C.sha256_file(THIS_DIR / filename) for filename in ACTIVE_FILES}

    # If a prior GraphGA result directory was deliberately orphaned after loss of the
    # v3.1 protocol files, record that engineering recovery without reading or using
    # the orphaned performance values. The scientific design below is unchanged.
    orphaned_graphga_dir = C.DEFAULT_RESULT_ROOT / "graphga_ORPHANED_PROTOCOL_20260823"
    engineering_recovery = {
        "prior_orphaned_graphga_directory_present": orphaned_graphga_dir.exists(),
        "prior_orphaned_graphga_directory": str(orphaned_graphga_dir.resolve()),
        "prior_orphaned_outputs_excluded_from_analysis": True,
        "performance_used_for_recovery_decisions": False,
        "reason": (
            "Re-freeze is permitted only as engineering recovery after loss of the "
            "v3.1 protocol files. No P0, scorer, seed, request cap, method parameter, "
            "endpoint, or analysis rule is changed."
        ),
    }
    scientific_status = (
        "ENGINEERING RECOVERY RE-FREEZE; SCIENTIFIC DESIGN UNCHANGED"
        if engineering_recovery["prior_orphaned_graphga_directory_present"]
        else "FROZEN BEFORE V3.1 EXTERNAL PERFORMANCE RUNS"
    )

    payload = {
        "created_utc": utc_now(),
        "protocol_version": C.PROTOCOL_VERSION,
        "methods": list(C.METHODS),
        "seeds": list(C.FORMAL_SEEDS),
        "scientific_status": scientific_status,
        "engineering_recovery": engineering_recovery,
        "resource_definition": {
            "primary_resource": "raw oracle score requests",
            "per_seed_cap": "same-seed completed DASR-TSQ Step-28 final total_oracle_requests_cumulative",
            "request_counting": "count before canonicalization and cache lookup",
            "p0_counts_toward_cap": True,
            "cache_hits_count_toward_cap": True,
            "invalid_requests_count_toward_cap": True,
            "batch_methods_atomic_boundary": True,
            "atomic_boundary_rule": "reject a native score batch before any scoring if the entire batch cannot fit the remaining request cap",
            "native_early_stopping_preserved": True,
            "actual_request_utilization_reported": True,
        },
        "matched_request_caps": cap_dict,
        "common_initial_information": {
            "p0_rows": 200,
            "p0_csv": str(p0_out.resolve()),
            "p0_csv_sha256": C.sha256_file(p0_out),
            "p0_exact_order_required": True,
        },
        "frozen_scorer": {
            "files": scoring_files,
            "p0_reproduction": scorer_summary,
            "scalar_reward": "existing project FrozenFitnessAdapter final_reward",
            "posthoc_objectives": ["pred_gsk3b_pic50", "pred_jnk3_pic50", "qed", "sa_reward"],
            "tier1_rule": "min(pred_gsk3b_pic50,pred_jnk3_pic50)>=6.0 and QED>=0.50 and SA_score<=6.0",
            "generic_scaffold": "generic Bemis-Murcko scaffold",
            "hypervolume_reference": [0.0, 0.0, 0.0, 0.0],
        },
        "method_specific_search_preserved": True,
        "graphga": graphga,
        "mars": mars,
        "mnopt": mnopt,
        "analysis": {
            "primary_object": "method-native population of 200 molecules at the last completed native update",
            "primary_endpoints": [
                "top10_mean_final_reward",
                "hypervolume_4d",
                "tier1_count",
                "unique_scaffolds",
            ],
            "secondary_endpoints": [
                "max_final_reward",
                "mean_final_reward",
                "pareto_front_size",
                "max_scaffold_occupancy",
                "actual_requests",
                "request_utilization",
                "model_evaluations",
                "cache_hits",
                "invalid_requests",
                "unique_valid_evaluations",
                "normalized_auc_top10_mean_final_reward",
            ],
            "trajectory_rule": "last completed native population at or before each resource point; after native early stop, carry the final completed population forward to the matched cap for resource-AUC only",
            "statistics": {
                "paired_seeds": list(C.FORMAL_SEEDS),
                "summary": "mean and sample SD",
                "paired_effects": "comparator minus DASR-TSQ, paired by seed",
                "bootstrap_resamples": 20000,
                "exact_signflip_two_sided": True,
                "small_n_boundary": "With five paired seeds, do not make a conventional exact two-sided p<0.05 claim.",
            },
        },
        "superseded_protocols": {
            "external_benchmark_10k_v2": {
                "path": str(C.LEGACY_V2_PROTOCOL_DIR.resolve()),
                "status": "SUPERSEDED FOR PERFORMANCE COMPARISON; retained only as provenance/method-freeze evidence",
                "reason": "10K unique-molecule budget is not equivalent to Step-28 raw oracle-request resource semantics",
            }
        },
        "historical_all_methods_gate_sha256": C.sha256_file(old_gate_path),
        "active_v31_code_sha256": active_hashes,
        "performance_used_to_select_or_tune_methods": False,
        "formal_external_runs_authorized": True,
    }
    C.write_json(protocol_path, payload)
    marker = {
        "frozen": True,
        "created_utc": utc_now(),
        "protocol_version": C.PROTOCOL_VERSION,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": C.sha256_file(protocol_path),
        "p0_sha256": C.sha256_file(p0_out),
        "caps_sha256": C.sha256_file(caps_path),
        "active_v31_code_sha256": active_hashes,
        "performance_used_for_freeze": False,
        "formal_external_runs_authorized": True,
    }
    C.write_json(marker_path, marker)

    print("=" * 88)
    print("EXTERNAL MATCHED STEP-28 V3.1 PROTOCOL FREEZE: PASS")
    print("=" * 88)
    print("P0 exact ordered identity across Step-28 seeds: PASS")
    print("P0 scorer reproduction: PASS")
    for row in caps.itertuples(index=False):
        print(f"seed {int(row.seed)} matched request cap = {int(row.request_cap):,}")
    print("GraphGA frozen-source verification: PASS")
    print("MARS frozen-source/method verification: PASS")
    print("MNopt frozen-core/snapshot verification: PASS")
    print("Protocol:", protocol_path)
    print("NO external optimization was run.")
    print("=" * 88)


if __name__ == "__main__":
    main()
