from __future__ import annotations

"""Cross-environment identity check for the frozen project scorer.

Purpose
-------
Compare the exact same frozen P0=200 molecules under:
1) the original project environment (.venv)
2) the isolated REINVENT4 environment (.venv_reinvent4)

This is a preflight/compatibility test only.  It performs no optimization and
does not alter the frozen scorer, P0, or REINVENT4 protocol.

PASS criteria
-------------
- exact same 200 canonical SMILES in the same order
- exact same scoring_status, scaffold, and Tier-1 flag
- all numeric score fields agree within atol=1e-10
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\pycharm\project\tsq_mcts_rebuild").resolve()
COMMON_V4 = (
    PROJECT_ROOT
    / "02_code"
    / "external_comparison"
    / "30_72_external_benchmark_common_v4.py"
)

PY_PROJECT = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PY_REINVENT = PROJECT_ROOT / ".venv_reinvent4" / "Scripts" / "python.exe"

OUT_ROOT = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_reinvent4_preflight"
)
PROJECT_JSON = OUT_ROOT / "scorer_identity_project_env.json"
REINVENT_JSON = OUT_ROOT / "scorer_identity_reinvent4_env.json"
REPORT_JSON = OUT_ROOT / "scorer_cross_environment_report.json"

EXPECTED_P0_SHA256 = (
    "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"
)
ATOL = 1e-10

NUMERIC_FIELDS = (
    "final_reward",
    "pred_gsk3b_pic50",
    "pred_jnk3_pic50",
    "dual_min_pic50",
    "qed",
    "sa_score",
    "sa_reward",
)
EXACT_FIELDS = (
    "canonical_smiles",
    "scoring_status",
    "scaffold",
    "tier1",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def python_runtime_metadata() -> dict:
    meta = {
        "python_executable": sys.executable,
        "python_version": sys.version,
    }

    try:
        import numpy as np
        meta["numpy_version"] = np.__version__
    except Exception as exc:
        meta["numpy_error"] = repr(exc)

    try:
        import rdkit
        meta["rdkit_version"] = rdkit.__version__
    except Exception as exc:
        meta["rdkit_error"] = repr(exc)

    try:
        import sklearn
        meta["sklearn_version"] = sklearn.__version__
    except Exception as exc:
        meta["sklearn_error"] = repr(exc)

    try:
        import joblib
        meta["joblib_version"] = joblib.__version__
    except Exception as exc:
        meta["joblib_error"] = repr(exc)

    return meta


def worker(out_json: Path) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    common = load_module("_reinvent4_scorer_identity_common_v4", COMMON_V4)
    p0_path, raw, canonical = common.load_frozen_p0()

    if sha256_file(Path(p0_path)) != EXPECTED_P0_SHA256:
        raise RuntimeError("Frozen P0 SHA256 mismatch")
    if len(raw) != 200 or len(canonical) != 200 or len(set(canonical)) != 200:
        raise RuntimeError("Frozen P0 identity check failed")

    scorer = common.FrozenProjectScorer()

    rows = []
    for index, smi in enumerate(canonical):
        rec = scorer.score_full(smi)
        rows.append(
            {
                "index": index,
                "canonical_smiles": str(rec.canonical_smiles),
                "scoring_status": str(rec.scoring_status),
                "final_reward": float(rec.final_reward),
                "pred_gsk3b_pic50": float(rec.pred_gsk3b_pic50),
                "pred_jnk3_pic50": float(rec.pred_jnk3_pic50),
                "dual_min_pic50": float(rec.dual_min_pic50),
                "qed": float(rec.qed),
                "sa_score": float(rec.sa_score),
                "sa_reward": float(rec.sa_reward),
                "scaffold": str(rec.scaffold),
                "tier1": bool(rec.tier1),
            }
        )

    payload = {
        "completed": True,
        "purpose": "frozen scorer cross-environment identity worker",
        "runtime": python_runtime_metadata(),
        "common_v4_path": str(COMMON_V4),
        "common_v4_sha256": sha256_file(COMMON_V4),
        "p0_path": str(Path(p0_path).resolve()),
        "p0_sha256": sha256_file(Path(p0_path)),
        "n_rows": len(rows),
        "rows": rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_json)

    print(f"WORKER PASS: {sys.executable}")
    print(f"rows       : {len(rows)}")
    print(f"output     : {out_json}")


def run_worker(python_exe: Path, out_json: Path) -> None:
    if not python_exe.exists():
        raise RuntimeError(f"Python executable not found: {python_exe}")

    cmd = [
        str(python_exe),
        str(Path(__file__).resolve()),
        "--worker",
        "--out-json",
        str(out_json),
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
    print("=" * 88)
    print(f"Worker: {python_exe}")
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        raise RuntimeError(
            f"Scorer worker failed under {python_exe} with return code {proc.returncode}"
        )


def finite_or_same_nonfinite(a: float, b: float) -> bool:
    if math.isfinite(a) and math.isfinite(b):
        return abs(a - b) <= ATOL
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def compare() -> None:
    a = json.loads(PROJECT_JSON.read_text(encoding="utf-8"))
    b = json.loads(REINVENT_JSON.read_text(encoding="utf-8"))

    if a["p0_sha256"] != EXPECTED_P0_SHA256 or b["p0_sha256"] != EXPECTED_P0_SHA256:
        raise RuntimeError("P0 SHA mismatch in worker outputs")
    if a["n_rows"] != 200 or b["n_rows"] != 200:
        raise RuntimeError("Expected exactly 200 scored P0 rows from each environment")

    exact_mismatches = []
    numeric_mismatches = []
    max_abs_diff = {field: 0.0 for field in NUMERIC_FIELDS}

    for row_a, row_b in zip(a["rows"], b["rows"]):
        idx = int(row_a["index"])
        if idx != int(row_b["index"]):
            exact_mismatches.append(
                {"index": idx, "field": "index", "project": idx, "reinvent4": row_b["index"]}
            )
            continue

        for field in EXACT_FIELDS:
            if row_a[field] != row_b[field]:
                exact_mismatches.append(
                    {
                        "index": idx,
                        "field": field,
                        "project": row_a[field],
                        "reinvent4": row_b[field],
                    }
                )

        for field in NUMERIC_FIELDS:
            va = float(row_a[field])
            vb = float(row_b[field])

            if math.isfinite(va) and math.isfinite(vb):
                diff = abs(va - vb)
                max_abs_diff[field] = max(max_abs_diff[field], diff)
            else:
                diff = None

            if not finite_or_same_nonfinite(va, vb):
                numeric_mismatches.append(
                    {
                        "index": idx,
                        "canonical_smiles": row_a["canonical_smiles"],
                        "field": field,
                        "project": va,
                        "reinvent4": vb,
                        "abs_diff": diff,
                    }
                )

    passed = not exact_mismatches and not numeric_mismatches

    report = {
        "completed": True,
        "passed": passed,
        "atol": ATOL,
        "n_molecules": 200,
        "project_runtime": a["runtime"],
        "reinvent4_runtime": b["runtime"],
        "p0_sha256": EXPECTED_P0_SHA256,
        "common_v4_sha256_project": a["common_v4_sha256"],
        "common_v4_sha256_reinvent4": b["common_v4_sha256"],
        "max_abs_difference": max_abs_diff,
        "n_exact_mismatches": len(exact_mismatches),
        "n_numeric_mismatches": len(numeric_mismatches),
        "exact_mismatches_first20": exact_mismatches[:20],
        "numeric_mismatches_first20": numeric_mismatches[:20],
        "project_output": str(PROJECT_JSON),
        "reinvent4_output": str(REINVENT_JSON),
    }

    tmp = REPORT_JSON.with_suffix(REPORT_JSON.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(REPORT_JSON)

    print("=" * 88)
    print("FROZEN SCORER CROSS-ENVIRONMENT CHECK")
    print("=" * 88)
    print("project Python   :", a["runtime"].get("python_executable"))
    print("project RDKit    :", a["runtime"].get("rdkit_version"))
    print("project sklearn  :", a["runtime"].get("sklearn_version"))
    print("REINVENT Python  :", b["runtime"].get("python_executable"))
    print("REINVENT RDKit   :", b["runtime"].get("rdkit_version"))
    print("REINVENT sklearn :", b["runtime"].get("sklearn_version"))
    print("P0 molecules     : 200")
    print("tolerance        :", ATOL)
    for field in NUMERIC_FIELDS:
        print(f"max |Δ| {field:18s}: {max_abs_diff[field]:.17g}")
    print("exact mismatches :", len(exact_mismatches))
    print("numeric mismatch :", len(numeric_mismatches))
    print("report            :", REPORT_JSON)

    if not passed:
        raise RuntimeError(
            "FROZEN SCORER CROSS-ENVIRONMENT CHECK FAILED. "
            "Do not run the REINVENT4 smoke."
        )

    print("SCORER CROSS-ENVIRONMENT IDENTITY PASS")
    print("smoke_ready = True")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if args.worker:
        if args.out_json is None:
            raise SystemExit("--out-json is required with --worker")
        worker(args.out_json.resolve())
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    run_worker(PY_PROJECT, PROJECT_JSON)
    run_worker(PY_REINVENT, REINVENT_JSON)
    compare()


if __name__ == "__main__":
    main()
