from __future__ import annotations

"""Run one frozen MARS seed under the final Step-28-matched raw-request cap.

This script itself runs in the historical ``mars_legacy`` environment. Scientific
property scoring is delegated to ``30_3_mars_oracle_bridge.py`` running under the
project .venv, so the frozen RF artifacts are never unpickled by the legacy env.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

import importlib.util

THIS_DIR = Path(__file__).resolve().parent
COMMON_PATH = THIS_DIR / "30_0_external_matched_common.py"


def load_common():
    name = "external_matched_common_v31_mars"
    spec = importlib.util.spec_from_file_location(name, COMMON_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(COMMON_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = load_common()
RDLogger.DisableLog("rdApp.*")


class MarsRequestBudgetReached(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_mars(mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class JsonlBridge:
    def __init__(self, project_python: Path, bridge_script: Path, protocol_dir: Path, seed: int, out: Path):
        self.stderr_path = out / "oracle_bridge.stderr.log"
        self.stderr_handle = self.stderr_path.open("w", encoding="utf-8")
        self.archive_path = out / "archive_unique.csv.gz"
        self.batch_log_path = out / "oracle_batch_log.json.gz"
        cmd = [
            str(project_python),
            str(bridge_script),
            "--protocol_dir", str(protocol_dir),
            "--seed", str(int(seed)),
            "--archive", str(self.archive_path),
            "--batch_log", str(self.batch_log_path),
        ]
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUTF8"] = "1"
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            bufsize=1,
            env=env,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Failed to open MARS oracle bridge pipes.")
        self.ready = self._read()
        if self.ready.get("status") != "ready":
            raise RuntimeError(f"MARS oracle bridge did not become ready: {self.ready}")

    def _read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            code = self.proc.poll()
            raise RuntimeError(
                f"MARS oracle bridge ended unexpectedly, returncode={code}. See {self.stderr_path}"
            )
        ans = json.loads(line)
        if ans.get("status") == "error":
            raise RuntimeError(f"MARS oracle bridge error: {ans}")
        return ans

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"MARS oracle bridge exited with {self.proc.returncode}. See {self.stderr_path}"
            )
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self._read()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.request({"cmd": "shutdown"})
            except Exception:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        self.stderr_handle.close()


class FrozenCommonMarsEstimator:
    """MARS estimator interface backed exclusively by the common project bridge."""

    def __init__(self, bridge: JsonlBridge) -> None:
        self.bridge = bridge
        self.calls = 0
        self.first_call_verified_p0_cache_only = False
        self.mars_to_project_canonical: dict[str, str] = {}

    def _map(self, rec: dict[str, Any]) -> dict[str, float]:
        if str(rec.get("scoring_status")) != "ok":
            return {"gsk3b": 0.0, "jnk3": 0.0, "qed": 0.0, "sa": 0.0}
        return {
            "gsk3b": clip01((float(rec["pred_gsk3b_pic50"]) - 5.0) / 3.0),
            "jnk3": clip01((float(rec["pred_jnk3_pic50"]) - 5.0) / 3.0),
            "qed": clip01(float(rec["qed"])),
            "sa": clip01(float(rec["sa_reward"])),
        }

    def get_scores(self, mols):
        self.calls += 1
        mars_smiles = [canonical_mars(mol) for mol in mols]
        ans = self.bridge.request({"cmd": "score_batch", "smiles": mars_smiles})
        if ans.get("status") == "budget_reached":
            raise MarsRequestBudgetReached(ans.get("message") or "matched request boundary reached")
        if ans.get("status") != "ok":
            raise RuntimeError(f"Unexpected MARS bridge response: {ans}")
        raw_scores = list(ans["scores"])
        if len(raw_scores) != len(mols):
            raise RuntimeError("MARS bridge returned a score count different from the native batch size.")
        for mars_smi, rec in zip(mars_smiles, raw_scores):
            project_can = str(rec.get("canonical_smiles") or "")
            if not project_can:
                raise RuntimeError("MARS native proposal produced an invalid molecule after native sanitization.")
            old = self.mars_to_project_canonical.get(mars_smi)
            if old is not None and old != project_can:
                raise RuntimeError("Cross-environment canonical mapping changed for the same MARS SMILES.")
            self.mars_to_project_canonical[mars_smi] = project_can
        if self.calls == 1:
            counters = ans["counters"]
            if int(counters["requests"]) != 400 or int(counters["cache_hits"]) < 200:
                raise RuntimeError(
                    "MARS initial P0 scoring did not reproduce the expected second 200-request cache-only pass."
                )
            self.first_call_verified_p0_cache_only = True
        return [self._map(rec) for rec in raw_scores]

    def project_canonical_population(self, mols) -> list[str]:
        out: list[str] = []
        for mol in mols:
            mars_smi = canonical_mars(mol)
            project_can = self.mars_to_project_canonical.get(mars_smi)
            if not project_can:
                raise RuntimeError(
                    "A MARS native population molecule has no common-oracle mapping; population capture is unsafe."
                )
            out.append(project_can)
        return out


def population_from_archive(smiles: list[str], archive: pd.DataFrame) -> pd.DataFrame:
    indexed = archive.set_index("canonical_smiles", drop=False)
    rows = []
    for smi in smiles:
        if smi not in indexed.index:
            raise RuntimeError(f"MARS population molecule missing from common archive: {smi}")
        row = indexed.loc[smi]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rows.append(row.to_dict())
    if len(rows) != 200:
        raise RuntimeError(f"MARS native population size is {len(rows)}, expected 200.")
    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=C.FORMAL_SEEDS)
    ap.add_argument("--protocol_dir", type=Path, default=C.DEFAULT_PROTOCOL_DIR)
    ap.add_argument("--output_root", type=Path, default=C.DEFAULT_RESULT_ROOT / "mars")
    ap.add_argument("--mars_repo", type=Path, default=C.PROJECT_ROOT / "08_external" / "MARS")
    ap.add_argument("--project_python", type=Path, default=C.PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    args = ap.parse_args()

    protocol, marker = C.load_protocol(args.protocol_dir)
    for filename in (
        "30_0_external_matched_common.py",
        "30_3_mars_oracle_bridge.py",
        "30_4_run_mars_matched.py",
    ):
        C.verify_active_code_hash(protocol, filename)

    seed = int(args.seed)
    cap = C.request_cap_for_seed(protocol, seed)
    out = args.output_root.resolve() / f"seed_{seed}"
    if C.validate_completed_task(out, "mars", seed, marker["protocol_sha256"]):
        print(f"MARS seed {seed}: ALREADY COMPLETE / HASH PASS")
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Incomplete/nonempty MARS output exists: {out}\n"
            "Inspect it manually; this runner never overwrites formal output."
        )
    out.mkdir(parents=True, exist_ok=True)

    repo = args.mars_repo.resolve()
    C.remove_python_cache_artifacts(repo)
    if C.git_commit(repo) != C.EXPECTED_MARS_COMMIT or C.git_is_dirty(repo):
        raise RuntimeError("MARS frozen repository commit/cleanliness mismatch.")
    if C.sha256_file(repo / "sampler.py") != C.EXPECTED_MARS_SAMPLER_SHA256:
        raise RuntimeError("MARS sampler.py differs from the audited frozen source.")

    freeze_path = Path(protocol["mars"]["method_freeze"]).resolve()
    if C.sha256_file(freeze_path) != protocol["mars"]["method_freeze_sha256"]:
        raise RuntimeError("MARS method-freeze JSON changed after v3.1 protocol freeze.")
    freeze = C.load_json(freeze_path)
    if C.sha256_file(repo / "main.py") != freeze["mars_source"]["main_py_sha256"]:
        raise RuntimeError("MARS main.py changed after method freeze.")

    # Historical environment check.
    import dgl
    if not str(torch.__version__).startswith("1.9.0"):
        raise RuntimeError(f"Unexpected MARS torch version: {torch.__version__}")
    if str(dgl.__version__) != "0.7.0":
        raise RuntimeError(f"Unexpected MARS DGL version: {dgl.__version__}")
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("MARS formal run must use the frozen CPU-only torch environment.")

    p0_df = pd.read_csv(Path(args.protocol_dir) / "formal_p0.csv")
    col = C.detect_smiles_column(p0_df)
    raw_p0 = p0_df[col].astype(str).tolist()
    if len(raw_p0) != 200:
        raise RuntimeError("Expected exact P0=200.")
    mols_init = [Chem.MolFromSmiles(smi) for smi in raw_p0]
    if any(mol is None for mol in mols_init):
        raise RuntimeError("MARS legacy RDKit cannot parse a frozen P0 molecule.")

    parent = str(repo.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    native_before = sorted(name for name in sys.modules if name.startswith("MARS.estimator"))
    if native_before:
        raise RuntimeError(f"MARS native estimator modules were already imported: {native_before}")

    from MARS.proposal.models.editor_basic import BasicEditor
    from MARS.proposal.proposal import Proposal_Editor
    from MARS.sampler import Sampler_SA

    native_after = sorted(name for name in sys.modules if name.startswith("MARS.estimator"))
    if native_after:
        raise RuntimeError("Importing MARS search machinery loaded forbidden native estimator modules.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = dict(freeze["mars_config"])
    config["device"] = torch.device("cpu")
    config["mols_ref"] = None
    runtime_assets_path = Path(freeze["runtime_assets"]["path"]).resolve()
    runtime_assets = C.load_json(runtime_assets_path)
    if C.sha256_file(runtime_assets_path) != freeze["runtime_assets"]["sha256"]:
        raise RuntimeError("MARS runtime-assets freeze changed.")
    if Path(config["data_dir"]).resolve() != Path(runtime_assets["runtime_data_dir"]).resolve():
        raise RuntimeError("MARS frozen data_dir mismatch.")
    for key in ("chembl_txt", "arms_pkl", "arms_smiles"):
        path = Path(runtime_assets["assets"][f"{key}_path"]).resolve()
        expected = runtime_assets["assets"][f"{key}_sha256"]
        if not path.exists() or C.sha256_file(path) != expected:
            raise RuntimeError(f"MARS frozen runtime asset missing/changed: {path}")

    project_python = args.project_python.resolve()
    bridge_script = THIS_DIR / "30_3_mars_oracle_bridge.py"
    for path in (project_python, bridge_script):
        if not path.exists():
            raise FileNotFoundError(path)

    bridge = JsonlBridge(project_python, bridge_script, Path(args.protocol_dir).resolve(), seed, out)
    snapshots: list[dict[str, Any]] = []
    boundary_stop = False
    returned_normally = False
    estimator: FrozenCommonMarsEstimator | None = None
    sampler = None
    final_server_counters: dict[str, Any] | None = None
    final_server_metadata: dict[str, Any] | None = None
    try:
        if bridge.ready.get("protocol_sha256") != marker["protocol_sha256"]:
            raise RuntimeError("MARS bridge protocol SHA mismatch.")
        if int(bridge.ready["counters"]["requests"]) != 200:
            raise RuntimeError("MARS bridge did not start after exactly 200 P0 prime requests.")

        estimator = FrozenCommonMarsEstimator(bridge)
        mars_p0 = [canonical_mars(mol) for mol in mols_init]
        # Populate the mapping on the real first estimator call; the explicit prime snapshot
        # uses the same raw order and is converted after that call verifies canonical identity.

        editor = BasicEditor(config).to(config["device"])
        proposal = Proposal_Editor(config, editor)
        sampler = Sampler_SA(config, proposal, estimator)

        original_record = sampler.record

        def wrapped_record(step, old_mols, old_dicts, acc_rates):
            # Preserve the official MARS record/early-stop behavior first.
            result = original_record(step, old_mols, old_dicts, acc_rates)
            counters_ans = bridge.request({"cmd": "counters"})
            project_population = estimator.project_canonical_population(old_mols)
            if len(project_population) != 200:
                raise RuntimeError("Captured MARS native population is not 200 molecules.")
            snapshots.append(
                {
                    "stage": "native_initial_scoring" if int(step) == -1 else "step_update",
                    "update": int(step),
                    "requests": int(counters_ans["counters"]["requests"]),
                    "canonical_smiles": project_population,
                }
            )
            return result

        sampler.record = wrapped_record
        run_dir = out / "mars_native"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            sampler.sample(str(run_dir), mols_init)
            returned_normally = True
        except MarsRequestBudgetReached as exc:
            boundary_stop = True
            print("[MARS] expected matched-request boundary stop:", exc, flush=True)

        if estimator is None or not estimator.first_call_verified_p0_cache_only:
            raise RuntimeError("MARS exact P0 second-pass/cache verification did not pass.")
        # After the first estimator call, exact cross-environment P0 mapping is known.
        p0_project = [estimator.mars_to_project_canonical.get(smi, "") for smi in mars_p0]
        if any(not x for x in p0_project) or len(set(p0_project)) != 200:
            raise RuntimeError("MARS/project P0 canonical mapping is not exact one-to-one.")
        if not snapshots or snapshots[0]["canonical_smiles"] != p0_project:
            raise RuntimeError("MARS initial native population is not exact ordered P0.")
        snapshots.insert(
            0,
            {
                "stage": "p0_prime",
                "update": -2,
                "requests": 200,
                "canonical_smiles": p0_project.copy(),
            },
        )
        final_server_counters = dict(bridge.request({"cmd": "counters"})["counters"])
        final_server_metadata = dict(bridge.request({"cmd": "metadata"})["metadata"])
        bridge.request({"cmd": "save"})
    finally:
        bridge.close()

    archive_path = out / "archive_unique.csv.gz"
    batch_log_path = out / "oracle_batch_log.json.gz"
    if not archive_path.exists() or not batch_log_path.exists():
        raise RuntimeError("MARS bridge did not persist archive/batch log.")
    archive = pd.read_csv(archive_path)
    batch_log = C.read_json_gz(batch_log_path)
    completed_batches = [b for b in batch_log if b.get("completed")]
    rejected_batches = [b for b in batch_log if b.get("rejected_before_scoring")]
    if final_server_counters is None or final_server_metadata is None:
        raise RuntimeError("MARS final project-oracle counters/metadata were not captured.")
    actual_requests = int(final_server_counters["requests"])
    if actual_requests > cap:
        raise RuntimeError("MARS exceeded its matched request cap.")
    if int(final_server_counters["request_budget"]) != cap:
        raise RuntimeError("MARS project-oracle request cap mismatch.")

    # Independent reconciliation: P0 prime contributes 200 requests and every completed
    # native batch contributes its full atomic size. A rejected batch contributes zero.
    total_returned_rows = sum(int(b.get("returned_count", 0)) for b in completed_batches)
    if 200 + total_returned_rows != actual_requests:
        raise RuntimeError(
            f"MARS batch-log request accounting mismatch: 200+{total_returned_rows} != {actual_requests}"
        )
    if rejected_batches and any(int(b.get("returned_count", 0)) != 0 for b in rejected_batches):
        raise RuntimeError("A rejected MARS atomic batch consumed/returned rows.")

    final_smiles = list(snapshots[-1]["canonical_smiles"])
    p0_project = list(snapshots[0]["canonical_smiles"])
    validation = C.validate_common_archive(archive, p0_project, actual_requests)
    final_population = population_from_archive(final_smiles, archive)
    final_metrics = C.population_metrics(final_population, compute_hypervolume=True)

    final_path = out / "final_population.csv"
    snapshots_path = out / "population_snapshots.json.gz"
    metadata_path = out / "scorer_broker_metadata.json"
    final_population.to_csv(final_path, index=False)
    C.write_json_gz(snapshots_path, snapshots)
    C.write_json(metadata_path, final_server_metadata)
    counter_summary = {key: int(value) for key, value in final_server_counters.items()}

    native_modules_final = sorted(name for name in sys.modules if name.startswith("MARS.estimator"))
    if native_modules_final:
        raise RuntimeError("Forbidden MARS native estimator/scorer module was imported during formal run.")

    native_reason = None
    if returned_normally and actual_requests < cap:
        if sampler is not None and int(getattr(sampler, "patience", 1)) <= 0:
            native_reason = "native_patience"
        else:
            native_reason = "native_num_step_ceiling"
    native_early = bool(returned_normally and actual_requests < cap)
    request_boundary = bool(boundary_stop or actual_requests == cap)

    completion = {
        "completed": True,
        "created_utc": utc_now(),
        "method": "mars",
        "seed": seed,
        "protocol_sha256": marker["protocol_sha256"],
        "request_cap": cap,
        "actual_requests": actual_requests,
        "request_utilization": actual_requests / cap,
        "request_cap_boundary_stop": request_boundary,
        "native_early_stop_before_request_cap": native_early,
        "native_stop_reason": native_reason,
        "atomic_batch_boundary": True,
        "counters": counter_summary,
        "n_completed_native_updates": int(sum(s["stage"] == "step_update" for s in snapshots)),
        "final_population_source": "exact old_mols passed to frozen Sampler.record after acceptance update",
        "archive_validation": validation,
        "final_metrics": final_metrics,
        "mars_commit": C.git_commit(repo),
        "mars_sampler_sha256": C.sha256_file(repo / "sampler.py"),
        "mars_method_freeze_sha256": C.sha256_file(freeze_path),
        "mars_environment_python": sys.executable,
        "project_oracle_python": str(project_python),
        "native_mars_property_scorers_used": False,
        "frozen_mars_config": freeze["mars_config"],
        "archive_sha256": C.sha256_file(archive_path),
        "final_population_sha256": C.sha256_file(final_path),
        "population_snapshots_sha256": C.sha256_file(snapshots_path),
        "oracle_batch_log_sha256": C.sha256_file(batch_log_path),
        "scorer_broker_metadata_sha256": C.sha256_file(metadata_path),
        "performance_used_for_tuning": False,
    }
    C.write_json(out / "TASK_COMPLETED.json", completion)

    print("=" * 88)
    print(f"MARS MATCHED seed {seed}: TECHNICAL PASS")
    print("=" * 88)
    print(f"request cap     : {cap:,}")
    print(f"actual requests : {actual_requests:,} ({actual_requests/cap:.2%})")
    print("request boundary:", request_boundary)
    print("native stop     :", native_early, native_reason or "")
    print("completed updates:", completion["n_completed_native_updates"])
    print("Native MARS property scorers used: NO")
    print("Output:", out)
    print("Do NOT tune MARS after observing this formal result.")


if __name__ == "__main__":
    main()
