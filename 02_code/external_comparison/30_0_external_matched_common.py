from __future__ import annotations

"""Common infrastructure for the final Step-28-matched external benchmark (v3.1).

Scientific resource definition
------------------------------
For seed s, GraphGA, MARS and MNopt receive a *maximum raw oracle-request cap*
equal to the final ``total_oracle_requests_cumulative`` of the already-completed
DASR–TSQ Step-28 run with the same seed.

A request is counted before canonicalization/cache lookup, exactly matching the
Step-28 CountingFitnessAdapter request semantics. The frozen P0=200 is included.
Batch-based methods use an atomic budget boundary: if a whole native score batch
cannot fit, that batch is rejected before any request is consumed. This prevents
partially scored generations/steps from being discarded after consuming budget.
"""

import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# Engineering-only console cleanup: suppress RDKit warning/deprecation chatter while
# preserving rdApp.error messages and Python exceptions.
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]

PROTOCOL_VERSION = "external_matched_step28_requests_v3_1"
FORMAL_SEEDS = (42, 43, 44, 45, 46)
METHODS = ("dasr_tsq", "graphga", "mars", "mnopt")
EXPECTED_METHODS = ("dasr_tsq", "graphga", "mars", "mnopt")

# Already-frozen project identities.
STEP28_PROTOCOL_SHA256 = "48974a63e13986423f337ab1b2a8f34b07c8e532f5b582453495272058125da6"
FROZEN_P0_SHA256 = "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"

EXPECTED_GRAPHGA_COMMIT = "63382d78890e910080ef9a9b3b6d04a4552aff85"
EXPECTED_GRAPHGA_RUN_SHA256 = "b7b64e3ac219ce37732ad8aeb76a0e64e18ee7a695f2fb5a7bfd215676772dc4"
EXPECTED_GRAPHGA_BASE_SHA256 = "c4e663fa0709fc90d1dd1439aa65f3b3fbc94de35c70e39d81121564f0edfed3"
EXPECTED_GRAPHGA_HPARAMS_SHA256 = "11cd42bbd86c7d4e2a2716e6c13eeac54132488e893b551ba2cc03b31c2b8289"

EXPECTED_MARS_COMMIT = "1a68d2a1a9ea9fb535675344715cc4df87abef0b"
EXPECTED_MARS_SAMPLER_SHA256 = "44104991e394849a9d77854fdcdbca2284ba8cc65552130c40f392b8869136c0"

EXPECTED_MNOPT_COMMIT = "44355e3f0d1afe14a62fa430c5f0f80b4d33d1dd"
EXPECTED_MNOPT_CORE_SHA256 = "70c32483e452a5dc1632fbaf0278858da42cce11d74bdebf96ef142177fded8c"

STEP14_PATH = PROJECT_ROOT / "02_code" / "core" / "14_tsq_nsga2_evolution.py"
GSK3B_ORACLE_PATH = PROJECT_ROOT / "03_models" / "final_oracles" / "final_gsk3b_rf_oracle.joblib"
JNK3_ORACLE_PATH = PROJECT_ROOT / "03_models" / "final_oracles" / "final_jnk3_rf_oracle.joblib"
FPSCORES_PATH = PROJECT_ROOT / "02_code" / "scoring" / "fpscores.pkl" / "myZip.pkl"

LEGACY_V2_PROTOCOL_DIR = (
    PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_10k_v2"
)
DEFAULT_PROTOCOL_DIR = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_matched_step28_v3_1"
)
DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT
    / "05_results"
    / "external_comparison"
    / "matched_step28_v3_1"
)

SCORE_COLUMNS = (
    "final_reward",
    "pred_gsk3b_pic50",
    "pred_jnk3_pic50",
    "dual_min_pic50",
    "qed",
    "sa_score",
    "sa_reward",
)


class RequestBudgetReached(RuntimeError):
    """Expected scientific stop at the matched request boundary."""


# -----------------------------------------------------------------------------
# Generic IO/provenance helpers
# -----------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".__tmp__.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json_gz(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".__tmp__.{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), default=str)
    os.replace(tmp, path)


def read_json_gz(path: str | Path) -> Any:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_module(name: str, path: str | Path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    # Required for dataclasses and several Python versions when dynamically importing.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def remove_python_cache_artifacts(root: str | Path) -> None:
    """Remove generated Python bytecode only; never remove tracked/source files."""
    root = Path(root)
    if not root.exists():
        return
    for cache_dir in sorted(root.rglob("__pycache__"), reverse=True):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
    for pattern in ("*.pyc", "*.pyo"):
        for path in root.rglob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def resolve_git_executable() -> Optional[str]:
    found = shutil.which("git")
    if found:
        return found
    candidates: list[Path] = []
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_name)
        if base:
            candidates.extend(
                [Path(base) / "Git" / "cmd" / "git.exe", Path(base) / "Git" / "bin" / "git.exe"]
            )
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.extend(
            [
                Path(local) / "Programs" / "Git" / "cmd" / "git.exe",
                Path(local) / "Programs" / "Git" / "bin" / "git.exe",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def git(repo: str | Path, *args: str) -> str:
    executable = resolve_git_executable()
    if not executable:
        raise RuntimeError("git executable was not found")
    proc = subprocess.run(
        [executable, "-C", str(Path(repo).resolve()), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def git_commit(repo: str | Path) -> str:
    return git(repo, "rev-parse", "HEAD")


def git_is_dirty(repo: str | Path) -> bool:
    return bool(git(repo, "status", "--porcelain"))


# -----------------------------------------------------------------------------
# Frozen project scorer and Step-28 reference discovery
# -----------------------------------------------------------------------------
_STEP14 = None


def step14():
    global _STEP14
    if _STEP14 is None:
        _STEP14 = load_module("external_matched_step14_v31", STEP14_PATH)
    return _STEP14


def canonicalize_smiles(smiles: str) -> Optional[str]:
    return step14().canonicalize_smiles(str(smiles))


def generic_scaffold(smiles: str) -> str:
    return step14().generic_murcko_scaffold_smiles(str(smiles))


def detect_smiles_column(df: pd.DataFrame) -> str:
    for column in ("smiles", "canonical_smiles", "standardized_smiles", "SMILES"):
        if column in df.columns:
            return column
    raise KeyError(f"No SMILES column. Columns={list(df.columns)}")


def audit_required_scoring_files() -> dict[str, dict[str, str]]:
    required = {
        "step14_outer": STEP14_PATH,
        "gsk3b_oracle": GSK3B_ORACLE_PATH,
        "jnk3_oracle": JNK3_ORACLE_PATH,
        "sa_fpscores": FPSCORES_PATH,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen scoring files:\n- " + "\n- ".join(missing))
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in required.items()
    }


def find_frozen_p0(search_root: str | Path | None = None) -> Path:
    preferred = LEGACY_V2_PROTOCOL_DIR / "formal_p0.csv"
    if preferred.exists() and sha256_file(preferred) == FROZEN_P0_SHA256:
        return preferred.resolve()
    root = (
        Path(search_root).resolve()
        if search_root
        else PROJECT_ROOT / "06_paper" / "experiment_records"
    )
    matches: list[Path] = []
    for path in root.rglob("formal_p0.csv"):
        try:
            if sha256_file(path) == FROZEN_P0_SHA256:
                matches.append(path.resolve())
        except OSError:
            continue
    if not matches:
        raise FileNotFoundError(
            f"Authoritative formal_p0.csv not found under {root}; expected SHA256={FROZEN_P0_SHA256}"
        )
    return sorted(matches, key=lambda p: (len(str(p)), str(p).lower()))[0]


def load_frozen_p0(path: str | Path | None = None) -> tuple[Path, list[str], list[str]]:
    p0_path = Path(path).resolve() if path else find_frozen_p0()
    if sha256_file(p0_path) != FROZEN_P0_SHA256:
        raise RuntimeError("formal_p0.csv SHA256 mismatch")
    df = pd.read_csv(p0_path)
    column = detect_smiles_column(df)
    raw = [str(value).strip() for value in df[column].tolist()]
    if len(raw) != 200 or any(not value for value in raw):
        raise RuntimeError(f"Frozen P0 must contain exactly 200 non-empty rows, got {len(raw)}")
    canonical = [canonicalize_smiles(value) for value in raw]
    if any(value is None for value in canonical) or len(set(canonical)) != 200:
        raise RuntimeError("Frozen P0 must contain exactly 200 valid unique canonical molecules")
    return p0_path, raw, [str(value) for value in canonical]


def _task_signature(path: Path) -> tuple[str, ...]:
    final_name = "final_population.csv"
    if not (path / final_name).exists() and (path / "population_gen049.csv").exists():
        final_name = "population_gen049.csv"
    names = ("history.csv", "population_initial.csv", final_name)
    signature = []
    for name in names:
        file_path = path / name
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        signature.append(sha256_file(file_path))
    return tuple(signature)


def find_step28_dasr_task_dirs(results_root: str | Path | None = None) -> dict[int, Path]:
    root = (
        Path(results_root).resolve()
        if results_root
        else PROJECT_ROOT / "05_results" / "dasr_tsq_end_to_end"
    )
    out: dict[int, Path] = {}
    for seed in FORMAL_SEEDS:
        wanted = f"step28_dasr_tsq_listwise_tsq10_seed{seed}"
        candidates: list[Path] = []
        for marker_path in root.rglob("TASK_COMPLETED.json"):
            if marker_path.parent.name != wanted:
                continue
            try:
                marker = load_json(marker_path)
            except Exception:
                continue
            if (
                marker.get("completed") is True
                and marker.get("protocol_sha256") == STEP28_PROTOCOL_SHA256
                and marker.get("task_id") == wanted
                and marker.get("variant") == "dasr_tsq_listwise_tsq10"
                and int(marker.get("seed", -1)) == int(seed)
            ):
                candidates.append(marker_path.parent.resolve())
        if not candidates:
            raise FileNotFoundError(
                f"No completed Step-28 DASR–TSQ task found for seed {seed} under {root}"
            )
        if len(candidates) > 1:
            signatures = {candidate: _task_signature(candidate) for candidate in candidates}
            if len(set(signatures.values())) != 1:
                raise RuntimeError(
                    f"Multiple non-identical Step-28 task copies found for seed {seed}. "
                    "Pass --step28_root pointing to the authoritative result tree.\n- "
                    + "\n- ".join(map(str, candidates))
                )
            candidates.sort(
                key=lambda p: (
                    0 if "step28_formal_v4_2" in str(p) else 1,
                    len(str(p)),
                    str(p).lower(),
                )
            )
        out[seed] = candidates[0]
    return out


def extract_step28_caps(task_dirs: dict[int, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, path in sorted(task_dirs.items()):
        marker_path = path / "TASK_COMPLETED.json"
        marker = load_json(marker_path)
        if (
            marker.get("completed") is not True
            or marker.get("protocol_sha256") != STEP28_PROTOCOL_SHA256
            or marker.get("task_id") != f"step28_dasr_tsq_listwise_tsq10_seed{seed}"
            or marker.get("variant") != "dasr_tsq_listwise_tsq10"
            or int(marker.get("seed", -1)) != int(seed)
        ):
            raise RuntimeError(f"Step-28 completion marker identity mismatch: {marker_path}")

        history_path = path / "history.csv"
        initial_path = path / "population_initial.csv"
        final_path = path / "final_population.csv"
        if not final_path.exists():
            final_path = path / "population_gen049.csv"
        files_record = dict(marker.get("files", {}))
        for file_path in (history_path, initial_path, final_path):
            rec = files_record.get(file_path.name)
            if not rec or str(rec.get("sha256", "")) != sha256_file(file_path):
                raise RuntimeError(
                    f"Step-28 task file hash differs from TASK_COMPLETED: {file_path}"
                )

        validation_path = path / "integration_validation.json"
        if not validation_path.exists():
            raise FileNotFoundError(validation_path)
        validation_rec = files_record.get("integration_validation.json")
        if validation_rec and str(validation_rec.get("sha256", "")) != sha256_file(validation_path):
            raise RuntimeError(f"Step-28 validation hash mismatch: {validation_path}")
        validation = load_json(validation_path)
        if validation.get("passed") is not True or validation.get("critical_errors") not in ([], None):
            raise RuntimeError(f"Step-28 integration validation is not clean for seed {seed}")

        history = pd.read_csv(history_path).sort_values("generation").reset_index(drop=True)
        if len(history) != 50 or int(history.iloc[-1]["generation"]) != 49:
            raise RuntimeError(f"Step-28 seed {seed} is not a complete 50-generation run")
        if "total_oracle_requests_cumulative" not in history.columns:
            raise RuntimeError(f"Step-28 seed {seed} history lacks total_oracle_requests_cumulative")
        requests = pd.to_numeric(history["total_oracle_requests_cumulative"], errors="raise").astype(int)
        if not requests.is_monotonic_increasing or requests.iloc[-1] <= 400:
            raise RuntimeError(f"Invalid Step-28 request trajectory for seed {seed}")
        rows.append(
            {
                "seed": int(seed),
                "request_cap": int(requests.iloc[-1]),
                "step28_model_evaluations_cumulative": int(
                    history.iloc[-1].get("model_evaluations_executed_cumulative", -1)
                ),
                "step28_unique_structures_scored_globally": int(
                    history.iloc[-1].get("unique_structures_scored_globally", -1)
                ),
                "step28_task_dir": str(path.resolve()),
                "task_completed_sha256": sha256_file(marker_path),
                "integration_validation_sha256": sha256_file(validation_path),
                "history_sha256": sha256_file(path / "history.csv"),
                "population_initial_sha256": sha256_file(path / "population_initial.csv"),
                "final_population_sha256": sha256_file(final_path),
            }
        )
    return pd.DataFrame(rows)


def audit_current_scorer_against_step28_p0(
    p0_path: str | Path,
    task_dirs: dict[int, Path],
    atol: float = 1e-10,
) -> tuple[dict[str, Any], pd.DataFrame]:
    _, _, p0_canonical = load_frozen_p0(p0_path)
    adapter = step14().FrozenFitnessAdapter()
    rescored: dict[str, dict[str, float]] = {}
    for canonical in p0_canonical:
        score = adapter.score(canonical)
        if str(score.get("scoring_status", "unknown")) != "ok":
            raise RuntimeError(f"Current frozen scorer failed for P0 molecule: {canonical}")
        rescored[canonical] = {column: float(score[column]) for column in SCORE_COLUMNS}

    details: list[dict[str, Any]] = []
    max_abs = {column: 0.0 for column in SCORE_COLUMNS}
    mismatches = {column: 0 for column in SCORE_COLUMNS}

    for seed, task_dir in sorted(task_dirs.items()):
        initial = pd.read_csv(task_dir / "population_initial.csv")
        column = detect_smiles_column(initial)
        canonical = [canonicalize_smiles(value) for value in initial[column].astype(str)]
        if any(value is None for value in canonical):
            raise RuntimeError(f"Step-28 seed {seed} contains an invalid P0 SMILES")
        canonical = [str(value) for value in canonical]
        # Critical v3.1 correction: exact ordered identity, not set equality.
        if canonical != p0_canonical:
            first_bad = next(
                (i for i, (a, b) in enumerate(zip(canonical, p0_canonical)) if a != b),
                None,
            )
            raise RuntimeError(
                f"Step-28 seed {seed} P0 order/identity mismatch at row {first_bad}"
            )

        indexed = initial.copy()
        indexed["_canonical"] = canonical
        indexed = indexed.set_index("_canonical", drop=False)
        for canonical_smiles in p0_canonical:
            row = indexed.loc[canonical_smiles]
            record: dict[str, Any] = {"seed": int(seed), "canonical_smiles": canonical_smiles}
            for score_column in SCORE_COLUMNS:
                stored = float(row[score_column])
                current = rescored[canonical_smiles][score_column]
                difference = abs(stored - current)
                passed = math.isclose(stored, current, rel_tol=0.0, abs_tol=atol)
                max_abs[score_column] = max(max_abs[score_column], difference)
                mismatches[score_column] += int(not passed)
                record[f"{score_column}_stored"] = stored
                record[f"{score_column}_current"] = current
                record[f"{score_column}_abs_diff"] = difference
                record[f"{score_column}_passed"] = bool(passed)
            details.append(record)

    summary = {
        "passed": all(value == 0 for value in mismatches.values()),
        "exact_ordered_p0_across_all_reference_seeds": True,
        "atol": float(atol),
        "n_reference_seeds": len(task_dirs),
        "n_p0": 200,
        "n_molecule_seed_pairs": 200 * len(task_dirs),
        "max_abs_difference": max_abs,
        "n_mismatches": mismatches,
    }
    return summary, pd.DataFrame(details)


# -----------------------------------------------------------------------------
# Common deterministic scientific scorer
# -----------------------------------------------------------------------------
@dataclass
class FullScore:
    canonical_smiles: str
    scoring_status: str
    final_reward: float
    pred_gsk3b_pic50: float
    pred_jnk3_pic50: float
    dual_min_pic50: float
    qed: float
    sa_score: float
    sa_reward: float
    scaffold: str
    tier1: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class FrozenProjectScorer:
    def __init__(self) -> None:
        self.adapter = step14().FrozenFitnessAdapter()
        self.cache: dict[str, FullScore] = {}
        self.__name__ = "frozen_project_final_reward"

    def score_full(self, smiles: str) -> FullScore:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        if canonical in self.cache:
            return self.cache[canonical]
        raw = self.adapter.score(canonical)
        status = str(raw.get("scoring_status", "unknown"))
        dual = float(raw.get("dual_min_pic50", float("nan")))
        qed = float(raw.get("qed", float("nan")))
        sa_score = float(raw.get("sa_score", float("nan")))
        record = FullScore(
            canonical_smiles=canonical,
            scoring_status=status,
            final_reward=float(raw["final_reward"]) if status == "ok" else 0.0,
            pred_gsk3b_pic50=float(raw.get("pred_gsk3b_pic50", float("nan"))),
            pred_jnk3_pic50=float(raw.get("pred_jnk3_pic50", float("nan"))),
            dual_min_pic50=dual,
            qed=qed,
            sa_score=sa_score,
            sa_reward=float(raw.get("sa_reward", float("nan"))),
            scaffold=generic_scaffold(canonical),
            tier1=bool(
                math.isfinite(dual)
                and math.isfinite(qed)
                and math.isfinite(sa_score)
                and dual >= 6.0
                and qed >= 0.50
                and sa_score <= 6.0
            ),
        )
        self.cache[canonical] = record
        return record

    def __call__(self, smiles: str) -> float:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return 0.0
        return float(self.score_full(canonical).final_reward)


# -----------------------------------------------------------------------------
# Request-budget broker
# -----------------------------------------------------------------------------
class RequestBudgetOracleBroker:
    """Common scorer/cache with Step-28-compatible raw request accounting.

    ``requests`` increments before canonicalization/cache lookup.
    ``model_evaluations`` and ``cache_hits`` follow the Step-28 CountingFitnessAdapter
    key semantics: canonical SMILES for valid molecules, ``__INVALID__::<raw>`` for
    invalid requests. The archive itself stores only first-seen valid canonical molecules.
    """

    def __init__(self, scorer: FrozenProjectScorer, request_budget: int, method: str) -> None:
        if int(request_budget) < 400:
            raise ValueError(
                "Matched request budget must be >=400 to include P0 priming and native initialization"
            )
        self.scorer = scorer
        self.request_budget = int(request_budget)
        self.method = str(method)

        self.requests = 0
        self.model_evaluations = 0
        self.cache_hits = 0
        self.invalid_requests = 0

        self.archive: list[dict[str, Any]] = []
        self._score_cache: dict[str, FullScore] = {}
        self._seen_request_keys: set[str] = set()
        self.mol_buffer: dict[str, list[float | int]] = {}
        self._phase = "method"
        self.batch_log: list[dict[str, Any]] = []

        # Compatibility attributes used by mol-opt GraphGA/BaseOptimizer.
        self.name = "frozen_project_final_reward"
        self.evaluator = scorer
        self.task_label = None
        self.output_dir = None
        self.freq_log = 100

    @property
    def remaining_requests(self) -> int:
        return max(0, self.request_budget - self.requests)

    @property
    def exhausted(self) -> bool:
        return self.requests >= self.request_budget

    @property
    def finish(self) -> bool:
        return self.exhausted

    @property
    def unique_count(self) -> int:
        return len(self.archive)

    @property
    def unique_smiles(self) -> set[str]:
        return set(self._score_cache)

    @property
    def budget(self) -> int:
        return self.request_budget

    def __len__(self) -> int:
        # GraphGA uses len(oracle) for its native early-stopping/finish logic.
        return int(self.requests)

    def assign_evaluator(self, evaluator) -> None:
        if not callable(evaluator):
            raise TypeError("External evaluator must be callable")
        self.evaluator = evaluator

    def sort_buffer(self) -> None:
        self.mol_buffer = dict(
            sorted(self.mol_buffer.items(), key=lambda item: item[1][0], reverse=True)
        )

    def save_result(self, suffix=None) -> None:
        return None

    def log_intermediate(self, mols=None, scores=None, finish=False) -> None:
        if self.requests and (
            self.requests % 10000 == 0 or self.exhausted or bool(finish)
        ):
            print(
                f"[{self.method}] requests={self.requests}/{self.request_budget} | "
                f"unique={self.unique_count} | cache_hits={self.cache_hits}",
                flush=True,
            )

    def prime_p0(self, smiles: Iterable[str]) -> None:
        if self.requests or self.unique_count:
            raise RuntimeError("P0 must be primed into an empty broker")
        values = [str(value) for value in smiles]
        if len(values) != 200:
            raise RuntimeError(f"Expected P0=200, got {len(values)}")
        self._phase = "initial_p0"
        for value in values:
            self.score(value)
        self._phase = "method"
        if (
            self.requests != 200
            or self.unique_count != 200
            or self.model_evaluations != 200
            or self.cache_hits != 0
        ):
            raise RuntimeError(
                "Expected exact P0 priming counters: requests=200, unique=200, "
                f"model_evaluations=200, cache_hits=0; got {self.counters()}"
            )

    def _invalid_record(self, raw: str) -> dict[str, Any]:
        return {
            "canonical_smiles": "",
            "scoring_status": "invalid_smiles",
            "final_reward": 0.0,
            "pred_gsk3b_pic50": float("nan"),
            "pred_jnk3_pic50": float("nan"),
            "dual_min_pic50": float("nan"),
            "qed": float("nan"),
            "sa_score": float("nan"),
            "sa_reward": float("nan"),
            "scaffold": "invalid_smiles",
            "tier1": False,
            "raw_smiles": raw,
        }

    def _new_unique(self, score: FullScore, request_index: int) -> None:
        row = {
            "unique_index": self.unique_count + 1,
            "request_index_first_seen": int(request_index),
            "method": self.method,
            "phase_first_seen": self._phase,
            **score.as_dict(),
        }
        self.archive.append(row)
        self._score_cache[score.canonical_smiles] = score
        self.mol_buffer[score.canonical_smiles] = [
            float(score.final_reward),
            int(request_index),
        ]

    def score(self, smiles: str) -> dict[str, Any]:
        if self.exhausted:
            raise RequestBudgetReached(
                f"{self.method}: matched request budget={self.request_budget} exhausted"
            )

        raw = str(smiles)
        self.requests += 1
        canonical = canonicalize_smiles(raw)
        key = canonical if canonical is not None else f"__INVALID__::{raw}"

        if key in self._seen_request_keys:
            self.cache_hits += 1
        else:
            self._seen_request_keys.add(key)
            self.model_evaluations += 1

        if canonical is None:
            self.invalid_requests += 1
            return self._invalid_record(raw)

        if canonical in self._score_cache:
            return self._score_cache[canonical].as_dict()

        score = self.scorer.score_full(canonical)
        self._new_unique(score, self.requests)
        return score.as_dict()

    def score_batch_atomic(self, smiles: Sequence[str]) -> list[dict[str, Any]]:
        values = [str(value) for value in smiles]
        batch_id = len(self.batch_log) + 1
        if len(values) > self.remaining_requests:
            self.batch_log.append(
                {
                    "batch_id": batch_id,
                    "phase": self._phase,
                    "requested_count": len(values),
                    "returned_count": 0,
                    "completed": False,
                    "rejected_before_scoring": True,
                    "request_start": self.requests + 1,
                    "request_end": self.requests,
                    "remaining_requests_before": self.remaining_requests,
                    "rows": [],
                }
            )
            raise RequestBudgetReached(
                f"{self.method}: next native batch size={len(values)} cannot fit "
                f"remaining_requests={self.remaining_requests}; batch rejected atomically"
            )

        start = self.requests + 1
        rows: list[dict[str, Any]] = []
        completed = False
        try:
            for raw in values:
                score = self.score(raw)
                rows.append(
                    {
                        "input_smiles": raw,
                        "canonical_smiles": str(score.get("canonical_smiles", "")),
                        "final_reward": float(score["final_reward"]),
                    }
                )
            completed = True
            return [self.score_cache_lookup(row["canonical_smiles"], row["input_smiles"]) for row in rows]
        finally:
            self.batch_log.append(
                {
                    "batch_id": batch_id,
                    "phase": self._phase,
                    "requested_count": len(values),
                    "returned_count": len(rows),
                    "completed": completed,
                    "rejected_before_scoring": False,
                    "request_start": start,
                    "request_end": self.requests,
                    "remaining_requests_after": self.remaining_requests,
                    "rows": rows,
                }
            )

    def score_cache_lookup(self, canonical: str, raw: str = "") -> dict[str, Any]:
        if canonical and canonical in self._score_cache:
            return self._score_cache[canonical].as_dict()
        if not canonical:
            return self._invalid_record(raw)
        raise KeyError(canonical)

    def __call__(self, smiles):
        if isinstance(smiles, (list, tuple, np.ndarray, pd.Series)):
            records = self.score_batch_atomic([str(value) for value in smiles])
            return [float(record["final_reward"]) for record in records]
        return float(self.score(str(smiles))["final_reward"])

    def counters(self) -> dict[str, int]:
        return {
            "requests": int(self.requests),
            "request_budget": int(self.request_budget),
            "remaining_requests": int(self.remaining_requests),
            "model_evaluations": int(self.model_evaluations),
            "cache_hits": int(self.cache_hits),
            "invalid_requests": int(self.invalid_requests),
            "unique_valid_evaluations": int(self.unique_count),
            "unique_request_keys_seen": int(len(self._seen_request_keys)),
        }

    def archive_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.archive)

    def save_archive(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_dataframe().to_csv(path, index=False, compression="gzip")

    def save_batch_log(self, path: str | Path) -> None:
        write_json_gz(path, self.batch_log)

    def metadata(self) -> dict[str, Any]:
        metadata = copy.deepcopy(self.scorer.adapter.metadata())
        metadata["external_matched_request_broker"] = {
            "method": self.method,
            **self.counters(),
        }
        return metadata


def lookup_existing_scores(
    broker: RequestBudgetOracleBroker, smiles: Sequence[str]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in smiles:
        canonical = canonicalize_smiles(str(raw))
        if canonical is None or canonical not in broker._score_cache:
            raise KeyError(f"Molecule is not present in broker cache: {raw}")
        out.append(broker._score_cache[canonical].as_dict())
    return out


def validate_common_archive(
    archive: pd.DataFrame,
    p0_canonical: Sequence[str],
    actual_requests: int,
) -> dict[str, Any]:
    required = {
        "unique_index",
        "request_index_first_seen",
        "canonical_smiles",
        "final_reward",
        "pred_gsk3b_pic50",
        "pred_jnk3_pic50",
        "dual_min_pic50",
        "qed",
        "sa_score",
        "sa_reward",
        "scaffold",
        "tier1",
    }
    missing = sorted(required - set(archive.columns))
    if missing:
        raise RuntimeError(f"Archive missing columns: {missing}")
    n = len(archive)
    if n < 200:
        raise RuntimeError(f"Archive has fewer than P0=200 rows: {n}")
    if archive["canonical_smiles"].nunique() != n:
        raise RuntimeError("Archive canonical_smiles are not unique")
    if archive["unique_index"].astype(int).tolist() != list(range(1, n + 1)):
        raise RuntimeError("Archive unique_index is not consecutive")
    if archive.iloc[:200]["canonical_smiles"].astype(str).tolist() != list(p0_canonical):
        raise RuntimeError("Archive first 200 canonical molecules are not exact ordered frozen P0")
    first_seen = archive["request_index_first_seen"].astype(int)
    if (first_seen < 1).any() or (first_seen > int(actual_requests)).any():
        raise RuntimeError("Archive request_index_first_seen exceeds actual request range")
    return {
        "passed": True,
        "n_unique_valid": int(n),
        "first_200_exact_ordered_p0": True,
        "unique_index_consecutive": True,
        "all_canonical_unique": True,
    }


# -----------------------------------------------------------------------------
# Protocol helpers
# -----------------------------------------------------------------------------
def load_protocol(
    protocol_dir: str | Path = DEFAULT_PROTOCOL_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = Path(protocol_dir).resolve()
    protocol_path = directory / "external_matched_protocol.json"
    marker_path = directory / "EXTERNAL_MATCHED_PROTOCOL_FROZEN.json"
    if not protocol_path.exists() or not marker_path.exists():
        raise FileNotFoundError("Matched v3.1 protocol has not been frozen")
    protocol = load_json(protocol_path)
    marker = load_json(marker_path)
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"Unexpected protocol version: {protocol.get('protocol_version')}")
    if sha256_file(protocol_path) != marker.get("protocol_sha256"):
        raise RuntimeError("Matched v3.1 protocol/marker hash mismatch")
    return protocol, marker


def request_cap_for_seed(protocol: dict[str, Any], seed: int) -> int:
    record = protocol.get("matched_request_caps", {}).get(str(int(seed)))
    if not record:
        raise KeyError(f"No matched request cap for seed {seed}")
    return int(record["request_cap"])


def verify_active_code_hash(
    protocol: dict[str, Any], filename: str, path: str | Path | None = None
) -> None:
    file_path = Path(path).resolve() if path else THIS_DIR / filename
    expected = protocol.get("active_v31_code_sha256", {}).get(filename)
    if not expected:
        raise RuntimeError(f"Protocol does not freeze active v3.1 code file: {filename}")
    observed = sha256_file(file_path)
    if observed != expected:
        raise RuntimeError(
            f"Active v3.1 code changed after freeze: {filename}\n"
            f"expected={expected}\nobserved={observed}"
        )


def validate_completed_task(
    task_dir: str | Path,
    method: str,
    seed: int,
    protocol_sha256: str,
) -> bool:
    directory = Path(task_dir)
    marker_path = directory / "TASK_COMPLETED.json"
    if not marker_path.exists():
        return False
    marker = load_json(marker_path)
    if (
        marker.get("completed") is not True
        or marker.get("method") != method
        or int(marker.get("seed", -1)) != int(seed)
        or marker.get("protocol_sha256") != protocol_sha256
    ):
        raise RuntimeError(f"Existing completion marker is inconsistent: {marker_path}")
    for key, filename in (
        ("archive_sha256", "archive_unique.csv.gz"),
        ("final_population_sha256", "final_population.csv"),
        ("population_snapshots_sha256", "population_snapshots.json.gz"),
    ):
        path = directory / filename
        expected = marker.get(key)
        if not path.exists() or not expected or sha256_file(path) != expected:
            raise RuntimeError(f"Existing completed task failed hash check: {path}")
    return True


# -----------------------------------------------------------------------------
# GraphGA adapter helpers. Search mechanics remain official GraphGA._optimize.
# -----------------------------------------------------------------------------
def build_graphga_runtime_without_tdc_constructor(
    GraphGA,
    *,
    broker: RequestBudgetOracleBroker,
    smi_file: str | Path,
    initial_smiles: Iterable[str],
    n_jobs: int,
    max_oracle_calls: int,
    freq_log: int,
    output_dir: str | Path,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = GraphGA.__new__(GraphGA)
    optimizer.model_name = "graph_ga"
    optimizer.n_jobs = int(n_jobs)
    optimizer.smi_file = str(Path(smi_file).resolve())
    optimizer.max_oracle_calls = int(max_oracle_calls)
    optimizer.freq_log = int(freq_log)
    optimizer.output_dir = str(output_dir)
    optimizer.log_results = False
    optimizer.oracle = broker
    optimizer.all_smiles = [str(value) for value in initial_smiles]
    return optimizer


def run_official_graphga_search(
    optimizer,
    *,
    scorer: FrozenProjectScorer,
    seed: int,
    patience: int,
    population_size: int,
    offspring_size: int,
    mutation_rate: float,
) -> None:
    import torch

    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    optimizer.seed = int(seed)
    optimizer.patience = int(patience)
    optimizer.oracle.task_label = f"graph_ga_frozen_project_final_reward_{seed}"
    optimizer._optimize(
        scorer,
        {
            "population_size": int(population_size),
            "offspring_size": int(offspring_size),
            "mutation_rate": float(mutation_rate),
        },
    )


def graphga_population_snapshots(
    broker: RequestBudgetOracleBroker,
    p0_canonical: Sequence[str],
    population_size: int = 200,
) -> list[dict[str, Any]]:
    """Reconstruct GraphGA native populations from its real oracle-call structure.

    Frozen GraphGA ``run.py`` does:
      1) score the starting population (200),
      2) each loop: append offspring, sanitize, score the whole candidate pool,
      3) sort that scored pool and retain top population_size.
    Therefore every completed list call after initialization uniquely determines the
    completed post-selection population. A batch rejected at the budget boundary is
    never treated as an update.
    """
    p0 = list(p0_canonical)
    snapshots: list[dict[str, Any]] = [
        {
            "stage": "p0_prime",
            "update": -2,
            "requests": 200,
            "canonical_smiles": p0.copy(),
        }
    ]
    completed_batches = [batch for batch in broker.batch_log if batch.get("completed")]
    if not completed_batches:
        raise RuntimeError("GraphGA produced no completed native score batch")

    init = completed_batches[0]
    init_rows = init.get("rows", [])
    init_can = [str(row.get("canonical_smiles", "")) for row in init_rows]
    if len(init_can) != population_size or init_can != p0:
        raise RuntimeError(
            "GraphGA first native score batch is not the exact ordered P0=200"
        )
    snapshots.append(
        {
            "stage": "native_initial_scoring",
            "update": -1,
            "requests": int(init["request_end"]),
            "canonical_smiles": p0.copy(),
        }
    )

    update_index = 0
    for batch in completed_batches[1:]:
        rows = list(batch.get("rows", []))
        if len(rows) < population_size:
            raise RuntimeError(
                f"Completed GraphGA candidate-pool batch has only {len(rows)} rows (<200)"
            )
        if any(not str(row.get("canonical_smiles", "")) for row in rows):
            raise RuntimeError("GraphGA candidate pool contains an invalid/empty canonical SMILES")
        # Exact frozen run.py semantics: sorted(..., key=score, reverse=True)[:population_size].
        # Python sort is stable, so source order is the tie-breaker.
        ranked = sorted(
            enumerate(rows),
            key=lambda item: (-float(item[1]["final_reward"]), item[0]),
        )[:population_size]
        population = [str(row["canonical_smiles"]) for _, row in ranked]
        snapshots.append(
            {
                "stage": "generation_update",
                "update": update_index,
                "requests": int(batch["request_end"]),
                "canonical_smiles": population,
            }
        )
        update_index += 1
    return snapshots


# -----------------------------------------------------------------------------
# Common population metrics/statistics (same definitions as Step 28 analysis)
# -----------------------------------------------------------------------------
def scores_for_population(
    smiles: Sequence[str],
    archive: pd.DataFrame,
    *,
    expected_size: int = 200,
) -> pd.DataFrame:
    index = archive.set_index("canonical_smiles", drop=False)
    rows: list[dict[str, Any]] = []
    for raw in smiles:
        canonical = canonicalize_smiles(str(raw))
        if canonical is None or canonical not in index.index:
            raise RuntimeError(
                f"Population molecule was not scored in the common archive: {raw}"
            )
        row = index.loc[canonical]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rows.append(row.to_dict())
    out = pd.DataFrame(rows).reset_index(drop=True)
    if len(out) != int(expected_size):
        raise RuntimeError(
            f"Population must contain {expected_size} rows, got {len(out)}"
        )
    return out


def pareto_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not keep[i]:
            continue
        dominated = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        if np.any(dominated):
            keep[i] = False
    return keep


def hypervolume(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    points = np.clip(points, 0.0, 1.0)
    points = points[np.all(points > 0.0, axis=1)]
    if not len(points):
        return 0.0
    points = np.unique(points, axis=0)
    points = points[pareto_mask(points)]

    def recurse(frame: np.ndarray, dim: int) -> float:
        if not len(frame):
            return 0.0
        if dim == 1:
            return float(np.max(frame[:, 0]))
        total = 0.0
        previous = 0.0
        for level in np.unique(frame[:, dim - 1]):
            active = frame[frame[:, dim - 1] >= level, : dim - 1]
            active = np.unique(active, axis=0)
            if len(active):
                active = active[pareto_mask(active)]
            total += float(level - previous) * recurse(active, dim - 1)
            previous = float(level)
        return total

    return float(recurse(points, points.shape[1]))


def normalized_objectives(population: pd.DataFrame) -> np.ndarray:
    g = np.clip((population["pred_gsk3b_pic50"].to_numpy(float) - 5.0) / 3.0, 0.0, 1.0)
    j = np.clip((population["pred_jnk3_pic50"].to_numpy(float) - 5.0) / 3.0, 0.0, 1.0)
    q = np.clip(population["qed"].to_numpy(float), 0.0, 1.0)
    s = np.clip((10.0 - population["sa_score"].to_numpy(float)) / 9.0, 0.0, 1.0)
    return np.column_stack([g, j, q, s])


def population_metrics(
    population: pd.DataFrame, compute_hypervolume: bool = True
) -> dict[str, Any]:
    rewards = np.sort(population["final_reward"].to_numpy(float))[::-1]
    dual = np.sort(population["dual_min_pic50"].to_numpy(float))[::-1]
    tier1 = (
        (population["dual_min_pic50"].to_numpy(float) >= 6.0)
        & (population["qed"].to_numpy(float) >= 0.50)
        & (population["sa_score"].to_numpy(float) <= 6.0)
    )
    points = normalized_objectives(population)
    mask = pareto_mask(points)
    scaffolds = population["scaffold"].astype(str)
    counts = scaffolds.value_counts()
    return {
        "max_final_reward": float(rewards[0]),
        "mean_final_reward": float(np.mean(rewards)),
        "top10_mean_final_reward": float(np.mean(rewards[:10])),
        "mean_dual_min_pic50": float(np.mean(dual)),
        "top10_mean_dual_min_pic50": float(np.mean(dual[:10])),
        "tier1_count": int(np.sum(tier1)),
        "tier1_rate": float(np.mean(tier1)),
        "unique_scaffolds": int(scaffolds.nunique()),
        "max_scaffold_occupancy": int(counts.max()),
        "no_scaffold_count": int(np.sum(scaffolds == "no_scaffold")),
        "pareto_front_size": int(np.sum(mask)),
        "hypervolume_4d": hypervolume(points) if compute_hypervolume else float("nan"),
    }


def normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    return float(np.trapz(y, x) / span) if span > 0 else float(y[-1])


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 20000, seed: int = 42
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    means: list[float] = []
    for mask in range(1 << len(values)):
        signs = np.array(
            [1.0 if (mask >> index) & 1 else -1.0 for index in range(len(values))]
        )
        means.append(abs(float(np.mean(values * signs))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))
