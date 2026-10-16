from __future__ import annotations

"""
MolecularGraphPareto NSGA-II v4 chemistry/initialization preflight.

DIAGNOSTIC ONLY:
- does NOT run NSGA-II optimization;
- does NOT compute benchmark performance;
- does NOT update any population;
- checks whether the frozen project P0 can be supplied to the published
  MolecularGraphPareto implementation while preserving its native chemical
  filters and population size;
- checks frozen four-objective scorer compatibility and parent-sampling weights.

Expected official source:
    08_external/MolecularGraphPareto_official
Expected frozen commit:
    826e533b1b3995a8944e7c5cefe087806ff8c03f
"""

import ast
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rdkit import Chem

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
EXTERNAL_CODE_DIR = THIS_FILE.parent
COMMON_PATH = EXTERNAL_CODE_DIR / "30_0_external_matched_common.py"

MGP_ROOT = PROJECT_ROOT / "08_external" / "MolecularGraphPareto_official"
MGP_NSGA_ROOT = MGP_ROOT / "nsga-ii"
FREEZE_INFO = PROJECT_ROOT / "08_external" / "_source_freeze" / "MolecularGraphPareto_SOURCE_FREEZE.txt"

EXPECTED_COMMIT = "826e533b1b3995a8944e7c5cefe087806ff8c03f"
EXPECTED_P0_SHA256 = "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"

OUT_DIR = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_preflight"
)
OUT_JSON = OUT_DIR / "MGP_P0_CHEMISTRY_PREFLIGHT.json"

SOURCE_FILES = (
    "nsga2.py",
    "configuration/config.yaml",
    "nsga2/base.py",
    "nsga2/infrastructure.py",
    "nsga2/mechanism.py",
    "nsga2/operations.py",
    "data/smarts/mutation_collection.tsv",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_freeze_info(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def install_hydra_path_stub(root: Path) -> None:
    """Provide only hydra.utils.to_absolute_path needed by official filter code.

    This is diagnostic-only. It avoids installing Hydra merely to exercise the
    published Arbiter during the chemistry/P0 preflight.
    """
    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")

    def to_absolute_path(value):
        p = Path(str(value))
        return str(p.resolve() if p.is_absolute() else (root / p).resolve())

    hydra_utils.to_absolute_path = to_absolute_path
    hydra.utils = hydra_utils
    sys.modules["hydra"] = hydra
    sys.modules["hydra.utils"] = hydra_utils


def normalized_objectives(score: dict) -> np.ndarray:
    # Same frozen normalization used by the project 4-D HV analysis.
    g = np.clip((float(score["pred_gsk3b_pic50"]) - 5.0) / 3.0, 0.0, 1.0)
    j = np.clip((float(score["pred_jnk3_pic50"]) - 5.0) / 3.0, 0.0, 1.0)
    q = np.clip(float(score["qed"]), 0.0, 1.0)
    s = np.clip(float(score["sa_reward"]), 0.0, 1.0)
    return np.asarray([g, j, q, s], dtype=float)


def geometric_weight(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=float)
    if np.any(vector < 0):
        return float("nan")
    return float(np.prod(vector) ** (1.0 / len(vector)))


def static_source_checks() -> dict:
    nsga_path = MGP_NSGA_ROOT / "nsga2.py"
    text = nsga_path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    checks = {
        "initial_population_loads_data_file": "self.load_from_database()" in text,
        "data_file_random_sample_uses_initial_size": ".sample(n=self.initial_size)" in text,
        "selection_uses_pareto_fronts": "fronts = self.calculate_fronts(molecules)" in text,
        "selection_truncates_by_crowding": "front.sort(key=lambda x: x.crowding_distance, reverse=True)" in text,
        "dominance_is_all_maximize": (
            "np.all(fitnesses_a >= fitnesses_b)" in text
            and "np.any(fitnesses_a > fitnesses_b)" in text
        ),
        "uses_python_random_choices": "random.choices(" in text,
        "uses_numpy_random_choice": "np.random.choice(" in text,
        "published_batch_branch_source_quirk_present": (
            'if self.task == "mpo_dap_kinases" or "mpo_anipsychotics":' in text
        ),
        "published_crowding_accumulation_source_quirk_present": (
            "pair[1].crowding_distance +=" in text
        ),
    }

    if not all(
        checks[k]
        for k in (
            "initial_population_loads_data_file",
            "data_file_random_sample_uses_initial_size",
            "selection_uses_pareto_fronts",
            "selection_truncates_by_crowding",
            "dominance_is_all_maximize",
            "uses_python_random_choices",
            "uses_numpy_random_choice",
        )
    ):
        raise RuntimeError(f"Unexpected MGP NSGA-II source structure: {checks}")

    return checks


def main() -> None:
    print("=" * 92)
    print("MolecularGraphPareto NSGA-II v4 P0/chemistry preflight")
    print("DIAGNOSTIC ONLY - no optimization / no performance metric")
    print("=" * 92)

    if not COMMON_PATH.exists():
        raise FileNotFoundError(COMMON_PATH)
    if not MGP_NSGA_ROOT.exists():
        raise FileNotFoundError(MGP_NSGA_ROOT)

    freeze = parse_freeze_info(FREEZE_INFO)
    observed_commit = freeze.get("Commit", "")
    if observed_commit != EXPECTED_COMMIT:
        raise RuntimeError(
            "MolecularGraphPareto frozen commit mismatch:\n"
            f"expected={EXPECTED_COMMIT}\nobserved={observed_commit}"
        )

    source_hashes = {}
    for rel in SOURCE_FILES:
        p = MGP_NSGA_ROOT / rel
        if not p.exists():
            raise FileNotFoundError(p)
        source_hashes[rel] = sha256_file(p)

    checks = static_source_checks()

    C = load_module("external_common_mgp_v4_preflight", COMMON_PATH)
    p0_path, p0_raw, p0_canonical = C.load_frozen_p0()

    if C.sha256_file(p0_path) != EXPECTED_P0_SHA256:
        raise RuntimeError("Frozen P0 SHA256 mismatch")

    # Install the minimal path-resolution stub before importing the official
    # infrastructure module. No official source file is edited.
    install_hydra_path_stub(MGP_NSGA_ROOT)

    if str(MGP_NSGA_ROOT) not in sys.path:
        sys.path.insert(0, str(MGP_NSGA_ROOT))

    # Clear possible stale imports from other runs.
    for name in list(sys.modules):
        if name == "nsga2" or name.startswith("nsga2."):
            del sys.modules[name]

    # Import dynamically after adding the frozen official source tree to
    # sys.path. Using importlib here is intentional: PyCharm cannot statically
    # resolve a package that is added to sys.path only at runtime, which makes
    # direct ``from nsga2...`` statements appear as red/unresolved even though
    # they are valid at runtime.
    base_mod = importlib.import_module("nsga2.base")
    infrastructure_mod = importlib.import_module("nsga2.infrastructure")
    Molecule = base_mod.Molecule
    Arbiter = infrastructure_mod.Arbiter

    # Feed all 200 canonical P0 molecules through the official native arbiter.
    arbiter_cfg = SimpleNamespace(rules=["Glaxo"])
    arbiter = Arbiter(arbiter_cfg)
    supplied = [Molecule(smi) for smi in p0_canonical]
    accepted = arbiter(supplied)
    accepted_smiles = [m.smiles for m in accepted]

    # Frozen common scorer compatibility. This is a scorer audit only; it does
    # not create an optimization trajectory.
    scorer = C.FrozenProjectScorer()
    score_rows = []
    vectors = []
    for smi in p0_canonical:
        row = scorer.score_full(smi).as_dict()
        if row["scoring_status"] != "ok":
            raise RuntimeError(f"Frozen scorer failed on P0 molecule: {smi}")
        vector = normalized_objectives(row)
        vectors.append(vector)
        score_rows.append(
            {
                "canonical_smiles": smi,
                "search_objectives": vector.tolist(),
                "mean_parent_weight": float(np.mean(vector)),
                "geometric_parent_weight": geometric_weight(vector),
            }
        )

    vectors_np = np.vstack(vectors)
    mean_weights = np.asarray([row["mean_parent_weight"] for row in score_rows], dtype=float)
    gmean_weights = np.asarray([row["geometric_parent_weight"] for row in score_rows], dtype=float)

    accepted_set = set(accepted_smiles)
    accepted_rows = [
        row for row in score_rows
        if row["canonical_smiles"] in accepted_set
    ]
    accepted_gmean_sum = float(
        sum(float(row["geometric_parent_weight"]) for row in accepted_rows)
    )

    # Native MGP paper population size is 100. Supplying all P0=200 and allowing
    # the native arbiter + NSGA-II selection to reduce to 100 is viable if at
    # least 100 P0 structures survive native chemistry filtering.
    same_p0_route_viable = len(accepted_smiles) >= 100
    parent_sampling_viable = bool(
        float(np.sum(mean_weights)) > 0.0
        and accepted_gmean_sum > 0.0
        and bool(np.all(np.isfinite(vectors_np)))
    )

    passed = bool(same_p0_route_viable and parent_sampling_viable)

    payload = {
        "created_utc": utc_now(),
        "completed": True,
        "diagnostic_only": True,
        "optimization_executed": False,
        "performance_metrics_computed": False,
        "passed": passed,
        "expected_official_commit": EXPECTED_COMMIT,
        "observed_freeze_record": freeze,
        "source_sha256": source_hashes,
        "static_source_checks": checks,
        "p0_path": str(p0_path.resolve()),
        "p0_sha256": C.sha256_file(p0_path),
        "p0_supplied_count": len(p0_canonical),
        "native_arbiter_accepted_count": len(accepted_smiles),
        "native_arbiter_rejected_count": len(p0_canonical) - len(accepted_smiles),
        "native_population_size_target": 100,
        "same_p0_route_viable": same_p0_route_viable,
        "search_objective_definition": {
            "gsk3b": "clip((pred_gsk3b_pic50 - 5.0)/3.0, 0, 1)",
            "jnk3": "clip((pred_jnk3_pic50 - 5.0)/3.0, 0, 1)",
            "qed": "clip(qed, 0, 1)",
            "sa_reward": "clip(sa_reward, 0, 1)",
            "direction": "maximize_all",
        },
        "objective_min": vectors_np.min(axis=0).tolist(),
        "objective_max": vectors_np.max(axis=0).tolist(),
        "mean_parent_weight_sum": float(np.sum(mean_weights)),
        "geometric_parent_weight_sum_all_p0": float(np.sum(gmean_weights)),
        "geometric_parent_weight_sum_native_accepted_p0": accepted_gmean_sum,
        "positive_geometric_weight_count_all_p0": int(np.sum(gmean_weights > 0)),
        "positive_geometric_weight_count_native_accepted_p0": int(
            sum(float(row["geometric_parent_weight"]) > 0 for row in accepted_rows)
        ),
        "parent_sampling_viable": parent_sampling_viable,
        "recommended_if_passed": (
            "Supply all frozen P0=200 as initial_size=200; preserve published "
            "MGP NSGA-II native population_size=100, batch_size=20 and generations=150. "
            "Allow the official native Arbiter and NSGA-II selection to reduce the "
            "supplied starting set to the native working population."
        ),
        "fallback_if_failed": (
            "Do not alter MGP chemistry filters merely to force P0 compatibility. "
            "Use the method-native GuacaMol initialization and report native "
            "initialization explicitly in the external benchmark protocol."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT_JSON)

    print(f"Frozen commit              : {observed_commit}")
    print(f"P0 supplied                : {len(p0_canonical)}")
    print(f"Native arbiter accepted    : {len(accepted_smiles)}")
    print(f"Native arbiter rejected    : {len(p0_canonical) - len(accepted_smiles)}")
    print(f"Native population target   : 100")
    print(f"Same-P0 route viable       : {same_p0_route_viable}")
    print(f"Parent sampling viable     : {parent_sampling_viable}")
    print(f"RESULT                     : {'PASS' if passed else 'FAIL'}")
    print(f"Output                     : {OUT_JSON}")
    print("=" * 92)

    if not passed:
        raise SystemExit(2)



if __name__ == "__main__":
    main()