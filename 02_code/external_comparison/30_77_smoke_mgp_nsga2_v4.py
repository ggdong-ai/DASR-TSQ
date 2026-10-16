from __future__ import annotations

"""
MolecularGraphPareto NSGA-II v4 one-update runtime smoke.

DIAGNOSTIC ONLY:
- uses the frozen official MolecularGraphPareto source without editing it;
- supplies frozen project P0=200;
- preserves native MGP Arbiter, mutation, crossover, Pareto sorting and
  crowding-distance selection;
- preserves published MGP working settings:
      population_size=100, batch_size=20;
- executes ONLY ONE offspring/update cycle;
- computes NO benchmark performance conclusion and performs NO tuning.

The script stubs only optional/import-time packages that are not involved in
the tested search trajectory (Hydra path/decorator, Dask client/bag,
pymoo/pygmo reporting imports, OmegaConf and Faith reporting import).
"""

import hashlib
import importlib.util
import json
import random
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
EXTERNAL_CODE_DIR = THIS_FILE.parent

COMMON_PATH = EXTERNAL_CODE_DIR / "30_0_external_matched_common.py"
MGP_ROOT = PROJECT_ROOT / "08_external" / "MolecularGraphPareto_official"
MGP_NSGA_ROOT = MGP_ROOT / "nsga-ii"
FREEZE_INFO = (
    PROJECT_ROOT
    / "08_external"
    / "_source_freeze"
    / "MolecularGraphPareto_SOURCE_FREEZE.txt"
)
PREFLIGHT_JSON = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_preflight"
    / "MGP_P0_CHEMISTRY_PREFLIGHT.json"
)

EXPECTED_COMMIT = "826e533b1b3995a8944e7c5cefe087806ff8c03f"
EXPECTED_P0_SHA256 = "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"

SEED = 42
POPULATION_SIZE = 100
BATCH_SIZE = 20
FORMAL_GENERATIONS_CONFIG = 150
SMOKE_UPDATE_CYCLES = 1
TECHNICAL_REQUEST_CAP = 100_000

OUT_DIR = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_runtime_smoke"
)
OUT_JSON = OUT_DIR / "MGP_ONE_UPDATE_RUNTIME_SMOKE.json"
P0_CSV = OUT_DIR / "frozen_p0_for_mgp_smoke.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_freeze_info(path: Path) -> dict[str, str]:
    out = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" in raw:
            key, value = raw.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def install_runtime_stubs(root: Path) -> None:
    """Install only import/runtime compatibility shims outside search mechanics."""

    # ---- hydra: official code only needs utils.to_absolute_path and @hydra.main
    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")

    def to_absolute_path(value):
        p = Path(str(value))
        return str(p.resolve() if p.is_absolute() else (root / p).resolve())

    def hydra_main(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    hydra_utils.to_absolute_path = to_absolute_path
    hydra.utils = hydra_utils
    hydra.main = hydra_main
    sys.modules["hydra"] = hydra
    sys.modules["hydra.utils"] = hydra_utils

    # ---- dask: official branch is always batch_score; Client is otherwise unused
    dask = types.ModuleType("dask")
    dask_bag = types.ModuleType("dask.bag")
    dask_distributed = types.ModuleType("dask.distributed")

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
        def close(self):
            return None

    class _DummyBag:
        @staticmethod
        def from_sequence(values):
            return list(values)
        @staticmethod
        def map(fn, values):
            class _Computed:
                def compute(self_nonlocal):
                    return [fn(v) for v in values]
            return _Computed()

    dask.bag = dask_bag
    dask_bag.from_sequence = _DummyBag.from_sequence
    dask_bag.map = _DummyBag.map
    dask_distributed.Client = DummyClient
    sys.modules["dask"] = dask
    sys.modules["dask.bag"] = dask_bag
    sys.modules["dask.distributed"] = dask_distributed

    # ---- reporting-only pymoo imports
    pymoo = types.ModuleType("pymoo")
    pymoo_factory = types.ModuleType("pymoo.factory")
    pymoo_util = types.ModuleType("pymoo.util")
    pymoo_ref_dirs = types.ModuleType("pymoo.util.ref_dirs")
    pymoo_energy = types.ModuleType("pymoo.util.ref_dirs.energy")

    def _unused_get_reference_directions(*args, **kwargs):
        raise RuntimeError("Reporting-only pymoo function unexpectedly called in smoke")

    class _UnusedRiesz:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Reporting-only pymoo class unexpectedly called in smoke")

    pymoo_factory.get_reference_directions = _unused_get_reference_directions
    pymoo_energy.RieszEnergyReferenceDirectionFactory = _UnusedRiesz
    sys.modules["pymoo"] = pymoo
    sys.modules["pymoo.factory"] = pymoo_factory
    sys.modules["pymoo.util"] = pymoo_util
    sys.modules["pymoo.util.ref_dirs"] = pymoo_ref_dirs
    sys.modules["pymoo.util.ref_dirs.energy"] = pymoo_energy

    # ---- pygmo is used only by official __call__ reporting; smoke uses manual loop
    pygmo = types.ModuleType("pygmo")
    sys.modules["pygmo"] = pygmo

    # ---- OmegaConf is only used by official CLI launcher
    omegaconf = types.ModuleType("omegaconf")
    class DictConfig(dict):
        pass
    class OmegaConf:
        @staticmethod
        def to_yaml(config):
            return str(config)
    omegaconf.DictConfig = DictConfig
    omegaconf.OmegaConf = OmegaConf
    sys.modules["omegaconf"] = omegaconf

    # ---- Faith is only used by internal-similarity reporting, not search
    indices = types.ModuleType("indices")
    faith_mod = types.ModuleType("indices.faith")
    class Faith:
        def __init__(self, *args, **kwargs):
            self.__dict__["Fai_1sim_wdis"] = float("nan")
    faith_mod.Faith = Faith
    indices.faith = faith_mod
    sys.modules["indices"] = indices
    sys.modules["indices.faith"] = faith_mod


def normalized_objectives(score: dict) -> np.ndarray:
    return np.asarray(
        [
            np.clip((float(score["pred_gsk3b_pic50"]) - 5.0) / 3.0, 0.0, 1.0),
            np.clip((float(score["pred_jnk3_pic50"]) - 5.0) / 3.0, 0.0, 1.0),
            np.clip(float(score["qed"]), 0.0, 1.0),
            np.clip(float(score["sa_reward"]), 0.0, 1.0),
        ],
        dtype=float,
    )


CURRENT_BROKER = None


class FrozenFitnessAdapter:
    """Drop-in replacement for official task-specific Fitness only."""

    def __init__(self, config_fitness) -> None:
        if CURRENT_BROKER is None:
            raise RuntimeError("CURRENT_BROKER is not initialized")
        self.broker = CURRENT_BROKER
        self.calls = 0

    def batch_score(self, molecules):
        for molecule in molecules:
            row = self.broker.score(molecule.smiles)
            if str(row.get("scoring_status")) != "ok":
                raise RuntimeError(f"Frozen scorer failed: {molecule.smiles}")
            molecule.fitnesses = normalized_objectives(row)
        return molecules

    def __call__(self, molecule):
        return self.batch_score([molecule])[0]


def install_fitness_stub() -> None:
    mechanism = types.ModuleType("nsga2.mechanism")
    mechanism.Fitness = FrozenFitnessAdapter
    sys.modules["nsga2.mechanism"] = mechanism


def import_official_nsga2_module():
    if str(MGP_NSGA_ROOT) not in sys.path:
        sys.path.insert(0, str(MGP_NSGA_ROOT))

    # Clear stale modules except our deliberate mechanism stub.
    for name in list(sys.modules):
        if name == "nsga2" or (
            name.startswith("nsga2.") and name != "nsga2.mechanism"
        ):
            del sys.modules[name]

    module_path = MGP_NSGA_ROOT / "nsga2.py"
    return load_module("mgp_official_nsga2_v4_smoke", module_path)


def main() -> None:
    global CURRENT_BROKER

    print("=" * 94)
    print("MolecularGraphPareto NSGA-II v4 ONE-UPDATE runtime smoke - seed 42")
    print("DIAGNOSTIC ONLY - no formal optimization result / no tuning")
    print("=" * 94)

    freeze = parse_freeze_info(FREEZE_INFO)
    if freeze.get("Commit") != EXPECTED_COMMIT:
        raise RuntimeError("Frozen MGP commit mismatch")

    preflight = load_json(PREFLIGHT_JSON)
    if preflight.get("passed") is not True:
        raise RuntimeError("MGP P0 chemistry preflight did not PASS")
    if preflight.get("p0_sha256") != EXPECTED_P0_SHA256:
        raise RuntimeError("Preflight P0 SHA256 mismatch")

    C = load_module("external_common_mgp_v4_smoke", COMMON_PATH)
    p0_path, _p0_raw, p0_canonical = C.load_frozen_p0()
    if C.sha256_file(p0_path) != EXPECTED_P0_SHA256:
        raise RuntimeError("Frozen P0 SHA256 mismatch")
    if len(p0_canonical) != 200 or len(set(p0_canonical)) != 200:
        raise RuntimeError("Expected P0=200 unique canonical molecules")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": list(p0_canonical)}).to_csv(P0_CSV, index=False)

    scorer = C.FrozenProjectScorer()
    broker = C.RequestBudgetOracleBroker(
        scorer=scorer,
        request_budget=TECHNICAL_REQUEST_CAP,
        method="mgp_nsga2_v4_smoke",
    )
    broker.prime_p0(p0_canonical)
    CURRENT_BROKER = broker

    random.seed(SEED)
    np.random.seed(SEED)

    install_runtime_stubs(MGP_NSGA_ROOT)
    install_fitness_stub()
    official = import_official_nsga2_module()

    config = SimpleNamespace(
        data_file=str(P0_CSV.resolve()),
        generations=FORMAL_GENERATIONS_CONFIG,
        batch_size=BATCH_SIZE,
        initial_size=200,
        population_size=POPULATION_SIZE,
        workers=1,
        threads=2,
        fitness=SimpleNamespace(task="frozen_project_four_objective"),
        arbiter=SimpleNamespace(rules=["Glaxo"]),
        mutator=SimpleNamespace(data_file="data/smarts/mutation_collection.tsv"),
    )

    engine = official.NSGA2(config)

    before_initial = broker.counters().copy()
    population = engine.initial_population()
    after_initial = broker.counters().copy()

    if len(population) != POPULATION_SIZE:
        raise RuntimeError(
            f"Native initial population must be {POPULATION_SIZE}, got {len(population)}"
        )
    if any(len(np.asarray(m.fitnesses)) != 4 for m in population):
        raise RuntimeError("Initial population does not have four objectives")

    # One native offspring/update cycle.
    broker._phase = "mgp_smoke_generation_1"
    before_update = broker.counters().copy()
    offspring = engine.generate_molecules(population)
    generated_after_arbiter = len(offspring)
    if generated_after_arbiter <= 0:
        raise RuntimeError("Native mutation/crossover produced no accepted offspring")

    offspring = engine.calculate_fitnesses(offspring)
    scored_offspring = len(offspring)
    next_population = engine.selection(population + offspring)
    after_update = broker.counters().copy()

    if len(next_population) != POPULATION_SIZE:
        raise RuntimeError(
            f"Post-update population must be {POPULATION_SIZE}, got {len(next_population)}"
        )
    if any(len(np.asarray(m.fitnesses)) != 4 for m in next_population):
        raise RuntimeError("Post-update population does not have four objectives")
    if len({m.smiles for m in next_population}) != len(next_population):
        raise RuntimeError("Unexpected duplicate SMILES in selected population")

    initial_vectors = np.vstack([np.asarray(m.fitnesses, dtype=float) for m in population])
    next_vectors = np.vstack(
        [np.asarray(m.fitnesses, dtype=float) for m in next_population]
    )
    if not np.all(np.isfinite(initial_vectors)):
        raise RuntimeError("Non-finite objective in initial population")
    if not np.all(np.isfinite(next_vectors)):
        raise RuntimeError("Non-finite objective in next population")
    if np.any(initial_vectors < 0) or np.any(initial_vectors > 1):
        raise RuntimeError("Initial objectives escaped [0,1]")
    if np.any(next_vectors < 0) or np.any(next_vectors > 1):
        raise RuntimeError("Next objectives escaped [0,1]")

    # Important native-source accounting identity: official calculate_fitnesses
    # adds len(molecules) to Fitness.calls; our adapter does not add separately.
    if int(engine.fitness.calls) != (
        int(preflight["native_arbiter_accepted_count"]) + scored_offspring
    ):
        raise RuntimeError(
            "Official Fitness.calls accounting mismatch: "
            f"{engine.fitness.calls}"
        )

    payload = {
        "created_utc": utc_now(),
        "completed": True,
        "diagnostic_only": True,
        "formal_performance_result": False,
        "seed": SEED,
        "official_commit": freeze.get("Commit"),
        "p0_sha256": C.sha256_file(p0_path),
        "preflight_sha256": sha256_file(PREFLIGHT_JSON),
        "scientific_config": {
            "initial_size": 200,
            "native_population_size": POPULATION_SIZE,
            "native_batch_size": BATCH_SIZE,
            "formal_generations_config": FORMAL_GENERATIONS_CONFIG,
            "smoke_update_cycles": SMOKE_UPDATE_CYCLES,
            "arbiter_rules": ["Glaxo"],
            "search_objectives": [
                "normalized_gsk3b",
                "normalized_jnk3",
                "qed",
                "sa_reward",
            ],
            "maximize_all": True,
        },
        "runtime_compatibility_only": {
            "hydra_stub": True,
            "dask_stub": True,
            "pymoo_import_stub": True,
            "pygmo_reporting_stub": True,
            "omegaconf_cli_stub": True,
            "faith_reporting_stub": True,
            "official_source_edited": False,
            "search_methods_used_from_official_source": [
                "initial_population",
                "load_from_database",
                "generate_molecules",
                "sample",
                "sample_pairs",
                "selection",
                "calculate_fronts",
                "crowding_distance",
                "Arbiter",
                "Mutator",
                "Crossover",
            ],
        },
        "initial_population_size": len(population),
        "native_arbiter_accepted_from_p0_expected": int(
            preflight["native_arbiter_accepted_count"]
        ),
        "offspring_after_native_arbiter": generated_after_arbiter,
        "offspring_scored": scored_offspring,
        "post_update_population_size": len(next_population),
        "official_fitness_calls": int(engine.fitness.calls),
        "broker_before_initial": before_initial,
        "broker_after_initial": after_initial,
        "broker_before_update": before_update,
        "broker_after_update": after_update,
        "final_broker_counters": broker.counters(),
        "initial_objective_min": initial_vectors.min(axis=0).tolist(),
        "initial_objective_max": initial_vectors.max(axis=0).tolist(),
        "post_update_objective_min": next_vectors.min(axis=0).tolist(),
        "post_update_objective_max": next_vectors.max(axis=0).tolist(),
        "passed": True,
        "next_step_if_passed": (
            "Freeze the formal MGP v4 protocol at official commit "
            "826e533b1b3995a8944e7c5cefe087806ff8c03f with P0=200 supplied, "
            "native Arbiter retained, population_size=100, batch_size=20, "
            "generations config=150, seeds 42-46, and the frozen four-objective scorer."
        ),
    }

    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(OUT_JSON)

    print(f"P0 common priming requests      : {before_initial['requests']}")
    print(f"Initial native population       : {len(population)}")
    print(
        "Initial native scoring calls   : "
        f"{after_initial['requests'] - before_initial['requests']}"
    )
    print(f"Offspring after native Arbiter  : {generated_after_arbiter}")
    print(f"Offspring scored                : {scored_offspring}")
    print(f"Post-update population          : {len(next_population)}")
    print(
        "Update raw requests            : "
        f"{after_update['requests'] - before_update['requests']}"
    )
    print(f"Official Fitness.calls          : {engine.fitness.calls}")
    print(f"RESULT                          : PASS")
    print(f"Output                          : {OUT_JSON}")
    print("=" * 94)

    try:
        engine.client.close()
    except Exception:
        pass



if __name__ == "__main__":
    main()