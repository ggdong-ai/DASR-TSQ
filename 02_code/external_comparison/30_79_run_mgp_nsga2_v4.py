from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CODE_DIR = THIS_FILE.parent

COMMON_PATH = CODE_DIR / "30_72_external_benchmark_common_v4.py"
MGP_ROOT = PROJECT_ROOT / "08_external" / "MolecularGraphPareto_official"
MGP_NSGA_ROOT = MGP_ROOT / "nsga-ii"
FREEZE_INFO = PROJECT_ROOT / "08_external" / "_source_freeze" / "MolecularGraphPareto_SOURCE_FREEZE.txt"
PROTOCOL_JSON = PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_v4_mgp_protocol" / "MGP_V4_PROTOCOL.json"

EXPECTED_COMMIT = "826e533b1b3995a8944e7c5cefe087806ff8c03f"
EXPECTED_P0_SHA256 = "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"

POPULATION_SIZE = 100
BATCH_SIZE = 20
GENERATIONS = 150
TECHNICAL_REQUEST_CAP = 1_000_000
FORMAL_SEEDS = {42, 43, 44, 45, 46}

CURRENT_BROKER = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
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


def parse_freeze(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def canonical_json_hash(payload: dict) -> str:
    tmp = dict(payload)
    expected = tmp.pop("protocol_sha256", None)
    raw = json.dumps(tmp, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    observed = hashlib.sha256(raw).hexdigest()
    if expected is not None and observed != expected:
        raise RuntimeError(f"Protocol self-hash mismatch: {observed} != {expected}")
    return observed


def install_runtime_stubs(root: Path) -> None:
    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    def to_absolute_path(value):
        p = Path(str(value))
        return str(p.resolve() if p.is_absolute() else (root / p).resolve())
    def hydra_main(*args, **kwargs):
        def deco(fn): return fn
        return deco
    hydra_utils.to_absolute_path = to_absolute_path
    hydra.utils = hydra_utils
    hydra.main = hydra_main
    sys.modules["hydra"] = hydra
    sys.modules["hydra.utils"] = hydra_utils

    dask = types.ModuleType("dask")
    dask_bag = types.ModuleType("dask.bag")
    dask_distributed = types.ModuleType("dask.distributed")
    class DummyClient:
        def __init__(self, *args, **kwargs): pass
        def close(self): return None
    class DummyComputed:
        def __init__(self, values): self.values = values
        def compute(self): return self.values
    dask_bag.from_sequence = lambda values: list(values)
    dask_bag.map = lambda fn, values: DummyComputed([fn(v) for v in values])
    dask_distributed.Client = DummyClient
    dask.bag = dask_bag
    sys.modules["dask"] = dask
    sys.modules["dask.bag"] = dask_bag
    sys.modules["dask.distributed"] = dask_distributed

    pymoo = types.ModuleType("pymoo")
    pymoo_factory = types.ModuleType("pymoo.factory")
    pymoo_util = types.ModuleType("pymoo.util")
    pymoo_ref_dirs = types.ModuleType("pymoo.util.ref_dirs")
    pymoo_energy = types.ModuleType("pymoo.util.ref_dirs.energy")
    def unused(*args, **kwargs):
        raise RuntimeError("Reporting-only pymoo function called during formal search")
    class UnusedRiesz:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Reporting-only pymoo class called during formal search")
    pymoo_factory.get_reference_directions = unused
    pymoo_energy.RieszEnergyReferenceDirectionFactory = UnusedRiesz
    for n, m in {
        "pymoo": pymoo,
        "pymoo.factory": pymoo_factory,
        "pymoo.util": pymoo_util,
        "pymoo.util.ref_dirs": pymoo_ref_dirs,
        "pymoo.util.ref_dirs.energy": pymoo_energy,
    }.items():
        sys.modules[n] = m

    sys.modules["pygmo"] = types.ModuleType("pygmo")

    omegaconf = types.ModuleType("omegaconf")
    class DictConfig(dict): pass
    class OmegaConf:
        @staticmethod
        def to_yaml(config): return str(config)
    omegaconf.DictConfig = DictConfig
    omegaconf.OmegaConf = OmegaConf
    sys.modules["omegaconf"] = omegaconf

    indices = types.ModuleType("indices")
    faith = types.ModuleType("indices.faith")
    class Faith:
        def __init__(self, *args, **kwargs):
            self.Fai_1sim_wdis = float("nan")
    faith.Faith = Faith
    indices.faith = faith
    sys.modules["indices"] = indices
    sys.modules["indices.faith"] = faith


def normalized_objectives(score: dict) -> np.ndarray:
    return np.asarray([
        np.clip((float(score["pred_gsk3b_pic50"]) - 5.0) / 3.0, 0.0, 1.0),
        np.clip((float(score["pred_jnk3_pic50"]) - 5.0) / 3.0, 0.0, 1.0),
        np.clip(float(score["qed"]), 0.0, 1.0),
        np.clip(float(score["sa_reward"]), 0.0, 1.0),
    ], dtype=float)


class FrozenFitnessAdapter:
    def __init__(self, config_fitness) -> None:
        if CURRENT_BROKER is None:
            raise RuntimeError("CURRENT_BROKER not initialized")
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


def import_official_nsga2():
    if str(MGP_NSGA_ROOT) not in sys.path:
        sys.path.insert(0, str(MGP_NSGA_ROOT))
    for name in list(sys.modules):
        if name == "nsga2" or (name.startswith("nsga2.") and name != "nsga2.mechanism"):
            del sys.modules[name]
    return load_module("mgp_official_nsga2_v4_formal", MGP_NSGA_ROOT / "nsga2.py")


def snapshot(C, broker, generation: int, population, offspring_after_arbiter=None):
    smiles = [m.smiles for m in population]
    records = C.lookup_existing_scores(broker, smiles)
    frame = pd.DataFrame(records)
    return {
        "generation": int(generation),
        "requests": int(broker.requests),
        "unique_valid_evaluations": int(broker.unique_count),
        "cache_hits": int(broker.cache_hits),
        "population_size": len(smiles),
        "canonical_smiles": smiles,
        "metrics": C.population_metrics(frame),
        "offspring_after_arbiter": None if offspring_after_arbiter is None else int(offspring_after_arbiter),
    }


def main(seed: int):
    global CURRENT_BROKER

    if seed not in FORMAL_SEEDS:
        raise ValueError(f"Formal seed must be one of {sorted(FORMAL_SEEDS)}")

    if not PROTOCOL_JSON.exists():
        raise FileNotFoundError(
            f"Run 30_78_freeze_mgp_nsga2_v4.py first: {PROTOCOL_JSON}"
        )
    protocol = json.loads(PROTOCOL_JSON.read_text(encoding="utf-8"))
    protocol_hash = canonical_json_hash(protocol)

    freeze = parse_freeze(FREEZE_INFO)
    if freeze.get("Commit") != EXPECTED_COMMIT:
        raise RuntimeError("MGP source commit mismatch")

    C = load_module(f"external_common_mgp_v4_seed_{seed}", COMMON_PATH)
    p0_path, _raw, p0 = C.load_frozen_p0()
    if C.sha256_file(p0_path) != EXPECTED_P0_SHA256:
        raise RuntimeError("P0 SHA256 mismatch")

    out_dir = (
        PROJECT_ROOT
        / "06_paper"
        / "experiment_records"
        / "external_benchmark_v4_mgp_formal"
        / f"seed_{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / "TASK_COMPLETED.json"
    if marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") == protocol_hash and old.get("completed") is True:
            print(f"Seed {seed} already completed under identical protocol. Skip.")
            return
        raise RuntimeError(f"Conflicting existing completion marker: {marker}")

    p0_csv = out_dir / "frozen_p0.csv"
    pd.DataFrame({"smiles": p0}).to_csv(p0_csv, index=False)

    scorer = C.FrozenProjectScorer()
    broker = C.RequestBudgetOracleBroker(
        scorer=scorer,
        technical_request_cap=TECHNICAL_REQUEST_CAP,
        method=f"mgp_nsga2_v4_seed_{seed}",
    )
    broker.prime_p0(p0)
    CURRENT_BROKER = broker

    random.seed(seed)
    np.random.seed(seed)

    install_runtime_stubs(MGP_NSGA_ROOT)
    install_fitness_stub()
    official = import_official_nsga2()

    config = SimpleNamespace(
        data_file=str(p0_csv.resolve()),
        generations=GENERATIONS,
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

    start = time.perf_counter()
    population = engine.initial_population()
    if len(population) != POPULATION_SIZE:
        raise RuntimeError(f"Initial population={len(population)} != {POPULATION_SIZE}")

    snapshots = [snapshot(C, broker, 0, population)]
    events = []

    print("=" * 90)
    print(f"MGP v4 FORMAL seed {seed}")
    print(f"initial population={len(population)} requests={broker.requests} unique={broker.unique_count}")
    print("=" * 90)

    for generation in range(1, GENERATIONS + 1):
        t0 = time.perf_counter()
        broker._phase = f"mgp_generation_{generation}"
        before = broker.counters().copy()

        offspring = engine.generate_molecules(population)
        n_offspring = len(offspring)
        if n_offspring <= 0:
            raise RuntimeError(f"Generation {generation}: no accepted offspring")
        offspring = engine.calculate_fitnesses(offspring)
        population = engine.selection(population + offspring)

        if len(population) != POPULATION_SIZE:
            raise RuntimeError(
                f"Generation {generation}: selected population={len(population)}"
            )
        smiles = [m.smiles for m in population]
        if len(set(smiles)) != POPULATION_SIZE:
            raise RuntimeError(f"Generation {generation}: duplicate selected SMILES")

        after = broker.counters().copy()
        event = {
            "generation": generation,
            "offspring_after_arbiter": n_offspring,
            "requests_used": after["requests"] - before["requests"],
            "model_evaluations_used": after["model_evaluations"] - before["model_evaluations"],
            "cache_hits_used": after["cache_hits"] - before["cache_hits"],
            "elapsed_seconds": time.perf_counter() - t0,
        }
        events.append(event)
        snapshots.append(snapshot(C, broker, generation, population, n_offspring))

        if generation == 1 or generation % 10 == 0 or generation == GENERATIONS:
            print(
                f"[seed {seed}] gen {generation:3d}/{GENERATIONS} | "
                f"offspring={n_offspring:3d} | requests={broker.requests:6d} | "
                f"unique={broker.unique_count:6d} | cache={broker.cache_hits:6d}"
            )

        if broker.exhausted:
            raise RuntimeError(
                "Technical safety cap was reached; formal run is invalid because "
                "the cap must remain nonbinding."
            )

    elapsed = time.perf_counter() - start

    final_smiles = [m.smiles for m in population]
    final_records = C.lookup_existing_scores(broker, final_smiles)
    final_df = pd.DataFrame(final_records)
    final_metrics = C.population_metrics(final_df)

    archive_path = out_dir / "archive_unique.csv.gz"
    final_path = out_dir / "final_population.csv"
    snapshots_path = out_dir / "population_snapshots.json.gz"
    events_path = out_dir / "generation_events.json.gz"
    batch_log_path = out_dir / "oracle_batch_log.json.gz"
    metrics_path = out_dir / "final_metrics.json"

    broker.save_archive(archive_path)
    final_df.to_csv(final_path, index=False)
    C.write_json_gz(snapshots_path, snapshots)
    C.write_json_gz(events_path, events)
    broker.save_batch_log(batch_log_path)
    C.write_json(
        metrics_path,
        {
            "seed": seed,
            "formal": True,
            "protocol_sha256": protocol_hash,
            "native_generations_completed": GENERATIONS,
            "final_metrics": final_metrics,
            "oracle_counters": broker.counters(),
            "official_fitness_calls": int(engine.fitness.calls),
            "elapsed_seconds": elapsed,
            "stop_reason": "native_generation_ceiling_150",
        },
    )

    completion = {
        "completed": True,
        "method": "MolecularGraphPareto-NSGA-II",
        "seed": seed,
        "protocol_sha256": protocol_hash,
        "official_commit": EXPECTED_COMMIT,
        "p0_sha256": EXPECTED_P0_SHA256,
        "native_generations_completed": GENERATIONS,
        "stop_reason": "native_generation_ceiling_150",
        "technical_safety_cap_nonbinding": not broker.exhausted,
        "oracle_counters": broker.counters(),
        "official_fitness_calls": int(engine.fitness.calls),
        "elapsed_seconds": elapsed,
        "archive_sha256": C.sha256_file(archive_path),
        "final_population_sha256": C.sha256_file(final_path),
        "population_snapshots_sha256": C.sha256_file(snapshots_path),
        "events_sha256": C.sha256_file(events_path),
        "batch_log_sha256": C.sha256_file(batch_log_path),
        "metrics_sha256": C.sha256_file(metrics_path),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    C.write_json(marker, completion)

    print("=" * 90)
    print(f"MGP seed {seed} COMPLETE")
    print(f"generations       : {GENERATIONS}")
    print(f"final population  : {len(final_df)}")
    print(f"raw requests      : {broker.requests}")
    print(f"unique evaluations: {broker.unique_count}")
    print(f"cache hits        : {broker.cache_hits}")
    print(f"elapsed seconds   : {elapsed:.1f}")
    print(f"output            : {out_dir}")
    print("=" * 90)

    try:
        engine.client.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(args.seed)
