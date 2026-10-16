from __future__ import annotations

"""
Formal MGP + TSQ(q=5) transferability run.

This is a post-hoc transferability analysis, not a replacement for the frozen
native MGP external baseline.

Only intervention relative to native MGP:
    environmental selection -> native MGP Pareto/crowding priority + TSQ q=5

All MGP generation/scoring machinery is imported from the already validated
formal MGP runner and frozen official source tree.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CODE_DIR = THIS_FILE.parent

BASE_RUNNER_PATH = CODE_DIR / "30_79_run_mgp_nsga2_v4.py"
ADAPTER_PATH = CODE_DIR / "30_81_mgp_tsq_transfer_adapter_v1.py"
PROTOCOL_JSON = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_tsq_transfer_protocol"
    / "MGP_TSQ_TRANSFER_PROTOCOL.json"
)
BASE_FORMAL_ROOT = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_formal"
)
OUT_ROOT = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_tsq_transfer_formal"
)

EXPECTED_COMMIT = "826e533b1b3995a8944e7c5cefe087806ff8c03f"
EXPECTED_P0_SHA256 = "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"

FORMAL_SEEDS = {42, 43, 44, 45, 46}
POPULATION_SIZE = 100
BATCH_SIZE = 20
GENERATIONS = 150
TRANSFER_Q = 5
TECHNICAL_REQUEST_CAP = 1_000_000


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
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_protocol_hash(payload: dict) -> str:
    tmp = dict(payload)
    expected = tmp.pop("protocol_sha256", None)
    raw = json.dumps(
        tmp,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    observed = hashlib.sha256(raw).hexdigest()
    if expected is not None and observed != expected:
        raise RuntimeError(
            f"Transfer protocol self-hash mismatch: {observed} != {expected}"
        )
    return observed


def read_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json_gz(path: Path, payload) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def snapshot(C, broker, generation: int, population, offspring_n=None, selection_diag=None):
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
        "offspring_after_native_arbiter": (
            None if offspring_n is None else int(offspring_n)
        ),
        "selection_diagnostics": selection_diag,
    }


def main(seed: int) -> None:
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"Formal seed must be one of {sorted(FORMAL_SEEDS)}")

    required = [BASE_RUNNER_PATH, ADAPTER_PATH, PROTOCOL_JSON]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))

    protocol = json.loads(PROTOCOL_JSON.read_text(encoding="utf-8"))
    protocol_hash = canonical_protocol_hash(protocol)

    if int(protocol["transfer_intervention"]["max_per_scaffold_q"]) != TRANSFER_Q:
        raise RuntimeError("Frozen transfer q mismatch")
    if int(protocol["scientific_config"]["native_population_size"]) != POPULATION_SIZE:
        raise RuntimeError("Frozen population size mismatch")
    if int(protocol["scientific_config"]["generations"]) != GENERATIONS:
        raise RuntimeError("Frozen generation count mismatch")
    if seed not in set(protocol["scientific_config"]["seeds"]):
        raise RuntimeError("Seed not contained in frozen transfer protocol")

    code_hashes = protocol.get("code_sha256", {})
    expected_runner_hash = code_hashes.get(THIS_FILE.name)
    if expected_runner_hash != sha256_file(THIS_FILE):
        raise RuntimeError(
            "Formal transfer runner changed after protocol freeze:\n"
            f"expected={expected_runner_hash}\ncurrent={sha256_file(THIS_FILE)}"
        )
    expected_adapter_hash = code_hashes.get(ADAPTER_PATH.name)
    if expected_adapter_hash != sha256_file(ADAPTER_PATH):
        raise RuntimeError("Transfer adapter changed after protocol freeze")

    BASE = load_module(
        f"mgp_v4_base_runner_for_tsq_formal_seed_{seed}",
        BASE_RUNNER_PATH,
    )
    ADAPTER = load_module(
        f"mgp_tsq_transfer_adapter_formal_seed_{seed}",
        ADAPTER_PATH,
    )

    # Freeze baseline runner identity too.
    expected_base_runner_hash = (
        protocol.get("baseline_identity", {}).get("baseline_runner_sha256")
    )
    if expected_base_runner_hash != sha256_file(BASE_RUNNER_PATH):
        raise RuntimeError("Baseline MGP formal runner changed after transfer freeze")
    if BASE.EXPECTED_COMMIT != EXPECTED_COMMIT:
        raise RuntimeError("Baseline runner MGP commit constant mismatch")
    if BASE.EXPECTED_P0_SHA256 != EXPECTED_P0_SHA256:
        raise RuntimeError("Baseline runner P0 constant mismatch")

    C = BASE.load_module(
        f"external_common_mgp_tsq_transfer_seed_{seed}",
        BASE.COMMON_PATH,
    )

    p0_path, _raw, p0 = C.load_frozen_p0()
    if C.sha256_file(p0_path) != EXPECTED_P0_SHA256:
        raise RuntimeError("Frozen P0 SHA256 mismatch")

    baseline_seed_dir = BASE_FORMAL_ROOT / f"seed_{seed}"
    baseline_marker_path = baseline_seed_dir / "TASK_COMPLETED.json"
    baseline_snapshots_path = baseline_seed_dir / "population_snapshots.json.gz"

    if not baseline_marker_path.exists():
        raise FileNotFoundError(baseline_marker_path)
    if not baseline_snapshots_path.exists():
        raise FileNotFoundError(baseline_snapshots_path)

    baseline_marker = json.loads(
        baseline_marker_path.read_text(encoding="utf-8")
    )
    if baseline_marker.get("completed") is not True:
        raise RuntimeError(f"Baseline MGP seed {seed} is not completed")
    if baseline_marker.get("official_commit") != EXPECTED_COMMIT:
        raise RuntimeError("Baseline seed official commit mismatch")
    if baseline_marker.get("p0_sha256") != EXPECTED_P0_SHA256:
        raise RuntimeError("Baseline seed P0 mismatch")
    if int(baseline_marker.get("native_generations_completed", -1)) != GENERATIONS:
        raise RuntimeError("Baseline seed generation count mismatch")

    out_dir = OUT_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    marker_path = out_dir / "TASK_COMPLETED.json"
    if marker_path.exists():
        old = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            old.get("completed") is True
            and old.get("protocol_sha256") == protocol_hash
        ):
            print(
                f"MGP+TSQ(q=5) seed {seed} already completed under "
                "identical protocol. Skip."
            )
            return
        raise RuntimeError(f"Conflicting completion marker: {marker_path}")

    p0_csv = out_dir / "frozen_p0.csv"
    pd.DataFrame({"smiles": p0}).to_csv(p0_csv, index=False)

    if not hasattr(C, "ExternalBenchmarkOracleBroker"):
        raise RuntimeError(
            "v4 common does not expose ExternalBenchmarkOracleBroker"
        )

    scorer = C.FrozenProjectScorer()
    broker = C.ExternalBenchmarkOracleBroker(
        scorer=scorer,
        technical_request_cap=TECHNICAL_REQUEST_CAP,
        method=f"mgp_tsq_transfer_q5_seed_{seed}",
    )
    broker.prime_p0(p0)

    # The fitness adapter lives in the imported baseline runner module.
    BASE.CURRENT_BROKER = broker

    random.seed(seed)
    np.random.seed(seed)

    BASE.install_runtime_stubs(BASE.MGP_NSGA_ROOT)
    BASE.install_fitness_stub()
    official = BASE.import_official_nsga2()

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
        raise RuntimeError(
            f"Initial population={len(population)} != {POPULATION_SIZE}"
        )

    # Paired-analysis integrity check: same seed + same P0 must reproduce the
    # exact baseline initial population before TSQ is ever applied.
    baseline_snapshots = read_json_gz(baseline_snapshots_path)
    if not baseline_snapshots:
        raise RuntimeError("Baseline MGP snapshots are empty")
    baseline_initial = list(
        baseline_snapshots[0].get("canonical_smiles", [])
    )
    transfer_initial = [m.smiles for m in population]
    if transfer_initial != baseline_initial:
        raise RuntimeError(
            "Initial population does not exactly match the completed native "
            f"MGP baseline for seed {seed}; paired transfer comparison invalid."
        )

    snapshots = [
        snapshot(
            C,
            broker,
            generation=0,
            population=population,
            offspring_n=None,
            selection_diag=None,
        )
    ]
    events = []

    print("=" * 96)
    print(f"MGP + TSQ(q=5) FORMAL seed {seed}")
    print(
        f"initial population={len(population)} | "
        f"requests={broker.requests} | unique={broker.unique_count}"
    )
    print("paired baseline initial population: EXACT MATCH")
    print("=" * 96)

    quota_binding_generations = 0
    quota_relaxation_generations = 0
    total_blocked_by_quota = 0

    for generation in range(1, GENERATIONS + 1):
        t0 = time.perf_counter()
        broker._phase = f"mgp_tsq_transfer_generation_{generation}"
        before = broker.counters().copy()

        offspring = engine.generate_molecules(population)
        n_offspring = len(offspring)
        if n_offspring <= 0:
            raise RuntimeError(
                f"Generation {generation}: no accepted offspring"
            )

        offspring = engine.calculate_fitnesses(offspring)

        population, selection_diag = ADAPTER.mgp_tsq_environmental_selection(
            engine=engine,
            molecules=population + offspring,
            scaffold_fn=C.generic_scaffold,
            max_per_scaffold=TRANSFER_Q,
            population_size=POPULATION_SIZE,
        )

        if len(population) != POPULATION_SIZE:
            raise RuntimeError(
                f"Generation {generation}: selected population="
                f"{len(population)} != {POPULATION_SIZE}"
            )

        smiles = [m.smiles for m in population]
        if len(set(smiles)) != POPULATION_SIZE:
            raise RuntimeError(
                f"Generation {generation}: duplicate selected SMILES"
            )

        if bool(selection_diag.get("quota_binding")):
            quota_binding_generations += 1
        if bool(selection_diag.get("quota_relaxation_used")):
            quota_relaxation_generations += 1
        total_blocked_by_quota += int(
            selection_diag.get("blocked_by_quota_count", 0)
        )

        after = broker.counters().copy()
        event = {
            "generation": generation,
            "offspring_after_native_arbiter": n_offspring,
            "requests_used": after["requests"] - before["requests"],
            "model_evaluations_used": (
                after["model_evaluations"] - before["model_evaluations"]
            ),
            "cache_hits_used": after["cache_hits"] - before["cache_hits"],
            "quota_binding": bool(selection_diag.get("quota_binding")),
            "quota_relaxation_used": bool(
                selection_diag.get("quota_relaxation_used")
            ),
            "strict_selected_count": int(
                selection_diag.get("strict_selected_count", -1)
            ),
            "quota_relaxed_count": int(
                selection_diag.get("quota_relaxed_count", -1)
            ),
            "blocked_by_quota_count": int(
                selection_diag.get("blocked_by_quota_count", 0)
            ),
            "final_unique_scaffolds": int(
                selection_diag.get("final_unique_scaffolds", -1)
            ),
            "final_max_scaffold_occupancy": int(
                selection_diag.get("final_max_scaffold_occupancy", -1)
            ),
            "elapsed_seconds": time.perf_counter() - t0,
        }
        events.append(event)

        snapshots.append(
            snapshot(
                C,
                broker,
                generation,
                population,
                n_offspring,
                selection_diag,
            )
        )

        if generation == 1 or generation % 10 == 0 or generation == GENERATIONS:
            print(
                f"[seed {seed}] gen {generation:3d}/{GENERATIONS} | "
                f"offspring={n_offspring:3d} | "
                f"requests={broker.requests:6d} | "
                f"unique={broker.unique_count:6d} | "
                f"scaffolds={selection_diag['final_unique_scaffolds']:3d} | "
                f"maxOcc={selection_diag['final_max_scaffold_occupancy']:2d} | "
                f"relax={selection_diag['quota_relaxed_count']:2d}"
            )

        if broker.exhausted:
            raise RuntimeError(
                "Technical safety cap was reached; formal run is invalid "
                "because the cap must remain nonbinding."
            )

    elapsed = time.perf_counter() - start

    final_smiles = [m.smiles for m in population]
    final_records = C.lookup_existing_scores(broker, final_smiles)
    final_df = pd.DataFrame(final_records)
    final_metrics = C.population_metrics(final_df)

    final_event = events[-1]
    transfer_diagnostics = {
        "max_per_scaffold_q": TRANSFER_Q,
        "relative_occupancy_quota": TRANSFER_Q / POPULATION_SIZE,
        "quota_binding_generations": quota_binding_generations,
        "quota_relaxation_generations": quota_relaxation_generations,
        "quota_relaxation_ever_used": quota_relaxation_generations > 0,
        "total_blocked_by_quota": total_blocked_by_quota,
        "final_generation_quota_relaxation_used": bool(
            final_event["quota_relaxation_used"]
        ),
        "final_generation_quota_relaxed_count": int(
            final_event["quota_relaxed_count"]
        ),
        "final_generation_unique_scaffolds": int(
            final_event["final_unique_scaffolds"]
        ),
        "final_generation_max_scaffold_occupancy": int(
            final_event["final_max_scaffold_occupancy"]
        ),
    }

    archive_path = out_dir / "archive_unique.csv.gz"
    final_path = out_dir / "final_population.csv"
    snapshots_path = out_dir / "population_snapshots.json.gz"
    events_path = out_dir / "generation_events.json.gz"
    batch_log_path = out_dir / "oracle_batch_log.json.gz"
    metrics_path = out_dir / "final_metrics.json"

    broker.save_archive(archive_path)
    final_df.to_csv(final_path, index=False)
    write_json_gz(snapshots_path, snapshots)
    write_json_gz(events_path, events)
    broker.save_batch_log(batch_log_path)

    metrics_payload = {
        "seed": seed,
        "formal": True,
        "analysis_status": "posthoc_transferability_analysis",
        "method": "MolecularGraphPareto-NSGA-II+TSQ(q=5)",
        "protocol_sha256": protocol_hash,
        "native_generations_completed": GENERATIONS,
        "final_metrics": final_metrics,
        "transfer_diagnostics": transfer_diagnostics,
        "oracle_counters": broker.counters(),
        "official_fitness_calls": int(engine.fitness.calls),
        "elapsed_seconds": elapsed,
        "stop_reason": "native_generation_ceiling_150",
        "paired_baseline": {
            "method": "MolecularGraphPareto-NSGA-II",
            "seed": seed,
            "task_completed_sha256": sha256_file(baseline_marker_path),
            "population_snapshots_sha256": sha256_file(
                baseline_snapshots_path
            ),
            "initial_population_exact_match": True,
        },
    }
    C.write_json(metrics_path, metrics_payload)

    completion = {
        "completed": True,
        "method": "MolecularGraphPareto-NSGA-II+TSQ(q=5)",
        "analysis_status": "posthoc_transferability_analysis",
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
        "transfer_diagnostics": transfer_diagnostics,
        "paired_baseline_task_completed_sha256": sha256_file(
            baseline_marker_path
        ),
        "paired_baseline_initial_population_exact_match": True,
        "archive_sha256": C.sha256_file(archive_path),
        "final_population_sha256": C.sha256_file(final_path),
        "population_snapshots_sha256": C.sha256_file(snapshots_path),
        "events_sha256": C.sha256_file(events_path),
        "batch_log_sha256": C.sha256_file(batch_log_path),
        "metrics_sha256": C.sha256_file(metrics_path),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    C.write_json(marker_path, completion)

    print("=" * 96)
    print(f"MGP + TSQ(q=5) seed {seed} COMPLETE")
    print(f"generations               : {GENERATIONS}")
    print(f"final population          : {len(final_df)}")
    print(f"final unique scaffolds    : {final_metrics['unique_scaffolds']}")
    print(f"final max occupancy       : {final_metrics['max_scaffold_occupancy']}")
    print(f"quota binding generations : {quota_binding_generations}/{GENERATIONS}")
    print(f"quota relaxation gens     : {quota_relaxation_generations}/{GENERATIONS}")
    print(f"raw requests              : {broker.requests}")
    print(f"unique evaluations        : {broker.unique_count}")
    print(f"cache hits                : {broker.cache_hits}")
    print(f"elapsed seconds           : {elapsed:.1f}")
    print(f"output                    : {out_dir}")
    print("=" * 96)

    try:
        engine.client.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(args.seed)
