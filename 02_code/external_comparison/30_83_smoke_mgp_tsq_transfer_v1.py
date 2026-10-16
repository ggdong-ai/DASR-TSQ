from __future__ import annotations

"""
One-update runtime smoke for MGP + TSQ transferability analysis.

DIAGNOSTIC ONLY:
- creates no formal benchmark conclusion;
- uses the exact frozen MGP initialization/generation/scoring path;
- scores one offspring pool only once;
- proves that the adapter with q=population_size reproduces native MGP
  selection EXACTLY (including order);
- then applies the prespecified transfer quota q=5 to the same candidate pool.
"""

import copy
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CODE_DIR = THIS_FILE.parent

BASE_RUNNER = CODE_DIR / "30_79_run_mgp_nsga2_v4.py"
ADAPTER_PATH = CODE_DIR / "30_81_mgp_tsq_transfer_adapter_v1.py"
PREFLIGHT_JSON = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_tsq_transfer_preflight"
    / "MGP_TSQ_TRANSFER_PREFLIGHT.json"
)

OUT_DIR = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_tsq_transfer_smoke"
)
OUT_JSON = OUT_DIR / "MGP_TSQ_ONE_UPDATE_SMOKE.json"
P0_CSV = OUT_DIR / "frozen_p0_for_mgp_tsq_smoke.csv"

SEED = 42
POPULATION_SIZE = 100
BATCH_SIZE = 20
FORMAL_GENERATIONS_REFERENCE = 150
TRANSFER_Q = 5
TECHNICAL_REQUEST_CAP = 100_000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not PREFLIGHT_JSON.exists():
        raise FileNotFoundError(
            f"Run 30_82_preflight_mgp_tsq_transfer_v1.py first: {PREFLIGHT_JSON}"
        )

    preflight = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise RuntimeError("MGP+TSQ static preflight did not PASS")

    BASE = load_module("mgp_v4_base_runner_for_tsq_smoke", BASE_RUNNER)
    ADAPTER = load_module("mgp_tsq_transfer_adapter_for_smoke", ADAPTER_PATH)
    C = BASE.load_module("mgp_tsq_smoke_common", BASE.COMMON_PATH)

    p0_path, _raw, p0 = C.load_frozen_p0()
    if C.sha256_file(p0_path) != BASE.EXPECTED_P0_SHA256:
        raise RuntimeError("Frozen P0 SHA256 mismatch")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": p0}).to_csv(P0_CSV, index=False)

    scorer = C.FrozenProjectScorer()
    broker = C.ExternalBenchmarkOracleBroker(
        scorer=scorer,
        technical_request_cap=TECHNICAL_REQUEST_CAP,
        method="mgp_tsq_transfer_smoke_seed42",
    )
    broker.prime_p0(p0)
    BASE.CURRENT_BROKER = broker

    random.seed(SEED)
    np.random.seed(SEED)

    BASE.install_runtime_stubs(BASE.MGP_NSGA_ROOT)
    BASE.install_fitness_stub()
    official = BASE.import_official_nsga2()

    config = SimpleNamespace(
        data_file=str(P0_CSV.resolve()),
        generations=FORMAL_GENERATIONS_REFERENCE,
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

    population = engine.initial_population()
    if len(population) != POPULATION_SIZE:
        raise RuntimeError(f"Initial population={len(population)} != {POPULATION_SIZE}")

    broker._phase = "mgp_tsq_transfer_smoke_generation_1"
    before = broker.counters().copy()

    offspring = engine.generate_molecules(population)
    n_offspring = len(offspring)
    if n_offspring <= 0:
        raise RuntimeError("Smoke generation produced no accepted offspring")
    offspring = engine.calculate_fitnesses(offspring)
    after_scoring = broker.counters().copy()

    combined = population + offspring

    # Three independent copies of the SAME already-scored candidate pool.
    native_pool = copy.deepcopy(combined)
    noquota_pool = copy.deepcopy(combined)
    tsq_pool = copy.deepcopy(combined)

    native_selected = engine.selection(native_pool)

    noquota_selected, noquota_diag = ADAPTER.mgp_tsq_environmental_selection(
        engine=engine,
        molecules=noquota_pool,
        scaffold_fn=C.generic_scaffold,
        max_per_scaffold=POPULATION_SIZE,
        population_size=POPULATION_SIZE,
    )

    native_smiles = [m.smiles for m in native_selected]
    noquota_smiles = [m.smiles for m in noquota_selected]

    exact_native_order_identity = native_smiles == noquota_smiles
    exact_native_set_identity = set(native_smiles) == set(noquota_smiles)

    if not exact_native_order_identity:
        raise RuntimeError(
            "Transfer adapter FAILED no-quota equivalence: q=100 does not "
            "reproduce native MGP selection order exactly."
        )

    tsq_selected, tsq_diag = ADAPTER.mgp_tsq_environmental_selection(
        engine=engine,
        molecules=tsq_pool,
        scaffold_fn=C.generic_scaffold,
        max_per_scaffold=TRANSFER_Q,
        population_size=POPULATION_SIZE,
    )

    if len(tsq_selected) != POPULATION_SIZE:
        raise RuntimeError(
            f"TSQ smoke selected {len(tsq_selected)} != {POPULATION_SIZE}"
        )

    tsq_smiles = [m.smiles for m in tsq_selected]
    unique_selected_smiles = len(set(tsq_smiles))
    if unique_selected_smiles != POPULATION_SIZE:
        raise RuntimeError(
            f"TSQ smoke population contains duplicate SMILES: "
            f"{unique_selected_smiles}/{POPULATION_SIZE}"
        )

    tsq_records = C.lookup_existing_scores(broker, tsq_smiles)
    tsq_df = pd.DataFrame(tsq_records)
    tsq_metrics = C.population_metrics(tsq_df)

    native_records = C.lookup_existing_scores(broker, native_smiles)
    native_df = pd.DataFrame(native_records)
    native_metrics = C.population_metrics(native_df)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "diagnostic_only": True,
        "formal_performance_result": False,
        "seed": SEED,
        "baseline_mgp_commit": BASE.EXPECTED_COMMIT,
        "p0_sha256": C.sha256_file(p0_path),
        "scientific_config_reference": {
            "initial_size": 200,
            "population_size": POPULATION_SIZE,
            "batch_size": BATCH_SIZE,
            "formal_generations_reference": FORMAL_GENERATIONS_REFERENCE,
            "transfer_max_per_scaffold": TRANSFER_Q,
            "relative_quota": TRANSFER_Q / POPULATION_SIZE,
            "arbiter_rules": ["Glaxo"],
            "search_objectives": [
                "clip((pred_gsk3b_pic50-5)/3,0,1)",
                "clip((pred_jnk3_pic50-5)/3,0,1)",
                "qed",
                "sa_reward",
            ],
        },
        "one_update_accounting": {
            "initial_population_size": len(population),
            "offspring_after_native_arbiter": n_offspring,
            "broker_before_update": before,
            "broker_after_scoring": after_scoring,
            "official_fitness_calls": int(engine.fitness.calls),
        },
        "adapter_noquota_equivalence": {
            "max_per_scaffold": POPULATION_SIZE,
            "exact_native_order_identity": exact_native_order_identity,
            "exact_native_set_identity": exact_native_set_identity,
            "native_selected_size": len(native_selected),
            "adapter_selected_size": len(noquota_selected),
            "adapter_diagnostics": noquota_diag,
        },
        "transfer_q5_one_update": {
            "selected_size": len(tsq_selected),
            "unique_selected_smiles": unique_selected_smiles,
            "selection_diagnostics": tsq_diag,
            "metrics": tsq_metrics,
        },
        "same_candidate_pool_native_reference": {
            "selected_size": len(native_selected),
            "metrics": native_metrics,
        },
        "passed": True,
        "formal_ready_without_review": (
            exact_native_order_identity
            and len(tsq_selected) == POPULATION_SIZE
            and unique_selected_smiles == POPULATION_SIZE
            and not bool(tsq_diag.get("quota_relaxation_used"))
        ),
        "review_note": (
            "If quota_relaxation_used is true, do not freeze the formal transfer "
            "protocol yet. Review whether q=5 can operate as the intended 5% "
            "occupancy quota under the MGP candidate pool."
        ),
    }

    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT_JSON)

    print("=" * 96)
    print("MGP + TSQ one-update smoke: PASS")
    print(f"Native/no-quota exact order : {exact_native_order_identity}")
    print(f"Offspring scored            : {n_offspring}")
    print(f"q=5 strict selected         : {tsq_diag['strict_selected_count']}")
    print(f"q=5 quota relaxed           : {tsq_diag['quota_relaxed_count']}")
    print(f"q=5 final unique scaffolds  : {tsq_diag['final_unique_scaffolds']}")
    print(f"q=5 final max occupancy     : {tsq_diag['final_max_scaffold_occupancy']}")
    print(f"Formal-ready without review : {payload['formal_ready_without_review']}")
    print(f"Output                      : {OUT_JSON}")
    print("=" * 96)

    try:
        engine.client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
