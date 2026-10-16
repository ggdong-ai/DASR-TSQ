"""
29_1_run_tsq_quota_sensitivity.py
=================================
Runner for TSQ scaffold-quota sensitivity analysis.

Two strictly separated modes are supported:

1) Smoke mode (--smoke)
   - q_max in {5, 10, 20}
   - G=5
   - outputs go to 05_results/tsq_quota_sensitivity_smoke/
   - engineering/behavior check only; never used for parameter reselection
     or final scientific conclusions.

2) Full sensitivity mode (default)
   - q_max in {5, 20}
   - G=50
   - seeds in {42, 43, 44, 45, 46}
   - outputs go to 05_results/tsq_quota_sensitivity/
   - q_max=10 is NOT rerun here; its five prespecified formal Step-28 runs
     are reused as the reference condition in the later analysis.

All scientific settings other than q_max are kept identical to the formal
DASR-TSQ configuration: fixed P0, frozen QSAR and DASR models, N=200,
Top-6 + Random-2, K=32, B=8, local oracle budget=101, max depth=5,
tournament size=5, and offspring multiplier=1.0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 0. Paths and fixed design
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Windows multiprocessing uses the spawn start method.  Worker processes must
# be able to import the engine module by the same module name that is recorded
# in pickled worker functions.  Put this directory on sys.path and load the
# engine under its real file stem (29_0_tsq_quota_sensitivity_engine).
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

ENGINE_PATH = THIS_DIR / "29_0_tsq_quota_sensitivity_engine.py"

DEFAULT_P0_PATH = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "dasr_step28_formal_protocol_frozen"
    / "formal_p0.csv"
)

# Compatibility fallback for projects that kept the amended-v4.2 directory name.
ALTERNATE_P0_PATH = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "dasr_step28_formal_protocol_amended_v4_2"
    / "formal_p0.csv"
)

DEFAULT_FREEZE_DIR = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "dasr_p0_safe_v2_test_lock"
)

FORMAL_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "05_results"
    / "tsq_quota_sensitivity"
)

SMOKE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "05_results"
    / "tsq_quota_sensitivity_smoke"
)

FORMAL_QUOTAS = (5, 20)
SMOKE_QUOTAS = (5, 10, 20)
FORMAL_SEEDS = (42, 43, 44, 45, 46)

POP_SIZE = 200
FORMAL_GENERATIONS = 50
SMOKE_GENERATIONS = 5
OFFSPRING_MULTIPLIER = 1.0
TOURNAMENT_K = 5
N_JOBS = 5

K_CANDIDATES = 32
B_ORACLE_CHILDREN = 8
TOP_RANKED_COUNT = 6
RANDOM_EXPLORATION_COUNT = 2
LOCAL_ORACLE_BUDGET = 101
LOCAL_MAX_DEPTH = 5
MAX_TREE_POLICY_STEPS = 3000
MAX_STAGNANT_STEPS = 100


# ============================================================
# 1. Utilities
# ============================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".__tmp__")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
    tmp.replace(path)



def load_module(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required engine was not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {path}")

    module = importlib.util.module_from_spec(spec)
    # Important for dataclasses and dynamically loaded modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



def resolve_p0_path(user_path: Path | None) -> Path:
    if user_path is not None:
        path = user_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Specified P0 does not exist: {path}")
        return path

    for candidate in (DEFAULT_P0_PATH, ALTERNATE_P0_PATH):
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "formal_p0.csv was not found in either expected formal-protocol directory. "
        "Pass the exact frozen file explicitly with --p0_path."
    )



def load_fixed_p0(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "smiles" not in frame.columns:
        raise ValueError(f"P0 file has no 'smiles' column: {path}")

    smiles = frame["smiles"].astype(str).tolist()

    if len(smiles) != POP_SIZE:
        raise ValueError(
            f"Expected exactly {POP_SIZE} molecules in fixed P0, got {len(smiles)}."
        )

    if len(set(smiles)) != POP_SIZE:
        raise ValueError("Fixed P0 contains duplicated SMILES.")

    if any(not item.strip() for item in smiles):
        raise ValueError("Fixed P0 contains an empty SMILES string.")

    return smiles


# ============================================================
# 2. Run one quota/seed condition
# ============================================================


def run_one(
    *,
    quota: int,
    seed: int,
    smoke: bool,
    p0_path: Path,
    freeze_dir: Path,
    clear_incomplete: bool,
) -> None:
    allowed_quotas = SMOKE_QUOTAS if smoke else FORMAL_QUOTAS

    if quota not in allowed_quotas:
        if smoke:
            raise ValueError(f"Smoke quota must be one of {SMOKE_QUOTAS}.")
        raise ValueError(
            f"Full sensitivity quota must be one of {FORMAL_QUOTAS}. "
            "q_max=10 must be reused from the prespecified formal Step-28 runs."
        )

    if seed not in FORMAL_SEEDS:
        raise ValueError(f"seed must be one of {FORMAL_SEEDS}.")

    if not ENGINE_PATH.exists():
        raise FileNotFoundError(
            f"Place 29_0_tsq_quota_sensitivity_engine.py next to this runner: {ENGINE_PATH}"
        )

    if not freeze_dir.exists():
        raise FileNotFoundError(f"Frozen DASR model directory does not exist: {freeze_dir}")

    generations = SMOKE_GENERATIONS if smoke else FORMAL_GENERATIONS
    output_root = SMOKE_OUTPUT_ROOT if smoke else FORMAL_OUTPUT_ROOT

    if smoke:
        task_id = f"q{quota}_seed{seed}"
    else:
        task_id = f"tsq_sensitivity_q{quota}_seed{seed}"

    final_dir = output_root / task_id
    running_dir = output_root / f"{task_id}.__running__"

    output_root.mkdir(parents=True, exist_ok=True)

    if final_dir.exists():
        raise FileExistsError(
            f"Completed/output directory already exists and will not be overwritten: {final_dir}"
        )

    if running_dir.exists():
        if clear_incomplete:
            shutil.rmtree(running_dir)
            print(f"Removed incomplete directory: {running_dir}")
        else:
            raise FileExistsError(
                f"Incomplete run directory exists: {running_dir}\n"
                "Inspect it first. If it is only a failed technical run, rerun with --clear_incomplete."
            )

    initial_smiles = load_fixed_p0(p0_path)
    engine = load_module(ENGINE_PATH.stem, ENGINE_PATH)

    # Fail early with a clear message if an incompatible engine file is present.
    required_fields = {
        "variant_name", "allocation_policy", "tsq_enabled", "tsq_quota",
        "local_oracle_budget", "local_max_depth", "raw_candidate_pool_size",
        "oracle_child_budget", "top_ranked_count", "random_exploration_count",
    }
    available_fields = set(getattr(engine.DASRTSQEvolutionConfig, "__dataclass_fields__", {}))
    missing_fields = sorted(required_fields - available_fields)
    if missing_fields:
        raise RuntimeError(
            "29_0/29_1 version mismatch. The engine is missing dataclass fields: "
            + ", ".join(missing_fields)
        )

    running_dir.mkdir(parents=True, exist_ok=False)

    preflight = {
        "task_id": task_id,
        "run_mode": "smoke_5_generation" if smoke else "full_50_generation",
        "scientific_role": (
            "engineering smoke test only"
            if smoke
            else "post-formal TSQ quota sensitivity"
        ),
        "quota": int(quota),
        "seed": int(seed),
        "generations": int(generations),
        "p0_path": str(p0_path),
        "p0_sha256": sha256_file(p0_path),
        "engine_path": str(ENGINE_PATH),
        "engine_sha256": sha256_file(ENGINE_PATH),
        "freeze_dir": str(freeze_dir),
        "fixed_settings": {
            "population": POP_SIZE,
            "offspring_multiplier": OFFSPRING_MULTIPLIER,
            "tournament_k": TOURNAMENT_K,
            "K": K_CANDIDATES,
            "B": B_ORACLE_CHILDREN,
            "allocation": "Top-6 + Random-2",
            "local_oracle_budget": LOCAL_ORACLE_BUDGET,
            "local_max_depth": LOCAL_MAX_DEPTH,
            "n_jobs": N_JOBS,
        },
        "result_use_policy": (
            "Do not use smoke results to reselect q_max or support final scientific conclusions."
            if smoke
            else
            "q_max=10 remains the prespecified formal reference; q_max=5 and q_max=20 are sensitivity conditions only."
        ),
    }
    write_json(running_dir / "SENSITIVITY_PREFLIGHT.json", preflight)

    config = engine.DASRTSQEvolutionConfig(
        # Outer evolutionary loop
        pop_size=POP_SIZE,
        generations=generations,
        offspring_multiplier=OFFSPRING_MULTIPLIER,
        tournament_k=TOURNAMENT_K,
        max_per_scaffold=quota,
        random_seed=seed,
        output_dir=str(running_dir),
        experiment_name=(
            "tsq_quota_sensitivity_smoke" if smoke else "tsq_quota_sensitivity"
        ),
        run_id=task_id,
        n_jobs=N_JOBS,

        # Sensitivity factor
        variant_name=f"dasr_tsq_q{quota}",
        allocation_policy="listwise_ranker_6plus2",
        tsq_enabled=True,
        tsq_quota=quota,

        # Frozen local-search settings
        local_oracle_budget=LOCAL_ORACLE_BUDGET,
        local_max_depth=LOCAL_MAX_DEPTH,
        raw_candidate_pool_size=K_CANDIDATES,
        oracle_child_budget=B_ORACLE_CHILDREN,
        top_ranked_count=TOP_RANKED_COUNT,
        random_exploration_count=RANDOM_EXPLORATION_COUNT,
        max_tree_policy_steps=MAX_TREE_POLICY_STEPS,
        max_stagnant_steps=MAX_STAGNANT_STEPS,

        # Frozen model location / execution
        freeze_dir=str(freeze_dir),
        device="cpu",
        save_local_evaluated_smiles=False,
    )

    # This validates that ONLY q_max changes among scientific parameters.
    config.validate_step28()

    evolution = engine.DASRTSQNSGA2Evolution(
        initial_population_smiles=initial_smiles,
        evolution_config=config,
    )

    try:
        history_df, final_df = evolution.run()

        if len(history_df) != generations:
            raise RuntimeError(
                f"Expected {generations} history rows, got {len(history_df)}."
            )

        if len(final_df) != POP_SIZE:
            raise RuntimeError(
                f"Expected final population size {POP_SIZE}, got {len(final_df)}."
            )

        completion = {
            "completed": True,
            "task_id": task_id,
            "run_mode": "smoke_5_generation" if smoke else "full_50_generation",
            "quota": int(quota),
            "seed": int(seed),
            "generations": int(generations),
            "p0_sha256": sha256_file(p0_path),
            "engine_sha256": sha256_file(ENGINE_PATH),
            "scientific_role": (
                "engineering smoke test only"
                if smoke
                else "post-formal TSQ quota sensitivity"
            ),
            "result_use_policy": (
                "engineering_check_only"
                if smoke
                else "post_formal_sensitivity_analysis"
            ),
        }
        write_json(running_dir / "SENSITIVITY_TASK_COMPLETED.json", completion)

        running_dir.replace(final_dir)

    except Exception:
        # Keep the .__running__ directory as technical evidence for inspection.
        raise

    print("=" * 78)
    print(f"Completed: {task_id}")
    print(f"Mode: {'SMOKE (G=5)' if smoke else 'FULL SENSITIVITY (G=50)'}")
    print(f"q_max={quota} | seed={seed}")
    print(f"Output: {final_dir}")
    print("=" * 78)


# ============================================================
# 3. CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one TSQ q_max sensitivity condition."
    )

    parser.add_argument(
        "--quota",
        type=int,
        required=True,
        help=(
            "TSQ quota. Smoke mode permits 5,10,20. Full mode permits 5,20; "
            "q=10 is reused from formal Step 28."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        choices=FORMAL_SEEDS,
        help="Stochastic seed; default=42.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run G=5 engineering smoke test and write to "
            "05_results/tsq_quota_sensitivity_smoke/."
        ),
    )

    parser.add_argument(
        "--p0_path",
        type=Path,
        default=None,
        help="Optional explicit path to the frozen formal_p0.csv.",
    )

    parser.add_argument(
        "--freeze_dir",
        type=Path,
        default=DEFAULT_FREEZE_DIR,
        help="Frozen DASR ranker/model directory.",
    )

    parser.add_argument(
        "--clear_incomplete",
        action="store_true",
        help=(
            "Delete only an existing .__running__ directory for this task. "
            "Completed output directories are never overwritten."
        ),
    )

    return parser



def main() -> None:
    args = build_parser().parse_args()

    p0_path = resolve_p0_path(args.p0_path)
    freeze_dir = args.freeze_dir.resolve()

    run_one(
        quota=int(args.quota),
        seed=int(args.seed),
        smoke=bool(args.smoke),
        p0_path=p0_path,
        freeze_dir=freeze_dir,
        clear_incomplete=bool(args.clear_incomplete),
    )


if __name__ == "__main__":
    main()