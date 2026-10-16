"""
29_0_tsq_quota_sensitivity_engine.py
====================================
Post-formal sensitivity engine for the TSQ scaffold quota q_max.

本文件用于 TSQ 配额敏感性分析。它从正式 Step 28 DASR–TSQ 端到端引擎派生，
但不修改原正式实验代码、协议或 q_max=10 的正式结果。

Sensitivity design
------------------
Only one scientific parameter is allowed to vary:

    TSQ quota q_max: 5, 10, or 20

All other settings are fixed to the formal DASR–TSQ configuration:

    allocation policy: Listwise DASR Top-6 + Random-2
    population size:   N=200
    generations:       G=50 for the full sensitivity study
                       G=5 only in explicit smoke mode
    tournament size:   5
    K:                 32
    B:                 8
    local oracle budget: 101 unique calls including the root
    local max depth:   5
    legacy P_success:  disabled

q_max=10 is the prespecified formal reference. q_max=5 and q_max=20 are
post-formal sensitivity conditions and must not be used to retune or reselect
the primary method. G=5 is permitted only for engineering
checks; those outputs must be stored separately and must not be interpreted
as the sensitivity-analysis result.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import importlib.metadata as importlib_metadata
import json
import math
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = PROJECT_ROOT / "02_code" / "core"
OUTER_PATH = CORE_DIR / "14_tsq_nsga2_evolution.py"
DASR_DIR = CORE_DIR / "27_dasr_mcts"


def _load(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not path.exists():
        raise FileNotFoundError(f"Required module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_outer = _load("step14_tsq_nsga2_for_tsq_sensitivity", OUTER_PATH)
_dasr = _load("step27_dasr_mcts_for_tsq_sensitivity", DASR_DIR / "27_1_dasr_mcts.py")
_rankers = _load(
    "step27_frozen_rankers_for_tsq_sensitivity",
    DASR_DIR / "27_0_frozen_sibling_rankers.py",
)

TSQNSGA2Config = _outer.TSQNSGA2Config
TSQNSGA2Evolution = _outer.TSQNSGA2Evolution
CandidateRecord = _outer.CandidateRecord
TSQMCTSConfig = _outer.TSQMCTSConfig
FrozenFitnessAdapter = _outer.FrozenFitnessAdapter
MoleculeMutator = _outer.MoleculeMutator
canonicalize_smiles = _outer.canonicalize_smiles
safe_float = _outer.safe_float
assign_nsga2_rank_and_crowding = _outer.assign_nsga2_rank_and_crowding
score_smiles_list = _outer.score_smiles_list
tsq_environmental_selection = _outer.tsq_environmental_selection

DASRMCTSConfig = _dasr.DASRMCTSConfig
DASRMCTS = _dasr.DASRMCTS
FrozenSiblingRankerBank = _rankers.FrozenSiblingRankerBank

TSQ_SENSITIVITY_VARIANTS: dict[str, dict[str, Any]] = {
    "dasr_tsq_q5": {
        "allocation_policy": "listwise_ranker_6plus2",
        "tsq_enabled": True,
        "tsq_quota": 5,
        "analysis_role": "post_formal_sensitivity",
        "scientific_role": "TSQ quota sensitivity: q_max=5",
    },
    "dasr_tsq_q10": {
        "allocation_policy": "listwise_ranker_6plus2",
        "tsq_enabled": True,
        "tsq_quota": 10,
        "analysis_role": "formal_reference",
        "scientific_role": "Prespecified formal reference: q_max=10",
    },
    "dasr_tsq_q20": {
        "allocation_policy": "listwise_ranker_6plus2",
        "tsq_enabled": True,
        "tsq_quota": 20,
        "analysis_role": "post_formal_sensitivity",
        "scientific_role": "TSQ quota sensitivity: q_max=20",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def software_environment() -> dict[str, Any]:
    """Return reproducibility metadata without importing optional packages."""
    packages: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "pandas",
        "rdkit",
        "torch",
        "scikit-learn",
        "joblib",
    ):
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
    }


def tier1_from_record(record: CandidateRecord) -> bool:
    score = record.score_record
    dual_min = safe_float(score.get("dual_min_pic50"), float("nan"))
    qed = safe_float(score.get("qed"), float("nan"))
    sa_score = safe_float(score.get("sa_score"), float("nan"))
    return bool(
        math.isfinite(dual_min)
        and math.isfinite(qed)
        and math.isfinite(sa_score)
        and dual_min >= 6.0
        and qed >= 0.50
        and sa_score <= 6.0
    )


class CountingFitnessAdapter:
    """Count requests, cache hits, and process-local model evaluations."""

    def __init__(self, base: Optional[Any] = None) -> None:
        self.base = base or FrozenFitnessAdapter()
        self.requests = 0
        self.model_evaluations = 0
        self.cache_hits = 0
        self._seen: set[str] = set()

    def score(self, smiles: str) -> dict[str, Any]:
        canonical = canonicalize_smiles(smiles)
        key = canonical if canonical is not None else f"__INVALID__::{smiles}"
        self.requests += 1
        if key in self._seen:
            self.cache_hits += 1
        else:
            self._seen.add(key)
            self.model_evaluations += 1
        return self.base.score(smiles)

    @property
    def unique_smiles(self) -> set[str]:
        return set(self._seen)

    def counters(self) -> dict[str, int]:
        return {
            "requests": int(self.requests),
            "model_evaluations": int(self.model_evaluations),
            "cache_hits": int(self.cache_hits),
            "unique_process_local_keys": int(len(self._seen)),
        }

    def metadata(self) -> dict[str, Any]:
        metadata = copy.deepcopy(self.base.metadata())
        metadata["tsq_sensitivity_counting"] = self.counters()
        return metadata


@dataclass
class DASRTSQEvolutionConfig(TSQNSGA2Config):
    """
    Outer evolution settings for the TSQ quota sensitivity experiment.

    Only tsq_quota may vary across q_max in {5, 10, 20}.
    All other scientific settings are fixed to the formal DASR–TSQ design.
    """

    variant_name: str = "dasr_tsq_q10"
    allocation_policy: str = "listwise_ranker_6plus2"
    tsq_enabled: bool = True
    tsq_quota: int = 10
    # Keep the inherited Step 14 quota metadata synchronized by default.
    max_per_scaffold: int = 10

    local_oracle_budget: int = 101
    local_max_depth: int = 5
    raw_candidate_pool_size: int = 32
    oracle_child_budget: int = 8
    top_ranked_count: int = 6
    random_exploration_count: int = 2
    max_tree_policy_steps: int = 3000
    max_stagnant_steps: int = 100

    freeze_dir: str = "06_paper/experiment_records/dasr_p0_safe_v2_test_lock"
    device: str = "cpu"
    save_local_evaluated_smiles: bool = False

    def validate_step28(self) -> None:
        """
        Keep the historical method name for compatibility with the inherited
        Step 28 engine, but validate the independent sensitivity design.
        """
        if self.variant_name not in TSQ_SENSITIVITY_VARIANTS:
            raise ValueError(
                f"Unknown TSQ sensitivity variant: {self.variant_name}"
            )

        expected = TSQ_SENSITIVITY_VARIANTS[self.variant_name]

        if self.allocation_policy != expected["allocation_policy"]:
            raise ValueError(
                "Sensitivity experiment must keep "
                "Listwise Top-6 + Random-2 unchanged."
            )

        if not bool(self.tsq_enabled):
            raise ValueError(
                "TSQ must remain enabled in the quota sensitivity experiment."
            )

        expected_quota = int(expected["tsq_quota"])
        if int(self.tsq_quota) != expected_quota:
            raise ValueError(
                f"Variant/quota mismatch: {self.variant_name} requires "
                f"q_max={expected_quota}, but received q_max={self.tsq_quota}."
            )

        if int(self.tsq_quota) not in {5, 10, 20}:
            raise ValueError(
                "Sensitivity quota must be one of {5, 10, 20}."
            )

        # Keep the inherited Step 14 quota field synchronized with tsq_quota.
        # The Step 29 runner should explicitly pass max_per_scaffold=tsq_quota.
        if int(self.max_per_scaffold) != int(self.tsq_quota):
            raise ValueError(
                "max_per_scaffold and tsq_quota must be identical in the "
                "sensitivity experiment."
            )

        if int(self.raw_candidate_pool_size) != 32:
            raise ValueError("Sensitivity study keeps K=32.")

        if int(self.oracle_child_budget) != 8:
            raise ValueError("Sensitivity study keeps B=8.")

        if (
            int(self.top_ranked_count) != 6
            or int(self.random_exploration_count) != 2
        ):
            raise ValueError(
                "Sensitivity study keeps Top-6 + Random-2."
            )

        if int(self.local_oracle_budget) != 101:
            raise ValueError(
                "Sensitivity study keeps local oracle budget=101."
            )

        if int(self.local_max_depth) != 5:
            raise ValueError(
                "Sensitivity study keeps local MCTS max depth=5."
            )

        if int(self.pop_size) != 200:
            raise ValueError(
                "Sensitivity study keeps population size N=200."
            )

        if int(self.generations) not in {5, 50}:
            raise ValueError(
                "TSQ quota sensitivity engine permits only G=5 (engineering smoke) "
                "or G=50 (full sensitivity)."
            )

        if float(self.offspring_multiplier) != 1.0:
            raise ValueError(
                "Sensitivity study keeps offspring_multiplier=1.0."
            )

        if int(self.tournament_k) != 5:
            raise ValueError(
                "Sensitivity study keeps tournament size=5."
            )

        if int(self.random_seed) not in {42, 43, 44, 45, 46}:
            raise ValueError(
                "Sensitivity study uses the same formal seeds: 42, 43, 44, 45, 46."
            )

        if int(self.n_jobs) < 1:
            raise ValueError("n_jobs must be >=1.")

# ---------------------------------------------------------------------------
# Worker-local frozen objects. They are loaded once per process.
# ---------------------------------------------------------------------------
_WORKER_RANKER: Any = None
_WORKER_FITNESS: Any = None
_WORKER_MUTATOR: Any = None


def _init_tsq_sensitivity_worker(freeze_dir: str, device: str) -> None:
    global _WORKER_RANKER, _WORKER_FITNESS, _WORKER_MUTATOR
    _WORKER_RANKER = FrozenSiblingRankerBank(Path(freeze_dir), device=device)
    _WORKER_FITNESS = CountingFitnessAdapter()
    _WORKER_MUTATOR = MoleculeMutator()


def _make_local_config(task: dict[str, Any]) -> DASRMCTSConfig:
    return DASRMCTSConfig(
        max_depth=int(task["local_max_depth"]),
        max_children_per_node=int(task["oracle_child_budget"]),
        mcts_iterations=int(task["max_tree_policy_steps"]),
        exploration_constant=math.sqrt(2.0),
        p_call=0.0,
        gamma=0.0,
        random_seed=int(task["random_seed"]),
        experiment_name="tsq_quota_sensitivity_inner_mcts",
        run_id=str(task["run_id"]),
        raw_candidate_pool_size=int(task["raw_candidate_pool_size"]),
        oracle_budget=int(task["local_oracle_budget"]),
        candidate_policy=str(task["allocation_policy"]),
        top_ranked_count=int(task["top_ranked_count"]),
        random_exploration_count=int(task["random_exploration_count"]),
        max_tree_policy_steps=int(task["max_tree_policy_steps"]),
        max_stagnant_steps=int(task["max_stagnant_steps"]),
    )


def _summarize_candidate_events(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "n_candidate_events": 0,
            "mean_raw_candidate_count": float("nan"),
            "max_raw_candidate_count": 0,
            "full_k_fraction": float("nan"),
            "more_than_b_fraction": float("nan"),
            "min_selected_candidate_count": 0,
            "max_selected_candidate_count": 0,
            "mean_selected_top_count": float("nan"),
            "mean_selected_random_count": float("nan"),
            "policy_count_violation_events": 0,
            "nonfinite_score_events": 0,
        }

    frame = pd.DataFrame(records)
    score_cols = [
        column for column in ("score_min", "score_mean", "score_max")
        if column in frame
    ]

    # Only events that actually called a learned ranker must contain finite scores.
    # A node with zero valid candidates has no ranker output by design, so its
    # score_min/score_mean/score_max fields are legitimately NaN and must not be
    # counted as a ranker failure. Random-8 also has no ranker scores by design.
    raw_counts = pd.to_numeric(
        frame["raw_candidate_count"],
        errors="raise",
    ).astype(int)
    scored_mask = (
        frame["candidate_policy"].astype(str).ne("random_8")
        & raw_counts.gt(0)
    )

    nonfinite = 0
    if bool(scored_mask.any()):
        if len(score_cols) != 3:
            nonfinite = int(scored_mask.sum())
        else:
            values = frame.loc[scored_mask, score_cols].to_numpy(
                dtype=np.float64
            )
            nonfinite = int(
                np.sum(~np.isfinite(values).all(axis=1))
            )

    violations = 0
    for row in frame.to_dict(orient="records"):
        policy = str(row["candidate_policy"])
        raw_count = int(row["raw_candidate_count"])
        selected_count = int(row["selected_candidate_count"])
        selected_top = int(row["selected_top_count"])
        selected_random = int(row["selected_random_count"])
        if selected_count > min(8, raw_count):
            violations += 1
            continue
        if policy == "random_8":
            if selected_top != 0 or selected_random != selected_count:
                violations += 1
        else:
            expected_top = min(6, selected_count)
            expected_random = selected_count - expected_top
            if selected_top != expected_top or selected_random != expected_random:
                violations += 1

    return {
        "n_candidate_events": int(len(frame)),
        "mean_raw_candidate_count": float(frame["raw_candidate_count"].mean()),
        "max_raw_candidate_count": int(frame["raw_candidate_count"].max()),
        "full_k_fraction": float(np.mean(frame["raw_candidate_count"] == 32)),
        "more_than_b_fraction": float(np.mean(frame["raw_candidate_count"] > 8)),
        "min_selected_candidate_count": int(frame["selected_candidate_count"].min()),
        "max_selected_candidate_count": int(frame["selected_candidate_count"].max()),
        "mean_selected_top_count": float(frame["selected_top_count"].mean()),
        "mean_selected_random_count": float(frame["selected_random_count"].mean()),
        "policy_count_violation_events": int(violations),
        "nonfinite_score_events": nonfinite,
    }


def _execute_local_task(
    task: dict[str, Any],
    ranker: FrozenSiblingRankerBank,
    fitness: CountingFitnessAdapter,
    mutator: MoleculeMutator,
) -> dict[str, Any]:
    before = fitness.counters()
    start = time.time()
    try:
        engine = DASRMCTS(
            config=_make_local_config(task),
            ranker_bank=ranker,
            fitness_adapter=fitness,
            mutator=mutator,
        )
        result = engine.search(str(task["parent_smiles"]))
        after = fitness.counters()
        child = canonicalize_smiles(result.best_node.smiles)
        if child is None:
            raise RuntimeError("DASR-MCTS returned an invalid best-node SMILES.")

        candidate_summary = _summarize_candidate_events(
            result.candidate_selection_records
        )
        summary = {
            "status": "ok",
            "error": "",
            "variant_name": task["variant_name"],
            "allocation_policy": task["allocation_policy"],
            "generation": int(task["generation"]),
            "child_index": int(task["child_index"]),
            "parent_smiles": str(task["parent_smiles"]),
            "child_smiles": child,
            "random_seed": int(task["random_seed"]),
            "run_id": str(task["run_id"]),
            "elapsed_seconds": float(time.time() - start),
            "local_oracle_budget": int(task["local_oracle_budget"]),
            "local_oracle_requests": int(result.summary["oracle_calls_used"]),
            "worker_model_evaluations": int(
                after["model_evaluations"] - before["model_evaluations"]
            ),
            "worker_cache_hits": int(after["cache_hits"] - before["cache_hits"]),
            "tree_policy_steps": int(result.summary["tree_policy_steps"]),
            "termination_reason": str(result.summary["termination_reason"]),
            "best_final_reward": float(result.summary["best_final_reward"]),
            "n_prior_gate_open_events": int(
                result.summary.get("n_prior_gate_open_events", 0)
            ),
            **candidate_summary,
        }
        evaluated_smiles = [
            str(row["smiles"]) for row in result.oracle_trajectory if row.get("smiles")
        ]
        return {
            "child_smiles": child,
            "summary": summary,
            "evaluated_smiles": evaluated_smiles,
        }
    except Exception as exc:
        after = fitness.counters()
        parent = str(task["parent_smiles"])
        summary = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "variant_name": task["variant_name"],
            "allocation_policy": task["allocation_policy"],
            "generation": int(task["generation"]),
            "child_index": int(task["child_index"]),
            "parent_smiles": parent,
            "child_smiles": parent,
            "random_seed": int(task["random_seed"]),
            "run_id": str(task["run_id"]),
            "elapsed_seconds": float(time.time() - start),
            "local_oracle_budget": int(task["local_oracle_budget"]),
            "local_oracle_requests": 0,
            "worker_model_evaluations": int(
                after["model_evaluations"] - before["model_evaluations"]
            ),
            "worker_cache_hits": int(after["cache_hits"] - before["cache_hits"]),
            "tree_policy_steps": 0,
            "termination_reason": "technical_error_parent_fallback",
            "best_final_reward": float("nan"),
            "n_prior_gate_open_events": 0,
            "n_candidate_events": 0,
            "mean_raw_candidate_count": float("nan"),
            "max_raw_candidate_count": 0,
            "full_k_fraction": float("nan"),
            "more_than_b_fraction": float("nan"),
            "min_selected_candidate_count": 0,
            "max_selected_candidate_count": 0,
            "mean_selected_top_count": float("nan"),
            "mean_selected_random_count": float("nan"),
            "policy_count_violation_events": 0,
            "nonfinite_score_events": 0,
        }
        return {"child_smiles": parent, "summary": summary, "evaluated_smiles": []}


def _run_tsq_sensitivity_worker(task: dict[str, Any]) -> dict[str, Any]:
    global _WORKER_RANKER, _WORKER_FITNESS, _WORKER_MUTATOR
    if _WORKER_RANKER is None or _WORKER_FITNESS is None or _WORKER_MUTATOR is None:
        raise RuntimeError("TSQ sensitivity worker was not initialized.")
    return _execute_local_task(
        task,
        ranker=_WORKER_RANKER,
        fitness=_WORKER_FITNESS,
        mutator=_WORKER_MUTATOR,
    )


class DASRTSQNSGA2Evolution(TSQNSGA2Evolution):
    """TSQ-NSGA-II outer loop with frozen Step 27 DASR-MCTS for quota sensitivity."""

    def __init__(
        self,
        initial_population_smiles: list[str],
        evolution_config: DASRTSQEvolutionConfig,
    ) -> None:
        evolution_config.validate_step28()

        # The base MCTS config is retained only for compatibility with Step 14
        # metadata. Actual local searches use DASRMCTSConfig constructed above.
        compatibility_config = TSQMCTSConfig(
            max_depth=evolution_config.local_max_depth,
            max_children_per_node=evolution_config.oracle_child_budget,
            mcts_iterations=evolution_config.max_tree_policy_steps,
            exploration_constant=math.sqrt(2.0),
            p_call=0.0,
            gamma=0.0,
            random_seed=evolution_config.random_seed,
            experiment_name="tsq_quota_sensitivity_compatibility_config",
            run_id=evolution_config.run_id,
        )
        super().__init__(
            initial_population_smiles=initial_population_smiles,
            evolution_config=evolution_config,
            mcts_config=compatibility_config,
        )
        self.config: DASRTSQEvolutionConfig = evolution_config
        self.fitness = CountingFitnessAdapter()
        self.mutator = MoleculeMutator()
        self.ranker_bank = FrozenSiblingRankerBank(
            Path(evolution_config.freeze_dir).resolve(),
            device=evolution_config.device,
        )
        self._serial_local_fitness = CountingFitnessAdapter()
        self._serial_local_mutator = MoleculeMutator()

        self._local_summary_rows: list[dict[str, Any]] = []
        self._all_local_evaluated_smiles: set[str] = set()
        self._executor: Optional[ProcessPoolExecutor] = None

    def _task_payload(
        self,
        parent_smiles: str,
        generation: int,
        child_index: int,
    ) -> dict[str, Any]:
        seed = int(self.config.random_seed) + int(generation) * 10000 + int(child_index)
        return {
            "variant_name": self.config.variant_name,
            "allocation_policy": self.config.allocation_policy,
            "parent_smiles": parent_smiles,
            "generation": generation,
            "child_index": child_index,
            "random_seed": seed,
            "run_id": (
                f"{self.config.run_id}_gen{generation:03d}_child{child_index:04d}"
            ),
            "local_oracle_budget": self.config.local_oracle_budget,
            "local_max_depth": self.config.local_max_depth,
            "raw_candidate_pool_size": self.config.raw_candidate_pool_size,
            "oracle_child_budget": self.config.oracle_child_budget,
            "top_ranked_count": self.config.top_ranked_count,
            "random_exploration_count": self.config.random_exploration_count,
            "max_tree_policy_steps": self.config.max_tree_policy_steps,
            "max_stagnant_steps": self.config.max_stagnant_steps,
        }

    def _generate_offspring(
        self,
        population_records: list[CandidateRecord],
        generation: int,
    ) -> list[str]:
        target_count = int(
            math.ceil(self.config.pop_size * self.config.offspring_multiplier)
        )
        tasks: list[dict[str, Any]] = []
        for child_index in range(target_count):
            parent = self._tournament_select_parent(population_records)
            tasks.append(
                self._task_payload(parent.smiles, generation, child_index)
            )

        if self.config.n_jobs <= 1:
            results = [
                _execute_local_task(
                    task,
                    ranker=self.ranker_bank,
                    fitness=self._serial_local_fitness,
                    mutator=self._serial_local_mutator,
                )
                for task in tasks
            ]
        else:
            if self._executor is None:
                raise RuntimeError("TSQ sensitivity process pool is not initialized.")
            results = list(self._executor.map(_run_tsq_sensitivity_worker, tasks, chunksize=1))

        children: list[str] = []
        for result in results:
            children.append(str(result["child_smiles"]))
            self._local_summary_rows.append(dict(result["summary"]))
            self._all_local_evaluated_smiles.update(
                canonical
                for smiles in result.get("evaluated_smiles", [])
                if (canonical := canonicalize_smiles(smiles)) is not None
            )
        return children

    def _generation_extra_summary(
        self,
        generation: int,
        selected_records: list[CandidateRecord],
        outer_before: dict[str, int],
    ) -> dict[str, Any]:
        final_rewards = sorted(
            [float(record.final_reward) for record in selected_records], reverse=True
        )
        top10 = final_rewards[: min(10, len(final_rewards))]
        finite_duals = sorted(
            [
                value
                for record in selected_records
                if math.isfinite(
                    value := safe_float(
                        record.score_record.get("dual_min_pic50"),
                        float("nan"),
                    )
                )
            ],
            reverse=True,
        )
        local = pd.DataFrame(
            [row for row in self._local_summary_rows if int(row["generation"]) == generation]
        )
        outer_after = self.fitness.counters()

        local_requests = int(local["local_oracle_requests"].sum()) if not local.empty else 0
        local_evaluations = (
            int(local["worker_model_evaluations"].sum()) if not local.empty else 0
        )
        local_hits = int(local["worker_cache_hits"].sum()) if not local.empty else 0
        local_errors = int(np.sum(local["status"] != "ok")) if not local.empty else 0

        outer_requests_delta = int(outer_after["requests"] - outer_before["requests"])
        outer_evaluations_delta = int(
            outer_after["model_evaluations"] - outer_before["model_evaluations"]
        )
        outer_hits_delta = int(outer_after["cache_hits"] - outer_before["cache_hits"])
        global_unique = len(
            self._all_local_evaluated_smiles
            | {
                key
                for key in self.fitness.unique_smiles
                if not key.startswith("__INVALID__::")
            }
        )

        return {
            "top10_mean_final_reward": float(np.mean(top10)) if top10 else float("nan"),
            "top10_mean_dual_min_pic50": (
                float(np.mean(finite_duals[: min(10, len(finite_duals))]))
                if finite_duals
                else float("nan")
            ),
            "tier1_count": int(sum(tier1_from_record(record) for record in selected_records)),
            "tier1_rate": float(
                np.mean([tier1_from_record(record) for record in selected_records])
            ),
            "no_scaffold_count": int(
                sum(record.scaffold == "no_scaffold" for record in selected_records)
            ),
            "no_scaffold_rate": float(
                np.mean([record.scaffold == "no_scaffold" for record in selected_records])
            ),
            "scaffold_failed_count": int(
                sum(record.scaffold == "scaffold_failed" for record in selected_records)
            ),
            "scaffold_failed_rate": float(
                np.mean([record.scaffold == "scaffold_failed" for record in selected_records])
            ),
            "invalid_scaffold_token_count": int(
                sum(record.scaffold == "invalid_smiles" for record in selected_records)
            ),
            "local_mcts_searches": int(len(local)),
            "local_mcts_errors": local_errors,
            "local_oracle_requests_generation": local_requests,
            "local_model_evaluations_generation": local_evaluations,
            "local_cache_hits_generation": local_hits,
            "outer_oracle_requests_generation": outer_requests_delta,
            "outer_model_evaluations_generation": outer_evaluations_delta,
            "outer_cache_hits_generation": outer_hits_delta,
            "total_oracle_requests_generation": local_requests + outer_requests_delta,
            "model_evaluations_executed_generation": (
                local_evaluations + outer_evaluations_delta
            ),
            "total_oracle_requests_cumulative": int(
                sum(row["local_oracle_requests"] for row in self._local_summary_rows)
                + outer_after["requests"]
            ),
            "model_evaluations_executed_cumulative": int(
                sum(row["worker_model_evaluations"] for row in self._local_summary_rows)
                + outer_after["model_evaluations"]
            ),
            "unique_structures_scored_globally": int(global_unique),
        }

    def _write_sensitivity_outputs(self, history_df: pd.DataFrame) -> dict[str, Any]:
        local_df = pd.DataFrame(self._local_summary_rows)
        local_df.to_csv(self.output_dir / "local_mcts_summary.csv", index=False)

        errors_df = (
            local_df[local_df["status"] != "ok"].copy()
            if not local_df.empty
            else pd.DataFrame(columns=["status", "error"])
        )
        errors_df.to_csv(self.output_dir / "local_mcts_errors.csv", index=False)

        critical_errors: list[str] = []
        warnings: list[str] = []
        expected_searches = int(
            math.ceil(self.config.pop_size * self.config.offspring_multiplier)
            * self.config.generations
        )
        if len(local_df) != expected_searches:
            critical_errors.append(
                f"local_search_count={len(local_df)} expected={expected_searches}"
            )
        if not errors_df.empty:
            critical_errors.append(f"local_mcts_errors={len(errors_df)}")
        if not local_df.empty:
            if int(local_df["local_oracle_requests"].max()) > self.config.local_oracle_budget:
                critical_errors.append("local_oracle_budget_exceeded")
            if int(local_df["max_raw_candidate_count"].max()) > 32:
                critical_errors.append("raw_candidate_pool_K32_exceeded")
            if int(local_df["max_selected_candidate_count"].max()) > 8:
                critical_errors.append("candidate_budget_B8_exceeded")
            if int(local_df["policy_count_violation_events"].sum()) != 0:
                critical_errors.append("candidate_allocation_policy_count_violation")
            if int(local_df["n_prior_gate_open_events"].sum()) != 0:
                critical_errors.append("legacy_p_success_prior_was_activated")
            if int(local_df["nonfinite_score_events"].sum()) != 0:
                critical_errors.append("nonfinite_ranker_scores_detected")

            exact_budget_rate = float(
                np.mean(
                    local_df["local_oracle_requests"]
                    == int(self.config.local_oracle_budget)
                )
            )
            if exact_budget_rate < 0.95:
                warnings.append(
                    f"Only {exact_budget_rate:.3f} of local searches used the exact oracle budget."
                )
        else:
            exact_budget_rate = 0.0

        if self.config.tsq_enabled:
            relaxed_total = int(history_df["quota_relaxed_count"].sum())
            if relaxed_total > 0:
                warnings.append(f"TSQ quota fallback selected {relaxed_total} individuals in total.")

        no_scaffold_max = int(history_df["no_scaffold_count"].max()) if not history_df.empty else 0
        scaffold_failed_total = int(history_df["scaffold_failed_count"].sum()) if not history_df.empty else 0
        invalid_scaffold_total = int(history_df["invalid_scaffold_token_count"].sum()) if not history_df.empty else 0
        if no_scaffold_max > 0:
            warnings.append(
                f"Acyclic no_scaffold category reached {no_scaffold_max} molecules in one generation; "
                "report this category explicitly in diversity analyses."
            )
        if scaffold_failed_total > 0 or invalid_scaffold_total > 0:
            critical_errors.append("scaffold_assignment_failures_detected")

        report = {
            "passed": len(critical_errors) == 0,
            "critical_errors": critical_errors,
            "warnings": warnings,
            "variant_name": self.config.variant_name,
            "allocation_policy": self.config.allocation_policy,
            "tsq_enabled": self.config.tsq_enabled,
            "tsq_quota": self.config.tsq_quota if self.config.tsq_enabled else None,
            "expected_local_searches": expected_searches,
            "observed_local_searches": int(len(local_df)),
            "exact_local_oracle_budget_rate": exact_budget_rate,
            "candidate_events_with_full_K32_rate": (
                float(local_df["full_k_fraction"].mean()) if not local_df.empty else float("nan")
            ),
            "candidate_events_with_more_than_B8_rate": (
                float(local_df["more_than_b_fraction"].mean()) if not local_df.empty else float("nan")
            ),
            "outer_fitness_counters": self.fitness.counters(),
            "serial_local_fitness_counters": self._serial_local_fitness.counters(),
            "ranker_metadata": self.ranker_bank.metadata(),
            "special_scaffold_audit": {
                "max_no_scaffold_count_per_generation": no_scaffold_max,
                "total_scaffold_failed_count_across_generations": scaffold_failed_total,
                "total_invalid_scaffold_token_count_across_generations": invalid_scaffold_total,
            },
        }
        write_json(self.output_dir / "integration_validation.json", report)
        return report

    def _run_main_loop(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        history_rows: list[dict[str, Any]] = []

        initial_records = self._score_current_population()
        assign_nsga2_rank_and_crowding(initial_records)
        self._save_population("initial", initial_records)

        for generation in range(self.config.generations):
            print(
                f"[{self.config.variant_name}] generation={generation} "
                f"population={len(self.population_smiles)}",
                flush=True,
            )
            outer_before = self.fitness.counters()
            population_records = self._score_current_population()
            assign_nsga2_rank_and_crowding(population_records)

            offspring_smiles = self._generate_offspring(
                population_records=population_records,
                generation=generation,
            )
            combined_records = score_smiles_list(
                self.population_smiles + offspring_smiles,
                self.fitness,
            )

            quota = self.config.tsq_quota if self.config.tsq_enabled else self.config.pop_size
            selected_records, diagnostics = tsq_environmental_selection(
                records=combined_records,
                pop_size=self.config.pop_size,
                max_per_scaffold=quota,
            )
            if len(selected_records) != self.config.pop_size:
                raise RuntimeError(
                    f"Generation {generation}: selected {len(selected_records)} "
                    f"instead of {self.config.pop_size}."
                )
            self.population_smiles = [record.smiles for record in selected_records]

            summary = self._summarize_generation(
                generation=generation,
                selected_records=selected_records,
                diagnostics=diagnostics,
            )
            summary.update(
                self._generation_extra_summary(
                    generation=generation,
                    selected_records=selected_records,
                    outer_before=outer_before,
                )
            )
            summary.update(
                {
                    "variant_name": self.config.variant_name,
                    "allocation_policy": self.config.allocation_policy,
                    "tsq_enabled": bool(self.config.tsq_enabled),
                    "tsq_quota": quota,
                }
            )
            history_rows.append(summary)
            self._save_population(f"gen{generation:03d}", selected_records)

            pd.DataFrame(history_rows).to_csv(
                self.output_dir / "history.csv", index=False
            )
            pd.DataFrame(self._local_summary_rows).to_csv(
                self.output_dir / "local_mcts_summary.csv", index=False
            )
            print(
                f"  meanF={summary['mean_final_reward']:.4f} | "
                f"top10F={summary['top10_mean_final_reward']:.4f} | "
                f"scaffolds={summary['n_unique_scaffolds']} | "
                f"tier1={summary['tier1_count']} | "
                f"local_oracle={summary['local_oracle_requests_generation']}",
                flush=True,
            )

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(self.output_dir / "history.csv", index=False)
        final_path = self.output_dir / f"population_gen{self.config.generations - 1:03d}.csv"
        final_df = pd.read_csv(final_path)
        final_df.to_csv(self.output_dir / "final_population.csv", index=False)

        metadata = {
            "scientific_status": (
                "Engineering smoke test for TSQ quota sensitivity"
                if int(self.config.generations) == 5
                else "Post-formal TSQ quota sensitivity analysis"
            ),
            "run_mode": "smoke_5_generation" if int(self.config.generations) == 5 else "full_50_generation",
            "sensitivity_status": (
                "Smoke run for code/behavior checking only; it must not be used "
                "to reselect q_max or support final scientific conclusions."
                if int(self.config.generations) == 5
                else
                "q_max=10 is the prespecified formal reference; q_max=5 and "
                "q_max=20 are post-formal sensitivity conditions and are not "
                "used for parameter reselection."
            ),
            "result_use_policy": (
                "engineering_check_only" if int(self.config.generations) == 5
                else "post_formal_sensitivity_analysis"
            ),
            "evolution_config": asdict(self.config),
            "variant_definition": TSQ_SENSITIVITY_VARIANTS[self.config.variant_name],
            "frozen_ranker_metadata": self.ranker_bank.metadata(),
            "outer_fitness_metadata": self.fitness.metadata(),
            "software_environment": software_environment(),
            "code_hashes": {
                "step14_outer": sha256_file(OUTER_PATH),
                "step27_ranker_loader": sha256_file(
                    DASR_DIR / "27_0_frozen_sibling_rankers.py"
                ),
                "step27_dasr_mcts": sha256_file(
                    DASR_DIR / "27_1_dasr_mcts.py"
                ),
                "tsq_sensitivity_engine": sha256_file(Path(__file__).resolve()),
            },
            "isolation_note": (
                "Only the TSQ scaffold quota is varied. P0, QSAR models, "
                "DASR ranker, molecular mutation rules, NSGA-II objectives, "
                f"population size N=200, generations G={self.config.generations}, tournament size=5, "
                "K=32, B=8, Top-6 + Random-2, local oracle budget=101, "
                "MCTS max depth=5, and stochastic seeds are unchanged."
            ),
        }
        write_json(self.output_dir / "run_metadata.json", metadata)
        validation = self._write_sensitivity_outputs(history_df)
        if not validation["passed"]:
            raise RuntimeError(
                "TSQ sensitivity integration validation failed: "
                + "; ".join(validation["critical_errors"])
            )
        return history_df, final_df

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._executor = None
        if self.config.n_jobs <= 1:
            return self._run_main_loop()

        with ProcessPoolExecutor(
            max_workers=self.config.n_jobs,
            initializer=_init_tsq_sensitivity_worker,
            initargs=(str(Path(self.config.freeze_dir).resolve()), self.config.device),
        ) as executor:
            self._executor = executor
            try:
                return self._run_main_loop()
            finally:
                self._executor = None


__all__ = [
    "DASRTSQEvolutionConfig",
    "DASRTSQNSGA2Evolution",
    "TSQ_SENSITIVITY_VARIANTS",
    "CountingFitnessAdapter",
]
