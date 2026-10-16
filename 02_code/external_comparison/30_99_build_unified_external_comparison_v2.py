from __future__ import annotations

"""Build the unified five-method external benchmark analysis (v2).

Methods:
    DASR-TSQ
    GraphGA
    MARS
    MGP (official/native NSGA-II baseline)
    REINVENT4-Mol2Mol

This script is analysis-only.  It does not run or modify any optimizer.

Important comparison rule
-------------------------
Native final populations differ in size (MGP=100; the other four=200).
Therefore:
  * reward and hypervolume metrics are used directly;
  * Tier-1, unique-scaffold and max-occupancy comparisons use RATES as the
    cross-method primary quantities;
  * raw counts are retained for provenance/SI.

Statistics:
  * formal paired seeds 42-46;
  * arithmetic mean + sample SD;
  * paired differences are comparator - DASR-TSQ;
  * percentile bootstrap 95% CI, 20,000 resamples;
  * exact two-sided sign-flip p-value is descriptive only;
  * with n=5 paired seeds, do NOT make a conventional exact p<0.05 claim.

Outputs are written to a new analysis directory and existing benchmark outputs
are never modified.
"""

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\pycharm\project\tsq_mcts_rebuild").resolve()

LEGACY_ANALYSIS_DIR = (
    PROJECT_ROOT / "05_results" / "external_comparison"
    / "matched_step28_v3_1" / "analysis"
)
LEGACY_RUN_METRICS = LEGACY_ANALYSIS_DIR / "external_run_metrics.csv"
LEGACY_COMPLETED = LEGACY_ANALYSIS_DIR / "EXTERNAL_ANALYSIS_COMPLETED.json"

MGP_ROOT = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_mgp_formal"
)

REINVENT_ROOT = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_reinvent4_formal"
)
REINVENT_PER_SEED = REINVENT_ROOT / "REINVENT4_5SEED_PER_SEED_METRICS.csv"
REINVENT_AGG_DONE = REINVENT_ROOT / "REINVENT4_5SEED_AGGREGATE_TASK_COMPLETED.json"

OUT_DIR = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_unified_5method_analysis_v2"
)

SEEDS = [42, 43, 44, 45, 46]

DISPLAY = {
    "dasr_tsq": "DASR-TSQ",
    "graphga": "GraphGA",
    "mars": "MARS",
    "mgp": "MGP",
    "reinvent4": "REINVENT4",
}

METHOD_ORDER = ["dasr_tsq", "graphga", "mars", "mgp", "reinvent4"]

# Higher is favorable except max occupancy rate and resource consumption.
PAIRED_METRICS = {
    "top10_mean_final_reward": +1,
    "hypervolume_4d": +1,
    "tier1_rate": +1,
    "unique_scaffold_rate": +1,
    "mean_final_reward": +1,
    "pareto_front_rate": +1,
    "max_scaffold_occupancy_rate": -1,
}

SUMMARY_METRICS = [
    "mean_final_reward",
    "top10_mean_final_reward",
    "max_final_reward",
    "hypervolume_4d",
    "tier1_rate",
    "unique_scaffold_rate",
    "max_scaffold_occupancy_rate",
    "pareto_front_rate",
    "actual_requests",
    "model_evaluations",
    "cache_hits",
    "unique_valid_evaluations",
    # Raw counts retained for SI/provenance:
    "tier1_count",
    "unique_scaffolds",
    "max_scaffold_occupancy",
    "pareto_front_size",
]


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap_ci(values: np.ndarray, seed: int, n_resamples: int = 20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values, size=(n_resamples, len(values)), replace=True
    ).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    vals = []
    for mask in range(1 << len(values)):
        signs = np.array(
            [1.0 if (mask >> i) & 1 else -1.0 for i in range(len(values))]
        )
        vals.append(abs(float(np.mean(values * signs))))
    return float(np.mean(np.asarray(vals) >= observed - 1e-15))


def normalize_row(
    *,
    method: str,
    seed: int,
    population_size: int,
    mean_final_reward,
    top10_mean_final_reward,
    max_final_reward,
    hypervolume_4d,
    tier1_count,
    unique_scaffolds,
    max_scaffold_occupancy,
    pareto_front_size,
    actual_requests,
    model_evaluations,
    cache_hits,
    unique_valid_evaluations,
    source: str,
) -> dict:
    n = int(population_size)
    require(n > 0, f"{method} seed {seed}: invalid population size")
    return {
        "method": method,
        "display_name": DISPLAY[method],
        "seed": int(seed),
        "native_population_size": n,
        "mean_final_reward": float(mean_final_reward),
        "top10_mean_final_reward": float(top10_mean_final_reward),
        "max_final_reward": float(max_final_reward),
        "hypervolume_4d": float(hypervolume_4d),
        "tier1_count": float(tier1_count),
        "tier1_rate": float(tier1_count) / n,
        "unique_scaffolds": float(unique_scaffolds),
        "unique_scaffold_rate": float(unique_scaffolds) / n,
        "max_scaffold_occupancy": float(max_scaffold_occupancy),
        "max_scaffold_occupancy_rate": float(max_scaffold_occupancy) / n,
        "pareto_front_size": float(pareto_front_size),
        "pareto_front_rate": float(pareto_front_size) / n,
        "actual_requests": float(actual_requests),
        "model_evaluations": float(model_evaluations),
        "cache_hits": float(cache_hits),
        "unique_valid_evaluations": float(unique_valid_evaluations),
        "source": source,
    }


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reconstruct_legacy_three_methods_from_frozen_raw() -> list[dict]:
    """Read the already-completed DASR-TSQ/GraphGA/MARS formal runs directly.

    This is a read-only fallback for projects where the historical aggregate
    analysis directory was later cleaned up.  It reuses the frozen v3.1
    analysis loaders, including their hash/protocol checks, and DOES NOT run
    any optimizer or write into the historical result tree.
    """
    code_dir = PROJECT_ROOT / "02_code" / "external_comparison"
    common_path = code_dir / "30_0_external_matched_common.py"
    legacy_analysis_path = code_dir / "30_10_analyze_external_matched.py"
    require(common_path.exists(), f"Missing frozen common helper: {common_path}")
    require(
        legacy_analysis_path.exists(),
        f"Missing frozen historical analysis helper: {legacy_analysis_path}",
    )

    A = _load_module(
        "external_matched_v31_readonly_helper_for_unified_v2",
        legacy_analysis_path,
    )
    C = A.C

    protocol, marker = C.load_protocol(C.DEFAULT_PROTOCOL_DIR)
    # Verify the historical code identities that the frozen protocol records.
    for filename in (
        "30_0_external_matched_common.py",
        "30_10_analyze_external_matched.py",
    ):
        C.verify_active_code_hash(protocol, filename)

    rows: list[dict] = []
    for seed in SEEDS:
        for method in ("dasr_tsq", "graphga", "mars"):
            if method == "dasr_tsq":
                final, _snapshots, completed = A.load_dasr_seed(protocol, seed)
            else:
                final, _snapshots, completed, _archive = A.load_external_seed(
                    protocol, marker, method, seed
                )

            fm = C.population_metrics(final, compute_hypervolume=True)
            counters = dict(completed.get("counters", {}))
            actual_requests = int(completed["actual_requests"])

            rows.append(
                normalize_row(
                    method=method,
                    seed=seed,
                    population_size=200,
                    mean_final_reward=fm["mean_final_reward"],
                    top10_mean_final_reward=fm["top10_mean_final_reward"],
                    max_final_reward=fm["max_final_reward"],
                    hypervolume_4d=fm["hypervolume_4d"],
                    tier1_count=fm["tier1_count"],
                    unique_scaffolds=fm["unique_scaffolds"],
                    max_scaffold_occupancy=fm["max_scaffold_occupancy"],
                    pareto_front_size=fm["pareto_front_size"],
                    actual_requests=actual_requests,
                    model_evaluations=counters.get(
                        "model_evaluations", math.nan
                    ),
                    cache_hits=counters.get("cache_hits", math.nan),
                    unique_valid_evaluations=counters.get(
                        "unique_valid_evaluations", math.nan
                    ),
                    source=(
                        "READ-ONLY reconstruction from frozen formal run via "
                        + str(legacy_analysis_path)
                    ),
                )
            )

    print(
        "Historical aggregate external_run_metrics.csv not found; "
        "DASR-TSQ/GraphGA/MARS metrics reconstructed READ-ONLY from "
        "their already-completed frozen formal runs."
    )
    return rows


def load_legacy_three_methods() -> list[dict]:
    # Preferred path: use the historical aggregate if it is still present.
    if LEGACY_RUN_METRICS.exists() and LEGACY_COMPLETED.exists():
        done = load_json(LEGACY_COMPLETED)
        require(
            done.get("completed") is True,
            "Historical matched analysis is not completed",
        )

        raw = pd.read_csv(LEGACY_RUN_METRICS)
        rows = []
        for method in ("dasr_tsq", "graphga", "mars"):
            sub = raw.loc[raw["method"] == method].copy()
            require(
                sorted(sub["seed"].astype(int).tolist()) == SEEDS,
                f"{method}: expected seeds 42-46 exactly once",
            )
            for _, r in sub.sort_values("seed").iterrows():
                rows.append(
                    normalize_row(
                        method=method,
                        seed=int(r["seed"]),
                        population_size=200,
                        mean_final_reward=r["final_mean_final_reward"],
                        top10_mean_final_reward=r[
                            "final_top10_mean_final_reward"
                        ],
                        max_final_reward=r["final_max_final_reward"],
                        hypervolume_4d=r["final_hypervolume_4d"],
                        tier1_count=r["final_tier1_count"],
                        unique_scaffolds=r["final_unique_scaffolds"],
                        max_scaffold_occupancy=r[
                            "final_max_scaffold_occupancy"
                        ],
                        pareto_front_size=r["final_pareto_front_size"],
                        actual_requests=r["actual_requests"],
                        model_evaluations=r["model_evaluations"],
                        cache_hits=r["cache_hits"],
                        unique_valid_evaluations=r[
                            "unique_valid_evaluations"
                        ],
                        source=str(LEGACY_RUN_METRICS),
                    )
                )
        return rows

    # Safe fallback: the aggregate folder may have been cleaned up, while the
    # formal method runs remain authoritative.  Reconstruct metrics in memory.
    return _reconstruct_legacy_three_methods_from_frozen_raw()


def load_mgp() -> tuple[list[dict], dict]:
    rows = []
    protocol_ids = set()
    input_hashes = {}

    for seed in SEEDS:
        task = MGP_ROOT / f"seed_{seed}"
        metrics_path = task / "final_metrics.json"
        done_path = task / "TASK_COMPLETED.json"
        require(metrics_path.exists(), f"Missing {metrics_path}")
        require(done_path.exists(), f"Missing {done_path}")

        d = load_json(metrics_path)
        done = load_json(done_path)
        require(done.get("completed") is True, f"MGP seed {seed} not completed")
        require(int(d.get("seed", -1)) == seed, f"MGP seed mismatch: {metrics_path}")
        require(d.get("formal") is True, f"MGP seed {seed} is not formal")
        require(
            int(d.get("native_generations_completed", -1)) == 150,
            f"MGP seed {seed}: expected 150 native generations",
        )

        protocol_ids.add(str(d.get("protocol_sha256")))
        fm = d["final_metrics"]
        oc = d["oracle_counters"]

        rows.append(
            normalize_row(
                method="mgp",
                seed=seed,
                population_size=100,
                mean_final_reward=fm["mean_final_reward"],
                top10_mean_final_reward=fm["top10_mean_final_reward"],
                max_final_reward=fm["max_final_reward"],
                hypervolume_4d=fm["hypervolume_4d"],
                tier1_count=fm["tier1_count"],
                unique_scaffolds=fm["unique_scaffolds"],
                max_scaffold_occupancy=fm["max_scaffold_occupancy"],
                pareto_front_size=fm["pareto_front_size"],
                actual_requests=oc["requests"],
                model_evaluations=oc["model_evaluations"],
                cache_hits=oc["cache_hits"],
                unique_valid_evaluations=oc["unique_valid_evaluations"],
                source=str(metrics_path),
            )
        )
        input_hashes[f"seed_{seed}_final_metrics_sha256"] = sha256_file(metrics_path)
        input_hashes[f"seed_{seed}_task_completed_sha256"] = sha256_file(done_path)

    require(len(protocol_ids) == 1, f"MGP protocol IDs differ across seeds: {protocol_ids}")
    return rows, {
        "protocol_id_recorded_in_final_metrics": next(iter(protocol_ids)),
        "input_hashes": input_hashes,
    }


def load_reinvent4() -> tuple[list[dict], dict]:
    require(REINVENT_PER_SEED.exists(), f"Missing {REINVENT_PER_SEED}")
    require(REINVENT_AGG_DONE.exists(), f"Missing {REINVENT_AGG_DONE}")

    done = load_json(REINVENT_AGG_DONE)
    require(done.get("completed") is True, "REINVENT4 aggregate is not completed")
    require(int(done.get("n_seeds", -1)) == 5, "REINVENT4 aggregate n_seeds != 5")
    require(done.get("seeds") == SEEDS, "REINVENT4 aggregate seed list mismatch")

    raw = pd.read_csv(REINVENT_PER_SEED)
    require(
        sorted(raw["seed"].astype(int).tolist()) == SEEDS,
        "REINVENT4 per-seed CSV does not contain seeds 42-46 exactly once",
    )

    rows = []
    for _, r in raw.sort_values("seed").iterrows():
        rows.append(
            normalize_row(
                method="reinvent4",
                seed=int(r["seed"]),
                population_size=200,
                mean_final_reward=r["mean_final_reward"],
                top10_mean_final_reward=r["top10_mean_final_reward"],
                max_final_reward=r["max_final_reward"],
                hypervolume_4d=r["hypervolume_4d"],
                tier1_count=r["tier1_count"],
                unique_scaffolds=r["unique_scaffolds"],
                max_scaffold_occupancy=r["max_scaffold_occupancy"],
                pareto_front_size=r["pareto_front_size"],
                actual_requests=r["requests"],
                model_evaluations=r["model_evaluations"],
                cache_hits=r["cache_hits"],
                unique_valid_evaluations=r["unique_valid_evaluations"],
                source=str(REINVENT_PER_SEED),
            )
        )

    return rows, {
        "formal_protocol_sha256": done["formal_protocol_sha256"],
        "per_seed_csv_sha256": sha256_file(REINVENT_PER_SEED),
        "aggregate_task_completed_sha256": sha256_file(REINVENT_AGG_DONE),
    }


def summarize(master: pd.DataFrame) -> pd.DataFrame:
    out = []
    for method in METHOD_ORDER:
        sub = master.loc[master["method"] == method]
        require(len(sub) == 5, f"{method}: expected 5 seeds")
        row = {
            "method": method,
            "display_name": DISPLAY[method],
            "n_seeds": 5,
            "native_population_size": int(sub["native_population_size"].iloc[0]),
        }
        require(
            sub["native_population_size"].nunique() == 1,
            f"{method}: population size changed across seeds",
        )
        for metric in SUMMARY_METRICS:
            vals = sub[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_sd"] = float(np.std(vals, ddof=1))
        out.append(row)
    return pd.DataFrame(out)


def paired_effects(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot_cache = {}
    effect_index = 0

    for metric, favorable_sign in PAIRED_METRICS.items():
        pivot = master.pivot(index="seed", columns="method", values=metric).sort_index()
        require(pivot.index.astype(int).tolist() == SEEDS, f"{metric}: seed mismatch")
        pivot_cache[metric] = pivot

        for comparator in ("graphga", "mars", "mgp", "reinvent4"):
            diff = (
                pivot[comparator].to_numpy(float)
                - pivot["dasr_tsq"].to_numpy(float)
            )
            lo, hi = bootstrap_ci(diff, seed=51000 + effect_index)
            sd = float(np.std(diff, ddof=1))
            favorable = diff * favorable_sign
            rows.append(
                {
                    "comparator": comparator,
                    "comparator_display": DISPLAY[comparator],
                    "reference": "dasr_tsq",
                    "reference_display": DISPLAY["dasr_tsq"],
                    "metric": metric,
                    "difference_definition": "comparator_minus_DASR-TSQ",
                    "favorable_sign": favorable_sign,
                    "n_pairs": 5,
                    "mean_difference": float(np.mean(diff)),
                    "sd_difference": sd,
                    "bootstrap_95ci_low": lo,
                    "bootstrap_95ci_high": hi,
                    "exact_signflip_p_two_sided_descriptive": exact_signflip_p(diff),
                    "paired_effect_size_dz": (
                        float(np.mean(diff) / sd) if sd > 0 else math.nan
                    ),
                    "n_favorable_seeds_for_comparator": int(np.sum(favorable > 0)),
                    "n_tied_seeds": int(np.sum(np.isclose(favorable, 0.0, atol=1e-12))),
                }
            )
            effect_index += 1

    return pd.DataFrame(rows)


def format_mean_sd(mean: float, sd: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def make_table_ready(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            {
                "Method": r["display_name"],
                "Native population": int(r["native_population_size"]),
                "Mean reward": format_mean_sd(
                    r["mean_final_reward_mean"], r["mean_final_reward_sd"], 4
                ),
                "Top-10 reward": format_mean_sd(
                    r["top10_mean_final_reward_mean"],
                    r["top10_mean_final_reward_sd"],
                    4,
                ),
                "4D HV": format_mean_sd(
                    r["hypervolume_4d_mean"], r["hypervolume_4d_sd"], 4
                ),
                "Tier-1 rate (%)": format_mean_sd(
                    100 * r["tier1_rate_mean"], 100 * r["tier1_rate_sd"], 2
                ),
                "Unique scaffold rate (%)": format_mean_sd(
                    100 * r["unique_scaffold_rate_mean"],
                    100 * r["unique_scaffold_rate_sd"],
                    2,
                ),
                "Max scaffold occupancy (%)": format_mean_sd(
                    100 * r["max_scaffold_occupancy_rate_mean"],
                    100 * r["max_scaffold_occupancy_rate_sd"],
                    2,
                ),
                "Pareto-front rate (%)": format_mean_sd(
                    100 * r["pareto_front_rate_mean"],
                    100 * r["pareto_front_rate_sd"],
                    2,
                ),
                "Actual requests": format_mean_sd(
                    r["actual_requests_mean"], r["actual_requests_sd"], 0
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if any(OUT_DIR.iterdir()):
        raise FileExistsError(
            f"Unified analysis output directory is nonempty: {OUT_DIR}\n"
            "Do not overwrite an existing analysis. Rename/move it only after provenance review."
        )

    rows = []
    rows.extend(load_legacy_three_methods())
    mgp_rows, mgp_prov = load_mgp()
    rows.extend(mgp_rows)
    reinvent_rows, reinvent_prov = load_reinvent4()
    rows.extend(reinvent_rows)

    master = pd.DataFrame(rows)
    master["method"] = pd.Categorical(
        master["method"], categories=METHOD_ORDER, ordered=True
    )
    master = master.sort_values(["method", "seed"]).reset_index(drop=True)
    master["method"] = master["method"].astype(str)

    require(len(master) == 25, f"Expected 25 method-seed rows, got {len(master)}")
    for method in METHOD_ORDER:
        sub = master.loc[master["method"] == method]
        require(
            sorted(sub["seed"].astype(int).tolist()) == SEEDS,
            f"{method}: seed coverage failed",
        )

    # Frozen/common endpoint sanity checks.
    require((master["tier1_rate"] >= 0).all() and (master["tier1_rate"] <= 1).all(),
            "Tier-1 rate outside [0,1]")
    require((master["unique_scaffold_rate"] >= 0).all() and
            (master["unique_scaffold_rate"] <= 1).all(),
            "Unique-scaffold rate outside [0,1]")
    require((master["max_scaffold_occupancy_rate"] > 0).all() and
            (master["max_scaffold_occupancy_rate"] <= 1).all(),
            "Max occupancy rate outside (0,1]")

    summary = summarize(master)
    effects = paired_effects(master)
    table_ready = make_table_ready(summary)

    # Long-form plot data for the four prespecified primary endpoints.
    primary_metrics = [
        "top10_mean_final_reward",
        "hypervolume_4d",
        "tier1_rate",
        "unique_scaffold_rate",
    ]
    primary_long = master[
        ["method", "display_name", "seed"] + primary_metrics
    ].melt(
        id_vars=["method", "display_name", "seed"],
        var_name="metric",
        value_name="value",
    )

    tradeoff = master[
        [
            "method",
            "display_name",
            "seed",
            "top10_mean_final_reward",
            "unique_scaffold_rate",
            "max_scaffold_occupancy_rate",
            "hypervolume_4d",
            "tier1_rate",
        ]
    ].copy()

    master_path = OUT_DIR / "UNIFIED_5METHOD_PER_SEED.csv"
    summary_path = OUT_DIR / "UNIFIED_5METHOD_MEAN_SD.csv"
    table_path = OUT_DIR / "UNIFIED_5METHOD_MAIN_TABLE_READY.csv"
    effects_path = OUT_DIR / "UNIFIED_5METHOD_PAIRED_EFFECTS_VS_DASR.csv"
    primary_path = OUT_DIR / "UNIFIED_5METHOD_PRIMARY_PLOT_DATA.csv"
    tradeoff_path = OUT_DIR / "UNIFIED_5METHOD_TRADEOFF_PLOT_DATA.csv"
    report_path = OUT_DIR / "UNIFIED_5METHOD_ANALYSIS_REPORT.json"
    done_path = OUT_DIR / "TASK_COMPLETED.json"

    master.to_csv(master_path, index=False)
    summary.to_csv(summary_path, index=False)
    table_ready.to_csv(table_path, index=False)
    effects.to_csv(effects_path, index=False)
    primary_long.to_csv(primary_path, index=False)
    tradeoff.to_csv(tradeoff_path, index=False)

    provenance = {
        "legacy_analysis_mode": (
            "historical_aggregate_csv"
            if LEGACY_RUN_METRICS.exists() and LEGACY_COMPLETED.exists()
            else "readonly_reconstruction_from_frozen_formal_runs"
        ),
        "legacy_analysis_completed_sha256": (
            sha256_file(LEGACY_COMPLETED) if LEGACY_COMPLETED.exists() else None
        ),
        "legacy_external_run_metrics_sha256": (
            sha256_file(LEGACY_RUN_METRICS)
            if LEGACY_RUN_METRICS.exists()
            else None
        ),
        "mgp": mgp_prov,
        "reinvent4": reinvent_prov,
    }

    report = {
        "completed": True,
        "analysis_version": "unified_external_5method_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methods": METHOD_ORDER,
        "display_names": DISPLAY,
        "seeds": SEEDS,
        "n_method_seed_runs": 25,
        "population_sizes": {
            "dasr_tsq": 200,
            "graphga": 200,
            "mars": 200,
            "mgp": 100,
            "reinvent4": 200,
        },
        "cross_method_rule": (
            "Because MGP uses its native population size of 100 while the other "
            "methods use 200, Tier-1/scaffold/Pareto structural comparisons are "
            "reported primarily as rates. Raw counts are retained only for provenance/SI."
        ),
        "primary_endpoints": [
            "top10_mean_final_reward",
            "hypervolume_4d",
            "tier1_rate",
            "unique_scaffold_rate",
        ],
        "secondary_endpoints": [
            "mean_final_reward",
            "max_final_reward",
            "max_scaffold_occupancy_rate",
            "pareto_front_rate",
            "actual_requests",
            "model_evaluations",
            "cache_hits",
            "unique_valid_evaluations",
        ],
        "paired_statistics": {
            "reference": "DASR-TSQ",
            "difference_definition": "comparator_minus_DASR-TSQ",
            "paired_seeds": SEEDS,
            "bootstrap_resamples": 20000,
            "exact_signflip_two_sided": True,
            "small_n_boundary": (
                "With five paired seeds, exact sign-flip p-values are descriptive; "
                "do not make a conventional exact two-sided p<0.05 claim."
            ),
        },
        "provenance": provenance,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    outputs = {
        "UNIFIED_5METHOD_PER_SEED.csv": sha256_file(master_path),
        "UNIFIED_5METHOD_MEAN_SD.csv": sha256_file(summary_path),
        "UNIFIED_5METHOD_MAIN_TABLE_READY.csv": sha256_file(table_path),
        "UNIFIED_5METHOD_PAIRED_EFFECTS_VS_DASR.csv": sha256_file(effects_path),
        "UNIFIED_5METHOD_PRIMARY_PLOT_DATA.csv": sha256_file(primary_path),
        "UNIFIED_5METHOD_TRADEOFF_PLOT_DATA.csv": sha256_file(tradeoff_path),
        "UNIFIED_5METHOD_ANALYSIS_REPORT.json": sha256_file(report_path),
    }

    completed = {
        "completed": True,
        "analysis_version": "unified_external_5method_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methods": METHOD_ORDER,
        "seeds": SEEDS,
        "n_method_seed_runs": 25,
        "outputs_sha256": outputs,
        "performance_based_parameter_change": False,
        "optimization_rerun": False,
    }
    done_path.write_text(
        json.dumps(completed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("UNIFIED FIVE-METHOD EXTERNAL ANALYSIS PASS")
    print("=" * 96)
    print(table_ready.to_string(index=False))
    print()
    print("Paired effects rows :", len(effects))
    print("Output directory    :", OUT_DIR)
    print("TASK_COMPLETED      :", done_path)
    print("unified_5method_analysis_complete = True")


if __name__ == "__main__":
    main()
