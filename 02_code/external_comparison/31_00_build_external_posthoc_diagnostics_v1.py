from __future__ import annotations

"""
Post-hoc diagnostics for the frozen five-method external benchmark.

ANALYSIS ONLY: this script does not run or modify any optimizer. It reads the
already-completed formal final populations, reconstructs Tier-1 component pass
rates and scaffold-occupancy diagnostics, and optionally reads the completed
MGP+TSQ transferability runs if present.

Recommended project location:
    02_code/external_comparison/31_00_build_external_posthoc_diagnostics_v1.py
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"D:\pycharm\project\tsq_mcts_rebuild").resolve()
CODE_DIR = PROJECT_ROOT / "02_code" / "external_comparison"
COMMON_PATH = CODE_DIR / "30_0_external_matched_common.py"
OUT_DIR = PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_posthoc_diagnostics_v1"
LEGACY_ROOT = PROJECT_ROOT / "05_results" / "external_comparison" / "matched_step28_v3_1"
MGP_ROOT = PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_v4_mgp_formal"
REINVENT_ROOT = PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_v4_reinvent4_formal"
MGP_TSQ_ROOT = PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_v4_mgp_tsq_transfer_formal"
UNIFIED_PER_SEED_CANDIDATES = [
    PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_unified_5method_analysis_v1" / "UNIFIED_5METHOD_PER_SEED.csv",
    PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_unified_5method_analysis_v2" / "UNIFIED_5METHOD_PER_SEED.csv",
]
SEEDS = [42, 43, 44, 45, 46]
METHOD_ORDER = ["dasr_tsq", "graphga", "mars", "mgp", "reinvent4"]
DISPLAY = {
    "p0": "P0", "dasr_tsq": "DASR-TSQ", "graphga": "GraphGA",
    "mars": "MARS", "mgp": "MGP", "reinvent4": "REINVENT4", "mgp_tsq": "MGP+TSQ",
}
EXPECTED_N = {"dasr_tsq": 200, "graphga": 200, "mars": 200, "mgp": 100, "reinvent4": 200, "mgp_tsq": 100}
ACTIVITY_THRESHOLD = 6.0
QED_THRESHOLD = 0.50
SA_THRESHOLD = 6.0
SENSITIVITY_THRESHOLDS = np.arange(6.0, 7.0001, 0.25)
TOP_SCAFFOLD_RANKS = 20


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load_module("posthoc_external_common_v31", COMMON_PATH)


def detect_column(df: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    if required:
        raise RuntimeError(f"Required column not found. aliases={aliases}; columns={list(df.columns)}")
    return None


def standardize_population_from_frame(raw: pd.DataFrame, expected_n: int) -> pd.DataFrame:
    require(len(raw) == expected_n, f"Expected {expected_n} rows, got {len(raw)}")
    g_col = detect_column(raw, ["pred_gsk3b_pic50", "gsk3b_pic50", "gsk3b"])
    j_col = detect_column(raw, ["pred_jnk3_pic50", "jnk3_pic50", "jnk3"])
    q_col = detect_column(raw, ["qed", "QED"])
    sa_col = detect_column(raw, ["sa_score", "SA_score", "sa"])
    dual_col = detect_column(raw, ["dual_min_pic50"], required=False)
    scaffold_col = detect_column(raw, ["scaffold", "generic_scaffold"], required=False)
    smiles_col = detect_column(raw, ["canonical_smiles", "smiles", "SMILES"], required=False)

    out = pd.DataFrame(index=np.arange(len(raw)))
    out["pred_gsk3b_pic50"] = pd.to_numeric(raw[g_col], errors="coerce")
    out["pred_jnk3_pic50"] = pd.to_numeric(raw[j_col], errors="coerce")
    out["qed"] = pd.to_numeric(raw[q_col], errors="coerce")
    out["sa_score"] = pd.to_numeric(raw[sa_col], errors="coerce")
    if dual_col is not None:
        out["dual_min_pic50"] = pd.to_numeric(raw[dual_col], errors="coerce")
    else:
        out["dual_min_pic50"] = np.minimum(out["pred_gsk3b_pic50"].to_numpy(float), out["pred_jnk3_pic50"].to_numpy(float))

    if scaffold_col is not None:
        scaffold = raw[scaffold_col].fillna("no_scaffold").astype(str).str.strip()
        scaffold = scaffold.replace({"": "no_scaffold", "nan": "no_scaffold", "None": "no_scaffold"})
        out["scaffold"] = scaffold
    else:
        require(smiles_col is not None, "Neither scaffold nor SMILES column is available")
        scaffolds = []
        for smi in raw[smiles_col].fillna("").astype(str):
            if not smi.strip():
                scaffolds.append("no_scaffold")
                continue
            try:
                value = C.generic_scaffold(smi)
            except Exception:
                value = "no_scaffold"
            scaffolds.append(value if value else "no_scaffold")
        out["scaffold"] = scaffolds
    return out


def standardize_population(path: Path, expected_n: int) -> pd.DataFrame:
    require(path.exists(), f"Missing final population: {path}")
    return standardize_population_from_frame(pd.read_csv(path), expected_n)


def find_dasr_final_paths() -> dict[int, Path]:
    task_dirs = C.find_step28_dasr_task_dirs()
    out = {}
    for seed, task in task_dirs.items():
        p1, p2 = task / "final_population.csv", task / "population_gen049.csv"
        out[int(seed)] = p1 if p1.exists() else p2
        require(out[int(seed)].exists(), f"DASR-TSQ final population missing for seed {seed}")
    return out


def five_method_final_paths() -> dict[tuple[str, int], Path]:
    paths = {}
    dasr = find_dasr_final_paths()
    for seed in SEEDS:
        paths[("dasr_tsq", seed)] = dasr[seed]
        paths[("graphga", seed)] = LEGACY_ROOT / "graphga" / f"seed_{seed}" / "final_population.csv"
        paths[("mars", seed)] = LEGACY_ROOT / "mars" / f"seed_{seed}" / "final_population.csv"
        paths[("mgp", seed)] = MGP_ROOT / f"seed_{seed}" / "final_population.csv"
        paths[("reinvent4", seed)] = REINVENT_ROOT / f"seed_{seed}" / "final_population.csv"
    for key, path in paths.items():
        require(path.exists(), f"Missing {key}: {path}")
    return paths


def load_p0_scores() -> tuple[pd.DataFrame, Path]:
    # Reuse frozen/scored P0 rows from a validated common archive: no new oracle evaluation.
    archive_path = LEGACY_ROOT / "graphga" / "seed_42" / "archive_unique.csv.gz"
    require(archive_path.exists(), f"Missing P0 source archive: {archive_path}")
    archive = pd.read_csv(archive_path)
    require(len(archive) >= 200, "GraphGA seed42 archive has fewer than 200 rows")
    p0 = archive.iloc[:200].copy().reset_index(drop=True)
    _, _, frozen_can = C.load_frozen_p0()
    require(p0["canonical_smiles"].astype(str).tolist() == list(frozen_can), "P0 archive rows do not exactly match frozen P0")
    return standardize_population_from_frame(p0, 200), archive_path


def tier1_component_row(method: str, seed, pop: pd.DataFrame) -> dict:
    g = pop["pred_gsk3b_pic50"].to_numpy(float)
    j = pop["pred_jnk3_pic50"].to_numpy(float)
    d = pop["dual_min_pic50"].to_numpy(float)
    q = pop["qed"].to_numpy(float)
    sa = pop["sa_score"].to_numpy(float)
    finite = np.isfinite(g) & np.isfinite(j) & np.isfinite(d) & np.isfinite(q) & np.isfinite(sa)
    pass_g = finite & (g >= ACTIVITY_THRESHOLD)
    pass_j = finite & (j >= ACTIVITY_THRESHOLD)
    pass_dual = finite & (d >= ACTIVITY_THRESHOLD)
    pass_qed = finite & (q >= QED_THRESHOLD)
    pass_sa = finite & (sa <= SA_THRESHOLD)
    tier1 = pass_dual & pass_qed & pass_sa
    dual_fail, qed_fail, sa_fail = ~pass_dual, ~pass_qed, ~pass_sa
    return {
        "method": method, "display_name": DISPLAY[method], "seed": seed, "n_population": len(pop),
        "gsk3b_pass_rate": float(np.mean(pass_g)), "jnk3_pass_rate": float(np.mean(pass_j)),
        "dual_activity_pass_rate": float(np.mean(pass_dual)), "qed_pass_rate": float(np.mean(pass_qed)),
        "sa_pass_rate": float(np.mean(pass_sa)), "tier1_rate": float(np.mean(tier1)),
        "dual_only_fail_rate": float(np.mean(dual_fail & ~qed_fail & ~sa_fail)),
        "qed_only_fail_rate": float(np.mean(~dual_fail & qed_fail & ~sa_fail)),
        "sa_only_fail_rate": float(np.mean(~dual_fail & ~qed_fail & sa_fail)),
        "multiple_criteria_fail_rate": float(np.mean((dual_fail.astype(int) + qed_fail.astype(int) + sa_fail.astype(int)) >= 2)),
    }


def sensitivity_rows(method: str, seed, pop: pd.DataFrame) -> list[dict]:
    d, q, sa = [pop[c].to_numpy(float) for c in ["dual_min_pic50", "qed", "sa_score"]]
    finite = np.isfinite(d) & np.isfinite(q) & np.isfinite(sa)
    rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        passed = finite & (d >= threshold) & (q >= QED_THRESHOLD) & (sa <= SA_THRESHOLD)
        rows.append({
            "method": method, "display_name": DISPLAY[method], "seed": seed,
            "dual_activity_threshold": float(threshold), "qed_threshold": QED_THRESHOLD,
            "sa_threshold": SA_THRESHOLD, "tier1_like_rate": float(np.mean(passed)),
        })
    return rows


def scaffold_rows(method: str, seed, pop: pd.DataFrame) -> tuple[dict, list[dict]]:
    counts = pop["scaffold"].fillna("no_scaffold").astype(str).value_counts(dropna=False).to_numpy(dtype=float)
    counts = np.sort(counts)[::-1]
    n = float(len(pop))
    p = counts / n
    entropy = float(-np.sum(p * np.log(p))) if len(p) else 0.0
    norm_entropy = float(entropy / np.log(len(p))) if len(p) > 1 else 0.0
    row = {
        "method": method, "display_name": DISPLAY[method], "seed": seed, "n_population": int(n),
        "unique_scaffolds": int(len(counts)), "unique_scaffold_rate": float(len(counts) / n),
        "max_scaffold_occupancy": int(counts[0]) if len(counts) else 0,
        "max_scaffold_occupancy_rate": float(p[0]) if len(p) else 0.0,
        "top3_scaffold_mass": float(np.sum(p[:3])), "top5_scaffold_mass": float(np.sum(p[:5])),
        "scaffold_hhi": float(np.sum(p**2)), "normalized_scaffold_entropy": norm_entropy,
    }
    ranks = []
    for rank in range(1, TOP_SCAFFOLD_RANKS + 1):
        ranks.append({
            "method": method, "display_name": DISPLAY[method], "seed": seed, "rank": rank,
            "occupancy_rate": float(p[rank - 1]) if rank <= len(p) else 0.0,
        })
    return row, ranks


def summarize_five_seed(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        require(len(sub) == 5, f"{method}: expected five rows")
        row = {"method": method, "display_name": DISPLAY[method], "n_seeds": 5}
        for metric in metrics:
            vals = sub[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_sd"] = float(np.std(vals, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        for threshold in SENSITIVITY_THRESHOLDS:
            sub = df[(df["method"] == method) & np.isclose(df["dual_activity_threshold"], threshold)]
            require(len(sub) == 5, f"{method} threshold {threshold}: expected five rows")
            vals = sub["tier1_like_rate"].to_numpy(float)
            rows.append({"method": method, "display_name": DISPLAY[method], "dual_activity_threshold": float(threshold), "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))})
    return pd.DataFrame(rows)


def unified_crosscheck(tier_df: pd.DataFrame, scaf_df: pd.DataFrame) -> dict:
    path = next((p for p in UNIFIED_PER_SEED_CANDIDATES if p.exists()), None)
    if path is None:
        return {"performed": False, "reason": "UNIFIED_5METHOD_PER_SEED.csv not found"}
    ref = pd.read_csv(path)
    max_tier = max_occ = 0.0
    for method in METHOD_ORDER:
        for seed in SEEDS:
            a = tier_df[(tier_df.method == method) & (tier_df.seed == seed)].iloc[0]
            b = scaf_df[(scaf_df.method == method) & (scaf_df.seed == seed)].iloc[0]
            r = ref[(ref["method"] == method) & (ref["seed"].astype(int) == seed)].iloc[0]
            max_tier = max(max_tier, abs(float(a["tier1_rate"]) - float(r["tier1_rate"])))
            max_occ = max(max_occ, abs(float(b["max_scaffold_occupancy_rate"]) - float(r["max_scaffold_occupancy_rate"])))
    require(max_tier <= 1e-12, f"Tier-1 cross-check mismatch: {max_tier}")
    require(max_occ <= 1e-12, f"Scaffold occupancy cross-check mismatch: {max_occ}")
    return {"performed": True, "source": str(path), "source_sha256": sha256_file(path), "max_abs_tier1_rate_difference": max_tier, "max_abs_max_occupancy_rate_difference": max_occ}


def analyze_mgp_tsq_transfer(scaf_primary: pd.DataFrame):
    needed = []
    for seed in SEEDS:
        needed += [MGP_TSQ_ROOT / f"seed_{seed}" / "final_population.csv", MGP_TSQ_ROOT / f"seed_{seed}" / "final_metrics.json", MGP_TSQ_ROOT / f"seed_{seed}" / "TASK_COMPLETED.json"]
    if not all(p.exists() for p in needed):
        return None, {"available": False, "reason": "Not all five formal MGP+TSQ transfer outputs are present"}
    rows = []
    for seed in SEEDS:
        pop_path = MGP_TSQ_ROOT / f"seed_{seed}" / "final_population.csv"
        metrics_path = MGP_TSQ_ROOT / f"seed_{seed}" / "final_metrics.json"
        done_path = MGP_TSQ_ROOT / f"seed_{seed}" / "TASK_COMPLETED.json"
        done, metrics = load_json(done_path), load_json(metrics_path)
        require(done.get("completed") is True, f"MGP+TSQ seed {seed} not completed")
        require(metrics.get("formal") is True, f"MGP+TSQ seed {seed} not formal")
        require(metrics.get("analysis_status") == "posthoc_transferability_analysis", f"MGP+TSQ seed {seed}: unexpected status")
        pop = standardize_population(pop_path, 100)
        scaf, _ = scaffold_rows("mgp_tsq", seed, pop)
        base = scaf_primary[(scaf_primary["method"] == "mgp") & (scaf_primary["seed"] == seed)].iloc[0]
        td = metrics["transfer_diagnostics"]
        rows.append({
            "seed": seed,
            "native_mgp_max_occupancy_rate": float(base["max_scaffold_occupancy_rate"]),
            "mgp_tsq_max_occupancy_rate": float(scaf["max_scaffold_occupancy_rate"]),
            "paired_delta_max_occupancy_rate": float(scaf["max_scaffold_occupancy_rate"] - base["max_scaffold_occupancy_rate"]),
            "native_mgp_unique_scaffold_rate": float(base["unique_scaffold_rate"]),
            "mgp_tsq_unique_scaffold_rate": float(scaf["unique_scaffold_rate"]),
            "paired_delta_unique_scaffold_rate": float(scaf["unique_scaffold_rate"] - base["unique_scaffold_rate"]),
            "quota_binding_generations": int(td["quota_binding_generations"]),
            "quota_binding_fraction": float(td["quota_binding_generations"]) / 150.0,
            "quota_relaxation_generations": int(td["quota_relaxation_generations"]),
            "quota_relaxation_ever_used": bool(td["quota_relaxation_ever_used"]),
            "total_blocked_by_quota": int(td["total_blocked_by_quota"]),
            "formal_population_size": 100, "q": 5, "nominal_occupancy_ceiling": 0.05,
        })
    return pd.DataFrame(rows), {"available": True, "root": str(MGP_TSQ_ROOT), "interpretation": "Post-hoc transferability analysis; not part of the prespecified five-method primary benchmark."}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = five_method_final_paths()
    tier_rows, sensitivity, scaffold_summary_rows, scaffold_rank_rows = [], [], [], []
    input_hashes = {}

    p0, p0_source = load_p0_scores()
    tier_rows.append(tier1_component_row("p0", "P0", p0))
    sensitivity.extend(sensitivity_rows("p0", "P0", p0))
    p0_scaf, p0_rank = scaffold_rows("p0", "P0", p0)
    scaffold_summary_rows.append(p0_scaf); scaffold_rank_rows.extend(p0_rank)
    input_hashes["p0_scored_archive"] = {"path": str(p0_source), "sha256": sha256_file(p0_source)}

    for method in METHOD_ORDER:
        for seed in SEEDS:
            path = paths[(method, seed)]
            pop = standardize_population(path, EXPECTED_N[method])
            tier_rows.append(tier1_component_row(method, seed, pop))
            sensitivity.extend(sensitivity_rows(method, seed, pop))
            scaf, rank = scaffold_rows(method, seed, pop)
            scaffold_summary_rows.append(scaf); scaffold_rank_rows.extend(rank)
            input_hashes[f"{method}_seed_{seed}"] = {"path": str(path), "sha256": sha256_file(path)}

    tier_df, sensitivity_df = pd.DataFrame(tier_rows), pd.DataFrame(sensitivity)
    scaf_df, rank_df = pd.DataFrame(scaffold_summary_rows), pd.DataFrame(scaffold_rank_rows)

    tier_metrics = ["gsk3b_pass_rate", "jnk3_pass_rate", "dual_activity_pass_rate", "qed_pass_rate", "sa_pass_rate", "tier1_rate", "dual_only_fail_rate", "qed_only_fail_rate", "sa_only_fail_rate", "multiple_criteria_fail_rate"]
    tier_summary = summarize_five_seed(tier_df[tier_df["method"].isin(METHOD_ORDER)].copy(), tier_metrics)
    p0_tier = tier_df[tier_df["method"] == "p0"].iloc[0]
    p0_summary = {"method": "p0", "display_name": "P0", "n_seeds": 0}
    for metric in tier_metrics:
        p0_summary[f"{metric}_mean"], p0_summary[f"{metric}_sd"] = float(p0_tier[metric]), 0.0
    tier_summary = pd.concat([pd.DataFrame([p0_summary]), tier_summary], ignore_index=True)

    sensitivity_summary = summarize_sensitivity(sensitivity_df[sensitivity_df["method"].isin(METHOD_ORDER)].copy())
    p0_sens = sensitivity_df[sensitivity_df["method"] == "p0"].copy()
    p0_sens_summary = pd.DataFrame({"method": "p0", "display_name": "P0", "dual_activity_threshold": p0_sens["dual_activity_threshold"].to_numpy(float), "mean": p0_sens["tier1_like_rate"].to_numpy(float), "sd": 0.0})
    sensitivity_summary = pd.concat([p0_sens_summary, sensitivity_summary], ignore_index=True)

    scaf_metrics = ["unique_scaffold_rate", "max_scaffold_occupancy_rate", "top3_scaffold_mass", "top5_scaffold_mass", "scaffold_hhi", "normalized_scaffold_entropy"]
    scaf_summary = summarize_five_seed(scaf_df[scaf_df["method"].isin(METHOD_ORDER)].copy(), scaf_metrics)
    p0_scaf_row = scaf_df[scaf_df["method"] == "p0"].iloc[0]
    p0_scaf_summary = {"method": "p0", "display_name": "P0", "n_seeds": 0}
    for metric in scaf_metrics:
        p0_scaf_summary[f"{metric}_mean"], p0_scaf_summary[f"{metric}_sd"] = float(p0_scaf_row[metric]), 0.0
    scaf_summary = pd.concat([pd.DataFrame([p0_scaf_summary]), scaf_summary], ignore_index=True)

    crosscheck = unified_crosscheck(tier_df[tier_df["method"].isin(METHOD_ORDER)].copy(), scaf_df[scaf_df["method"].isin(METHOD_ORDER)].copy())
    transfer_df, transfer_meta = analyze_mgp_tsq_transfer(scaf_df[scaf_df["method"].isin(METHOD_ORDER)].copy())

    outputs = {
        "TIER1_COMPONENT_PER_SEED.csv": tier_df,
        "TIER1_COMPONENT_SUMMARY.csv": tier_summary,
        "TIER1_THRESHOLD_SENSITIVITY_PER_SEED.csv": sensitivity_df,
        "TIER1_THRESHOLD_SENSITIVITY_SUMMARY.csv": sensitivity_summary,
        "SCAFFOLD_OCCUPANCY_PER_SEED.csv": scaf_df,
        "SCAFFOLD_OCCUPANCY_SUMMARY.csv": scaf_summary,
        "SCAFFOLD_RANK_PROFILE_PER_SEED.csv": rank_df,
    }
    if transfer_df is not None:
        outputs["MGP_TSQ_TRANSFER_SCAFFOLD_DIAGNOSTICS.csv"] = transfer_df

    output_hashes = {}
    for name, frame in outputs.items():
        path = OUT_DIR / name
        frame.to_csv(path, index=False)
        output_hashes[name] = sha256_file(path)

    report = {
        "completed": True,
        "analysis_name": "external_benchmark_posthoc_diagnostics_v1",
        "analysis_only": True,
        "optimization_rerun": False,
        "formal_benchmark_modified": False,
        "tier1_rule": {"pred_gsk3b_pic50_min": 6.0, "pred_jnk3_pic50_min": 6.0, "dual_min_pic50_min": 6.0, "qed_min": 0.50, "sa_score_max": 6.0},
        "tier1_threshold_sensitivity": {"posthoc": True, "dual_activity_thresholds": [float(x) for x in SENSITIVITY_THRESHOLDS], "qed_threshold_fixed": 0.50, "sa_threshold_fixed": 6.0, "purpose": "diagnose threshold saturation; not a replacement primary endpoint"},
        "scaffold_diagnostics": {"rank_profile_top_k": TOP_SCAFFOLD_RANKS, "metrics": scaf_metrics},
        "crosscheck_against_unified_formal_metrics": crosscheck,
        "mgp_tsq_transfer": transfer_meta,
        "inputs": input_hashes,
        "outputs_sha256": output_hashes,
    }
    report_path = OUT_DIR / "POSTHOC_DIAGNOSTICS_REPORT.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 96)
    print("POST-HOC EXTERNAL DIAGNOSTICS PASS")
    print("=" * 96)
    print("No optimizer was rerun.")
    print("Output:", OUT_DIR)
    print("MGP+TSQ formal transfer diagnostics available:", transfer_meta["available"])


if __name__ == "__main__":
    main()
