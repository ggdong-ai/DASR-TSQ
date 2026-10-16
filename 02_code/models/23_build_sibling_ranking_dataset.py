from __future__ import annotations

"""
23_build_sibling_ranking_dataset.py
===================================
Build a parent-conditioned sibling-candidate dataset for decision-aligned
molecular MCTS expansion.

核心思想：
- 一个 parent 分子对应一个 sibling group；
- 使用现有 MoleculeMutator 为该 parent 生成一组局部变异 children；
- 使用冻结 fitness oracle 评价 parent 和 children；
- 保存 parent-child 局部变化，而不是立即训练模型；
- 按 parent Murcko scaffold 划分 train/validation/test，避免 scaffold 泄漏。

This script deliberately does NOT modify the production MCTS yet. The learned
ranking idea must first pass an offline feasibility check before integration.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORE_MCTS_DIR = PROJECT_ROOT / "02_code" / "core" / "13_mcts"
DEFAULT_GSK3B_CSV = PROJECT_ROOT / "01_data" / "clean" / "gsk3b_modeling_dataset.csv"
DEFAULT_JNK3_CSV = PROJECT_ROOT / "01_data" / "clean" / "jnk3_modeling_dataset.csv"
DEFAULT_SMOKE_DIR = PROJECT_ROOT / "05_results" / "sibling_ranking_dataset_smoke"
DEFAULT_FORMAL_DIR = PROJECT_ROOT / "01_data" / "sibling_ranking"


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _load_module(module_name: str, path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required Python module was not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_mcts_modules():
    """Load the current Step-13 common, expansion, and evaluation modules."""
    common_path = CORE_MCTS_DIR / "13_0_mcts_common.py"
    expansion_path = CORE_MCTS_DIR / "13_2_mcts_expansion.py"
    evaluation_path = CORE_MCTS_DIR / "13_3_mcts_evaluation.py"

    # 13_2 and 13_3 explicitly reuse the module name "mcts_common".
    common = _load_module("mcts_common", common_path)
    expansion = _load_module("mcts_expansion_for_sibling_dataset", expansion_path)
    evaluation = _load_module("mcts_evaluation_for_sibling_dataset", evaluation_path)
    return common, expansion, evaluation


def canonicalize_smiles(smiles: Any) -> str | None:
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def murcko_scaffold_smiles(
    smiles: str,
    make_generic: bool = False,
) -> str:
    """
    返回不包含立体化学信息的 Bemis-Murcko scaffold SMILES。

    某些包含双键 E/Z 信息的分子在删除侧链后，scaffold 对象中可能残留无效的双键 stereo 引用，导致 RDKit 在 canonical SMILES序列化时报告：

        Pre-condition Violation: bad bond stereo

    本项目的 scaffold 分组本来就不区分立体异构体，因此在 scaffold提取前后主动移除 stereo 信息。
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return "__INVALID__"

    # 使用副本，避免修改输入分子对象。
    mol_no_stereo = Chem.Mol(mol)

    # 在删除侧链之前清除原子和双键的立体信息。
    Chem.RemoveStereochemistry(mol_no_stereo)

    scaffold = MurckoScaffold.GetScaffoldForMol(mol_no_stereo)

    if scaffold.GetNumAtoms() == 0:
        return "__ACYCLIC__"

    if make_generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)

    # GetScaffoldForMol 或 MakeScaffoldGeneric 会生成新的分子对象，
    # 因此在 canonicalize 前再次清理 stereo 信息。
    Chem.RemoveStereochemistry(scaffold)

    # 更新分子的属性缓存，避免编辑后的属性信息不一致。
    scaffold.UpdatePropertyCache(strict=False)

    try:
        return Chem.MolToSmiles(
            scaffold,
            canonical=True,
            isomericSmiles=False,
        )

    except RuntimeError as exc:
        # 若仍有极少数异常，保留具体来源 SMILES，便于定位，
        # 而不是静默地将错误 scaffold 混入数据集。
        raise RuntimeError(
            "Failed to serialize a stereo-cleaned Murcko scaffold. "
            f"make_generic={make_generic}; "
            f"source_smiles={smiles}"
        ) from exc


def stable_hash_int(text: str, seed: int = 0) -> int:
    payload = f"{seed}|{text}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


# -----------------------------------------------------------------------------
# Scorers
# -----------------------------------------------------------------------------
class DeterministicMockScorer:
    """
    Deterministic test scorer used only to verify code plumbing.

    It must never be used to create formal scientific results. Formal mode blocks
    this scorer explicitly.
    """

    def score(self, smiles: str) -> dict[str, Any]:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return {
                "canonical_smiles": None,
                "valid_smiles": False,
                "scoring_status": "invalid_smiles",
                "pred_gsk3b_pic50": np.nan,
                "pred_jnk3_pic50": np.nan,
                "dual_min_pic50": np.nan,
                "dual_mean_pic50": np.nan,
                "dual_activity_reward": 0.0,
                "qed": np.nan,
                "sa_score": np.nan,
                "sa_reward": 0.0,
                "final_reward": 0.0,
            }

        mol = Chem.MolFromSmiles(canonical)
        assert mol is not None
        qed = float(QED.qed(mol))
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        rings = int(rdMolDescriptors.CalcNumRings(mol))
        heavy = int(mol.GetNumHeavyAtoms())

        jitter = (stable_hash_int(canonical, seed=17) % 1000) / 1000.0
        pred_gsk3b = float(np.clip(5.2 + 1.6 * qed - 0.0012 * abs(mw - 360.0) + 0.15 * jitter, 4.0, 8.5))
        pred_jnk3 = float(np.clip(5.1 + 1.5 * qed - 0.0010 * abs(mw - 340.0) + 0.12 * (1.0 - jitter), 4.0, 8.5))
        dual_min = min(pred_gsk3b, pred_jnk3)
        dual_mean = 0.5 * (pred_gsk3b + pred_jnk3)
        dual_reward = float(np.clip((dual_min - 5.0) / 3.0, 0.0, 1.0))
        sa_score = float(np.clip(2.0 + 0.06 * heavy + 0.12 * rings + 0.08 * max(logp, 0.0), 1.0, 10.0))
        sa_reward = float(np.clip((10.0 - sa_score) / 9.0, 0.0, 1.0))
        final_reward = float(np.clip(0.60 * dual_reward + 0.20 * qed + 0.20 * sa_reward, 0.0, 1.0))

        return {
            "canonical_smiles": canonical,
            "valid_smiles": True,
            "scoring_status": "ok_mock",
            "pred_gsk3b_pic50": pred_gsk3b,
            "pred_jnk3_pic50": pred_jnk3,
            "dual_min_pic50": dual_min,
            "dual_mean_pic50": dual_mean,
            "dual_activity_reward": dual_reward,
            "qed": qed,
            "sa_score": sa_score,
            "sa_reward": sa_reward,
            "final_reward": final_reward,
            "molecular_weight": mw,
            "logp": logp,
            "heavy_atom_count": heavy,
            "ring_count": rings,
        }

    def metadata(self) -> dict[str, Any]:
        return {"scorer": "deterministic_mock", "formal_results_allowed": False}


def build_scorer(scorer_name: str, evaluation_module):
    if scorer_name == "mock":
        return DeterministicMockScorer()
    if scorer_name != "frozen":
        raise ValueError(f"Unknown scorer: {scorer_name}")
    return evaluation_module.FrozenFitnessAdapter()


# -----------------------------------------------------------------------------
# Parent preparation and scaffold split
# -----------------------------------------------------------------------------
REQUIRED_INPUT_COLUMNS = {"standardized_smiles"}


def load_parent_pool(gsk3b_csv: Path, jnk3_csv: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_name, path in (("gsk3b", gsk3b_csv), ("jnk3", jnk3_csv)):
        if not path.exists():
            raise FileNotFoundError(f"Input dataset was not found: {path}")
        frame = pd.read_csv(path)
        missing = REQUIRED_INPUT_COLUMNS.difference(frame.columns)
        if missing:
            raise KeyError(f"{path} is missing required columns: {sorted(missing)}")
        frame = frame.copy()
        frame["source_target"] = source_name
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    merged["parent_smiles"] = merged["standardized_smiles"].map(canonicalize_smiles)
    merged = merged[merged["parent_smiles"].notna()].copy()

    # Consolidate molecules that occur in both target-specific modeling datasets.
    rows: list[dict[str, Any]] = []
    for parent_smiles, group in merged.groupby("parent_smiles", sort=True):
        source_targets = ";".join(sorted(set(group["source_target"].astype(str))))
        supplied_scaffolds = [
            str(v).strip()
            for v in group.get("scaffold_group", pd.Series(dtype=object)).dropna().tolist()
            if str(v).strip()
        ]
        parent_scaffold = supplied_scaffolds[0] if supplied_scaffolds else murcko_scaffold_smiles(parent_smiles)
        rows.append(
            {
                "parent_smiles": parent_smiles,
                "source_targets": source_targets,
                "parent_scaffold": parent_scaffold,
                "parent_generic_scaffold": murcko_scaffold_smiles(parent_smiles, make_generic=True),
            }
        )

    parent_df = pd.DataFrame(rows)
    if parent_df.empty:
        raise RuntimeError("No valid unique parent molecules were found.")
    return parent_df


def scaffold_balanced_sample(parent_df: pd.DataFrame, max_parents: int | None, seed: int) -> pd.DataFrame:
    """Round-robin sampling across scaffolds to avoid domination by large series."""
    if max_parents is None or max_parents <= 0 or max_parents >= len(parent_df):
        max_parents = len(parent_df)

    groups: dict[str, list[dict[str, Any]]] = {}
    for scaffold, group in parent_df.groupby("parent_scaffold", sort=True):
        records = group.to_dict("records")
        rng = random.Random(stable_hash_int(str(scaffold), seed))
        rng.shuffle(records)
        groups[str(scaffold)] = records

    scaffold_order = sorted(groups)
    random.Random(seed).shuffle(scaffold_order)

    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < max_parents:
        added_this_round = 0
        for scaffold in scaffold_order:
            records = groups[scaffold]
            if round_index < len(records):
                selected.append(records[round_index])
                added_this_round += 1
                if len(selected) >= max_parents:
                    break
        if added_this_round == 0:
            break
        round_index += 1

    result = pd.DataFrame(selected)
    result = result.drop_duplicates("parent_smiles").reset_index(drop=True)
    return result


def assign_scaffold_splits(parent_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Deterministically assign complete parent-scaffold groups to train/validation/test.

    A largest-group-first greedy allocation keeps parent counts reasonably balanced
    while guaranteeing zero scaffold overlap.
    """
    split_fracs = {"train": 0.70, "validation": 0.15, "test": 0.15}
    target_counts = {name: frac * len(parent_df) for name, frac in split_fracs.items()}

    scaffold_sizes = parent_df.groupby("parent_scaffold").size().to_dict()
    scaffolds = list(scaffold_sizes)
    tie_rng = random.Random(seed)
    tie_noise = {scaf: tie_rng.random() for scaf in scaffolds}
    scaffolds.sort(key=lambda s: (-scaffold_sizes[s], tie_noise[s], str(s)))

    assigned_counts = Counter({name: 0 for name in split_fracs})
    scaffold_to_split: dict[str, str] = {}

    for scaffold in scaffolds:
        size = int(scaffold_sizes[scaffold])

        def allocation_key(split_name: str) -> tuple[float, float, str]:
            remaining = target_counts[split_name] - assigned_counts[split_name]
            relative_fill = assigned_counts[split_name] / max(target_counts[split_name], 1.0)
            return (-remaining, relative_fill, split_name)

        chosen = min(split_fracs, key=allocation_key)
        scaffold_to_split[str(scaffold)] = chosen
        assigned_counts[chosen] += size

    result = parent_df.copy()
    result["split"] = result["parent_scaffold"].astype(str).map(scaffold_to_split)
    if result["split"].isna().any():
        raise RuntimeError("Failed to assign one or more parent scaffolds to a split.")
    return result


# -----------------------------------------------------------------------------
# Pair construction
# -----------------------------------------------------------------------------
SCORE_FIELDS = (
    "pred_gsk3b_pic50",
    "pred_jnk3_pic50",
    "dual_min_pic50",
    "dual_mean_pic50",
    "dual_activity_reward",
    "qed",
    "sa_score",
    "sa_reward",
    "final_reward",
)


def get_float(record: dict[str, Any], key: str) -> float:
    try:
        value = float(record.get(key, np.nan))
    except (TypeError, ValueError):
        return float("nan")
    return value


def is_score_ok(record: dict[str, Any], scorer_name: str) -> bool:
    status = str(record.get("scoring_status", ""))
    accepted = {"ok"} if scorer_name == "frozen" else {"ok_mock"}
    return status in accepted and math.isfinite(get_float(record, "final_reward"))


def tier1_label(record: dict[str, Any]) -> int:
    dual_min = get_float(record, "dual_min_pic50")
    qed = get_float(record, "qed")
    sa_score = get_float(record, "sa_score")
    return int(
        math.isfinite(dual_min)
        and math.isfinite(qed)
        and math.isfinite(sa_score)
        and dual_min >= 6.0
        and qed >= 0.50
        and sa_score <= 6.0
    )


def build_pair_row(
    parent_row: dict[str, Any],
    child_index: int,
    child_smiles: str,
    action_name: str,
    parent_score: dict[str, Any],
    child_score: dict[str, Any],
) -> dict[str, Any]:
    parent_smiles = str(parent_row["parent_smiles"])
    group_id = hashlib.sha256(parent_smiles.encode("utf-8")).hexdigest()[:16]

    row: dict[str, Any] = {
        "sibling_group_id": group_id,
        "split": parent_row["split"],
        "source_targets": parent_row["source_targets"],
        "parent_smiles": parent_smiles,
        "parent_scaffold": parent_row["parent_scaffold"],
        "parent_generic_scaffold": parent_row["parent_generic_scaffold"],
        "child_index": int(child_index),
        "child_smiles": child_smiles,
        "child_scaffold": murcko_scaffold_smiles(child_smiles),
        "child_generic_scaffold": murcko_scaffold_smiles(child_smiles, make_generic=True),
        "mutation_action": action_name,
        "parent_scoring_status": parent_score.get("scoring_status"),
        "child_scoring_status": child_score.get("scoring_status"),
    }

    for field in SCORE_FIELDS:
        row[f"parent_{field}"] = get_float(parent_score, field)
        row[f"child_{field}"] = get_float(child_score, field)

    row["delta_final_reward"] = row["child_final_reward"] - row["parent_final_reward"]
    row["delta_dual_min_pic50"] = row["child_dual_min_pic50"] - row["parent_dual_min_pic50"]
    row["delta_dual_activity_reward"] = (
        row["child_dual_activity_reward"] - row["parent_dual_activity_reward"]
    )
    row["delta_qed"] = row["child_qed"] - row["parent_qed"]
    row["delta_sa_reward"] = row["child_sa_reward"] - row["parent_sa_reward"]
    row["delta_sa_score"] = row["child_sa_score"] - row["parent_sa_score"]

    tolerance = 1e-12
    objective_deltas = [
        row["delta_dual_activity_reward"],
        row["delta_qed"],
        row["delta_sa_reward"],
    ]
    row["strict_pareto_improve"] = int(
        all(delta >= -tolerance for delta in objective_deltas)
        and any(delta > tolerance for delta in objective_deltas)
    )
    row["scalar_improve_000"] = int(row["delta_final_reward"] > 0.00)
    row["scalar_improve_001"] = int(row["delta_final_reward"] >= 0.01)
    row["scalar_improve_002"] = int(row["delta_final_reward"] >= 0.02)
    row["parent_tier1"] = tier1_label(parent_score)
    row["child_tier1"] = tier1_label(child_score)
    row["tier1_gain"] = int(row["child_tier1"] == 1 and row["parent_tier1"] == 0)
    row["scaffold_changed"] = int(row["parent_scaffold"] != row["child_scaffold"])
    return row


def append_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_completed_parents(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_completed_parents(path: Path, parents: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for parent in parents:
            handle.write(f"{parent}\n")


# -----------------------------------------------------------------------------
# Finalization and reports
# -----------------------------------------------------------------------------
def remove_cross_split_child_overlap(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_counts = frame.groupby("child_smiles")["split"].nunique()
    overlapping_children = set(split_counts[split_counts > 1].index)

    overlap_report = (
        frame[frame["child_smiles"].isin(overlapping_children)]
        .groupby("child_smiles")
        .agg(
            n_rows=("child_smiles", "size"),
            n_splits=("split", "nunique"),
            splits=("split", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )

    clean = frame[~frame["child_smiles"].isin(overlapping_children)].copy()
    return clean, overlap_report


def add_within_group_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["rank_delta_final_reward"] = (
        result.groupby("sibling_group_id")["delta_final_reward"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    result["rank_child_final_reward"] = (
        result.groupby("sibling_group_id")["child_final_reward"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return result


def make_label_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    label_columns = [
        "strict_pareto_improve",
        "scalar_improve_000",
        "scalar_improve_001",
        "scalar_improve_002",
        "child_tier1",
        "tier1_gain",
        "scaffold_changed",
    ]
    rows: list[dict[str, Any]] = []
    for split_name in ["all", "train", "validation", "test"]:
        sub = frame if split_name == "all" else frame[frame["split"] == split_name]
        for label in label_columns:
            positive = int(sub[label].sum()) if len(sub) else 0
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "n_rows": int(len(sub)),
                    "n_positive": positive,
                    "positive_rate": positive / len(sub) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(
    frame: pd.DataFrame,
    parent_manifest: pd.DataFrame,
    args: argparse.Namespace,
    elapsed_seconds: float,
    scorer_metadata: dict[str, Any],
    overlap_removed: int,
) -> dict[str, Any]:
    group_sizes = frame.groupby("sibling_group_id").size()
    summary: dict[str, Any] = {
        "dataset_role": "parent-conditioned sibling-ranking dataset for pre-oracle MCTS expansion",
        "scientific_status": "offline feasibility dataset; not yet an online MCTS result",
        "mode": args.mode,
        "scorer": args.scorer,
        "random_seed": args.seed,
        "children_per_parent_requested": args.children_per_parent,
        "minimum_children_per_parent": args.min_children_per_parent,
        "n_parent_manifest": int(len(parent_manifest)),
        "n_parent_groups_retained": int(frame["sibling_group_id"].nunique()),
        "n_transition_rows": int(len(frame)),
        "n_unique_children": int(frame["child_smiles"].nunique()),
        "n_unique_parent_scaffolds": int(frame["parent_scaffold"].nunique()),
        "cross_split_child_rows_removed": int(overlap_removed),
        "group_size": {
            "min": int(group_sizes.min()) if len(group_sizes) else 0,
            "median": float(group_sizes.median()) if len(group_sizes) else 0.0,
            "mean": float(group_sizes.mean()) if len(group_sizes) else 0.0,
            "max": int(group_sizes.max()) if len(group_sizes) else 0,
        },
        "split_rows": frame["split"].value_counts().sort_index().to_dict(),
        "split_parent_groups": frame.groupby("split")["sibling_group_id"].nunique().to_dict(),
        "split_parent_scaffolds": frame.groupby("split")["parent_scaffold"].nunique().to_dict(),
        "mutation_action_counts": frame["mutation_action"].value_counts().to_dict(),
        "elapsed_seconds": elapsed_seconds,
        "scorer_metadata": scorer_metadata,
        "input_files": {
            "gsk3b_csv": str(Path(args.gsk3b_csv).resolve()),
            "gsk3b_sha256": sha256_file(Path(args.gsk3b_csv)),
            "jnk3_csv": str(Path(args.jnk3_csv).resolve()),
            "jnk3_sha256": sha256_file(Path(args.jnk3_csv)),
        },
        "code_files": {
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "mcts_common": str((CORE_MCTS_DIR / "13_0_mcts_common.py").resolve()),
            "mcts_common_sha256": sha256_file(CORE_MCTS_DIR / "13_0_mcts_common.py"),
            "mcts_expansion": str((CORE_MCTS_DIR / "13_2_mcts_expansion.py").resolve()),
            "mcts_expansion_sha256": sha256_file(CORE_MCTS_DIR / "13_2_mcts_expansion.py"),
            "mcts_evaluation": str((CORE_MCTS_DIR / "13_3_mcts_evaluation.py").resolve()),
            "mcts_evaluation_sha256": sha256_file(CORE_MCTS_DIR / "13_3_mcts_evaluation.py"),
        },
    }
    return summary


def finalize_dataset(
    partial_csv: Path,
    parent_manifest_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    elapsed_seconds: float,
    scorer_metadata: dict[str, Any],
) -> None:
    if not partial_csv.exists():
        raise RuntimeError("No transition rows were produced; partial CSV does not exist.")

    frame = pd.read_csv(partial_csv)
    if frame.empty:
        raise RuntimeError("No transition rows were produced.")

    # Remove exact duplicate transitions, then keep only sufficiently large sibling groups.
    frame = frame.drop_duplicates(["parent_smiles", "child_smiles"], keep="first").copy()
    group_sizes = frame.groupby("sibling_group_id").size()
    valid_groups = set(group_sizes[group_sizes >= args.min_children_per_parent].index)
    frame = frame[frame["sibling_group_id"].isin(valid_groups)].copy()

    before_overlap_removal = len(frame)
    frame, overlap_report = remove_cross_split_child_overlap(frame)

    # Removing cross-split child overlaps can make a group too small; filter again.
    group_sizes = frame.groupby("sibling_group_id").size()
    valid_groups = set(group_sizes[group_sizes >= args.min_children_per_parent].index)
    frame = frame[frame["sibling_group_id"].isin(valid_groups)].copy()

    if frame.empty:
        raise RuntimeError(
            "No sibling groups remained after minimum-size and leakage filters. "
            "Reduce --min_children_per_parent or inspect the mutator output."
        )

    frame = add_within_group_ranks(frame)
    frame = frame.sort_values(
        ["split", "parent_scaffold", "parent_smiles", "child_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    final_csv = output_dir / "transition_pairs.csv"
    frame.to_csv(final_csv, index=False)
    for split_name in ("train", "validation", "test"):
        frame[frame["split"] == split_name].to_csv(output_dir / f"{split_name}.csv", index=False)

    overlap_report.to_csv(output_dir / "removed_cross_split_child_overlap.csv", index=False)
    make_label_distribution(frame).to_csv(output_dir / "label_distribution.csv", index=False)

    action_distribution = (
        frame.groupby(["split", "mutation_action"])
        .size()
        .rename("n_rows")
        .reset_index()
    )
    action_distribution.to_csv(output_dir / "mutation_action_distribution.csv", index=False)

    parent_manifest = pd.read_csv(parent_manifest_path)
    retained_parents = set(frame["parent_smiles"])
    parent_manifest["retained_in_final_dataset"] = parent_manifest["parent_smiles"].isin(retained_parents)
    parent_manifest.to_csv(parent_manifest_path, index=False)

    summary = summarize_dataset(
        frame=frame,
        parent_manifest=parent_manifest,
        args=args,
        elapsed_seconds=elapsed_seconds,
        scorer_metadata=scorer_metadata,
        overlap_removed=before_overlap_removal - len(frame),
    )
    write_json(output_dir / "dataset_summary.json", summary)

    print("\n" + "=" * 88)
    print("Sibling-ranking dataset construction finished")
    print(f"Output directory: {output_dir}")
    print(f"Retained parent groups: {frame['sibling_group_id'].nunique()}")
    print(f"Transition rows: {len(frame)}")
    print(f"Median children/group: {frame.groupby('sibling_group_id').size().median():.1f}")
    print(f"Cross-split overlapping child rows removed: {before_overlap_removal - len(frame)}")
    print("Next mandatory step: run 23_1_validate_sibling_ranking_dataset.py")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main construction loop
# -----------------------------------------------------------------------------
def build_dataset(args: argparse.Namespace) -> None:
    if not args.show_rdkit_errors:
        # The mutator deliberately attempts many edits; rejected valence/kekulization
        # candidates are expected and are filtered out. Keep the console readable.
        RDLogger.DisableLog("rdApp.error")

    if args.mode == "formal" and args.scorer != "frozen":
        raise ValueError("Formal mode requires --scorer frozen. Mock scoring is forbidden.")

    output_dir = Path(args.output_dir).resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partial_csv = output_dir / "transition_pairs.partial.csv"
    completed_path = output_dir / "completed_parents.txt"
    parent_manifest_path = output_dir / "parent_manifest.csv"
    run_config_path = output_dir / "run_config.json"

    common, expansion, evaluation = _load_mcts_modules()
    mutator = expansion.MoleculeMutator()
    scorer = build_scorer(args.scorer, evaluation)

    if parent_manifest_path.exists() and args.resume:
        parent_manifest = pd.read_csv(parent_manifest_path)
    else:
        parent_pool = load_parent_pool(Path(args.gsk3b_csv), Path(args.jnk3_csv))
        parent_manifest = scaffold_balanced_sample(parent_pool, args.max_parents, args.seed)
        parent_manifest = assign_scaffold_splits(parent_manifest, args.seed)
        parent_manifest.insert(0, "parent_order", np.arange(len(parent_manifest), dtype=int))
        parent_manifest.to_csv(parent_manifest_path, index=False)

    run_config = {
        "script": str(Path(__file__).resolve()),
        "project_root": str(PROJECT_ROOT),
        "core_mcts_dir": str(CORE_MCTS_DIR),
        "arguments": vars(args),
        "parent_manifest_rows": int(len(parent_manifest)),
    }
    write_json(run_config_path, run_config)

    completed = load_completed_parents(completed_path) if args.resume else set()
    if not args.resume:
        partial_csv.unlink(missing_ok=True)
        completed_path.unlink(missing_ok=True)

    start_time = time.time()
    buffered_rows: list[dict[str, Any]] = []
    buffered_completed: list[str] = []
    n_scored_children = 0
    n_failed_parents = 0

    print("=" * 88)
    print("Building parent-conditioned sibling groups")
    print(f"Mode: {args.mode}")
    print(f"Scorer: {args.scorer}")
    print(f"Parents in manifest: {len(parent_manifest)}")
    print(f"Requested children per parent: {args.children_per_parent}")
    print(f"Minimum retained children per parent: {args.min_children_per_parent}")
    print(f"Output: {output_dir}")
    print(f"Resume: {args.resume}; already completed: {len(completed)}")
    print("=" * 88)

    for index, parent in parent_manifest.sort_values("parent_order").iterrows():
        parent_smiles = str(parent["parent_smiles"])
        if parent_smiles in completed:
            continue

        parent_score = scorer.score(parent_smiles)
        if not is_score_ok(parent_score, args.scorer):
            n_failed_parents += 1
            buffered_completed.append(parent_smiles)
            continue

        per_parent_seed = stable_hash_int(parent_smiles, args.seed) % (2**31 - 1)
        rng = random.Random(per_parent_seed)
        candidates = mutator.generate_candidates(
            smiles=parent_smiles,
            max_candidates=args.children_per_parent,
            rng=rng,
        )

        group_rows: list[dict[str, Any]] = []
        for child_index, candidate in enumerate(candidates):
            child_score = scorer.score(candidate.smiles)
            if not is_score_ok(child_score, args.scorer):
                continue
            group_rows.append(
                build_pair_row(
                    parent_row=parent.to_dict(),
                    child_index=child_index,
                    child_smiles=candidate.smiles,
                    action_name=candidate.action_name,
                    parent_score=parent_score,
                    child_score=child_score,
                )
            )
            n_scored_children += 1

        # Keep all valid rows in the partial file. Minimum group-size filtering is done
        # at finalization so interrupted runs remain auditable.
        buffered_rows.extend(group_rows)
        buffered_completed.append(parent_smiles)

        processed_count = len(completed) + len(buffered_completed)
        if len(buffered_completed) >= args.checkpoint_every:
            append_rows_csv(partial_csv, buffered_rows)
            append_completed_parents(completed_path, buffered_completed)
            completed.update(buffered_completed)
            buffered_rows.clear()
            buffered_completed.clear()

            elapsed = time.time() - start_time
            print(
                f"Processed {processed_count}/{len(parent_manifest)} parents | "
                f"scored children={n_scored_children} | elapsed={elapsed/60:.1f} min"
            )

    append_rows_csv(partial_csv, buffered_rows)
    append_completed_parents(completed_path, buffered_completed)

    elapsed_seconds = time.time() - start_time
    scorer_metadata = scorer.metadata() if hasattr(scorer, "metadata") else {"scorer": args.scorer}
    scorer_metadata["failed_parent_scores"] = n_failed_parents

    finalize_dataset(
        partial_csv=partial_csv,
        parent_manifest_path=parent_manifest_path,
        output_dir=output_dir,
        args=args,
        elapsed_seconds=elapsed_seconds,
        scorer_metadata=scorer_metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build parent-conditioned sibling groups for decision-aligned molecular MCTS."
    )
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--scorer", choices=("frozen", "mock"), default="frozen")
    parser.add_argument("--gsk3b_csv", type=Path, default=DEFAULT_GSK3B_CSV)
    parser.add_argument("--jnk3_csv", type=Path, default=DEFAULT_JNK3_CSV)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_parents", type=int, default=None)
    parser.add_argument("--children_per_parent", type=int, default=None)
    parser.add_argument("--min_children_per_parent", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--show_rdkit_errors",
        action="store_true",
        help="Show expected RDKit sanitization errors produced by rejected mutation attempts.",
    )
    return parser


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "smoke":
        args.max_parents = 30 if args.max_parents is None else args.max_parents
        args.children_per_parent = 16 if args.children_per_parent is None else args.children_per_parent
        args.min_children_per_parent = 4 if args.min_children_per_parent is None else args.min_children_per_parent
        args.output_dir = DEFAULT_SMOKE_DIR if args.output_dir is None else args.output_dir
    else:
        args.max_parents = 2000 if args.max_parents is None else args.max_parents
        args.children_per_parent = 32 if args.children_per_parent is None else args.children_per_parent
        args.min_children_per_parent = 8 if args.min_children_per_parent is None else args.min_children_per_parent
        args.output_dir = DEFAULT_FORMAL_DIR if args.output_dir is None else args.output_dir

    if args.max_parents < 1:
        raise ValueError("--max_parents must be >= 1.")
    if args.children_per_parent < 2:
        raise ValueError("--children_per_parent must be >= 2.")
    if not 1 <= args.min_children_per_parent <= args.children_per_parent:
        raise ValueError(
            "--min_children_per_parent must be between 1 and --children_per_parent."
        )
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint_every must be >= 1.")
    return args


if __name__ == "__main__":
    parsed = resolve_mode_defaults(build_parser().parse_args())
    build_dataset(parsed)
