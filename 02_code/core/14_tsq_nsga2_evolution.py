"""
1. A fixed cold-start initial population P0 is selected directly from real curated molecules.
   固定 cold-start 初始种群 P0 直接从真实清洗分子中筛选。

2. P0 construction uses low-fitness percentile filtering and scaffold-aware sampling.
   P0 构建使用低适应度分位数筛选和 scaffold-aware sampling。

3. No MCTS, P_success prior, NSGA-II optimization, or reward-based ranking is used during P0 construction.
   P0 构建阶段不使用 MCTS、不使用 P_success、不使用 NSGA-II 优化，也不按 reward 排序筛选。

4. During evolution, parent molecules are selected by tournament selection.
   进化阶段通过锦标赛选择父代分子。

5. Each selected parent is locally optimized by the MCTS engine from Step 13.
   每个父代分子由步骤 13 的  MCTS 局部优化。

6. The next population is selected using NSGA-II rank, crowding distance, and TSQ scaffold quota.
   下一代种群通过 NSGA-II rank、拥挤度距离和 TSQ scaffold quota 选择。
"""

from __future__ import annotations
import copy
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from concurrent.futures import ProcessPoolExecutor


# ============================================================
# 0. Load Step 13 modules
# ============================================================

_THIS_DIR = Path(__file__).resolve().parent


def _load_numbered_module(module_name: str, file_name: str):
    """
    Load a numbered local Python file as a module.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = _THIS_DIR / file_name

    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


_common = _load_numbered_module("mcts_common", "13_mcts/13_0_mcts_common.py")
_expansion = _load_numbered_module("mcts_expansion", "13_mcts/13_2_mcts_expansion.py")
_evaluation = _load_numbered_module("mcts_evaluation", "13_mcts/13_3_mcts_evaluation.py")
_mcts_core = _load_numbered_module("step13_mcts_core", "13_mcts/13_mcts_core.py")


TSQMCTS = _mcts_core.TSQMCTS
TSQMCTSConfig = getattr(_mcts_core, "TSQMCTSConfig", _common.TSQMCTSConfig)

FrozenFitnessAdapter = getattr(
    _mcts_core,
    "FrozenFitnessAdapter",
    _evaluation.FrozenFitnessAdapter,
)

PSuccessPrior = getattr(
    _mcts_core,
    "PSuccessPrior",
    _evaluation.PSuccessPrior,
)

MoleculeMutator = getattr(
    _mcts_core,
    "MoleculeMutator",
    _expansion.MoleculeMutator,
)

canonicalize_smiles = getattr(
    _mcts_core,
    "canonicalize_smiles",
    _common.canonicalize_smiles,
)

safe_float = getattr(
    _mcts_core,
    "safe_float",
    _common.safe_float,
)


# ============================================================
# 1. Configuration and candidate record
# 1. 配置类和候选分子记录
# ============================================================

@dataclass
class TSQNSGA2Config:
    """
    Configuration for the outer TSQ-NSGA-II evolutionary loop.
    外层 TSQ-NSGA-II 种群进化流程配置。
    """

    # EN: Number of molecules retained in each generation.
    # CN: 每一代保留的分子数量。
    pop_size: int = 200

    # EN: Maximum number of outer evolutionary generations.
    # CN: 最大外层演化代数。
    generations: int = 50

    # EN: Offspring pool size relative to population size.
    # CN: 子代数量相对于种群大小的倍数。
    offspring_multiplier: float = 1.0

    # EN: Tournament size. Larger values impose stronger selection pressure.
    # CN: 锦标赛规模。数值越大，选择压力越强。
    tournament_k: int = 5

    # EN: TSQ quota during environmental selection.
    # EN: At most this many survivors are allowed from the same generic scaffold.
    # CN: 环境选择阶段的 TSQ quota。
    # CN: 同一个 generic scaffold 最多允许保留这么多个体。
    max_per_scaffold: int = 20

    # EN: Random seed for the outer evolutionary loop.
    # CN: 外层进化流程的随机种子。
    random_seed: int = 42

    # EN: Output directory.
    # CN: 输出目录。
    output_dir: str = "04_runs/tsq_nsga2"

    experiment_name: str = "tsq_nsga2_outer_evolution"
    run_id: str = "run_001"

    # EN: Number of parallel worker processes for local MCTS offspring generation.
    # CN: 用于并行生成 offspring 的 MCTS 子进程数量。
    n_jobs: int = 1


@dataclass
class InitialPopulationConfig:
    """
    Configuration for fixed cold-start P0 construction.
    固定 cold-start 初始种群 P0 构建配置。
    """

    # EN: Number of molecules in fixed initial population P0.
    # CN: 固定初始种群 P0 的分子数量。
    pop_size: int = 200

    # EN: Low-fitness percentile window based on frozen final_reward.
    # CN: 基于 frozen final_reward 的低适应度分位数窗口。
    low_percentile: float = 0.10
    high_percentile: float = 0.40

    # EN: Maximum number of molecules selected from the same scaffold during strict P0 sampling.
    # CN: 严格 P0 采样阶段，同一个 scaffold 最多选多少个分子。
    initial_max_per_scaffold: int = 5

    # EN: If strict quota cannot fill P0, relax the upper percentile threshold.
    # CN: 如果严格 quota 无法填满 P0，则逐步放宽上分位阈值。
    relax_step: float = 0.10
    max_high_percentile: float = 0.70

    random_seed: int = 42


@dataclass
class CandidateRecord:
    """
    Scored candidate molecule used by NSGA-II and TSQ.
    供 NSGA-II 和 TSQ 使用的已评分候选分子记录。
    """

    smiles: str

    # EN: Scalar frozen reward used for diagnostics and tie-breaking.
    # CN: 冻结标量 reward，用于诊断和平局处理。
    final_reward: float

    # EN: Multi-objective vector. All objectives are maximized.
    # CN: 多目标向量。所有目标均按最大化处理。
    objectives: tuple[float, float, float, float]

    # EN: Generic Bemis-Murcko scaffold used for TSQ.
    # CN: 用于 TSQ 的 generic Bemis-Murcko scaffold。
    scaffold: str

    # EN: Full score dictionary returned by the frozen fitness function.
    # CN: 冻结评分函数返回的完整评分字典。
    score_record: dict[str, Any] = field(default_factory=dict)

    # EN: NSGA-II rank. Lower is better.
    # CN: NSGA-II rank，数值越小越好。
    rank: int = 10**9

    # EN: Crowding distance. Higher is better.
    # CN: 拥挤度距离，数值越大越好。
    crowding_distance: float = 0.0


# ============================================================
# 2. Scoring and scaffold utilities
# 2. 评分和 scaffold 工具函数
# ============================================================

def generic_murcko_scaffold_smiles(smiles: str) -> str:
    """
    Compute generic Bemis-Murcko scaffold SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return "invalid_smiles"

    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)

        if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
            return "no_scaffold"

        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)

        if generic is None or generic.GetNumAtoms() == 0:
            return "no_scaffold"

        return Chem.MolToSmiles(
            generic,
            canonical=True,
            isomericSmiles=True,
        )

    except Exception:
        return "scaffold_failed"


def build_candidate_record(
    smiles: str,
    fitness: FrozenFitnessAdapter,
) -> Optional[CandidateRecord]:
    """
    Score one molecule and convert it into a CandidateRecord.
    评价一个分子，并转换为 CandidateRecord。
    """
    canonical = canonicalize_smiles(smiles)

    if canonical is None:
        return None

    score = fitness.score(canonical)

    if str(score.get("scoring_status", "unknown")) != "ok":
        return None

    final_reward = safe_float(score.get("final_reward"), default=float("-inf"))

    pred_gsk3b = safe_float(score.get("pred_gsk3b_pic50"), default=float("-inf"))
    pred_jnk3 = safe_float(score.get("pred_jnk3_pic50"), default=float("-inf"))
    qed = safe_float(score.get("qed"), default=float("-inf"))
    sa_reward = safe_float(score.get("sa_reward"), default=float("-inf"))

    objectives = (
        pred_gsk3b,
        pred_jnk3,
        qed,
        sa_reward,
    )

    scaffold = generic_murcko_scaffold_smiles(canonical)

    return CandidateRecord(
        smiles=canonical,
        final_reward=final_reward,
        objectives=objectives,
        scaffold=scaffold,
        score_record=copy.deepcopy(score),
    )


def score_smiles_list(
    smiles_list: list[str],
    fitness: FrozenFitnessAdapter,
) -> list[CandidateRecord]:
    """
    Score a list of SMILES and remove invalid molecules.
    """
    records: list[CandidateRecord] = []

    for smi in smiles_list:
        record = build_candidate_record(smi, fitness)

        if record is not None:
            records.append(record)

    return records


def deduplicate_records(records: list[CandidateRecord]) -> list[CandidateRecord]:
    """
    Deduplicate records by canonical SMILES.
    按 canonical SMILES 去重。

    If duplicates occur, keep the one with higher final_reward.
    如果出现重复分子，保留 final_reward 更高的记录。
    """
    best_by_smiles: dict[str, CandidateRecord] = {}

    for record in records:
        old = best_by_smiles.get(record.smiles)

        if old is None or record.final_reward > old.final_reward:
            best_by_smiles[record.smiles] = record

    return list(best_by_smiles.values())


# ============================================================
# 3. Fixed cold-start initial population P0
# 3. 固定 cold-start 初始种群 P0
# ============================================================

def strict_quota_capacity(
    records: list[CandidateRecord],
    max_per_scaffold: int,
) -> int:
    """
    Compute the maximum number of molecules selectable under strict scaffold quota.
    计算严格 scaffold quota 下最多可以选出的分子数量。

    Example:
    示例：

    If scaffold counts are 10, 3, and 2, and max_per_scaffold = 5,
    the strict quota capacity is 5 + 3 + 2 = 10.
    如果三个 scaffold 分别包含 10、3、2 个分子，且 max_per_scaffold = 5，
    则严格 quota 容量为 5 + 3 + 2 = 10。
    """
    records = deduplicate_records(records)

    scaffold_counts: dict[str, int] = {}

    for record in records:
        scaffold_counts[record.scaffold] = scaffold_counts.get(record.scaffold, 0) + 1

    return sum(
        min(count, max_per_scaffold)
        for count in scaffold_counts.values()
    )


def scaffold_aware_sample(
    records: list[CandidateRecord],
    n_required: int,
    max_per_scaffold: int,
    rng: random.Random,
) -> tuple[list[CandidateRecord], dict[str, Any]]:
    """
    Sample records with scaffold-aware round-robin filtering.
    使用 scaffold-aware round-robin filtering 采样候选分子。

    This function is intended for cold-start P0 construction.
    本函数用于 cold-start P0 构建。

    It does not sort molecules by final_reward.
    它不按 final_reward 对分子排序。

    Selection rule:
    选择规则：

    1. Group molecules by generic Bemis-Murcko scaffold.
       按 generic Bemis-Murcko scaffold 分组。

    2. Shuffle molecules inside each scaffold using a fixed seed.
       使用固定随机种子打乱每个 scaffold 内的分子。

    3. Shuffle scaffold order using the same fixed seed.
       使用同一个固定随机种子打乱 scaffold 顺序。

    4. Select molecules in a round-robin manner across scaffolds.
       在不同 scaffold 之间轮转采样。

    5. During strict sampling, no scaffold contributes more than max_per_scaffold molecules.
       严格采样阶段，同一个 scaffold 最多贡献 max_per_scaffold 个分子。

    6. If strict sampling cannot fill the population, quota is relaxed without duplicating molecules.
       如果严格采样无法填满种群，则放宽 quota，但不复制分子。
    """
    records = deduplicate_records(records)

    groups: dict[str, list[CandidateRecord]] = {}

    for record in records:
        groups.setdefault(record.scaffold, []).append(record)

    for scaffold_records in groups.values():
        rng.shuffle(scaffold_records)

    scaffold_keys = list(groups.keys())
    rng.shuffle(scaffold_keys)

    selected: list[CandidateRecord] = []
    selected_smiles: set[str] = set()
    scaffold_counts: dict[str, int] = {}

    # EN: Strict quota round-robin sampling.
    # CN: 严格 quota 下的轮转采样。
    while len(selected) < n_required:
        progress = False

        for scaffold in scaffold_keys:
            if len(selected) >= n_required:
                break

            count = scaffold_counts.get(scaffold, 0)

            if count >= max_per_scaffold:
                continue

            if not groups[scaffold]:
                continue

            record = groups[scaffold].pop(0)

            if record.smiles in selected_smiles:
                continue

            selected.append(record)
            selected_smiles.add(record.smiles)
            scaffold_counts[scaffold] = count + 1
            progress = True

        if not progress:
            break

    strict_selected_count = len(selected)

    # EN: Fallback without duplication.
    # CN: 无复制兜底。
    if len(selected) < n_required:
        remaining: list[CandidateRecord] = []

        for scaffold in scaffold_keys:
            remaining.extend(groups[scaffold])

        rng.shuffle(remaining)

        for record in remaining:
            if len(selected) >= n_required:
                break

            if record.smiles in selected_smiles:
                continue

            selected.append(record)
            selected_smiles.add(record.smiles)
            scaffold_counts[record.scaffold] = scaffold_counts.get(record.scaffold, 0) + 1

    diagnostics = {
        "n_input_records": len(records),
        "n_selected": len(selected),
        "n_unique_scaffolds": len(scaffold_counts),
        "strict_selected_count": strict_selected_count,
        "quota_relaxed_count": len(selected) - strict_selected_count,
        "max_scaffold_occupancy": max(scaffold_counts.values()) if scaffold_counts else 0,
        "strict_quota_capacity": strict_quota_capacity(records, max_per_scaffold),
        "scaffold_counts": scaffold_counts,
    }

    return selected, diagnostics


def prepare_initial_population_from_pool(
    candidate_smiles: list[str],
    fitness: FrozenFitnessAdapter,
    config: InitialPopulationConfig,
    output_csv: Optional[str | Path] = None,
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    """
    Construct a fixed cold-start initial population P0 with N = config.pop_size.
    构建固定 cold-start 初始种群 P0，规模为 config.pop_size。

    This function should be run once to create a fixed P0 shared by all compared methods.
    该函数应只运行一次，用于创建所有对比方法共享的固定 P0。

    P0 construction steps:

    1. Score all valid candidate molecules with the frozen final_reward function.
       使用 frozen final_reward 评价所有有效候选分子。

    2. Define a low-fitness percentile window.
       定义低适应度分位数窗口。

    3. Check whether this window has enough scaffold-diverse molecules under strict quota.
       检查该窗口在严格 scaffold quota 下是否足够选满 P0。

    4. If not enough, gradually relax the upper percentile threshold.
       如果不足，则逐步放宽上分位阈值。

    5. Use scaffold-aware round-robin sampling to obtain P0.
       使用 scaffold-aware round-robin sampling 获得 P0。。
    """
    if not (0.0 <= config.low_percentile < config.high_percentile <= 1.0):
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 1.")

    if not (config.high_percentile <= config.max_high_percentile <= 1.0):
        raise ValueError("max_high_percentile must be >= high_percentile and <= 1.")

    rng = random.Random(config.random_seed)

    all_records = score_smiles_list(candidate_smiles, fitness)
    all_records = deduplicate_records(all_records)

    if len(all_records) < config.pop_size:
        raise ValueError(
            f"Only {len(all_records)} valid unique molecules are available, "
            f"but pop_size={config.pop_size}."
        )

    rewards = np.array(
        [record.final_reward for record in all_records],
        dtype=float,
    )

    low_p = config.low_percentile
    high_p = config.high_percentile

    selected_band_records: list[CandidateRecord] = []
    selected_capacity = 0
    low_cut = float("nan")
    high_cut = float("nan")

    while high_p <= config.max_high_percentile + 1e-12:
        low_cut = float(np.quantile(rewards, low_p))
        high_cut = float(np.quantile(rewards, high_p))

        selected_band_records = [
            record
            for record in all_records
            if low_cut <= record.final_reward <= high_cut
        ]

        selected_capacity = strict_quota_capacity(
            records=selected_band_records,
            max_per_scaffold=config.initial_max_per_scaffold,
        )

        if len(selected_band_records) >= config.pop_size and selected_capacity >= config.pop_size:
            break

        high_p += config.relax_step

    if len(selected_band_records) < config.pop_size or selected_capacity < config.pop_size:
        raise ValueError(
            "Not enough scaffold-diverse molecules in the low-fitness percentile region. "
            f"Available records in final band: {len(selected_band_records)}. "
            f"Strict quota capacity: {selected_capacity}. "
            f"Required: {config.pop_size}. "
            "Consider increasing max_high_percentile or relaxing initial_max_per_scaffold."
        )

    selected_records, diagnostics = scaffold_aware_sample(
        records=selected_band_records,
        n_required=config.pop_size,
        max_per_scaffold=config.initial_max_per_scaffold,
        rng=rng,
    )

    if len(selected_records) != config.pop_size:
        raise RuntimeError(
            f"P0 construction selected {len(selected_records)} molecules, "
            f"but expected {config.pop_size}."
        )

    rows: list[dict[str, Any]] = []

    for record in selected_records:
        score = record.score_record

        rows.append(
            {
                "smiles": record.smiles,
                "final_reward": record.final_reward,
                "pred_gsk3b_pic50": safe_float(score.get("pred_gsk3b_pic50")),
                "pred_jnk3_pic50": safe_float(score.get("pred_jnk3_pic50")),
                "dual_min_pic50": safe_float(score.get("dual_min_pic50")),
                "qed": safe_float(score.get("qed")),
                "sa_score": safe_float(score.get("sa_score")),
                "sa_reward": safe_float(score.get("sa_reward")),
                "scaffold": record.scaffold,
            }
        )

    p0_df = pd.DataFrame(rows)

    diagnostics.update(
        {
            "low_percentile": low_p,
            "final_high_percentile": high_p,
            "low_cut": low_cut,
            "high_cut": high_cut,
            "n_all_valid_unique_records": len(all_records),
            "n_low_fitness_band_records": len(selected_band_records),
            "strict_quota_capacity_final_band": selected_capacity,
            "p0_mean_final_reward": float(p0_df["final_reward"].mean()),
            "p0_max_final_reward": float(p0_df["final_reward"].max()),
            "p0_min_final_reward": float(p0_df["final_reward"].min()),
            "p0_unique_scaffolds": int(p0_df["scaffold"].nunique()),
            "random_seed": config.random_seed,
        }
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        p0_df.to_csv(output_csv, index=False)

        diagnostics_path = output_csv.with_suffix(".diagnostics.json")
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    return p0_df["smiles"].tolist(), p0_df, diagnostics


# ============================================================
# 4. NSGA-II functions
# 4. NSGA-II 函数
# ============================================================

def dominates(a: CandidateRecord, b: CandidateRecord) -> bool:
    """
    Return whether candidate a Pareto-dominates candidate b.
    判断候选 a 是否 Pareto 支配候选 b。

    All objectives are maximized.
    所有目标均为最大化。
    """
    return (
        all(x >= y for x, y in zip(a.objectives, b.objectives))
        and any(x > y for x, y in zip(a.objectives, b.objectives))
    )


def fast_non_dominated_sort(records: list[CandidateRecord]) -> list[list[int]]:
    """
    Fast non-dominated sorting used by NSGA-II.
    NSGA-II 使用的快速非支配排序。
    """
    n = len(records)

    dominated_sets: list[list[int]] = [[] for _ in range(n)]
    domination_counts = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            if dominates(records[i], records[j]):
                dominated_sets[i].append(j)

            elif dominates(records[j], records[i]):
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            records[i].rank = 0
            fronts[0].append(i)

    current = 0

    while current < len(fronts) and fronts[current]:
        next_front: list[int] = []

        for i in fronts[current]:
            for j in dominated_sets[i]:
                domination_counts[j] -= 1

                if domination_counts[j] == 0:
                    records[j].rank = current + 1
                    next_front.append(j)

        current += 1

        if next_front:
            fronts.append(next_front)

    return fronts


def assign_crowding_distance(
    records: list[CandidateRecord],
    front: list[int],
) -> None:
    """
    Assign crowding distance within one Pareto front.
    为一个 Pareto front 内的候选分子分配拥挤度距离。
    """
    if not front:
        return

    if len(front) <= 2:
        for idx in front:
            records[idx].crowding_distance = float("inf")
        return

    for idx in front:
        records[idx].crowding_distance = 0.0

    n_objectives = len(records[front[0]].objectives)

    for m in range(n_objectives):
        front_sorted = sorted(
            front,
            key=lambda idx: records[idx].objectives[m],
        )

        records[front_sorted[0]].crowding_distance = float("inf")
        records[front_sorted[-1]].crowding_distance = float("inf")

        min_value = records[front_sorted[0]].objectives[m]
        max_value = records[front_sorted[-1]].objectives[m]
        value_range = max_value - min_value

        if value_range <= 1e-12:
            continue

        for pos in range(1, len(front_sorted) - 1):
            prev_value = records[front_sorted[pos - 1]].objectives[m]
            next_value = records[front_sorted[pos + 1]].objectives[m]

            records[front_sorted[pos]].crowding_distance += (
                next_value - prev_value
            ) / value_range


def assign_nsga2_rank_and_crowding(records: list[CandidateRecord]) -> list[list[int]]:
    """
    Assign NSGA-II rank and crowding distance to all candidates.
    为所有候选分子分配 NSGA-II rank 和拥挤度距离。
    """
    fronts = fast_non_dominated_sort(records)

    for front in fronts:
        assign_crowding_distance(records, front)

    return fronts


# ============================================================
# 5. TSQ environmental selection
# 5. TSQ 环境选择
# ============================================================

def tsq_environmental_selection(
    records: list[CandidateRecord],
    pop_size: int,
    max_per_scaffold: int,
) -> tuple[list[CandidateRecord], dict[str, Any]]:
    """
    Select next population using NSGA-II priority plus TSQ scaffold quota.
    使用 NSGA-II 优先级 + TSQ scaffold quota 选择下一代种群。

    Selection priority:
    选择优先级：

    1. Lower NSGA-II rank.
       更低的 NSGA-II rank。

    2. Higher crowding distance.
       更高的拥挤度距离。

    3. Higher frozen final_reward as a tie-breaker.
       更高的 frozen final_reward 作为平局处理。

    TSQ rule:
    TSQ 规则：

    During the strict pass, no scaffold contributes more than max_per_scaffold survivors.
    严格筛选阶段，同一个 scaffold 不超过 max_per_scaffold 个幸存个体。

    If the strict quota cannot fill the population, the quota is relaxed without duplicating molecules.
    如果严格 quota 无法填满种群，则放宽 quota，但不复制分子。
    """
    records = deduplicate_records(records)

    if not records:
        return [], {
            "n_input_records": 0,
            "n_selected": 0,
            "strict_selected_count": 0,
            "quota_relaxed_count": 0,
            "n_unique_scaffolds": 0,
            "max_scaffold_occupancy": 0,
            "scaffold_counts": {},
        }

    assign_nsga2_rank_and_crowding(records)

    ordered = sorted(
        records,
        key=lambda r: (
            r.rank,
            -r.crowding_distance,
            -r.final_reward,
            r.smiles,
        ),
    )

    selected: list[CandidateRecord] = []
    selected_smiles: set[str] = set()
    scaffold_counts: dict[str, int] = {}

    # Strict TSQ quota pass.
    for record in ordered:
        if len(selected) >= pop_size:
            break

        if record.smiles in selected_smiles:
            continue

        count = scaffold_counts.get(record.scaffold, 0)

        if count < max_per_scaffold:
            selected.append(record)
            selected_smiles.add(record.smiles)
            scaffold_counts[record.scaffold] = count + 1

    strict_selected_count = len(selected)

    # EN: Fallback without duplication.
    # CN: 不复制分子的兜底阶段。
    if len(selected) < pop_size:
        for record in ordered:
            if len(selected) >= pop_size:
                break

            if record.smiles in selected_smiles:
                continue

            selected.append(record)
            selected_smiles.add(record.smiles)
            scaffold_counts[record.scaffold] = scaffold_counts.get(record.scaffold, 0) + 1

    diagnostics = {
        "n_input_records": len(records),
        "n_selected": len(selected),
        "strict_selected_count": strict_selected_count,
        "quota_relaxed_count": len(selected) - strict_selected_count,
        "n_unique_scaffolds": len(scaffold_counts),
        "max_scaffold_occupancy": max(scaffold_counts.values()) if scaffold_counts else 0,
        "scaffold_counts": scaffold_counts,
    }

    return selected, diagnostics


# EN: Worker-local objects. Each process loads its own models once.
# CN: 每个子进程各自加载一次模型，避免每个任务重复加载。
_WORKER_FITNESS = None
_WORKER_PRIOR = None
_WORKER_MUTATOR = None


def _init_mcts_worker():
    """
    EN: Initialize model objects once per worker process.
    CN: 每个子进程初始化一次模型对象。
    """
    global _WORKER_FITNESS, _WORKER_PRIOR, _WORKER_MUTATOR

    _WORKER_FITNESS = FrozenFitnessAdapter()
    _WORKER_PRIOR = PSuccessPrior()
    _WORKER_MUTATOR = MoleculeMutator()


def _run_local_mcts_worker(task: dict) -> str:
    """
    EN: Run one local MCTS search in a worker process.
    CN: 在子进程中运行一次局部 MCTS 搜索。
    """
    global _WORKER_FITNESS, _WORKER_PRIOR, _WORKER_MUTATOR

    parent_smiles = task["parent_smiles"]

    try:
        mcts_config_payload = dict(task["mcts_config"])
        mcts_config_payload["random_seed"] = task["random_seed"]
        mcts_config_payload["run_id"] = task["run_id"]

        local_config = TSQMCTSConfig(**mcts_config_payload)

        engine = TSQMCTS(
            config=local_config,
            fitness_adapter=_WORKER_FITNESS,
            prior=_WORKER_PRIOR,
            mutator=_WORKER_MUTATOR,
        )

        result = engine.search(parent_smiles)
        child_smiles = result.best_node.smiles

        can = canonicalize_smiles(child_smiles)
        if can is None:
            return parent_smiles

        return can

    except Exception:
        # EN: Fall back to parent if local MCTS fails.
        # CN: 如果局部 MCTS 失败，则退回 parent，保证外层进化不中断。
        return parent_smiles


# ============================================================
# 6. Outer TSQ-NSGA-II evolution engine
# 6. 外层 TSQ-NSGA-II 进化引擎
# ============================================================

class TSQNSGA2Evolution:
    """
    Outer evolutionary algorithm using prior-throttled MCTS and TSQ-NSGA-II selection.
    使用 prior-throttled MCTS 和 TSQ-NSGA-II 选择的外层进化算法。
    """

    def __init__(
        self,
        initial_population_smiles: list[str],
        evolution_config: TSQNSGA2Config,
        mcts_config: Optional[TSQMCTSConfig] = None,
    ) -> None:
        self.config = evolution_config
        self.rng = random.Random(evolution_config.random_seed)

        self.output_dir = Path(evolution_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fitness = FrozenFitnessAdapter()
        self.prior = PSuccessPrior()
        self.mutator = MoleculeMutator()

        self.mcts_config = mcts_config or TSQMCTSConfig(
            experiment_name="inner_prior_throttled_mcts",
            run_id=evolution_config.run_id,
            random_seed=evolution_config.random_seed,
        )

        canonical_population: list[str] = []

        for smi in initial_population_smiles:
            canonical = canonicalize_smiles(smi)

            if canonical is not None:
                canonical_population.append(canonical)

        canonical_population = list(dict.fromkeys(canonical_population))

        if len(canonical_population) != self.config.pop_size:
            raise ValueError(
                f"Initial population must contain exactly {self.config.pop_size} "
                f"unique valid molecules. Got {len(canonical_population)}."
            )

        self.population_smiles = canonical_population

    def _score_current_population(self) -> list[CandidateRecord]:
        # Score the current population.
        return score_smiles_list(self.population_smiles, self.fitness)

    def _tournament_select_parent(
        self,
        population_records: list[CandidateRecord],
    ) -> CandidateRecord:
        """
        Select one parent by tournament selection.
        通过锦标赛选择一个父代分子。

        Lower rank is better.
        更低 rank 更好。

        Higher crowding distance is better.
        更高 crowding distance 更好。

        Higher final_reward is used as a tie-breaker.
        更高 final_reward 用作平局处理。
        """
        if not population_records:
            raise ValueError("Population records are empty.")

        k = min(self.config.tournament_k, len(population_records))
        competitors = self.rng.sample(population_records, k)

        return max(
            competitors,
            key=lambda r: (
                -r.rank,
                r.crowding_distance,
                r.final_reward,
            ),
        )

    def _run_local_mcts(
        self,
        parent_smiles: str,
        generation: int,
        child_index: int,
    ) -> str:
        """
        Run one local MCTS search from a parent molecule.
        从一个父代分子出发运行一次局部 MCTS。
        """
        local_config = replace(
            self.mcts_config,
            experiment_name="inner_prior_throttled_mcts",
            run_id=f"{self.config.run_id}_gen{generation}_child{child_index}",
            random_seed=self.config.random_seed + generation * 10000 + child_index,
        )

        engine = TSQMCTS(
            config=local_config,
            fitness_adapter=self.fitness,
            prior=self.prior,
            mutator=self.mutator,
        )

        result = engine.search(parent_smiles)

        return result.best_node.smiles

    def _generate_offspring(
            self,
            population_records: list[CandidateRecord],
            generation: int,
    ) -> list[str]:
        """
        EN: Generate offspring molecules by running local MCTS from selected parents.
        CN: 从选择出的 parent 出发，通过局部 MCTS 生成 offspring。
        """
        target_count = int(math.ceil(self.config.pop_size * self.config.offspring_multiplier))

        # EN: Select parents in the main process to keep tournament selection deterministic.
        # CN: 在主进程中完成 parent 选择，保证 tournament selection 的随机性可复现。
        tasks = []

        if hasattr(self.mcts_config, "__dataclass_fields__"):
            base_mcts_config = asdict(self.mcts_config)
        else:
            base_mcts_config = dict(vars(self.mcts_config))

        for child_index in range(target_count):
            parent = self._tournament_select_parent(population_records)

            local_seed = (
                    int(self.config.random_seed)
                    + int(generation) * 10000
                    + int(child_index)
            )

            local_run_id = f"{self.config.run_id}_gen{generation:03d}_child{child_index:04d}"

            tasks.append(
                {
                    "parent_smiles": parent.smiles,
                    "generation": generation,
                    "child_index": child_index,
                    "random_seed": local_seed,
                    "run_id": local_run_id,
                    "mcts_config": base_mcts_config,
                }
            )

        # EN: Serial mode for debugging and exact compatibility.
        # CN: 串行模式，便于调试并保持旧逻辑兼容。
        if self.config.n_jobs <= 1:
            offspring_smiles = []
            for task in tasks:
                child = self._run_local_mcts(
                    parent_smiles=task["parent_smiles"],
                    generation=generation,
                    child_index=task["child_index"],
                )
                offspring_smiles.append(child)
            return offspring_smiles

        # EN: Parallel mode. The executor is created once in run().
        # CN: 并行模式。进程池在 run() 中只创建一次。
        if getattr(self, "_executor", None) is None:
            raise RuntimeError("Parallel executor is not initialized.")

        offspring_smiles = list(
            self._executor.map(
                _run_local_mcts_worker,
                tasks,
                chunksize=1,
            )
        )

        return offspring_smiles

    def _summarize_generation(
        self,
        generation: int,
        selected_records: list[CandidateRecord],
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Summarize one generation.
        汇总一代种群统计指标。
        """
        final_rewards = [r.final_reward for r in selected_records]
        qeds = [r.objectives[2] for r in selected_records]
        sa_rewards = [r.objectives[3] for r in selected_records]
        dual_mins = [
            min(r.objectives[0], r.objectives[1])
            for r in selected_records
        ]

        return {
            "generation": generation,
            "max_final_reward": float(np.max(final_rewards)) if final_rewards else float("nan"),
            "mean_final_reward": float(np.mean(final_rewards)) if final_rewards else float("nan"),
            "mean_dual_min_pic50": float(np.mean(dual_mins)) if dual_mins else float("nan"),
            "mean_qed": float(np.mean(qeds)) if qeds else float("nan"),
            "mean_sa_reward": float(np.mean(sa_rewards)) if sa_rewards else float("nan"),
            "n_population": len(selected_records),
            "n_unique_smiles": len({r.smiles for r in selected_records}),
            "n_unique_scaffolds": diagnostics.get("n_unique_scaffolds", 0),
            "max_scaffold_occupancy": diagnostics.get("max_scaffold_occupancy", 0),
            "quota_relaxed_count": diagnostics.get("quota_relaxed_count", 0),
        }

    def _save_population(
        self,
        generation_label: str,
        records: list[CandidateRecord],
    ) -> None:
        """
        Save population records to CSV.
        将种群记录保存为 CSV。
        """
        rows: list[dict[str, Any]] = []

        for record in records:
            score = record.score_record

            rows.append(
                {
                    "generation": generation_label,
                    "smiles": record.smiles,
                    "final_reward": record.final_reward,
                    "pred_gsk3b_pic50": safe_float(score.get("pred_gsk3b_pic50")),
                    "pred_jnk3_pic50": safe_float(score.get("pred_jnk3_pic50")),
                    "dual_min_pic50": safe_float(score.get("dual_min_pic50")),
                    "qed": safe_float(score.get("qed")),
                    "sa_score": safe_float(score.get("sa_score")),
                    "sa_reward": safe_float(score.get("sa_reward")),
                    "scaffold": record.scaffold,
                    "rank": record.rank,
                    "crowding_distance": record.crowding_distance,
                    "experiment_name": self.config.experiment_name,
                    "run_id": self.config.run_id,
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(
            self.output_dir / f"population_{generation_label}.csv",
            index=False,
        )

    def _run_main_loop(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        EN: Original main evolution loop.
        CN: 原来的主进化循环。

        Returns:
        返回：

        history_df:
            Generation-level summary.
            每一代的统计摘要。

        final_df:
            Final selected population.
            最终筛选后的种群。
        """
        history_rows: list[dict[str, Any]] = []

        # EN: Save fixed initial population P0.
        # CN: 保存固定初始种群 P0。
        initial_records = self._score_current_population()
        assign_nsga2_rank_and_crowding(initial_records)
        self._save_population("initial", initial_records)

        for generation in range(self.config.generations):
            print(f"[Generation {generation}] population={len(self.population_smiles)}")

            population_records = self._score_current_population()
            assign_nsga2_rank_and_crowding(population_records)

            offspring_smiles = self._generate_offspring(
                population_records=population_records,
                generation=generation,
            )

            combined_smiles = self.population_smiles + offspring_smiles
            combined_records = score_smiles_list(combined_smiles, self.fitness)

            selected_records, diagnostics = tsq_environmental_selection(
                records=combined_records,
                pop_size=self.config.pop_size,
                max_per_scaffold=self.config.max_per_scaffold,
            )

            if len(selected_records) < self.config.pop_size:
                raise RuntimeError(
                    f"Only {len(selected_records)} molecules selected at generation {generation}. "
                    f"Expected {self.config.pop_size}."
                )

            self.population_smiles = [r.smiles for r in selected_records]

            generation_summary = self._summarize_generation(
                generation=generation,
                selected_records=selected_records,
                diagnostics=diagnostics,
            )

            history_rows.append(generation_summary)

            print(
                f"  Max reward={generation_summary['max_final_reward']:.4f} | "
                f"Mean reward={generation_summary['mean_final_reward']:.4f} | "
                f"Mean dual-min pIC50={generation_summary['mean_dual_min_pic50']:.4f} | "
                f"Scaffolds={generation_summary['n_unique_scaffolds']} | "
                f"Quota relaxed={generation_summary['quota_relaxed_count']}"
            )

            self._save_population(f"gen{generation:03d}", selected_records)

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(
            self.output_dir / "history.csv",
            index=False,
        )

        final_path = self.output_dir / f"population_gen{self.config.generations - 1:03d}.csv"
        final_df = pd.read_csv(final_path)
        final_df.to_csv(
            self.output_dir / "final_population.csv",
            index=False,
        )

        metadata = {
            "evolution_config": asdict(self.config),
            "mcts_config": asdict(self.mcts_config),
            "note": (
                "Initial population P0 is fixed and selected directly from real curated molecules. "
                "P0 construction uses low-fitness percentile filtering and scaffold-aware sampling. "
                "No MCTS, P_success, NSGA-II optimization, or reward-based ranking is used during initialization. "
                "Outer evolution uses TSQ-NSGA-II. Inner local optimization uses prior-throttled MCTS. "
                "P_success affects MCTS branch selection but not final_reward."
            ),
        }

        with open(self.output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return history_df, final_df

    def run(self):
        """
        EN: Run the outer evolution loop, optionally with parallel local MCTS.
        CN: 运行外层进化流程，可选择并行局部 MCTS。
        """
        self._executor = None

        if self.config.n_jobs <= 1:
            return self._run_main_loop()

        with ProcessPoolExecutor(
                max_workers=self.config.n_jobs,
                initializer=_init_mcts_worker,
        ) as executor:
            self._executor = executor
            try:
                return self._run_main_loop()
            finally:
                self._executor = None


__all__ = [
    "TSQNSGA2Config",
    "InitialPopulationConfig",
    "CandidateRecord",
    "FrozenFitnessAdapter",
    "TSQMCTSConfig",
    "prepare_initial_population_from_pool",
    "TSQNSGA2Evolution",
    "tsq_environmental_selection",
    "generic_murcko_scaffold_smiles",
    "scaffold_aware_sample",
    "strict_quota_capacity",
]


if __name__ == "__main__":
    print(
        "This module defines Step 14: TSQ-NSGA-II outer evolution. "
        "Use 15_run_tsq_nsga2_experiments.py to prepare P0 and launch experiments."
    )