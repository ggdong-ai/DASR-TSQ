"""
Common utilities and data structures for the prior-throttled MCTS local search engine.
prior-throttled MCTS 的公共工具函数和数据结构

This module contains no MCTS decision logic. It only defines shared constants,
canonicalization helpers, configuration, node/result containers, and mutation
candidate containers.
本模块不包含 MCTS 决策逻辑，只定义共享常量、SMILES 标准化工具、配置类、节点/结果容器，以及突变候选容器
"""

from __future__ import annotations
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors


_THIS_DIR = Path(__file__).resolve().parent

# EN: Project root: tsq_mcts_rebuild
# CN: 项目根目录：tsq_mcts_rebuild
PROJECT_ROOT = _THIS_DIR.parents[2]

# EN: Code root: tsq_mcts_rebuild/02_code
# CN: 代码根目录：tsq_mcts_rebuild/02_code
CODE_ROOT = PROJECT_ROOT / "02_code"

# EN: Frozen scoring function used by MCTS evaluation.
# CN: MCTS evaluation 使用的冻结评分函数。
SCORING_FUNCTION_PATH = CODE_ROOT / "scoring" / "10_fitness_function.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "03_models"
P_SUCCESS_MODEL_PATH = MODEL_DIR / "p_success" / "p_success_model.pt"

MORGAN_RADIUS = 3
MORGAN_FP_SIZE = 1024

# EN: Loose applicability-domain limits inherited from the curated QSAR modeling dataset.
# EN: These are not drug-likeness criteria.
# CN: 继承自严格清洗后 QSAR 建模数据的宽松适用域上限。
# CN: 这些不是类药性判据。
MAX_MOLECULAR_WEIGHT = 800.0
MAX_HEAVY_ATOMS = 60

# EN: Numerical stability constant.
# CN: 数值稳定性常量。
EPSILON = 1e-12


def load_numbered_module(module_name: str, file_path: Path):
    """
    Load a local Python file whose filename starts with a number.
    加载文件名以数字开头的本地 Python 脚本
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required module not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def canonicalize_smiles(smiles: str | None) -> str | None:
    """Return canonical isomeric SMILES. Invalid input returns None."""
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def safe_float(value: Any, default: float = float("nan")) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def molecule_in_model_domain(mol: Chem.Mol) -> bool:
    """
    Apply a basic applicability-domain filter consistent with the cleaned modeling data.
    应用与清洗后建模数据一致的基础适用域过滤条件。
    """
    if mol is None:
        return False
    if Descriptors.MolWt(mol) > MAX_MOLECULAR_WEIGHT:
        return False
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        return False
    return True


@dataclass(frozen=True)
class MutationCandidate:
    """
    A sanitized local mutation candidate.
    突变候选容器保存：突变后分子的 canonical SMILES，产生这个分子的突变动作名称
    """
    smiles: str
    action_name: str


@dataclass
class TSQMCTSConfig:
    """
    Configuration for prior-throttled MCTS.

    p_call and gamma are different:
    - p_call: probability of opening the prior gate for an eligible local selection event.
    - gamma: additive strength of P_success when the prior gate is open.
    """

    max_depth: int = 5
    max_children_per_node: int = 8
    mcts_iterations: int = 100 # 仅用于调试的小规模默认值；正式实验应使用统一的 oracle 评价预算覆盖该参数
    exploration_constant: float = math.sqrt(2.0)
    p_call: float = 0.50  #每次符合条件的局部选择事件中，打开 P_success 先验门控的概率
    gamma: float = 0.50   # 如果先验门控打开，P_success 被加到 UCT 上的强度
    random_seed: int = 42
    experiment_name: str = "prior_throttled_mcts"
    run_id: str = "run_001"

    def validate(self) -> None: #检查配置是否合法
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if self.max_children_per_node < 1:
            raise ValueError("max_children_per_node must be >= 1.")
        if self.mcts_iterations < 1:
            raise ValueError("mcts_iterations must be >= 1.")
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant cannot be negative.")
        if not 0.0 <= self.p_call <= 1.0:
            raise ValueError("p_call must be in [0, 1].")
        if self.gamma < 0:
            raise ValueError("gamma cannot be negative.")


@dataclass
class MCTSNode:
    """A node in the molecular MCTS tree."""

    node_id: int
    smiles: str
    parent: Optional["MCTSNode"]
    depth: int
    mutation_action: str

    children: list["MCTSNode"] = field(default_factory=list)      # 子节点列表
    pending_candidates: Optional[list[MutationCandidate]] = None  # 已生成但还没扩展成子节点的候选

    visits: int = 0  # 访问次数
    total_reward: float = 0.0

    p_success: float = float("nan")
    prior_status: str = "not_scored"

    score_record: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_reward(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.total_reward / self.visits


@dataclass
class MCTSResult:
    """
    Complete output of one single-root MCTS search.
    单个 root 分子的一次完整 MCTS 搜索结果。
    """

    root: MCTSNode
    best_node: MCTSNode
    node_records: list[dict[str, Any]]
    selection_records: list[dict[str, Any]]
    summary: dict[str, Any]