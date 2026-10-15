"""
This module contains two conceptually different components:
本模块包含两个概念上不同的组件：

1. FrozenFitnessAdapter:
   Calls the frozen scoring function to obtain final_reward.
   调用冻结评分函数以获得 final_reward。

2. PSuccessPrior:
   Computes P_success as a search prior.
   计算作为搜索先验的 P_success。

P_success is not part of final_reward.
P_success 不参与 final_reward。
"""

from __future__ import annotations
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def _load_common_module():
    """
    EN: Load the shared common module once.
    CN: 只加载一次公共模块。
    """
    module_name = "mcts_common"

    if module_name in sys.modules:
        return sys.modules[module_name]

    this_dir = Path(__file__).resolve().parent
    common_path = this_dir / "13_0_mcts_common.py"

    spec = importlib.util.spec_from_file_location(module_name, common_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load common MCTS module from: {common_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


_common = _load_common_module()

MORGAN_FP_SIZE = _common.MORGAN_FP_SIZE
MORGAN_RADIUS = _common.MORGAN_RADIUS
P_SUCCESS_MODEL_PATH = _common.P_SUCCESS_MODEL_PATH
SCORING_FUNCTION_PATH = _common.SCORING_FUNCTION_PATH
canonicalize_smiles = _common.canonicalize_smiles
load_numbered_module = _common.load_numbered_module
safe_float = _common.safe_float
MCTSNode = _common.MCTSNode


class FrozenFitnessAdapter:
    """
    Wrapper around the frozen 10_fitness_function.py scorer.
    对冻结的 10_fitness_function.py 评分器进行封装。
    """

    def __init__(self, scoring_function_path: Path = SCORING_FUNCTION_PATH) -> None:
        module = load_numbered_module(
            module_name="frozen_fitness_function_for_prior_throttled_mcts",
            file_path=scoring_function_path,
        )

        if not hasattr(module, "FitnessScorer"):
            raise AttributeError(f"FitnessScorer not found in {scoring_function_path}")

        self._scorer = module.FitnessScorer()

        # EN: Cache scores by canonical SMILES to avoid repeated oracle calls.
        # CN: 按 canonical SMILES 缓存评分结果，避免重复调用 oracle。
        self._cache: dict[str, dict[str, Any]] = {}

    def score(self, smiles: str) -> dict[str, Any]:
        """
        Score one molecule using the frozen fitness function.
        使用冻结适应度函数为一个分子打分。
        """
        canonical = canonicalize_smiles(smiles)

        if canonical is None:
            return copy.deepcopy(self._scorer.score_smiles(smiles))

        if canonical not in self._cache:
            self._cache[canonical] = self._scorer.score_smiles(canonical)

        return copy.deepcopy(self._cache[canonical])

    def metadata(self) -> dict[str, Any]:
        """
        Return metadata for reproducibility.
        返回可复现性记录所需的元信息。
        """
        scorer_metadata = {}

        if hasattr(self._scorer, "metadata"):
            scorer_metadata = self._scorer.metadata()

        return {
            "scoring_function_path": str(SCORING_FUNCTION_PATH),
            "fitness_metadata": scorer_metadata,
            "cache_size": len(self._cache),
        }


class PSuccessNet(nn.Module):
    """
    MLP architecture consistent with 12_train_p_success.py.
    与 12_train_p_success.py 保持一致的 MLP 网络结构。
    """

    def __init__(
        self,
        input_dim: int = MORGAN_FP_SIZE,
        hidden_layers: tuple[int, int, int] = (512, 128, 32),
        dropout_1: float = 0.30,
        dropout_2: float = 0.20,
    ) -> None:
        super().__init__()

        h1, h2, h3 = hidden_layers

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_torch_artifact(path: Path) -> Any:
    """
    Load a PyTorch artifact with compatibility across PyTorch versions.
    兼容不同 PyTorch 版本加载模型文件。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"P_success model not found: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class PSuccessPrior:
    """
    Safe inference wrapper for P_success prior probabilities.
    P_success 先验概率的安全推理封装。

    P_success is used only during selection and never changes final_reward.
    P_success 只在 selection 阶段使用，永远不改变 final_reward。
    """

    def __init__(self, model_path: Path = P_SUCCESS_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        artifact = load_torch_artifact(self.model_path)

        if isinstance(artifact, dict) and "state_dict" in artifact:
            state_dict = artifact["state_dict"]
            architecture = artifact.get("architecture", {})
        elif isinstance(artifact, dict):
            state_dict = artifact
            architecture = {}
        else:
            raise TypeError("Unsupported P_success artifact format.")

        hidden_layers = tuple(architecture.get("hidden_layers", [512, 128, 32]))

        if len(hidden_layers) != 3:
            raise ValueError(f"P_success expects three hidden layers, got {hidden_layers}")

        dropout = architecture.get("dropout", [0.30, 0.20])

        if len(dropout) != 2:
            dropout = [0.30, 0.20]

        input_dim = int(architecture.get("input_dim", MORGAN_FP_SIZE))

        if input_dim != MORGAN_FP_SIZE:
            raise ValueError(
                f"P_success input_dim={input_dim}, but MCTS uses {MORGAN_FP_SIZE}."
            )

        self.model = PSuccessNet(
            input_dim=input_dim,
            hidden_layers=(
                int(hidden_layers[0]),
                int(hidden_layers[1]),
                int(hidden_layers[2]),
            ),
            dropout_1=float(dropout[0]),
            dropout_2=float(dropout[1]),
        )

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS,
            fpSize=MORGAN_FP_SIZE,
        )

        # EN: Cache prior predictions by canonical SMILES.
        # CN: 按 canonical SMILES 缓存先验预测结果。
        self._cache: dict[str, dict[str, Any]] = {}

        self.artifact_metadata = {
            "model_version": artifact.get("model_version") if isinstance(artifact, dict) else None,
            "model_role": artifact.get("model_role") if isinstance(artifact, dict) else None,
            "primary_success_criterion": artifact.get("primary_success_criterion") if isinstance(artifact, dict) else None,
        }

    def _fingerprint(self, canonical_smiles: str) -> np.ndarray:
        """
        Convert canonical SMILES into a Morgan bit fingerprint.
        将 canonical SMILES 转换为 Morgan bit 指纹。
        """
        mol = Chem.MolFromSmiles(canonical_smiles)

        if mol is None:
            raise ValueError(f"Invalid canonical SMILES: {canonical_smiles}")

        fp = self._generator.GetFingerprint(mol)
        array = np.zeros((MORGAN_FP_SIZE,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, array)

        return array.reshape(1, -1)

    def predict(self, smiles: str) -> dict[str, Any]:
        """
        Predict P_success for one molecule.
        预测一个分子的 P_success。
        """
        canonical = canonicalize_smiles(smiles)

        if canonical is None:
            return {
                "prior_status": "invalid_smiles",
                "canonical_smiles": None,
                "p_success": float("nan"),
            }

        if canonical in self._cache:
            return copy.deepcopy(self._cache[canonical])

        fingerprint = self._fingerprint(canonical)

        with torch.no_grad():
            logits = self.model(torch.tensor(fingerprint, dtype=torch.float32))
            probability = float(torch.sigmoid(logits).item())

        result = {
            "prior_status": "ok",
            "canonical_smiles": canonical,
            "p_success": probability,
        }

        self._cache[canonical] = result

        return copy.deepcopy(result)

    def metadata(self) -> dict[str, Any]:
        """
        Return prior-model metadata for reproducibility.
        返回先验模型元信息，用于可复现性记录。
        """
        return {
            "p_success_model_path": str(self.model_path),
            "molecular_representation": {
                "type": "Morgan bit fingerprint",
                "radius": MORGAN_RADIUS,
                "n_bits": MORGAN_FP_SIZE,
            },
            "artifact_metadata": self.artifact_metadata,
            "cache_size": len(self._cache),
        }


def evaluate_node(
    node: MCTSNode,
    fitness: FrozenFitnessAdapter,
    best_state: dict[str, Any],
) -> float:
    """
    Directly evaluate a leaf node using the frozen fitness function.
    使用冻结适应度函数直接评价一个叶节点。

    The returned reward is final_reward, not P_success.
    返回的 reward 是 final_reward，而不是 P_success。
    """
    score_record = fitness.score(node.smiles)
    node.score_record = copy.deepcopy(score_record)

    scoring_status = str(score_record.get("scoring_status", "unknown"))
    final_reward = safe_float(score_record.get("final_reward"), default=0.0)

    if scoring_status != "ok" or not np.isfinite(final_reward):
        final_reward = 0.0

    if final_reward > float(best_state.get("best_reward", -float("inf"))):
        best_state["best_reward"] = final_reward
        best_state["best_node"] = node

    return final_reward