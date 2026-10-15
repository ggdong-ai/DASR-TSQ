from __future__ import annotations

"""
本脚本只进行“离线训练与 validation 早停”，不会读取 test.csv 的内容，
也不会修改任何 MCTS 文件。所有模型使用 Step 23 已冻结的同一份
parent-conditioned sibling 数据集。

训练/保存的方法
---------------
B1 action_only    : 每种 mutation action 在训练集上的平均 ΔF（非神经网络基线）
B2 absolute_child : 仅根据 child FP + action 预测 child_final_reward
M1 pointwise_delta: 根据 parent/child/XOR/action 预测 delta_final_reward
M2 pairwise_ranker: RankNet，同一 parent 内比较两个 children
M3 listwise_ranker: ListNet，同一 parent 的完整 sibling list 直接优化

说明
----
1. Random baseline 不需要训练；脚本会在 validation_predictions.csv 中保存一列
   固定种子的 random_score，供 Step 25 统一评价。
2. test.csv 仅记录文件哈希，不会被 pandas 读取，防止调参阶段窥视测试集。
3. 模型选择只使用 validation NDCG@8，并使用 early stopping。
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import rdkit
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


# ============================================================
# 0. Paths and constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "01_data" / "sibling_ranking"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "03_models" / "dasr_sibling_rankers"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "05_results" / "dasr_sibling_rankers"

MORGAN_RADIUS = 3
MORGAN_FP_SIZE = 1024
EPS = 1e-12

SUPPORTED_MODELS = (
    "action_only",
    "absolute_child",
    "pointwise_delta",
    "pairwise_ranker",
    "listwise_ranker",
)

# Step 24 只读取这些列，避免无谓占用内存。
REQUIRED_COLUMNS = [
    "sibling_group_id",
    "split",
    "parent_smiles",
    "parent_scaffold",
    "child_smiles",
    "mutation_action",
    "parent_final_reward",
    "child_final_reward",
    "delta_final_reward",
    "scalar_improve_001",
    "strict_pareto_improve",
    "child_tier1",
    "tier1_gain",
    "rank_delta_final_reward",
]


# ============================================================
# 1. Reproducibility and utility functions
# ============================================================


def set_all_seeds(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # warn_only=True：若某个算子没有完全确定性实现，只警告而不中断。
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    """计算文件 SHA256；文件不存在时返回 None。"""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    """只清理 Step 24 自己的输出目录，不触碰冻结数据。"""
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}\n"
                "Use --overwrite only when you intentionally want to rerun Step 24."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_hidden_dims(values: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(v) for v in values)
    if not dims or any(v <= 0 for v in dims):
        raise ValueError("--hidden_dims must contain one or more positive integers.")
    return dims


# ============================================================
# 2. Configuration
# ============================================================


@dataclass(frozen=True)
class TrainConfig:
    mode: str
    seed: int
    device: str
    top_k: int
    max_epochs: int
    patience: int
    batch_size: int
    group_batch_size: int
    pairs_per_group: int
    learning_rate: float
    weight_decay: float
    hidden_dims: tuple[int, ...]
    dropout: float
    target_temperature: float
    gradient_clip_norm: float
    smoke_train_groups: int
    smoke_validation_groups: int


# ============================================================
# 3. Data loading and validation
# ============================================================


def load_split_csv(path: Path, expected_split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required split file was not found: {path}")

    header = pd.read_csv(path, nrows=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in header.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError(f"{path.name} is empty.")

    observed_splits = set(df["split"].astype(str).unique())
    if observed_splits != {expected_split}:
        raise ValueError(
            f"{path.name} contains unexpected split labels: {observed_splits}; "
            f"expected only {expected_split!r}."
        )

    text_cols = [
        "sibling_group_id",
        "parent_smiles",
        "parent_scaffold",
        "child_smiles",
        "mutation_action",
    ]
    numeric_cols = [
        "parent_final_reward",
        "child_final_reward",
        "delta_final_reward",
        "scalar_improve_001",
        "strict_pareto_improve",
        "child_tier1",
        "tier1_gain",
        "rank_delta_final_reward",
    ]

    for column in text_cols:
        if df[column].isna().any():
            raise ValueError(f"{path.name}: NaN detected in {column}.")
        df[column] = df[column].astype(str)

    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{path.name}: non-finite values detected in {column}.")

    # 每个 sibling group 必须只对应一个 parent。
    parents_per_group = df.groupby("sibling_group_id")["parent_smiles"].nunique()
    if int(parents_per_group.max()) != 1:
        bad = parents_per_group[parents_per_group != 1].head().to_dict()
        raise ValueError(f"A sibling group maps to multiple parents: {bad}")

    return df.reset_index(drop=True)


def select_smoke_groups(df: pd.DataFrame, n_groups: int, seed: int) -> pd.DataFrame:
    """从已冻结 split 中按 group 抽取小规模数据，仅用于代码通路测试。"""
    group_ids = np.asarray(sorted(df["sibling_group_id"].unique()), dtype=object)
    if n_groups >= len(group_ids):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(group_ids, size=n_groups, replace=False).tolist())
    return df[df["sibling_group_id"].isin(chosen)].reset_index(drop=True)


def validate_train_validation_separation(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    group_overlap = set(train_df["sibling_group_id"]) & set(val_df["sibling_group_id"])
    parent_overlap = set(train_df["parent_smiles"]) & set(val_df["parent_smiles"])
    scaffold_overlap = set(train_df["parent_scaffold"]) & set(val_df["parent_scaffold"])
    child_overlap = set(train_df["child_smiles"]) & set(val_df["child_smiles"])

    errors: list[str] = []
    if group_overlap:
        errors.append(f"group overlap={len(group_overlap)}")
    if parent_overlap:
        errors.append(f"parent overlap={len(parent_overlap)}")
    if scaffold_overlap:
        errors.append(f"parent scaffold overlap={len(scaffold_overlap)}")
    if child_overlap:
        errors.append(f"exact child overlap={len(child_overlap)}")

    if errors:
        raise ValueError("Train/validation leakage detected: " + "; ".join(errors))


# ============================================================
# 4. Fingerprint feature bank
# ============================================================


class FeatureBank:
    """
    在 train + validation 的 SMILES 上一次性建立 Morgan FP 缓存。

    注意：这里不会读取 test.csv。FP 使用与冻结 oracle 一致的 radius=3、1024 bits。
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        action_vocab: list[str],
        show_rdkit_errors: bool,
    ) -> None:
        self.action_vocab = list(action_vocab)
        self.action_to_index = {action: i for i, action in enumerate(self.action_vocab)}
        self.unknown_action_index = self.action_to_index["__UNK__"]

        if not show_rdkit_errors:
            RDLogger.DisableLog("rdApp.warning")

        all_smiles = pd.concat(
            [
                train_df["parent_smiles"],
                train_df["child_smiles"],
                val_df["parent_smiles"],
                val_df["child_smiles"],
            ],
            ignore_index=True,
        ).drop_duplicates()
        self.smiles_list = all_smiles.astype(str).tolist()
        self.smiles_to_index = {s: i for i, s in enumerate(self.smiles_list)}

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS,
            fpSize=MORGAN_FP_SIZE,
        )
        matrix = np.zeros((len(self.smiles_list), MORGAN_FP_SIZE), dtype=np.uint8)

        print(f"Building Morgan fingerprint cache for {len(self.smiles_list):,} unique SMILES ...")
        start = time.time()
        for i, smiles in enumerate(self.smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES encountered in frozen dataset: {smiles}")
            fp = generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, matrix[i])
            if (i + 1) % 10000 == 0:
                print(f"  fingerprinted {i + 1:,}/{len(self.smiles_list):,}")

        self.fp_matrix = matrix
        print(f"Fingerprint cache finished in {time.time() - start:.1f} s")

    @property
    def n_actions(self) -> int:
        return len(self.action_vocab)

    @property
    def pair_input_dim(self) -> int:
        return MORGAN_FP_SIZE * 3 + self.n_actions

    @property
    def child_input_dim(self) -> int:
        return MORGAN_FP_SIZE + self.n_actions

    def encode_dataframe(self, df: pd.DataFrame) -> "EncodedRows":
        parent_fp_index = np.asarray(
            [self.smiles_to_index[s] for s in df["parent_smiles"].astype(str)],
            dtype=np.int32,
        )
        child_fp_index = np.asarray(
            [self.smiles_to_index[s] for s in df["child_smiles"].astype(str)],
            dtype=np.int32,
        )
        action_index = np.asarray(
            [self.action_to_index.get(a, self.unknown_action_index) for a in df["mutation_action"].astype(str)],
            dtype=np.int16,
        )
        return EncodedRows(
            parent_fp_index=parent_fp_index,
            child_fp_index=child_fp_index,
            action_index=action_index,
            child_final_reward=df["child_final_reward"].to_numpy(dtype=np.float32),
            delta_final_reward=df["delta_final_reward"].to_numpy(dtype=np.float32),
        )

    def pair_features(
        self,
        encoded: "EncodedRows",
        row_indices: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        p_np = self.fp_matrix[encoded.parent_fp_index[row_indices]]
        c_np = self.fp_matrix[encoded.child_fp_index[row_indices]]
        xor_np = np.bitwise_xor(p_np, c_np)

        p = torch.from_numpy(p_np).to(device=device, dtype=torch.float32)
        c = torch.from_numpy(c_np).to(device=device, dtype=torch.float32)
        xor = torch.from_numpy(xor_np).to(device=device, dtype=torch.float32)
        action = torch.from_numpy(encoded.action_index[row_indices].astype(np.int64)).to(device)
        action_one_hot = F.one_hot(action, num_classes=self.n_actions).to(torch.float32)
        return torch.cat([p, c, xor, action_one_hot], dim=1)

    def child_features(
        self,
        encoded: "EncodedRows",
        row_indices: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        c_np = self.fp_matrix[encoded.child_fp_index[row_indices]]
        c = torch.from_numpy(c_np).to(device=device, dtype=torch.float32)
        action = torch.from_numpy(encoded.action_index[row_indices].astype(np.int64)).to(device)
        action_one_hot = F.one_hot(action, num_classes=self.n_actions).to(torch.float32)
        return torch.cat([c, action_one_hot], dim=1)


@dataclass
class EncodedRows:
    parent_fp_index: np.ndarray
    child_fp_index: np.ndarray
    action_index: np.ndarray
    child_final_reward: np.ndarray
    delta_final_reward: np.ndarray


# ============================================================
# 5. Neural scoring model
# ============================================================


class ScoringMLP(nn.Module):
    """输入一个候选的特征，输出一个标量分数。"""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ============================================================
# 6. Ranking metrics used only for validation early stopping
# ============================================================


def _dcg(relevance: np.ndarray) -> float:
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2, dtype=np.float64))
    return float(np.sum(relevance.astype(np.float64) / discounts))


def compute_group_metrics(
    df: pd.DataFrame,
    scores: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    """
    计算 group-macro 指标；每个 parent group 权重相同。

    NDCG relevance 使用 group 内平移后的 ΔF：rel = ΔF - min(ΔF)，
    因而不会因负 ΔF 导致非标准 DCG。
    """
    if len(df) != len(scores):
        raise ValueError("scores length does not match dataframe length.")

    score_series = pd.Series(scores, index=df.index)
    ndcg_values: list[float] = []
    best_recall_values: list[float] = []
    regret_values: list[float] = []
    selected_delta_values: list[float] = []
    selected_improve_values: list[float] = []
    selected_pareto_values: list[float] = []
    selected_tier1_gain_values: list[float] = []

    for _, group in df.groupby("sibling_group_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        y = group["delta_final_reward"].to_numpy(dtype=np.float64)
        s = score_series.loc[idx].to_numpy(dtype=np.float64)
        k = min(top_k, len(group))

        # 稳定排序：分数相同则按原行号，保证可复现。
        predicted_order = np.lexsort((idx, -s))[:k]
        ideal_order = np.lexsort((idx, -y))[:k]

        relevance = y - float(np.min(y))
        ideal_dcg = _dcg(relevance[ideal_order])
        ndcg = 1.0 if ideal_dcg <= EPS else _dcg(relevance[predicted_order]) / ideal_dcg
        ndcg_values.append(float(ndcg))

        true_max = float(np.max(y))
        selected_max = float(np.max(y[predicted_order]))
        regret_values.append(true_max - selected_max)
        best_recall_values.append(float(np.isclose(selected_max, true_max, atol=1e-12)))

        selected_delta_values.append(float(np.mean(y[predicted_order])))
        selected_improve_values.append(
            float(np.mean(group["scalar_improve_001"].to_numpy(dtype=np.float64)[predicted_order]))
        )
        selected_pareto_values.append(
            float(np.mean(group["strict_pareto_improve"].to_numpy(dtype=np.float64)[predicted_order]))
        )
        selected_tier1_gain_values.append(
            float(np.mean(group["tier1_gain"].to_numpy(dtype=np.float64)[predicted_order]))
        )

    return {
        f"ndcg_at_{top_k}": float(np.mean(ndcg_values)),
        f"best_child_recall_at_{top_k}": float(np.mean(best_recall_values)),
        f"regret_at_{top_k}": float(np.mean(regret_values)),
        f"selected_mean_delta_at_{top_k}": float(np.mean(selected_delta_values)),
        f"selected_scalar_improve_001_rate_at_{top_k}": float(np.mean(selected_improve_values)),
        f"selected_pareto_rate_at_{top_k}": float(np.mean(selected_pareto_values)),
        f"selected_tier1_gain_rate_at_{top_k}": float(np.mean(selected_tier1_gain_values)),
    }


# ============================================================
# 7. Common prediction and artifact helpers
# ============================================================


def predict_rows(
    model: nn.Module,
    feature_bank: FeatureBank,
    encoded: EncodedRows,
    n_rows: int,
    batch_size: int,
    device: torch.device,
    feature_kind: str,
) -> np.ndarray:
    model.eval()
    output = np.empty(n_rows, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            indices = np.arange(start, end, dtype=np.int64)
            if feature_kind == "child":
                x = feature_bank.child_features(encoded, indices, device)
            elif feature_kind == "pair":
                x = feature_bank.pair_features(encoded, indices, device)
            else:
                raise ValueError(f"Unknown feature_kind: {feature_kind}")
            output[start:end] = model(x).detach().cpu().numpy().astype(np.float32)
    return output


def save_model_artifact(
    path: Path,
    model_name: str,
    model: nn.Module,
    input_dim: int,
    feature_kind: str,
    config: TrainConfig,
    action_vocab: list[str],
    target_mean: float | None,
    target_std: float | None,
    best_epoch: int,
    best_validation_metrics: dict[str, float],
    data_hashes: dict[str, str | None],
) -> None:
    artifact = {
        "artifact_version": 1,
        "scientific_role": "offline sibling-ranking model; not an activity oracle",
        "model_name": model_name,
        "model_class": "ScoringMLP",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "architecture": {
            "input_dim": input_dim,
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
            "output": "single scalar score",
        },
        "feature_spec": {
            "kind": feature_kind,
            "morgan_radius": MORGAN_RADIUS,
            "morgan_fp_size": MORGAN_FP_SIZE,
            "pair_features": "parent_fp || child_fp || xor_fp || mutation_action_one_hot",
            "child_features": "child_fp || mutation_action_one_hot",
            "action_vocab": action_vocab,
        },
        "target_scaler": {
            "mean": target_mean,
            "std": target_std,
        },
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "train_config": asdict(config),
        "data_hashes": data_hashes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


# ============================================================
# 8. B1 Action-only baseline
# ============================================================


def fit_action_only(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray]:
    action_means = (
        train_df.groupby("mutation_action")["delta_final_reward"].mean().sort_index().to_dict()
    )
    action_counts = train_df["mutation_action"].value_counts().sort_index().to_dict()
    global_mean = float(train_df["delta_final_reward"].mean())
    scores = val_df["mutation_action"].map(action_means).fillna(global_mean).to_numpy(dtype=np.float32)
    artifact = {
        "model_name": "action_only",
        "scientific_role": "baseline detecting whether structural models only learn action priors",
        "target": "mean train delta_final_reward by mutation_action",
        "global_mean": global_mean,
        "action_means": {str(k): float(v) for k, v in action_means.items()},
        "action_counts": {str(k): int(v) for k, v in action_counts.items()},
    }
    return artifact, scores


# ============================================================
# 9. Pointwise regression training (absolute child / delta)
# ============================================================


def train_pointwise_regressor(
    *,
    model_name: str,
    feature_kind: str,
    target_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_encoded: EncodedRows,
    val_encoded: EncodedRows,
    feature_bank: FeatureBank,
    config: TrainConfig,
    device: torch.device,
    data_hashes: dict[str, str | None],
    model_dir: Path,
    result_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if target_name == "child_final_reward":
        train_target = train_encoded.child_final_reward.astype(np.float32)
        val_target = val_encoded.child_final_reward.astype(np.float32)
    elif target_name == "delta_final_reward":
        train_target = train_encoded.delta_final_reward.astype(np.float32)
        val_target = val_encoded.delta_final_reward.astype(np.float32)
    else:
        raise ValueError(f"Unsupported target: {target_name}")

    target_mean = float(np.mean(train_target))
    target_std = float(np.std(train_target))
    if target_std < EPS:
        raise ValueError(f"Training target {target_name} has near-zero standard deviation.")

    input_dim = (
        feature_bank.child_input_dim if feature_kind == "child" else feature_bank.pair_input_dim
    )
    model = ScoringMLP(input_dim, config.hidden_dims, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model_seed_offset = {"absolute_child": 101, "pointwise_delta": 202}[model_name]
    rng = np.random.default_rng(config.seed + model_seed_offset)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_ndcg = -math.inf
    best_metrics: dict[str, float] = {}
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = rng.permutation(len(train_df)).astype(np.int64)
        epoch_loss_sum = 0.0
        epoch_count = 0

        for start in range(0, len(order), config.batch_size):
            row_indices = order[start : start + config.batch_size]
            if feature_kind == "child":
                x = feature_bank.child_features(train_encoded, row_indices, device)
            else:
                x = feature_bank.pair_features(train_encoded, row_indices, device)

            y_np = (train_target[row_indices] - target_mean) / target_std
            y = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = F.smooth_l1_loss(prediction, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            epoch_loss_sum += float(loss.detach().cpu()) * len(row_indices)
            epoch_count += len(row_indices)

        train_loss = epoch_loss_sum / max(1, epoch_count)
        val_standardized = predict_rows(
            model,
            feature_bank,
            val_encoded,
            len(val_df),
            config.batch_size,
            device,
            feature_kind,
        )
        val_prediction = val_standardized * target_std + target_mean
        val_rmse = float(np.sqrt(np.mean((val_prediction - val_target) ** 2)))
        metrics = compute_group_metrics(val_df, val_prediction, config.top_k)
        ndcg_key = f"ndcg_at_{config.top_k}"
        current_ndcg = metrics[ndcg_key]

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_rmse": val_rmse,
            **metrics,
        }
        history.append(history_row)
        print(
            f"[{model_name}] epoch {epoch:03d} | loss={train_loss:.6f} | "
            f"val_RMSE={val_rmse:.6f} | val_NDCG@{config.top_k}={current_ndcg:.6f}"
        )

        if current_ndcg > best_ndcg + 1e-8:
            best_ndcg = current_ndcg
            best_epoch = epoch
            best_metrics = {"validation_rmse": val_rmse, **metrics}
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"[{model_name}] early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"No valid model state was produced for {model_name}.")
    model.load_state_dict(best_state)

    best_standardized = predict_rows(
        model,
        feature_bank,
        val_encoded,
        len(val_df),
        config.batch_size,
        device,
        feature_kind,
    )
    best_prediction = best_standardized * target_std + target_mean

    pd.DataFrame(history).to_csv(result_dir / f"{model_name}_history.csv", index=False)
    save_model_artifact(
        path=model_dir / f"{model_name}.pt",
        model_name=model_name,
        model=model,
        input_dim=input_dim,
        feature_kind=feature_kind,
        config=config,
        action_vocab=feature_bank.action_vocab,
        target_mean=target_mean,
        target_std=target_std,
        best_epoch=best_epoch,
        best_validation_metrics=best_metrics,
        data_hashes=data_hashes,
    )

    summary = {
        "model_name": model_name,
        "target": target_name,
        "feature_kind": feature_kind,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
    }
    return best_prediction.astype(np.float32), summary


# ============================================================
# 10. M2 Pairwise RankNet
# ============================================================


def build_group_row_indices(df: pd.DataFrame) -> list[np.ndarray]:
    return [group.index.to_numpy(dtype=np.int64) for _, group in df.groupby("sibling_group_id", sort=False)]


def sample_pair_indices(
    group_indices: list[np.ndarray],
    targets: np.ndarray,
    pairs_per_group: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    每组包含一个 best-vs-worst 锚点，其余为均匀随机非平局 pairs。
    这样既保证强排序信号，也不会只训练极端样本。
    """
    left: list[int] = []
    right: list[int] = []
    labels: list[float] = []

    for idx in group_indices:
        y = targets[idx]
        best_local = int(np.argmax(y))
        worst_local = int(np.argmin(y))
        if not np.isclose(y[best_local], y[worst_local], atol=1e-12):
            left.append(int(idx[best_local]))
            right.append(int(idx[worst_local]))
            labels.append(1.0)

        required = max(0, pairs_per_group - 1)
        added = 0
        attempts = 0
        max_attempts = max(100, required * 20)
        while added < required and attempts < max_attempts:
            attempts += 1
            a, b = rng.choice(len(idx), size=2, replace=False)
            ya = float(y[a])
            yb = float(y[b])
            if np.isclose(ya, yb, atol=1e-12):
                continue
            left.append(int(idx[a]))
            right.append(int(idx[b]))
            labels.append(1.0 if ya > yb else 0.0)
            added += 1

    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(labels, dtype=np.float32),
    )


def train_pairwise_ranker(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_encoded: EncodedRows,
    val_encoded: EncodedRows,
    feature_bank: FeatureBank,
    config: TrainConfig,
    device: torch.device,
    data_hashes: dict[str, str | None],
    model_dir: Path,
    result_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_name = "pairwise_ranker"
    model = ScoringMLP(feature_bank.pair_input_dim, config.hidden_dims, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    group_indices = build_group_row_indices(train_df)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_ndcg = -math.inf
    best_metrics: dict[str, float] = {}
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
        rng = np.random.default_rng(config.seed + 1000 + epoch)
        left, right, labels = sample_pair_indices(
            group_indices,
            train_encoded.delta_final_reward,
            config.pairs_per_group,
            rng,
        )
        if len(left) == 0:
            raise RuntimeError("Pairwise sampling produced zero non-tied pairs.")
        order = rng.permutation(len(left))

        model.train()
        loss_sum = 0.0
        count = 0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            left_idx = left[batch]
            right_idx = right[batch]
            label = torch.from_numpy(labels[batch]).to(device=device, dtype=torch.float32)

            combined_idx = np.concatenate([left_idx, right_idx]).astype(np.int64)
            x_pair = feature_bank.pair_features(train_encoded, combined_idx, device)

            optimizer.zero_grad(set_to_none=True)
            combined_score = model(x_pair)
            split_at = len(left_idx)
            logit = combined_score[:split_at] - combined_score[split_at:]
            loss = F.binary_cross_entropy_with_logits(logit, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            loss_sum += float(loss.detach().cpu()) * len(batch)
            count += len(batch)

        train_loss = loss_sum / max(1, count)
        val_scores = predict_rows(
            model,
            feature_bank,
            val_encoded,
            len(val_df),
            config.batch_size,
            device,
            "pair",
        )
        metrics = compute_group_metrics(val_df, val_scores, config.top_k)
        current_ndcg = metrics[f"ndcg_at_{config.top_k}"]
        history.append(
            {
                "epoch": epoch,
                "n_training_pairs": len(left),
                "train_loss": train_loss,
                **metrics,
            }
        )
        print(
            f"[{model_name}] epoch {epoch:03d} | pairs={len(left):,} | "
            f"loss={train_loss:.6f} | val_NDCG@{config.top_k}={current_ndcg:.6f}"
        )

        if current_ndcg > best_ndcg + 1e-8:
            best_ndcg = current_ndcg
            best_epoch = epoch
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"[{model_name}] early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("No valid pairwise model state was produced.")
    model.load_state_dict(best_state)
    best_scores = predict_rows(
        model,
        feature_bank,
        val_encoded,
        len(val_df),
        config.batch_size,
        device,
        "pair",
    )

    pd.DataFrame(history).to_csv(result_dir / f"{model_name}_history.csv", index=False)
    save_model_artifact(
        path=model_dir / f"{model_name}.pt",
        model_name=model_name,
        model=model,
        input_dim=feature_bank.pair_input_dim,
        feature_kind="pair",
        config=config,
        action_vocab=feature_bank.action_vocab,
        target_mean=None,
        target_std=None,
        best_epoch=best_epoch,
        best_validation_metrics=best_metrics,
        data_hashes=data_hashes,
    )
    summary = {
        "model_name": model_name,
        "loss": "RankNet binary cross-entropy on within-parent pairs",
        "pair_sampling": "1 best-vs-worst anchor plus uniform random non-tied pairs per group per epoch",
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
    }
    return best_scores.astype(np.float32), summary


# ============================================================
# 11. M3 Listwise ListNet
# ============================================================


def train_listwise_ranker(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_encoded: EncodedRows,
    val_encoded: EncodedRows,
    feature_bank: FeatureBank,
    config: TrainConfig,
    device: torch.device,
    data_hashes: dict[str, str | None],
    model_dir: Path,
    result_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_name = "listwise_ranker"
    model = ScoringMLP(feature_bank.pair_input_dim, config.hidden_dims, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    group_indices = build_group_row_indices(train_df)
    global_delta_std = float(np.std(train_encoded.delta_final_reward))
    if global_delta_std < EPS:
        raise ValueError("delta_final_reward has near-zero standard deviation.")

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_ndcg = -math.inf
    best_metrics: dict[str, float] = {}
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
        rng = np.random.default_rng(config.seed + 2000 + epoch)
        group_order = rng.permutation(len(group_indices))
        model.train()
        loss_sum = 0.0
        group_count = 0

        for start in range(0, len(group_order), config.group_batch_size):
            batch_group_ids = group_order[start : start + config.group_batch_size]
            groups = [group_indices[i] for i in batch_group_ids]
            flat_indices = np.concatenate(groups).astype(np.int64)
            x = feature_bank.pair_features(train_encoded, flat_indices, device)

            optimizer.zero_grad(set_to_none=True)
            flat_scores = model(x)

            losses: list[torch.Tensor] = []
            offset = 0
            for group_idx in groups:
                length = len(group_idx)
                scores = flat_scores[offset : offset + length]
                y_np = train_encoded.delta_final_reward[group_idx].astype(np.float32)
                y_np = (y_np - float(np.mean(y_np))) / (
                    global_delta_std * config.target_temperature + EPS
                )
                y_np = np.clip(y_np, -8.0, 8.0)
                target_logits = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)
                target_prob = F.softmax(target_logits, dim=0)
                log_pred_prob = F.log_softmax(scores, dim=0)
                losses.append(-(target_prob * log_pred_prob).sum())
                offset += length

            loss = torch.stack(losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            loss_sum += float(loss.detach().cpu()) * len(groups)
            group_count += len(groups)

        train_loss = loss_sum / max(1, group_count)
        val_scores = predict_rows(
            model,
            feature_bank,
            val_encoded,
            len(val_df),
            config.batch_size,
            device,
            "pair",
        )
        metrics = compute_group_metrics(val_df, val_scores, config.top_k)
        current_ndcg = metrics[f"ndcg_at_{config.top_k}"]
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(
            f"[{model_name}] epoch {epoch:03d} | loss={train_loss:.6f} | "
            f"val_NDCG@{config.top_k}={current_ndcg:.6f}"
        )

        if current_ndcg > best_ndcg + 1e-8:
            best_ndcg = current_ndcg
            best_epoch = epoch
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"[{model_name}] early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("No valid listwise model state was produced.")
    model.load_state_dict(best_state)
    best_scores = predict_rows(
        model,
        feature_bank,
        val_encoded,
        len(val_df),
        config.batch_size,
        device,
        "pair",
    )

    pd.DataFrame(history).to_csv(result_dir / f"{model_name}_history.csv", index=False)
    save_model_artifact(
        path=model_dir / f"{model_name}.pt",
        model_name=model_name,
        model=model,
        input_dim=feature_bank.pair_input_dim,
        feature_kind="pair",
        config=config,
        action_vocab=feature_bank.action_vocab,
        target_mean=None,
        target_std=None,
        best_epoch=best_epoch,
        best_validation_metrics=best_metrics,
        data_hashes=data_hashes,
    )
    summary = {
        "model_name": model_name,
        "loss": "ListNet top-one cross-entropy over each complete sibling group",
        "target_logits": "group-centered delta_final_reward / (global train delta std * target_temperature)",
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
    }
    return best_scores.astype(np.float32), summary


# ============================================================
# 12. Main workflow
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DASR parent-conditioned sibling-ranking models without touching test labels."
    )
    parser.add_argument("--mode", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--result_root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--group_batch_size", type=int, default=16)
    parser.add_argument("--pairs_per_group", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 32])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--target_temperature", type=float, default=1.0)
    parser.add_argument("--gradient_clip_norm", type=float, default=5.0)
    parser.add_argument("--smoke_train_groups", type=int, default=120)
    parser.add_argument("--smoke_validation_groups", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show_rdkit_errors", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    max_epochs = args.max_epochs if args.max_epochs is not None else (3 if args.mode == "smoke" else 60)
    patience = args.patience if args.patience is not None else (2 if args.mode == "smoke" else 10)
    config = TrainConfig(
        mode=args.mode,
        seed=args.seed,
        device=args.device,
        top_k=args.top_k,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=args.batch_size,
        group_batch_size=args.group_batch_size,
        pairs_per_group=args.pairs_per_group,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        target_temperature=args.target_temperature,
        gradient_clip_norm=args.gradient_clip_norm,
        smoke_train_groups=args.smoke_train_groups,
        smoke_validation_groups=args.smoke_validation_groups,
    )

    set_all_seeds(config.seed)
    device = resolve_device(config.device)

    data_dir = args.data_dir.resolve()
    model_dir = (args.model_root / f"step24_{args.mode}").resolve()
    result_dir = (args.result_root / f"step24_{args.mode}").resolve()
    prepare_output_dir(model_dir, args.overwrite)
    prepare_output_dir(result_dir, args.overwrite)

    train_path = data_dir / "train.csv"
    validation_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"
    validation_report_path = data_dir / "validation_report.json"

    # Formal 训练前必须确认 Step 23 验证报告通过。
    if not validation_report_path.exists():
        raise FileNotFoundError(f"Missing validation report: {validation_report_path}")
    with validation_report_path.open("r", encoding="utf-8") as handle:
        validation_report = json.load(handle)
    if not bool(validation_report.get("passed", False)):
        raise RuntimeError("Step 23 validation_report.json does not have passed=true.")
    if validation_report.get("critical_errors"):
        raise RuntimeError(
            f"Step 23 validation report contains critical errors: "
            f"{validation_report['critical_errors']}"
        )

    print("Loading frozen train and validation splits ...")
    train_df = load_split_csv(train_path, "train")
    val_df = load_split_csv(validation_path, "validation")
    validate_train_validation_separation(train_df, val_df)

    if args.mode == "smoke":
        train_df = select_smoke_groups(train_df, config.smoke_train_groups, config.seed)
        val_df = select_smoke_groups(val_df, config.smoke_validation_groups, config.seed + 1)
        validate_train_validation_separation(train_df, val_df)

    train_group_count = int(train_df["sibling_group_id"].nunique())
    val_group_count = int(val_df["sibling_group_id"].nunique())
    if train_group_count < 2 or val_group_count < 2:
        raise ValueError("Too few groups for training or validation.")
    if int(val_df.groupby("sibling_group_id").size().min()) < config.top_k:
        raise ValueError(
            f"At least one validation group has fewer than top_k={config.top_k} candidates."
        )

    print("=" * 88)
    print("Step 24: DASR sibling-ranker training")
    print(f"Mode: {config.mode}")
    print(f"Device: {device}")
    print(f"Train: {len(train_df):,} rows, {train_group_count:,} groups")
    print(f"Validation: {len(val_df):,} rows, {val_group_count:,} groups")
    print(f"Models: {args.models}")
    print(f"Model output: {model_dir}")
    print(f"Result output: {result_dir}")
    print("Test CSV content is NOT loaded in Step 24.")
    print("=" * 88)

    action_vocab = sorted(train_df["mutation_action"].unique().tolist()) + ["__UNK__"]
    feature_bank = FeatureBank(train_df, val_df, action_vocab, args.show_rdkit_errors)
    train_encoded = feature_bank.encode_dataframe(train_df)
    val_encoded = feature_bank.encode_dataframe(val_df)

    data_hashes = {
        "train_csv": sha256_file(train_path),
        "validation_csv": sha256_file(validation_path),
        "test_csv_not_loaded": sha256_file(test_path),
        "validation_report_json": sha256_file(validation_report_path),
        "step24_script": sha256_file(Path(__file__).resolve()),
    }

    selected_models = list(dict.fromkeys(args.models))
    validation_predictions = val_df[
        [
            "sibling_group_id",
            "parent_smiles",
            "child_smiles",
            "mutation_action",
            "child_final_reward",
            "delta_final_reward",
            "scalar_improve_001",
            "strict_pareto_improve",
            "child_tier1",
            "tier1_gain",
            "rank_delta_final_reward",
        ]
    ].copy()

    # 固定 random baseline；正式不确定性与重复随机基线由 Step 25 完成。
    random_rng = np.random.default_rng(config.seed)
    validation_predictions["random_score"] = random_rng.random(len(val_df))

    random_metrics = compute_group_metrics(
        val_df,
        validation_predictions["random_score"].to_numpy(dtype=np.float32),
        config.top_k,
    )
    summaries: list[dict[str, Any]] = [
        {
            "model_name": "random_baseline",
            "seed": config.seed,
            "best_validation_metrics": random_metrics,
        }
    ]
    print(
        f"[random_baseline] validation NDCG@{config.top_k}="
        f"{random_metrics[f'ndcg_at_{config.top_k}']:.6f}"
    )

    if "action_only" in selected_models:
        artifact, scores = fit_action_only(train_df, val_df)
        write_json(model_dir / "action_only.json", artifact)
        metrics = compute_group_metrics(val_df, scores, config.top_k)
        summaries.append({"model_name": "action_only", "best_validation_metrics": metrics})
        validation_predictions["action_only_score"] = scores
        print(f"[action_only] validation NDCG@{config.top_k}={metrics[f'ndcg_at_{config.top_k}']:.6f}")

    if "absolute_child" in selected_models:
        scores, summary = train_pointwise_regressor(
            model_name="absolute_child",
            feature_kind="child",
            target_name="child_final_reward",
            train_df=train_df,
            val_df=val_df,
            train_encoded=train_encoded,
            val_encoded=val_encoded,
            feature_bank=feature_bank,
            config=config,
            device=device,
            data_hashes=data_hashes,
            model_dir=model_dir,
            result_dir=result_dir,
        )
        validation_predictions["absolute_child_score"] = scores
        summaries.append(summary)

    if "pointwise_delta" in selected_models:
        scores, summary = train_pointwise_regressor(
            model_name="pointwise_delta",
            feature_kind="pair",
            target_name="delta_final_reward",
            train_df=train_df,
            val_df=val_df,
            train_encoded=train_encoded,
            val_encoded=val_encoded,
            feature_bank=feature_bank,
            config=config,
            device=device,
            data_hashes=data_hashes,
            model_dir=model_dir,
            result_dir=result_dir,
        )
        validation_predictions["pointwise_delta_score"] = scores
        summaries.append(summary)

    if "pairwise_ranker" in selected_models:
        scores, summary = train_pairwise_ranker(
            train_df=train_df,
            val_df=val_df,
            train_encoded=train_encoded,
            val_encoded=val_encoded,
            feature_bank=feature_bank,
            config=config,
            device=device,
            data_hashes=data_hashes,
            model_dir=model_dir,
            result_dir=result_dir,
        )
        validation_predictions["pairwise_ranker_score"] = scores
        summaries.append(summary)

    if "listwise_ranker" in selected_models:
        scores, summary = train_listwise_ranker(
            train_df=train_df,
            val_df=val_df,
            train_encoded=train_encoded,
            val_encoded=val_encoded,
            feature_bank=feature_bank,
            config=config,
            device=device,
            data_hashes=data_hashes,
            model_dir=model_dir,
            result_dir=result_dir,
        )
        validation_predictions["listwise_ranker_score"] = scores
        summaries.append(summary)

    validation_predictions.to_csv(result_dir / "validation_predictions.csv", index=False)
    pd.json_normalize(summaries, sep=".").to_csv(
        result_dir / "validation_training_summary.csv", index=False
    )

    run_config = {
        "scientific_status": "offline training and validation model selection; test labels untouched",
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "result_dir": str(result_dir),
        "selected_models": selected_models,
        "train_config": asdict(config),
        "train_rows": len(train_df),
        "train_groups": train_group_count,
        "validation_rows": len(val_df),
        "validation_groups": val_group_count,
        "action_vocab": action_vocab,
        "feature_dimensions": {
            "absolute_child": feature_bank.child_input_dim,
            "parent_conditioned": feature_bank.pair_input_dim,
        },
        "data_hashes": data_hashes,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
            "torch": torch.__version__,
        },
        "test_policy": (
            "test.csv was not loaded; only its SHA256 was recorded. "
            "Do not evaluate test until Step 25 validation-based model selection is frozen."
        ),
    }
    write_json(result_dir / "step24_run_config.json", run_config)
    write_json(result_dir / "step24_training_summary.json", summaries)

    print("=" * 88)
    print("Step 24 training finished successfully")
    print(f"Models: {model_dir}")
    print(f"Results: {result_dir}")
    print("Mandatory next step: inspect Step 24 smoke outputs before running --mode formal.")
    print("Do not modify MCTS and do not evaluate test.csv yet.")
    print("=" * 88)


if __name__ == "__main__":
    main()