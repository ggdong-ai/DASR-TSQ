"""
27_0_frozen_sibling_rankers.py
================================
Frozen inference utilities for Step 27 DASR-MCTS online expansion.
Step 27 DASR-MCTS 在线扩展使用的冻结 sibling-ranker 推理工具。

Scientific constraints
----------------------
1. This module only loads the artifacts frozen after Steps 24-26.
2. It never trains, calibrates, fine-tunes, or selects a model.
3. Ranking is pre-oracle: no GSK3β/JNK3 oracle score is queried here.
4. The feature definitions exactly match Step 24:
   - child model: child_fp || action_one_hot
   - parent-conditioned model: parent_fp || child_fp || xor_fp || action_one_hot
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

MORGAN_RADIUS = 3
MORGAN_FP_SIZE = 1024
SUPPORTED_NEURAL_MODELS = (
    "absolute_child",
    "pointwise_delta",
    "pairwise_ranker",
    "listwise_ranker",
)
SUPPORTED_RANKERS = ("action_only",) + SUPPORTED_NEURAL_MODELS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_smiles(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_torch_artifact(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen ranker artifact not found: {path}")
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict) or "state_dict" not in artifact:
        raise TypeError(f"Unsupported frozen artifact format: {path}")
    return artifact


class ScoringMLP(nn.Module):
    """Architecture frozen by Step 24. Step 27 performs inference only."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, int(hidden)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = int(hidden)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class FrozenSiblingRankerBank:
    """
    Load all frozen Step 24 rankers once and score online mutation candidates.
    一次加载全部冻结排序器，并对在线 mutation candidates 进行预 oracle 排序。
    """

    def __init__(
        self,
        freeze_dir: Path,
        device: str = "cpu",
        show_rdkit_warnings: bool = False,
    ) -> None:
        self.freeze_dir = Path(freeze_dir).resolve()
        if not self.freeze_dir.exists():
            raise FileNotFoundError(f"Freeze directory not found: {self.freeze_dir}")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

        if not show_rdkit_warnings:
            RDLogger.DisableLog("rdApp.warning")

        action_path = self.freeze_dir / "action_only.json"
        if not action_path.exists():
            raise FileNotFoundError(f"Missing action-only artifact: {action_path}")
        with action_path.open("r", encoding="utf-8") as handle:
            action_artifact = json.load(handle)
        self.action_global_mean = float(action_artifact["global_mean"])
        self.action_means = {
            str(key): float(value)
            for key, value in action_artifact.get("action_means", {}).items()
        }

        self.models: dict[str, nn.Module] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}
        action_vocab_reference: list[str] | None = None

        for model_name in SUPPORTED_NEURAL_MODELS:
            path = self.freeze_dir / f"{model_name}.pt"
            artifact = load_torch_artifact(path)
            observed_name = str(artifact.get("model_name", ""))
            if observed_name != model_name:
                raise ValueError(
                    f"Artifact name mismatch for {path.name}: {observed_name!r}"
                )

            architecture = artifact.get("architecture", {})
            input_dim = int(architecture["input_dim"])
            hidden_dims = [int(x) for x in architecture["hidden_dims"]]
            dropout = float(architecture["dropout"])
            feature_spec = artifact.get("feature_spec", {})
            if int(feature_spec.get("morgan_radius", -1)) != MORGAN_RADIUS:
                raise ValueError(f"Morgan radius mismatch in {path}")
            if int(feature_spec.get("morgan_fp_size", -1)) != MORGAN_FP_SIZE:
                raise ValueError(f"Morgan FP size mismatch in {path}")

            action_vocab = [str(x) for x in feature_spec.get("action_vocab", [])]
            if "__UNK__" not in action_vocab:
                raise ValueError(f"__UNK__ is missing from action vocabulary in {path}")
            if action_vocab_reference is None:
                action_vocab_reference = action_vocab
            elif action_vocab != action_vocab_reference:
                raise ValueError("Frozen ranker artifacts have inconsistent action vocabularies.")

            expected_kind = "child" if model_name == "absolute_child" else "pair"
            observed_kind = str(feature_spec.get("kind", ""))
            if observed_kind != expected_kind:
                raise ValueError(
                    f"Feature kind mismatch for {model_name}: {observed_kind!r}"
                )

            expected_dim = (
                MORGAN_FP_SIZE + len(action_vocab)
                if expected_kind == "child"
                else MORGAN_FP_SIZE * 3 + len(action_vocab)
            )
            if input_dim != expected_dim:
                raise ValueError(
                    f"Input dimension mismatch for {model_name}: "
                    f"artifact={input_dim}, expected={expected_dim}"
                )

            model = ScoringMLP(input_dim, hidden_dims, dropout)
            model.load_state_dict(artifact["state_dict"], strict=True)
            model.to(self.device)
            model.eval()
            self.models[model_name] = model
            self.artifacts[model_name] = artifact

        assert action_vocab_reference is not None
        self.action_vocab = action_vocab_reference
        self.action_to_index = {
            action: index for index, action in enumerate(self.action_vocab)
        }
        self.unknown_action_index = self.action_to_index["__UNK__"]

        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS,
            fpSize=MORGAN_FP_SIZE,
        )
        self.fp_cache: dict[str, np.ndarray] = {}
        self.score_cache: dict[tuple[str, str, str, str], float] = {}

        self.file_hashes = {
            "action_only.json": sha256_file(action_path),
            **{
                f"{model_name}.pt": sha256_file(
                    self.freeze_dir / f"{model_name}.pt"
                )
                for model_name in SUPPORTED_NEURAL_MODELS
            },
        }

    def _fingerprint(self, smiles: str) -> np.ndarray:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            raise ValueError(f"Invalid SMILES for frozen ranker inference: {smiles}")
        cached = self.fp_cache.get(canonical)
        if cached is not None:
            return cached
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            raise ValueError(f"RDKit failed to parse canonical SMILES: {canonical}")
        fp = self.generator.GetFingerprint(mol)
        array = np.zeros((MORGAN_FP_SIZE,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, array)
        self.fp_cache[canonical] = array
        return array

    def _action_indices(self, actions: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [
                self.action_to_index.get(str(action), self.unknown_action_index)
                for action in actions
            ],
            dtype=np.int64,
        )

    def _feature_tensor(
        self,
        model_name: str,
        parent_smiles: str,
        child_smiles: Sequence[str],
        actions: Sequence[str],
    ) -> torch.Tensor:
        child_fp = np.stack(
            [self._fingerprint(smiles) for smiles in child_smiles], axis=0
        )
        action_index = torch.from_numpy(self._action_indices(actions)).to(self.device)
        action_one_hot = F.one_hot(
            action_index,
            num_classes=len(self.action_vocab),
        ).to(torch.float32)
        child_tensor = torch.from_numpy(child_fp).to(
            device=self.device,
            dtype=torch.float32,
        )

        if model_name == "absolute_child":
            return torch.cat([child_tensor, action_one_hot], dim=1)

        parent_fp = self._fingerprint(parent_smiles)
        parent_matrix = np.repeat(parent_fp.reshape(1, -1), len(child_smiles), axis=0)
        xor_matrix = np.bitwise_xor(parent_matrix, child_fp)
        parent_tensor = torch.from_numpy(parent_matrix).to(
            device=self.device,
            dtype=torch.float32,
        )
        xor_tensor = torch.from_numpy(xor_matrix).to(
            device=self.device,
            dtype=torch.float32,
        )
        return torch.cat(
            [parent_tensor, child_tensor, xor_tensor, action_one_hot],
            dim=1,
        )

    def score_candidates(
        self,
        model_name: str,
        parent_smiles: str,
        candidates: Sequence[Any],
    ) -> np.ndarray:
        """Return one frozen pre-oracle score per candidate in input order."""
        if model_name not in SUPPORTED_RANKERS:
            raise ValueError(
                f"Unsupported ranker {model_name!r}; expected {SUPPORTED_RANKERS}"
            )
        if not candidates:
            return np.empty((0,), dtype=np.float64)

        parent = canonicalize_smiles(parent_smiles)
        if parent is None:
            raise ValueError(f"Invalid parent SMILES: {parent_smiles}")

        child_smiles = [str(candidate.smiles) for candidate in candidates]
        actions = [str(candidate.action_name) for candidate in candidates]
        output = np.empty(len(candidates), dtype=np.float64)
        missing_indices: list[int] = []

        for index, (child, action) in enumerate(zip(child_smiles, actions)):
            key = (model_name, parent, child, action)
            cached = self.score_cache.get(key)
            if cached is None:
                missing_indices.append(index)
            else:
                output[index] = cached

        if model_name == "action_only":
            for index in missing_indices:
                score = self.action_means.get(
                    actions[index],
                    self.action_global_mean,
                )
                output[index] = float(score)
                self.score_cache[(model_name, parent, child_smiles[index], actions[index])] = float(score)
            return output

        if missing_indices:
            missing_children = [child_smiles[index] for index in missing_indices]
            missing_actions = [actions[index] for index in missing_indices]
            x = self._feature_tensor(
                model_name,
                parent,
                missing_children,
                missing_actions,
            )
            model = self.models[model_name]
            with torch.no_grad():
                predicted = model(x).detach().cpu().numpy().astype(np.float64)

            # Pointwise/absolute artifacts predict standardized targets. Affine
            # unscaling does not change ranking, but makes logged scores interpretable.
            scaler = self.artifacts[model_name].get("target_scaler", {})
            target_mean = scaler.get("mean")
            target_std = scaler.get("std")
            if target_mean is not None and target_std is not None:
                predicted = predicted * float(target_std) + float(target_mean)

            for local_index, row_index in enumerate(missing_indices):
                score = float(predicted[local_index])
                output[row_index] = score
                self.score_cache[
                    (
                        model_name,
                        parent,
                        child_smiles[row_index],
                        actions[row_index],
                    )
                ] = score

        if not np.isfinite(output).all():
            raise RuntimeError(f"Non-finite score produced by {model_name}")
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "freeze_dir": str(self.freeze_dir),
            "device": str(self.device),
            "supported_rankers": list(SUPPORTED_RANKERS),
            "morgan_radius": MORGAN_RADIUS,
            "morgan_fp_size": MORGAN_FP_SIZE,
            "action_vocab": list(self.action_vocab),
            "file_hashes": copy.deepcopy(self.file_hashes),
            "fingerprint_cache_size": len(self.fp_cache),
            "score_cache_size": len(self.score_cache),
            "scientific_role": "frozen pre-oracle candidate rankers; no training or oracle scoring",
        }
