from __future__ import annotations

"""
Final GSK3β/JNK3 QSAR oracle predictor

This module loads final oracle joblib files from Step 06 and predicts pIC50.

Compatible with:
    1. joblib.dump(model, path)
    2. joblib.dump({"model": model, "metadata": ...}, path)
"""

import json
import sys
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


MORGAN_RADIUS = 3
MORGAN_FP_SIZE = 1024


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_DIR


def load_sklearn_model_from_artifact(path: str | Path) -> Any:
    """Load sklearn estimator from direct object or dictionary artifact"""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Oracle file not found: {path}")

    artifact = joblib.load(path)

    if hasattr(artifact, "predict"):
        return artifact

    if isinstance(artifact, dict):
        candidate_keys = [
            "model",
            "estimator",
            "oracle",
            "regressor",
            "predictor",
            "sklearn_model",
            "final_model",
        ]

        for key in candidate_keys:
            if key in artifact and hasattr(artifact[key], "predict"):
                return artifact[key]

        for value in artifact.values():
            if hasattr(value, "predict"):
                return value

        raise TypeError(
            f"Dictionary oracle artifact has no object with predict().\n"
            f"File: {path}\n"
            f"Keys: {list(artifact.keys())}"
        )

    raise TypeError(f"Unrecognized oracle artifact type: {type(artifact)}. File: {path}")


def smiles_to_morgan_array(smiles: str) -> np.ndarray | None:
    """Convert SMILES to shape=(1, 1024) Morgan fingerprint array"""
    if smiles is None:
        return None

    smiles = str(smiles).strip()
    if smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_FP_SIZE,
    )

    fingerprint = generator.GetFingerprint(mol)

    array = np.zeros((MORGAN_FP_SIZE,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return array.reshape(1, -1)


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES; invalid input returns None"""
    if smiles is None:
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


class FinalOraclePredictor:
    """Wrapper for final GSK3β and JNK3 QSAR oracles"""

    def __init__(
        self,
        gsk3b_oracle_path: str | Path | None = None,
        jnk3_oracle_path: str | Path | None = None,
    ) -> None:
        self.gsk3b_oracle_path = Path(
            gsk3b_oracle_path
            if gsk3b_oracle_path is not None
            else MODEL_DIR / "final_oracles" / "final_gsk3b_rf_oracle.joblib"
        )

        self.jnk3_oracle_path = Path(
            jnk3_oracle_path
            if jnk3_oracle_path is not None
            else MODEL_DIR / "final_oracles" / "final_jnk3_rf_oracle.joblib"
        )

        self.gsk3b_model = load_sklearn_model_from_artifact(self.gsk3b_oracle_path)
        self.jnk3_model = load_sklearn_model_from_artifact(self.jnk3_oracle_path)

    def predict_smiles(self, smiles: str) -> dict[str, Any]:
        """Predict GSK3β/JNK3 pIC50 for one SMILES"""
        x = smiles_to_morgan_array(smiles)
        canonical = canonicalize_smiles(smiles)

        if x is None or canonical is None:
            return {
                "oracle_status": "invalid_smiles",
                "canonical_smiles": None,
                "pred_gsk3b_pic50": np.nan,
                "pred_jnk3_pic50": np.nan,
                "dual_min_pic50": np.nan,
                "dual_mean_pic50": np.nan,
            }

        pred_gsk3b = float(self.gsk3b_model.predict(x)[0])
        pred_jnk3 = float(self.jnk3_model.predict(x)[0])

        return {
            "oracle_status": "ok",
            "canonical_smiles": canonical,
            "pred_gsk3b_pic50": pred_gsk3b,
            "pred_jnk3_pic50": pred_jnk3,
            "dual_min_pic50": min(pred_gsk3b, pred_jnk3),
            "dual_mean_pic50": (pred_gsk3b + pred_jnk3) / 2.0,
        }

    def metadata(self) -> dict[str, Any]:
        """Metadata for scoring_config.json"""
        return {
            "gsk3b_oracle_path": str(self.gsk3b_oracle_path),
            "jnk3_oracle_path": str(self.jnk3_oracle_path),
            "molecular_representation": {
                "type": "Morgan bit fingerprint",
                "radius": MORGAN_RADIUS,
                "n_bits": MORGAN_FP_SIZE,
            },
        }


def save_oracle_predictor_metadata(path: str | Path) -> None:
    """Save default predictor metadata"""
    predictor = FinalOraclePredictor()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(predictor.metadata(), handle, indent=2, ensure_ascii=False)