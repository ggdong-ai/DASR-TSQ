from __future__ import annotations

"""
The file names intentionally start with numbers for workflow clarity.
Because Python cannot use normal import syntax for modules starting with digits,
this script loads 08_sa_score.py and 09_final_oracle_predictor.py via importlib.

Primary reward profile:
    dual_min_pic50 = min(pred_gsk3b_pic50, pred_jnk3_pic50)

    dual_activity_reward = clip((dual_min_pic50 - 5.0) / (8.0 - 5.0), 0, 1)

    SA_reward = clip((10 - SA_score) / 9, 0, 1)

    final_reward = 0.60 * dual_activity_reward
                 + 0.20 * QED
                 + 0.20 * SA_reward
"""

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_local_module(module_name: str, file_name: str):
    """Load local module from a numbered Python file"""
    path = SCORING_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required module file: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


sa_module = load_local_module("sa_score_module", "08_sa_score.py")
oracle_module = load_local_module("final_oracle_predictor_module", "09_final_oracle_predictor.py")
calculate_sa_score = sa_module.calculate_sa_score
sa_score_to_reward = sa_module.sa_score_to_reward
get_sa_score_metadata = sa_module.get_sa_score_metadata
FinalOraclePredictor = oracle_module.FinalOraclePredictor


@dataclass(frozen=True)
class RewardConfig:
    """Reward configuration for scalarized TSQ-MCTS scoring."""
    profile_name: str = "activity_primary"
    activity_lower_pic50: float = 5.0
    activity_upper_pic50: float = 8.0
    weight_dual_activity: float = 0.60
    weight_qed: float = 0.20
    weight_sa: float = 0.20

    def validate(self) -> None:
        if self.activity_upper_pic50 <= self.activity_lower_pic50:
            raise ValueError("activity_upper_pic50 must be larger than activity_lower_pic50.")

        weight_sum = self.weight_dual_activity + self.weight_qed + self.weight_sa

        if abs(weight_sum - 1.0) > 1e-8:
            raise ValueError(f"Reward weights must sum to 1.0; current sum={weight_sum}.")


SENSITIVITY_PROFILES = {
    "activity_primary": {
        "weight_dual_activity": 0.60,
        "weight_qed": 0.20,
        "weight_sa": 0.20,
        "rationale": (
            "Primary profile. Dual-target activity is the main objective, "
            "while QED and SA remain explicit soft constraints."
        ),
    },
    "activity_strict": {
        "weight_dual_activity": 0.70,
        "weight_qed": 0.15,
        "weight_sa": 0.15,
        "rationale": "Sensitivity profile with stronger pressure on dual-target activity.",
    },
    "balanced": {
        "weight_dual_activity": 0.50,
        "weight_qed": 0.25,
        "weight_sa": 0.25,
        "rationale": "Sensitivity profile balancing activity, drug-likeness and accessibility.",
    },
}


def make_reward_config(profile_name: str = "activity_primary") -> RewardConfig:
    """Create reward config from a named scoring profile"""
    if profile_name not in SENSITIVITY_PROFILES:
        raise KeyError(
            f"Unknown reward profile: {profile_name}. "
            f"Available profiles: {list(SENSITIVITY_PROFILES)}"
        )

    profile = SENSITIVITY_PROFILES[profile_name]

    config = RewardConfig(
        profile_name=profile_name,
        weight_dual_activity=float(profile["weight_dual_activity"]),
        weight_qed=float(profile["weight_qed"]),
        weight_sa=float(profile["weight_sa"]),
    )
    config.validate()
    return config


def canonicalize_to_mol(smiles: str) -> tuple[Chem.Mol | None, str | None]:
    """SMILES -> RDKit Mol and canonical SMILES"""
    if smiles is None:
        return None, None

    smiles = str(smiles).strip()
    if smiles == "":
        return None, None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    canonical = Chem.MolToSmiles(mol, canonical=True)
    return mol, canonical


def clip01(value: float) -> float:
    """Clip float into [0, 1]"""
    return float(np.clip(float(value), 0.0, 1.0))


def normalize_dual_activity(dual_min_pic50: float, config: RewardConfig) -> float:
    """Map dual_min_pic50 to [0, 1]"""
    if np.isnan(dual_min_pic50):
        return 0.0

    raw = (
        float(dual_min_pic50) - config.activity_lower_pic50
    ) / (
        config.activity_upper_pic50 - config.activity_lower_pic50
    )

    return clip01(raw)


class FitnessScorer:
    """TSQ-MCTS scoring class"""

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        predictor: Any | None = None,
        fpscores_path: str | Path | None = None,
    ) -> None:
        self.reward_config = reward_config or make_reward_config("activity_primary")
        self.reward_config.validate()

        self.predictor = predictor or FinalOraclePredictor()
        self.fpscores_path = fpscores_path

        # Fail early if SA fpscores are missing
        get_sa_score_metadata(fpscores_path)

    def score_smiles(self, smiles: str) -> dict[str, Any]:
        """Score one SMILES and return all components"""
        mol, canonical = canonicalize_to_mol(smiles)

        base = {
            "input_smiles": smiles,
            "canonical_smiles": canonical,
            "reward_profile": self.reward_config.profile_name,
        }

        if mol is None or canonical is None:
            return {
                **base,
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
                "molecular_weight": np.nan,
                "logp": np.nan,
                "heavy_atom_count": np.nan,
                "tpsa": np.nan,
                "hbd": np.nan,
                "hba": np.nan,
                "rotatable_bonds": np.nan,
                "ring_count": np.nan,
            }

        oracle_result = self.predictor.predict_smiles(canonical)

        if oracle_result["oracle_status"] != "ok":
            return {
                **base,
                "canonical_smiles": canonical,
                "valid_smiles": True,
                "scoring_status": "oracle_failed",
                "pred_gsk3b_pic50": np.nan,
                "pred_jnk3_pic50": np.nan,
                "dual_min_pic50": np.nan,
                "dual_mean_pic50": np.nan,
                "dual_activity_reward": 0.0,
                "qed": np.nan,
                "sa_score": np.nan,
                "sa_reward": 0.0,
                "final_reward": 0.0,
                "molecular_weight": float(Descriptors.MolWt(mol)),
                "logp": float(Descriptors.MolLogP(mol)),
                "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
                "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
                "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
                "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
                "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
            }

        qed_value = float(QED.qed(mol))
        sa_value = float(calculate_sa_score(mol, fpscores_path=self.fpscores_path))
        sa_reward = sa_score_to_reward(sa_value)

        dual_activity_reward = normalize_dual_activity(
            oracle_result["dual_min_pic50"],
            self.reward_config,
        )

        final_reward = (
            self.reward_config.weight_dual_activity * dual_activity_reward
            + self.reward_config.weight_qed * qed_value
            + self.reward_config.weight_sa * sa_reward
        )

        final_reward = clip01(final_reward)

        return {
            **base,
            "canonical_smiles": canonical,
            "valid_smiles": True,
            "scoring_status": "ok",
            "pred_gsk3b_pic50": oracle_result["pred_gsk3b_pic50"],
            "pred_jnk3_pic50": oracle_result["pred_jnk3_pic50"],
            "dual_min_pic50": oracle_result["dual_min_pic50"],
            "dual_mean_pic50": oracle_result["dual_mean_pic50"],
            "dual_activity_reward": dual_activity_reward,
            "qed": qed_value,
            "sa_score": sa_value,
            "sa_reward": sa_reward,
            "final_reward": final_reward,
            "molecular_weight": float(Descriptors.MolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
            "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
            "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
            "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        }

    def metadata(self) -> dict[str, Any]:
        """Metadata for scoring_config.json."""
        return {
            "primary_profile": self.reward_config.profile_name,
            "reward_config": asdict(self.reward_config),
            "sensitivity_profiles": SENSITIVITY_PROFILES,
            "reward_formula": {
                "dual_min_pic50": "min(pred_gsk3b_pic50, pred_jnk3_pic50)",
                "dual_activity_reward": (
                    "clip((dual_min_pic50 - activity_lower_pic50) / "
                    "(activity_upper_pic50 - activity_lower_pic50), 0, 1)"
                ),
                "sa_reward": "clip((10 - SA_score) / 9, 0, 1)",
                "final_reward": (
                    "weight_dual_activity * dual_activity_reward "
                    "+ weight_qed * QED + weight_sa * SA_reward"
                ),
            },
            "primary_reward_interpretation": (
                "The primary profile is activity-centered because the main manuscript objective is "
                "dual-target GSK3β/JNK3 molecular core. QED and SA are retained as soft constraints."
            ),
            "qed_source": "RDKit QED implementation",
            "sa_score_source": "Ertl-Schuffenhauer synthetic accessibility score",
            "sa_module": get_sa_score_metadata(self.fpscores_path),
            "oracle_predictor": self.predictor.metadata(),
            "p_success_policy": {
                "used_in_main_pipeline": False,
                "reason": (
                    "The legacy P_success model was trained on old combined_training_data.csv and old labels. "
                    "It is not used in the main pipeline to avoid inconsistency and potential leakage. "
                    "It may be redesigned later as an ablation module using current frozen data and validation."
                ),
            },
        }


def score_smiles(smiles: str) -> dict[str, Any]:
    """Convenience scoring function. For MCTS, instantiate FitnessScorer once."""
    scorer = FitnessScorer()
    return scorer.score_smiles(smiles)


def save_scoring_config(path: str | Path) -> None:
    """Save default scoring configuration."""
    scorer = FitnessScorer()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scorer.metadata(), handle, indent=2, ensure_ascii=False)