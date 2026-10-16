from __future__ import annotations

"""
Fitness/scoring sanity check.

This script tests the new scoring stack before TSQ-MCTS pilot core.

It verifies:
    1. final oracle predictor works;
    2. SA score can be computed;
    3. QED can be computed;
    4. all reward components are within expected ranges;
    5. invalid SMILES are handled safely;
    6. scoring_config.json records formulas, paths, versions, and references.
"""

import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn


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


fitness_module = load_local_module("fitness_function_module", "10_fitness_function.py")
FitnessScorer = fitness_module.FitnessScorer

from config import CLEAN_DATA_DIR, RESULT_DIR


OUTPUT_DIR = RESULT_DIR / "scoring_sanity_check"

N_SAMPLE_PER_TARGET = 20
RANDOM_SEED = 42

INPUT_DATASETS = {
    "gsk3b": CLEAN_DATA_DIR / "gsk3b_modeling_dataset.csv",
    "jnk3": CLEAN_DATA_DIR / "jnk3_modeling_dataset.csv",
}

EXTERNAL_VALID_SMILES = [
    ("benzene", "c1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("phenol", "Oc1ccccc1"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]

INVALID_SMILES = [
    ("invalid_text", "not_a_smiles"),
    ("unclosed_ring", "C1CC"),
    ("empty_string", ""),
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_project_sample_smiles() -> pd.DataFrame:
    """Sample molecules from current frozen modeling datasets"""
    frames = []

    for target, path in INPUT_DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Input data not found: {path}")

        df = pd.read_csv(path)

        required = {"structure_key", "standardized_smiles", "pic50_median"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{target} missing columns: {sorted(missing)}")

        sample_n = min(N_SAMPLE_PER_TARGET, len(df))
        sample = df.sample(n=sample_n, random_state=RANDOM_SEED)[
            ["structure_key", "standardized_smiles", "pic50_median"]
        ].copy()

        sample = sample.rename(columns={"standardized_smiles": "smiles"})
        sample.insert(0, "source", f"{target}_modeling_sample")
        sample["reference_target"] = target

        frames.append(sample)

    return pd.concat(frames, ignore_index=True, sort=False)


def build_test_table() -> pd.DataFrame:
    """Build test input table."""
    project_samples = load_project_sample_smiles()

    external = pd.DataFrame(
        [
            {
                "source": "external_valid_smiles",
                "reference_target": "none",
                "structure_key": name,
                "smiles": smiles,
            }
            for name, smiles in EXTERNAL_VALID_SMILES
        ]
    )

    invalid = pd.DataFrame(
        [
            {
                "source": "invalid_smiles_test",
                "reference_target": "none",
                "structure_key": name,
                "smiles": smiles,
            }
            for name, smiles in INVALID_SMILES
        ]
    )

    table = pd.concat([project_samples, external, invalid], ignore_index=True, sort=False)
    table.insert(0, "test_id", range(1, len(table) + 1))

    return table


def run_scoring(table: pd.DataFrame, scorer: FitnessScorer) -> pd.DataFrame:
    """Score all test SMILES."""
    rows = []

    for _, row in table.iterrows():
        score = scorer.score_smiles(row["smiles"])
        output = row.to_dict()
        output.update(score)
        rows.append(output)

    return pd.DataFrame(rows)


def summarize_results(scored: pd.DataFrame, scorer: FitnessScorer) -> pd.DataFrame:
    """Summarize sanity-check results."""
    valid = scored[scored["scoring_status"] == "ok"].copy()
    invalid = scored[scored["scoring_status"] == "invalid_smiles"].copy()

    expected_invalid_count = len(INVALID_SMILES)

    required_valid_columns = [
        "pred_gsk3b_pic50",
        "pred_jnk3_pic50",
        "dual_min_pic50",
        "dual_mean_pic50",
        "dual_activity_reward",
        "qed",
        "sa_score",
        "sa_reward",
        "final_reward",
        "molecular_weight",
        "logp",
        "heavy_atom_count",
    ]

    valid_no_missing = bool(
        not valid.empty
        and valid[required_valid_columns].notna().all().all()
    )

    final_reward_range_passed = bool(
        not valid.empty
        and valid["final_reward"].between(0.0, 1.0).all()
    )

    component_reward_range_passed = bool(
        not valid.empty
        and valid["dual_activity_reward"].between(0.0, 1.0).all()
        and valid["qed"].between(0.0, 1.0).all()
        and valid["sa_reward"].between(0.0, 1.0).all()
    )

    sa_score_range_passed = bool(
        not valid.empty
        and valid["sa_score"].between(1.0, 10.0).all()
    )

    invalid_handling_passed = bool(
        len(invalid) == expected_invalid_count
        and invalid["final_reward"].fillna(0.0).eq(0.0).all()
    )

    weight_sum = (
        scorer.reward_config.weight_dual_activity
        + scorer.reward_config.weight_qed
        + scorer.reward_config.weight_sa
    )

    weights_passed = bool(abs(weight_sum - 1.0) < 1e-8)

    summary = {
        "n_total_inputs": len(scored),
        "n_valid_scored": len(valid),
        "n_invalid_detected": len(invalid),
        "expected_invalid": expected_invalid_count,
        "invalid_handling_passed": invalid_handling_passed,
        "valid_no_missing": valid_no_missing,
        "final_reward_range_passed": final_reward_range_passed,
        "component_reward_range_passed": component_reward_range_passed,
        "sa_score_range_passed": sa_score_range_passed,
        "reward_profile": scorer.reward_config.profile_name,
        "weight_dual_activity": scorer.reward_config.weight_dual_activity,
        "weight_qed": scorer.reward_config.weight_qed,
        "weight_sa": scorer.reward_config.weight_sa,
        "weight_sum": weight_sum,
        "weights_passed": weights_passed,
        "final_reward_min": float(valid["final_reward"].min()) if not valid.empty else np.nan,
        "final_reward_max": float(valid["final_reward"].max()) if not valid.empty else np.nan,
        "dual_min_pic50_min": float(valid["dual_min_pic50"].min()) if not valid.empty else np.nan,
        "dual_min_pic50_max": float(valid["dual_min_pic50"].max()) if not valid.empty else np.nan,
        "qed_min": float(valid["qed"].min()) if not valid.empty else np.nan,
        "qed_max": float(valid["qed"].max()) if not valid.empty else np.nan,
        "sa_score_min": float(valid["sa_score"].min()) if not valid.empty else np.nan,
        "sa_score_max": float(valid["sa_score"].max()) if not valid.empty else np.nan,
    }

    summary["overall_passed"] = bool(
        invalid_handling_passed
        and valid_no_missing
        and final_reward_range_passed
        and component_reward_range_passed
        and sa_score_range_passed
        and weights_passed
    )

    return pd.DataFrame([summary])


def save_config(scorer: FitnessScorer, summary_path: Path, predictions_path: Path) -> None:
    """Save complete scoring config"""
    config: dict[str, Any] = {
        "purpose": "Sanity check for TSQ-MCTS scoring function before pilot molecular core.",
        "important_note": (
            "This is not a molecule core experiment and not a QSAR performance evaluation."
        ),
        "scoring_metadata": scorer.metadata(),
        "test_inputs": {
            "n_sample_per_target": N_SAMPLE_PER_TARGET,
            "external_valid_smiles": EXTERNAL_VALID_SMILES,
            "invalid_smiles": INVALID_SMILES,
        },
        "outputs": {
            "summary": str(summary_path),
            "predictions": str(predictions_path),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit.__version__,
            "joblib": joblib.__version__,
        },
        "method_references": [
            {
                "topic": "QED drug-likeness",
                "doi": "10.1038/nchem.1243",
            },
            {
                "topic": "Synthetic accessibility score",
                "doi": "10.1186/1758-2946-1-8",
            },
            {
                "topic": "GuacaMol molecular core benchmark",
                "doi": "10.1021/acs.jcim.8b00839",
            },
            {
                "topic": "ChemTS MCTS molecular core",
                "doi": "10.1080/14686996.2017.1401424",
            },
            {
                "topic": "REINVENT4 multi-component generative molecular design",
                "doi": "10.1186/s13321-024-00812-5",
            },
            {
                "topic": "Reward hacking risk in multi-objective molecular core",
                "doi": "10.1038/s41467-025-57582-3",
            },
        ],
    }

    config_path = OUTPUT_DIR / "scoring_config.json"

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def main() -> None:
    ensure_output_dir()

    print("Initializing scorer ...")
    scorer = FitnessScorer()

    print("Building test table ...")
    table = build_test_table()

    print("Running scoring ...")
    scored = run_scoring(table, scorer)

    predictions_path = OUTPUT_DIR / "scoring_sanity_check_predictions.csv"
    summary_path = OUTPUT_DIR / "scoring_sanity_check_summary.csv"

    scored.to_csv(predictions_path, index=False)

    summary = summarize_results(scored, scorer)
    summary.to_csv(summary_path, index=False)

    save_config(scorer, summary_path, predictions_path)

    print("\n=== Fitness/scoring sanity check finished ===")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")
    print(f"Config: {OUTPUT_DIR / 'scoring_config.json'}")

    if bool(summary.loc[0, "overall_passed"]):
        print("Overall status: PASSED")
    else:
        raise RuntimeError(
            "Fitness/scoring sanity check failed. Please inspect scoring_sanity_check_summary.csv."
        )


if __name__ == "__main__":
    main()