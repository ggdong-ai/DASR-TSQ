from __future__ import annotations

"""
训练最终 QSAR activity oracle

1. 读取第五步已经确定好的最终模型选择：
   - GSK3β：Random Forest Regressor
   - JNK3：Random Forest Regressor

2. 不再划分数据，不再做 CV；直接用全部数据训练最终模型：
   - 01_data/clean/gsk3b_modeling_dataset.csv
   - 01_data/clean/jnk3_modeling_dataset.csv

3. 使用与前面全部流程一致的分子表示：
   - Morgan fingerprint
   - radius = 3
   - nBits = 1024

4. 保存后续 TSQ-MCTS 可以直接调用的最终模型包：
   - final_gsk3b_rf_oracle.joblib
   - final_jnk3_rf_oracle.joblib

5. 论文用途：说明最终生成算法调用的 oracle 如何训练
"""

import hashlib
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
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr



# ============================================================
# 0. 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import CLEAN_DATA_DIR, MODEL_DIR, RESULT_DIR


# ============================================================
# 1. 固定参数
# ============================================================
MORGAN_RADIUS = 3
MORGAN_FP_SIZE = 1024
RANDOM_SEED = 42

# 第五步 scaffold CV 结果显示 RF 是两个靶点的推荐最终 oracle
# 多数外层 scaffold fold 的 inner CV 选择 n_estimators=600；
# 其余关键参数保持与第五步 RF 参数网格一致
FINAL_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}

TARGET_CONFIG = {
    "gsk3b": {
        "input_file": CLEAN_DATA_DIR / "gsk3b_modeling_dataset.csv",
        "output_model": MODEL_DIR / "final_oracles" / "final_gsk3b_rf_oracle.joblib",
        "endpoint": "GSK3β pIC50",
    },
    "jnk3": {
        "input_file": CLEAN_DATA_DIR / "jnk3_modeling_dataset.csv",
        "output_model": MODEL_DIR / "final_oracles" / "final_jnk3_rf_oracle.joblib",
        "endpoint": "JNK3 pIC50",
    },
}

OUTPUT_DIR = RESULT_DIR / "final_oracles"

REQUIRED_COLUMNS = {
    "structure_key",
    "standardized_smiles",
    "pic50_median",
    "active_label_6_0",
    "active_label_6_5",
    "active_label_7_0",
}



# ============================================================
# 2. 工具函数
# ============================================================
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input(dataframe: pd.DataFrame, target: str) -> None:
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise KeyError(f"{target} 输入数据缺少字段：{sorted(missing)}")

    if dataframe.empty:
        raise ValueError(f"{target} 输入数据为空。")

    if dataframe["structure_key"].isna().any():
        raise ValueError(f"{target} 存在空 structure_key。")

    if dataframe["structure_key"].duplicated().any():
        duplicate_count = int(dataframe["structure_key"].duplicated().sum())
        raise ValueError(f"{target} 存在重复 structure_key：{duplicate_count} 条。")

    if dataframe["standardized_smiles"].isna().any():
        raise ValueError(f"{target} 存在空 standardized_smiles。")

    if dataframe["pic50_median"].isna().any():
        raise ValueError(f"{target} 存在空 pic50_median。")


def smiles_to_morgan_array(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit 无法解析 SMILES：{smiles}")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_FP_SIZE,
    )
    fingerprint = generator.GetFingerprint(mol)

    array = np.zeros((MORGAN_FP_SIZE,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def dataframe_to_fingerprints(dataframe: pd.DataFrame) -> np.ndarray:
    arrays = [
        smiles_to_morgan_array(smiles)
        for smiles in dataframe["standardized_smiles"]
    ]
    return np.asarray(arrays, dtype=np.float32)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        value = spearmanr(y_true, y_pred).correlation
        return float(value) if value is not None else float("nan")
    except Exception:
        return float("nan")


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        value = pearsonr(y_true, y_pred).statistic
        return float(value)
    except Exception:
        return float("nan")


def calculate_in_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    训练集内指标只用于 sanity check，不用于论文性能报告
    论文性能必须使用第五步 nested CV 外层测试集结果
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    spearman = safe_spearman(y_true, y_pred)
    pearson = safe_pearson(y_true, y_pred)

    return {
        "in_sample_rmse_not_for_reporting": rmse,
        "in_sample_mae_not_for_reporting": mae,
        "in_sample_r2_not_for_reporting": r2,
        "in_sample_spearman_not_for_reporting": spearman,
        "in_sample_pearson_not_for_reporting": pearson,
    }



# ============================================================
# 3. 训练最终 oracle
# ============================================================
def train_final_oracle(target: str, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    input_file: Path = config["input_file"]
    output_model: Path = config["output_model"]

    if not input_file.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_file}")

    dataframe = pd.read_csv(input_file)
    validate_input(dataframe, target)

    x = dataframe_to_fingerprints(dataframe)
    y = dataframe["pic50_median"].to_numpy(dtype=float)

    model = RandomForestRegressor(**FINAL_RF_PARAMS)
    model.fit(x, y)

    y_pred = model.predict(x)
    sanity_metrics = calculate_in_sample_metrics(y, y_pred)

    output_model.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "target": target,
        "endpoint": config["endpoint"],
        "model_type": "RandomForestRegressor",
        "model": model,
        "model_params": FINAL_RF_PARAMS,
        "molecular_representation": {
            "type": "Morgan bit fingerprint",
            "radius": MORGAN_RADIUS,
            "n_bits": MORGAN_FP_SIZE,
        },
        "training_data": {
            "path": str(input_file),
            "sha256": sha256_file(input_file),
            "n_structures": int(len(dataframe)),
            "n_unique_structure_key": int(dataframe["structure_key"].nunique()),
            "pic50_mean": float(dataframe["pic50_median"].mean()),
            "pic50_std": float(dataframe["pic50_median"].std(ddof=1)),
            "active_fraction_6_0": float(dataframe["active_label_6_0"].mean()),
            "active_fraction_6_5": float(dataframe["active_label_6_5"].mean()),
            "active_fraction_7_0": float(dataframe["active_label_7_0"].mean()),
        },
        "performance_note": (
            "This final oracle was trained on all available modeling data. "
            "Its in-sample metrics are only sanity checks and must not be reported "
            "as predictive performance. Report nested CV performance from Step 05 instead."
        ),
        "in_sample_sanity_metrics": sanity_metrics,
    }

    joblib.dump(model_bundle, output_model)

    prediction_table = dataframe[
        [
            "structure_key",
            "standardized_smiles",
            "pic50_median",
            "active_label_6_0",
            "active_label_6_5",
            "active_label_7_0",
        ]
    ].copy()
    prediction_table.insert(0, "target", target)
    prediction_table["final_oracle_in_sample_prediction"] = y_pred
    prediction_table["absolute_error_not_for_reporting"] = np.abs(y - y_pred)

    summary = {
        "target": target,
        "endpoint": config["endpoint"],
        "model_type": "RandomForestRegressor",
        "model_path": str(output_model),
        "input_file": str(input_file),
        "input_sha256": sha256_file(input_file),
        "n_structures": int(len(dataframe)),
        "morgan_radius": MORGAN_RADIUS,
        "morgan_n_bits": MORGAN_FP_SIZE,
        "model_params_json": json.dumps(FINAL_RF_PARAMS, ensure_ascii=False),
    }
    summary.update(sanity_metrics)

    return summary, prediction_table


def predict_smiles_with_oracle(smiles_list: list[str], oracle_path: str | Path) -> pd.DataFrame:
    """
    后续 TSQ-MCTS 可以复用这个函数的逻辑
    示例：result = predict_smiles_with_oracle(["CCO", "c1ccccc1"], "final_gsk3b_rf_oracle.joblib")
    """
    bundle = joblib.load(oracle_path)
    model = bundle["model"]

    rows = []
    for smiles in smiles_list:
        try:
            fp = smiles_to_morgan_array(smiles).reshape(1, -1)
            pred = float(model.predict(fp)[0])
            rows.append(
                {
                    "smiles": smiles,
                    "target": bundle["target"],
                    "predicted_pic50": pred,
                    "valid_smiles": True,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "smiles": smiles,
                    "target": bundle.get("target", "unknown"),
                    "predicted_pic50": np.nan,
                    "valid_smiles": False,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 4. 主程序
# ============================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "final_oracles").mkdir(parents=True, exist_ok=True)

    summaries = []
    prediction_tables = []

    for target, cfg in TARGET_CONFIG.items():
        print(f"Training final oracle for {target} ...")
        summary, prediction_table = train_final_oracle(target, cfg)
        summaries.append(summary)
        prediction_tables.append(prediction_table)
        print(f"Finished {target}: {summary['model_path']}")

    summary_dataframe = pd.DataFrame(summaries)
    predictions_dataframe = pd.concat(prediction_tables, ignore_index=True)

    summary_path = OUTPUT_DIR / "final_oracle_training_summary.csv"
    prediction_path = OUTPUT_DIR / "final_oracle_in_sample_predictions_not_for_reporting.csv"
    config_path = OUTPUT_DIR / "final_oracle_config.json"

    summary_dataframe.to_csv(summary_path, index=False)
    predictions_dataframe.to_csv(prediction_path, index=False)

    config = {
        "purpose": (
            "Train final deployment QSAR activity oracles for TSQ-MCTS using all "
            "final modeling data after Step 05 nested-CV model selection."
        ),
        "important_note": (
            "This step does not estimate predictive performance. Predictive performance "
            "must be taken from Step 05 nested cross-validation results."
        ),
        "selected_oracles": {
            "gsk3b": "RandomForestRegressor",
            "jnk3": "RandomForestRegressor",
        },
        "selection_basis": (
            "Random Forest was selected based on scaffold-CV performance in Step 05. "
            "For GSK3β it achieved the best overall scaffold-CV metrics. For JNK3, "
            "HistGradientBoosting had slightly lower RMSE, but Random Forest showed "
            "stronger ranking-related metrics and was selected for oracle consistency."
        ),
        "molecular_representation": {
            "type": "Morgan bit fingerprint",
            "radius": MORGAN_RADIUS,
            "n_bits": MORGAN_FP_SIZE,
        },
        "final_model_params": FINAL_RF_PARAMS,
        "outputs": {
            "summary": str(summary_path),
            "in_sample_predictions_not_for_reporting": str(prediction_path),
            "gsk3b_oracle": str(TARGET_CONFIG["gsk3b"]["output_model"]),
            "jnk3_oracle": str(TARGET_CONFIG["jnk3"]["output_model"]),
        },
        "training_inputs": {
            target: {
                "path": str(cfg["input_file"]),
                "sha256": sha256_file(cfg["input_file"]),
            }
            for target, cfg in TARGET_CONFIG.items()
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit.__version__,
            "joblib": joblib.__version__,
        },
    }

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n=== Final oracle training finished ===")
    print(f"Summary: {summary_path}")
    print(f"Config: {config_path}")
    print(f"In-sample sanity predictions: {prediction_path}")
    print(f"GSK3β oracle: {TARGET_CONFIG['gsk3b']['output_model']}")
    print(f"JNK3 oracle: {TARGET_CONFIG['jnk3']['output_model']}")
    print("注意：in-sample 指标只用于 sanity check，不可作为论文预测性能。")



if __name__ == "__main__":
    main()