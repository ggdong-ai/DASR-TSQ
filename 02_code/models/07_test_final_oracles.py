from __future__ import annotations

"""
最终 QSAR activity oracle 调用测试

检查内容：
1. final_gsk3b_rf_oracle.joblib 是否存在并能加载；
2. final_jnk3_rf_oracle.joblib 是否存在并能加载；
3. 输入合法 SMILES 是否能得到两个靶点的 predicted pIC50；
4. 输入非法 SMILES 是否会被识别并跳过，而不是导致程序崩溃；
5. 同一个 SMILES 连续预测两次是否完全一致；
6. 预测值是否处于合理范围；
7. 后续 TSQ-MCTS 必须使用的 Morgan fingerprint 参数是否仍为 radius=3、nBits=1024。
"""

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

# 最终 oracle 路径
ORACLE_DIR = MODEL_DIR / "final_oracles"
GSK3B_ORACLE_PATH = ORACLE_DIR / "final_gsk3b_rf_oracle.joblib"
JNK3_ORACLE_PATH = ORACLE_DIR / "final_jnk3_rf_oracle.joblib"

# 最终建模数据，用来抽取一小部分真实项目分子进行调用测试
INPUT_DATASETS = {
    "gsk3b": CLEAN_DATA_DIR / "gsk3b_modeling_dataset.csv",
    "jnk3": CLEAN_DATA_DIR / "jnk3_modeling_dataset.csv",
}

# 输出目录
OUTPUT_DIR = RESULT_DIR / "oracle_sanity_check"

# 合法测试 SMILES：包含简单小分子和常见药物样结构
# 这里只用于测试 oracle 是否可调用，不用于论文性能评价
EXTERNAL_TEST_SMILES = [
    ("benzene", "c1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("phenol", "Oc1ccccc1"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]

# 非法 SMILES：用于确认程序能识别无效输入并跳过
INVALID_TEST_SMILES = [
    ("invalid_text", "not_a_smiles"),
    ("unclosed_ring", "C1CC"),
    ("empty_string", ""),
]

# 预测值硬性合理范围。
# 由于训练数据已经限定 pIC50 大致在 2-12，本范围只用于发现明显异常
HARD_MIN_PIC50 = -2.0
HARD_MAX_PIC50 = 15.0

# 从每个建模数据集中抽取多少个真实分子做调用测试
N_SAMPLE_PER_TARGET = 20
RANDOM_SEED = 42



# ============================================================
# 2. 基础工具
# ============================================================
def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(path: Path) -> Any:
    """
    加载 joblib oracle

    兼容两种保存方式：
    1. 直接保存 sklearn 模型：joblib.dump(model, path)
    2. 保存带元信息的字典： joblib.dump({"model": model, "metadata": ...}, path)
    如果是第二种情况，本函数会自动从字典中取出真正带 predict 方法的模型
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到 oracle 模型文件：{path}")

    artifact = joblib.load(path)

    # 情况 1：joblib 文件本身就是 sklearn 模型
    if hasattr(artifact, "predict"):
        print(f"Loaded sklearn estimator directly: {path}")
        return artifact

    # 情况 2：joblib 文件是一个 dict，模型被包在其中
    if isinstance(artifact, dict):
        print(f"Loaded dictionary artifact: {path}")
        print(f"Available keys: {list(artifact.keys())}")

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
                print(f"Using artifact['{key}'] as oracle model.")
                return artifact[key]

        # 如果键名不在候选列表里，就遍历所有 value，找带 predict 的对象
        for key, value in artifact.items():
            if hasattr(value, "predict"):
                print(f"Using artifact['{key}'] as oracle model.")
                return value

        raise TypeError(
            f"joblib 文件是字典，但没有找到带 predict 方法的模型对象。\n"
            f"文件：{path}\n"
            f"可用 keys：{list(artifact.keys())}"
        )

    raise TypeError(
        f"无法识别的 oracle 文件类型：{type(artifact)}。\n"
        f"文件：{path}\n"
        f"该对象没有 predict 方法，也不是 dict artifact。"
    )


def smiles_to_morgan_array(smiles: str) -> np.ndarray | None:
    """
    将 SMILES 转换为 Morgan fingerprint 数组
    合法 SMILES 返回 shape=(1, 1024) 的二维数组；非法 SMILES 返回 None。
    """
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

    if array.shape[0] != MORGAN_FP_SIZE:
        raise RuntimeError(
            f"指纹长度异常：{array.shape[0]}，应为 {MORGAN_FP_SIZE}"
        )

    return array.reshape(1, -1)


def predict_one_smiles(
    smiles: str,
    gsk3b_model: Any,
    jnk3_model: Any,
) -> dict[str, Any]:
    """对一个 SMILES 同时调用 GSK3β 和 JNK3 oracle"""
    x = smiles_to_morgan_array(smiles)

    if x is None:
        return {
            "status": "invalid_smiles",
            "pred_gsk3b_pic50": np.nan,
            "pred_jnk3_pic50": np.nan,
            "dual_min_pic50": np.nan,
            "dual_mean_pic50": np.nan,
        }

    pred_gsk3b = float(gsk3b_model.predict(x)[0])
    pred_jnk3 = float(jnk3_model.predict(x)[0])

    return {
        "status": "ok",
        "pred_gsk3b_pic50": pred_gsk3b,
        "pred_jnk3_pic50": pred_jnk3,
        "dual_min_pic50": min(pred_gsk3b, pred_jnk3),
        "dual_mean_pic50": (pred_gsk3b + pred_jnk3) / 2.0,
    }


def load_project_test_smiles() -> pd.DataFrame:
    """
    从 GSK3β 和 JNK3 最终建模数据中各抽取一小部分分子。这些分子只用于调用测试，不用于重新评价模型性能。
    """
    frames = []

    for target, path in INPUT_DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到建模数据：{path}")

        dataframe = pd.read_csv(path)

        required = {"structure_key", "standardized_smiles", "pic50_median"}
        missing = required - set(dataframe.columns)
        if missing:
            raise KeyError(f"{target} 建模数据缺少字段：{sorted(missing)}")

        sample_n = min(N_SAMPLE_PER_TARGET, len(dataframe))

        sample = dataframe.sample(
            n=sample_n,
            random_state=RANDOM_SEED,
        )[
            ["structure_key", "standardized_smiles", "pic50_median"]
        ].copy()

        sample.insert(0, "source", f"{target}_modeling_sample")
        sample = sample.rename(
            columns={
                "standardized_smiles": "smiles",
                "pic50_median": f"observed_{target}_pic50",
            }
        )

        frames.append(sample)

    project_smiles = pd.concat(frames, ignore_index=True, sort=False)

    return project_smiles


def make_test_input_table() -> pd.DataFrame:
    """构建本次 sanity check 的全部输入 SMILES"""
    project_smiles = load_project_test_smiles()

    external = pd.DataFrame(
        [
            {
                "source": "external_valid_smiles",
                "structure_key": name,
                "smiles": smiles,
            }
            for name, smiles in EXTERNAL_TEST_SMILES
        ]
    )

    invalid = pd.DataFrame(
        [
            {
                "source": "invalid_smiles_test",
                "structure_key": name,
                "smiles": smiles,
            }
            for name, smiles in INVALID_TEST_SMILES
        ]
    )

    table = pd.concat(
        [project_smiles, external, invalid],
        ignore_index=True,
        sort=False,
    )

    table.insert(0, "test_id", range(1, len(table) + 1))

    return table



# ============================================================
# 3. 检查逻辑
# ============================================================
def run_predictions(test_table: pd.DataFrame, gsk3b_model: Any, jnk3_model: Any) -> pd.DataFrame:
    """对测试表中的全部 SMILES 运行 oracle"""
    rows = []

    for _, row in test_table.iterrows():
        prediction = predict_one_smiles(
            row["smiles"],
            gsk3b_model=gsk3b_model,
            jnk3_model=jnk3_model,
        )

        output = row.to_dict()
        output.update(prediction)
        rows.append(output)

    return pd.DataFrame(rows)


def check_determinism(
    valid_smiles: list[str],
    gsk3b_model: Any,
    jnk3_model: Any,
) -> dict[str, Any]:
    """检查同一个 SMILES 连续预测两次是否一致"""
    max_abs_diff = 0.0

    for smiles in valid_smiles:
        first = predict_one_smiles(smiles, gsk3b_model, jnk3_model)
        second = predict_one_smiles(smiles, gsk3b_model, jnk3_model)

        for key in ["pred_gsk3b_pic50", "pred_jnk3_pic50"]:
            diff = abs(first[key] - second[key])
            max_abs_diff = max(max_abs_diff, diff)

    return {
        "determinism_max_abs_diff": max_abs_diff,
        "determinism_passed": bool(max_abs_diff == 0.0),
    }


def summarize_checks(predictions: pd.DataFrame, determinism_result: dict[str, Any]) -> pd.DataFrame:
    """生成本次检查汇总表"""
    valid = predictions[predictions["status"] == "ok"].copy()
    invalid = predictions[predictions["status"] == "invalid_smiles"].copy()

    if valid.empty:
        raise RuntimeError("没有任何合法 SMILES 成功预测。")

    gsk3b_values = valid["pred_gsk3b_pic50"].to_numpy(dtype=float)
    jnk3_values = valid["pred_jnk3_pic50"].to_numpy(dtype=float)

    hard_range_passed = bool(
        np.all(gsk3b_values >= HARD_MIN_PIC50)
        and np.all(gsk3b_values <= HARD_MAX_PIC50)
        and np.all(jnk3_values >= HARD_MIN_PIC50)
        and np.all(jnk3_values <= HARD_MAX_PIC50)
    )

    expected_invalid_count = len(INVALID_TEST_SMILES)

    invalid_handling_passed = bool(
        len(invalid) == expected_invalid_count
        and invalid["pred_gsk3b_pic50"].isna().all()
        and invalid["pred_jnk3_pic50"].isna().all()
    )

    summary = {
        "morgan_radius": MORGAN_RADIUS,
        "morgan_fp_size": MORGAN_FP_SIZE,
        "n_total_inputs": len(predictions),
        "n_valid_predictions": len(valid),
        "n_invalid_smiles_detected": len(invalid),
        "expected_invalid_smiles": expected_invalid_count,
        "invalid_handling_passed": invalid_handling_passed,
        "gsk3b_pred_min": float(np.min(gsk3b_values)),
        "gsk3b_pred_max": float(np.max(gsk3b_values)),
        "gsk3b_pred_mean": float(np.mean(gsk3b_values)),
        "jnk3_pred_min": float(np.min(jnk3_values)),
        "jnk3_pred_max": float(np.max(jnk3_values)),
        "jnk3_pred_mean": float(np.mean(jnk3_values)),
        "hard_range_min": HARD_MIN_PIC50,
        "hard_range_max": HARD_MAX_PIC50,
        "hard_range_passed": hard_range_passed,
    }

    summary.update(determinism_result)

    summary["overall_passed"] = bool(
        invalid_handling_passed
        and hard_range_passed
        and determinism_result["determinism_passed"]
    )

    return pd.DataFrame([summary])


def save_config(summary_path: Path, predictions_path: Path) -> None:
    """保存本次 sanity check 的配置记录"""
    config = {
        "purpose": (
            "Deployment sanity check for final GSK3β and JNK3 QSAR oracles "
            "before TSQ-MCTS molecular core."
        ),
        "important_note": (
            "This script is not a predictive performance evaluation. "
            "Predictive performance must be taken from Step 05 nested CV."
        ),
        "molecular_representation": {
            "type": "Morgan bit fingerprint",
            "radius": MORGAN_RADIUS,
            "n_bits": MORGAN_FP_SIZE,
        },
        "oracle_paths": {
            "gsk3b": str(GSK3B_ORACLE_PATH),
            "jnk3": str(JNK3_ORACLE_PATH),
        },
        "input_datasets_for_sampling": {
            key: str(value)
            for key, value in INPUT_DATASETS.items()
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
    }

    with open(
        OUTPUT_DIR / "oracle_sanity_check_config.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)



# ============================================================
# 4. 主程序
# ============================================================
def main() -> None:
    ensure_output_dir()

    print("Loading final oracles ...")
    gsk3b_model = load_model(GSK3B_ORACLE_PATH)
    jnk3_model = load_model(JNK3_ORACLE_PATH)

    print("Building test SMILES table ...")
    test_table = make_test_input_table()

    print("Running predictions ...")
    predictions = run_predictions(
        test_table,
        gsk3b_model=gsk3b_model,
        jnk3_model=jnk3_model,
    )

    valid_smiles = predictions.loc[
        predictions["status"] == "ok",
        "smiles",
    ].astype(str).tolist()

    determinism_result = check_determinism(
        valid_smiles=valid_smiles,
        gsk3b_model=gsk3b_model,
        jnk3_model=jnk3_model,
    )

    summary = summarize_checks(
        predictions=predictions,
        determinism_result=determinism_result,
    )

    predictions_path = OUTPUT_DIR / "oracle_sanity_check_predictions.csv"
    summary_path = OUTPUT_DIR / "oracle_sanity_check_summary.csv"

    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)

    save_config(
        summary_path=summary_path,
        predictions_path=predictions_path,
    )

    print("\n=== Oracle sanity check finished ===")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")
    print(f"Config: {OUTPUT_DIR / 'oracle_sanity_check_config.json'}")

    if bool(summary.loc[0, "overall_passed"]):
        print("Overall status: PASSED")
    else:
        raise RuntimeError(
            "Oracle sanity check failed. "
            "请查看 oracle_sanity_check_summary.csv。"
        )



if __name__ == "__main__":
    main()