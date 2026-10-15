from __future__ import annotations

"""
GSK3β / JNK3 单靶点 QSAR oracle 的外层交叉验证训练

1. 对每个靶点、每种划分、每个外层折，训练以下回归模型：
   - Random Forest Regressor：稳健、非线性、QSAR中常用
   - Support Vector Regressor：经典小样本非线性回归方法
   - HistGradientBoosting Regressor：现代梯度提升树基线

2. 用途：模型验证、模型比较、选择 oracle

3. 输入：第四步固定好的 scaffold/random 外层 train/test folds

4. 方法：outer 5-fold CV + inner 3-fold CV

5. 输出：交叉验证性能、逐分子预测、最佳参数、候选模型

6. 论文用途：报告 QSAR 模型性能
"""

import hashlib
import json
import math
import platform
import sys
import warnings
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR



# ============================================================
# 0. 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_DIR, RESULT_DIR, SPLIT_DATA_DIR



# ============================================================
# 1. 固定参数
# ============================================================
SPLIT_ROOT = SPLIT_DATA_DIR / "paper_cv"
QSAR_RESULT_DIR = RESULT_DIR / "qsar"
QSAR_MODEL_DIR = MODEL_DIR / "qsar_cv_models"

MORGAN_RADIUS = 3
MORGAN_FP_SIZE = 1024

OUTER_FOLDS = [1, 2, 3, 4, 5]
SPLIT_METHODS = ["scaffold", "random"]
TARGETS = ["gsk3b", "jnk3"]

INNER_N_SPLITS = 3
RANDOM_SEED = 42
N_ACTIVITY_BINS_INNER = 5

ACTIVITY_THRESHOLDS = {
    "6_0": 6.0,
    "6_5": 6.5,
    "7_0": 7.0,
}

MODELS_TO_RUN = [
    "random_forest",
    "svr_rbf",
    "hist_gradient_boosting",
]

# 正式论文实验必须为 False。只在检查代码能否跑通时临时改为 True
QUICK_DEBUG = False



# ============================================================
# 2. 模型与参数网格
# ============================================================
def get_model_space() -> dict[str, dict[str, Any]]:
    """
    返回模型和超参数网格

    当前网格是传统 QSAR baseline 的第一轮论文实验设置：
    不追求极端大规模搜索，重点是可复现、不过拟合、外层测试集不参与调参
    """
    return {
        "random_forest": {
            "estimator": RandomForestRegressor(
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            "param_grid": {
                "n_estimators": [300, 600],
                "max_depth": [None, 20],
                "min_samples_leaf": [1, 3],
                "max_features": ["sqrt"],
            },
        },
        "svr_rbf": {
            "estimator": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf")),
                ]
            ),
            "param_grid": {
                "model__C": [1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.01],
                "model__epsilon": [0.1, 0.2],
            },
        },
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingRegressor(
                loss="squared_error",
                random_state=RANDOM_SEED,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            ),
            "param_grid": {
                "learning_rate": [0.03, 0.06, 0.10],
                "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 0.1],
                "max_iter": [300],
            },
        },
    }



# ============================================================
# 3. 工具函数
# ============================================================
def sha256_file(path: Path) -> str:
    """计算文件 SHA256，记录本次实验使用的确切输入文件"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_directories() -> None:
    """创建输出目录"""
    (QSAR_RESULT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (QSAR_RESULT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    QSAR_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_similarity_table() -> pd.DataFrame:
    """读取第四步生成的测试分子到训练集最大相似度表"""
    path = SPLIT_ROOT / "similarity_per_molecule.csv"
    if not path.exists():
        warnings.warn(
            f"未找到相似性文件：{path}。预测结果中将没有 max_tanimoto_to_train。"
        )
        return pd.DataFrame()

    table = pd.read_csv(path)
    required = {
        "target",
        "split_method",
        "outer_fold",
        "structure_key",
        "max_tanimoto_to_train",
    }
    missing = required - set(table.columns)
    if missing:
        warnings.warn(f"相似性文件缺少字段 {sorted(missing)}，将不进行相似性合并。")
        return pd.DataFrame()
    return table


def load_fold_data(
    target: str,
    split_method: str,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """读取一个外层折的训练集和测试集"""
    fold_dir = SPLIT_ROOT / split_method / f"fold_{outer_fold}"
    train_path = fold_dir / f"{target}_train.csv"
    test_path = fold_dir / f"{target}_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"找不到训练集：{train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"找不到测试集：{test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_fold_data(train, test, target, split_method, outer_fold)
    return train, test, train_path, test_path


def validate_fold_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    split_method: str,
    outer_fold: int,
) -> None:
    """再次检查一个外层折，防止文件被手工误改"""
    required = {
        "structure_key",
        "standardized_smiles",
        "pic50_median",
        "active_label_6_0",
        "active_label_6_5",
        "active_label_7_0",
    }

    for name, dataframe in [("train", train), ("test", test)]:
        missing = required - set(dataframe.columns)
        if missing:
            raise KeyError(
                f"{target} {split_method} fold {outer_fold} {name} 缺少字段：{sorted(missing)}"
            )
        if dataframe.empty:
            raise ValueError(f"{target} {split_method} fold {outer_fold} {name} 为空。")
        if dataframe["structure_key"].duplicated().any():
            raise ValueError(f"{target} {split_method} fold {outer_fold} {name} 存在重复结构。")

    overlap = set(train["structure_key"].astype(str)) & set(test["structure_key"].astype(str))
    if overlap:
        raise RuntimeError(
            f"{target} {split_method} fold {outer_fold} 发生结构泄漏，重叠数：{len(overlap)}"
        )

    if split_method == "scaffold" and "cv_scaffold_group" in train.columns:
        scaffold_overlap = set(train["cv_scaffold_group"].astype(str)) & set(
            test["cv_scaffold_group"].astype(str)
        )
        if scaffold_overlap:
            raise RuntimeError(
                f"{target} scaffold fold {outer_fold} 发生骨架泄漏，重叠数：{len(scaffold_overlap)}"
            )



# ============================================================
# 4. 分子指纹
# ============================================================
def smiles_to_morgan_array(smiles: str) -> np.ndarray:
    """
    将单个 SMILES 转成 Morgan fingerprint 数组

    本项目统一使用 radius=3、nBits=1024
    该设置必须与数据划分相似性审计、QSAR 训练和后续 MCTS oracle 保持一致
    """
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
    """将 standardized_smiles 全部转换为指纹矩阵。"""
    arrays = [smiles_to_morgan_array(smiles) for smiles in dataframe["standardized_smiles"]]
    return np.asarray(arrays, dtype=np.float32)



# ============================================================
# 5. 内层 CV 与指标计算
# ============================================================
def make_inner_cv_splits(train_dataframe: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    在外层训练集内部建立 inner 3-fold CV

    优先按 pic50_median 五分位区间分层；若不可行，回退到 active_label_6_0；
    若仍不可行，使用普通 KFold。整个过程不接触外层测试集
    """
    y = train_dataframe["pic50_median"].to_numpy()

    try:
        labels = pd.qcut(
            train_dataframe["pic50_median"],
            q=N_ACTIVITY_BINS_INNER,
            labels=False,
            duplicates="drop",
        )
        labels = pd.Series(labels).astype(int)
        counts = labels.value_counts()
        if labels.nunique() >= 2 and counts.min() >= INNER_N_SPLITS:
            splitter = StratifiedKFold(
                n_splits=INNER_N_SPLITS,
                shuffle=True,
                random_state=RANDOM_SEED,
            )
            dummy_x = np.zeros((len(y), 1))
            return list(splitter.split(dummy_x, labels))
    except Exception:
        pass

    try:
        labels = train_dataframe["active_label_6_0"].astype(int)
        counts = labels.value_counts()
        if labels.nunique() >= 2 and counts.min() >= INNER_N_SPLITS:
            splitter = StratifiedKFold(
                n_splits=INNER_N_SPLITS,
                shuffle=True,
                random_state=RANDOM_SEED,
            )
            dummy_x = np.zeros((len(y), 1))
            return list(splitter.split(dummy_x, labels))
    except Exception:
        pass

    splitter = KFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    dummy_x = np.zeros((len(y), 1))
    return list(splitter.split(dummy_x))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        value = spearmanr(y_true, y_pred).correlation
        return float(value) if value is not None else np.nan
    except Exception:
        return np.nan


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        value = pearsonr(y_true, y_pred).statistic
        return float(value)
    except Exception:
        return np.nan


def safe_auc(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true_binary)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true_binary, y_score))
    except Exception:
        return np.nan


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算回归指标和按不同 pIC50 阈值的分类辅助指标"""
    metrics: dict[str, float] = {
        "rmse": rmse_score(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "spearman": safe_spearman(y_true, y_pred),
        "pearson": safe_pearson(y_true, y_pred),
    }

    for suffix, threshold in ACTIVITY_THRESHOLDS.items():
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)

        metrics[f"auc_{suffix}"] = safe_auc(y_true_binary, y_pred)
        if len(np.unique(y_true_binary)) < 2:
            metrics[f"balanced_accuracy_{suffix}"] = np.nan
        else:
            metrics[f"balanced_accuracy_{suffix}"] = float(
                balanced_accuracy_score(y_true_binary, y_pred_binary)
            )

        metrics[f"precision_{suffix}"] = float(
            precision_score(y_true_binary, y_pred_binary, zero_division=0)
        )
        metrics[f"recall_{suffix}"] = float(
            recall_score(y_true_binary, y_pred_binary, zero_division=0)
        )
        metrics[f"f1_{suffix}"] = float(
            f1_score(y_true_binary, y_pred_binary, zero_division=0)
        )

    return metrics



# ============================================================
# 6. 单个模型训练
# ============================================================
def fit_one_model(
    model_name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Any, dict[str, Any], float, pd.DataFrame]:
    """
    在外层训练集上执行 GridSearchCV
    refit=True 表示找到最佳参数后，自动用完整外层训练集重训该模型
    """
    grid = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv_splits,
        refit=True,
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )

    grid.fit(x_train, y_train)

    best_estimator = grid.best_estimator_
    best_params = grid.best_params_
    best_inner_rmse = float(-grid.best_score_)

    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.insert(0, "model_name", model_name)

    return best_estimator, best_params, best_inner_rmse, cv_results



# ============================================================
# 7. 单个外层折训练与评价
# ============================================================
def run_one_outer_fold(
    target: str,
    split_method: str,
    outer_fold: int,
    similarity_table: pd.DataFrame,
    model_space: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    """训练并评价一个靶点、一个划分方法、一个外层折的全部模型。"""
    train, test, train_path, test_path = load_fold_data(
        target=target,
        split_method=split_method,
        outer_fold=outer_fold,
    )

    x_train = dataframe_to_fingerprints(train)
    x_test = dataframe_to_fingerprints(test)

    y_train = train["pic50_median"].to_numpy(dtype=float)
    y_test = test["pic50_median"].to_numpy(dtype=float)

    inner_cv_splits = make_inner_cv_splits(train)

    fold_metric_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    grid_result_tables: list[pd.DataFrame] = []

    for model_name in MODELS_TO_RUN:
        if model_name not in model_space:
            raise KeyError(f"未知模型：{model_name}")

        print(
            f"Training target={target}, split={split_method}, "
            f"outer_fold={outer_fold}, model={model_name}"
        )

        model_info = model_space[model_name]
        best_estimator, best_params, best_inner_rmse, cv_results = fit_one_model(
            model_name=model_name,
            estimator=model_info["estimator"],
            param_grid=model_info["param_grid"],
            x_train=x_train,
            y_train=y_train,
            inner_cv_splits=inner_cv_splits,
        )

        y_pred = best_estimator.predict(x_test)
        metrics = compute_metrics(y_test, y_pred)

        model_output_dir = QSAR_MODEL_DIR / target / split_method / f"fold_{outer_fold}"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_output_dir / f"{model_name}.pkl"
        joblib.dump(best_estimator, model_path)

        metric_row: dict[str, Any] = {
            "target": target,
            "split_method": split_method,
            "outer_fold": outer_fold,
            "model_name": model_name,
            "n_train": len(train),
            "n_test": len(test),
            "inner_cv_splits": INNER_N_SPLITS,
            "best_inner_rmse": best_inner_rmse,
            "best_params_json": json.dumps(best_params, ensure_ascii=False),
            "model_path": str(model_path),
            "train_path": str(train_path),
            "test_path": str(test_path),
        }
        metric_row.update(metrics)
        fold_metric_rows.append(metric_row)

        predictions = test[
            [
                "structure_key",
                "standardized_smiles",
                "pic50_median",
                "active_label_6_0",
                "active_label_6_5",
                "active_label_7_0",
            ]
        ].copy()
        predictions.insert(1, "target", target)
        predictions.insert(2, "split_method", split_method)
        predictions.insert(3, "outer_fold", outer_fold)
        predictions.insert(4, "model_name", model_name)
        predictions = predictions.rename(columns={"pic50_median": "observed_pic50"})
        predictions["predicted_pic50"] = y_pred
        predictions["absolute_error"] = np.abs(
            predictions["observed_pic50"].to_numpy(dtype=float) - y_pred
        )
        predictions["squared_error"] = (
            predictions["observed_pic50"].to_numpy(dtype=float) - y_pred
        ) ** 2

        for suffix, threshold in ACTIVITY_THRESHOLDS.items():
            predictions[f"pred_active_{suffix}"] = (
                predictions["predicted_pic50"] >= threshold
            ).astype(int)

        if not similarity_table.empty:
            merge_columns = [
                "target",
                "split_method",
                "outer_fold",
                "structure_key",
                "max_tanimoto_to_train",
            ]
            predictions = predictions.merge(
                similarity_table[merge_columns],
                on=["target", "split_method", "outer_fold", "structure_key"],
                how="left",
            )

        prediction_tables.append(predictions)

        cv_results.insert(0, "target", target)
        cv_results.insert(1, "split_method", split_method)
        cv_results.insert(2, "outer_fold", outer_fold)
        cv_results.insert(3, "best_params_json", json.dumps(best_params))
        grid_result_tables.append(cv_results)

    return fold_metric_rows, prediction_tables, grid_result_tables



# ============================================================
# 8. 汇总结果
# ============================================================
def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """按 target × split_method × model 汇总五折外层测试性能。"""
    metric_columns = [
        "rmse",
        "mae",
        "r2",
        "spearman",
        "pearson",
        "auc_6_0",
        "auc_6_5",
        "auc_7_0",
        "balanced_accuracy_6_0",
        "balanced_accuracy_6_5",
        "balanced_accuracy_7_0",
    ]

    rows = []
    group_columns = ["target", "split_method", "model_name"]

    for keys, group in fold_metrics.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["n_outer_folds"] = len(group)

        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_min"] = np.nan
                row[f"{metric}_max"] = np.nan
            else:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
                row[f"{metric}_median"] = float(values.median())
                row[f"{metric}_min"] = float(values.min())
                row[f"{metric}_max"] = float(values.max())
        rows.append(row)

    return pd.DataFrame(rows)


def select_best_models(metric_summary: pd.DataFrame) -> pd.DataFrame:
    """
    根据 scaffold CV 的平均 RMSE 和 Spearman 推荐候选 oracle。
    最终是否采用某模型，还要结合稳定性、误差—相似性关系和后续生成任务需求
    """
    scaffold = metric_summary[metric_summary["split_method"] == "scaffold"].copy()
    if scaffold.empty:
        return pd.DataFrame()

    scaffold["rank_rmse"] = scaffold.groupby("target")["rmse_mean"].rank(
        method="min",
        ascending=True,
    )
    scaffold["rank_spearman"] = scaffold.groupby("target")["spearman_mean"].rank(
        method="min",
        ascending=False,
    )
    scaffold["combined_rank"] = scaffold["rank_rmse"] + scaffold["rank_spearman"]

    scaffold = scaffold.sort_values(
        ["target", "combined_rank", "rmse_mean", "spearman_mean"],
        ascending=[True, True, True, False],
    )
    return scaffold



# ============================================================
# 9. 主程序
# ============================================================
def main() -> None:
    ensure_directories()

    if QUICK_DEBUG:
        targets = ["gsk3b"]
        split_methods = ["scaffold"]
        outer_folds = [1]
        warnings.warn(
            "QUICK_DEBUG=True：当前只运行 gsk3b/scaffold/fold_1。正式论文实验必须设置为 False。"
        )
    else:
        targets = TARGETS
        split_methods = SPLIT_METHODS
        outer_folds = OUTER_FOLDS

    model_space = get_model_space()
    similarity_table = load_similarity_table()

    all_metric_rows: list[dict[str, Any]] = []
    all_prediction_tables: list[pd.DataFrame] = []
    all_grid_result_tables: list[pd.DataFrame] = []
    input_files: dict[str, dict[str, str]] = {}

    for target in targets:
        for split_method in split_methods:
            for outer_fold in outer_folds:
                train_path = SPLIT_ROOT / split_method / f"fold_{outer_fold}" / f"{target}_train.csv"
                test_path = SPLIT_ROOT / split_method / f"fold_{outer_fold}" / f"{target}_test.csv"

                input_files[f"{target}_{split_method}_fold_{outer_fold}_train"] = {
                    "path": str(train_path),
                    "sha256": sha256_file(train_path),
                }
                input_files[f"{target}_{split_method}_fold_{outer_fold}_test"] = {
                    "path": str(test_path),
                    "sha256": sha256_file(test_path),
                }

                metric_rows, prediction_tables, grid_result_tables = run_one_outer_fold(
                    target=target,
                    split_method=split_method,
                    outer_fold=outer_fold,
                    similarity_table=similarity_table,
                    model_space=model_space,
                )
                all_metric_rows.extend(metric_rows)
                all_prediction_tables.extend(prediction_tables)
                all_grid_result_tables.extend(grid_result_tables)

    fold_metrics = pd.DataFrame(all_metric_rows)
    predictions = pd.concat(all_prediction_tables, ignore_index=True)
    grid_results = pd.concat(all_grid_result_tables, ignore_index=True)
    metric_summary = summarize_metrics(fold_metrics)
    best_model_candidates = select_best_models(metric_summary)

    fold_metrics_path = QSAR_RESULT_DIR / "metrics" / "qsar_fold_metrics.csv"
    metric_summary_path = QSAR_RESULT_DIR / "metrics" / "qsar_metric_summary.csv"
    best_candidates_path = QSAR_RESULT_DIR / "metrics" / "qsar_best_model_candidates.csv"
    predictions_path = QSAR_RESULT_DIR / "predictions" / "qsar_predictions.csv"
    grid_results_path = QSAR_RESULT_DIR / "metrics" / "qsar_gridsearch_results.csv"
    config_path = QSAR_RESULT_DIR / "metrics" / "qsar_training_config.json"

    fold_metrics.to_csv(fold_metrics_path, index=False)
    metric_summary.to_csv(metric_summary_path, index=False)
    best_model_candidates.to_csv(best_candidates_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    grid_results.to_csv(grid_results_path, index=False)

    config = {
        "purpose": "Nested cross-validation training and evaluation of single-target QSAR pIC50 regression models for GSK3β and JNK3.",
        "molecular_representation": {
            "type": "Morgan bit fingerprint",
            "radius": MORGAN_RADIUS,
            "n_bits": MORGAN_FP_SIZE,
            "reason": "Use the same 1024-bit radius-3 Morgan fingerprint setting as the original manuscript and data-splitting similarity audit.",
        },
        "outer_cv": {
            "primary_evaluation": "5-fold Bemis-Murcko scaffold CV",
            "secondary_evaluation": "5-fold stratified random CV",
            "outer_folds": OUTER_FOLDS,
            "split_root": str(SPLIT_ROOT),
        },
        "inner_cv": {
            "n_splits": INNER_N_SPLITS,
            "stratification": "five quantile bins of pic50_median; fallback to active_label_6_0; fallback to KFold if stratification is impossible",
            "rule": "Inner CV uses only the outer-training data. Outer test folds are never used for hyperparameter selection.",
        },
        "models": {
            name: {
                "estimator": str(info["estimator"]),
                "param_grid": info["param_grid"],
            }
            for name, info in model_space.items()
            if name in MODELS_TO_RUN
        },
        "outputs": {
            "fold_metrics": str(fold_metrics_path),
            "metric_summary": str(metric_summary_path),
            "best_model_candidates": str(best_candidates_path),
            "predictions": str(predictions_path),
            "gridsearch_results": str(grid_results_path),
            "cv_models": str(QSAR_MODEL_DIR),
        },
        "input_files": input_files,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit.__version__,
            "joblib": joblib.__version__,
        },
        "method_references": [
            {"topic": "OECD QSAR validation principles"},
            {"topic": "Bemis-Murcko molecular frameworks", "doi": "10.1021/jm9602928"},
            {"topic": "MoleculeNet molecular ML benchmark", "doi": "10.1039/C7SC02664A"},
            {"topic": "Molecular OOD evaluation", "doi": "10.1021/acs.jcim.3c01774"},
            {"topic": "Similarity leakage and DataSAIL context", "doi": "10.1038/s41467-025-58606-8"},
        ],
    }

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n=== QSAR nested CV training finished ===")
    print(f"Fold metrics: {fold_metrics_path}")
    print(f"Metric summary: {metric_summary_path}")
    print(f"Best model candidates: {best_candidates_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Grid-search results: {grid_results_path}")
    print(f"Training config: {config_path}")
    print(f"CV models: {QSAR_MODEL_DIR}")



if __name__ == "__main__":
    main()