"""
This module propagates the frozen final_reward from the evaluated leaf node back to all ancestors along the selected path.
本模块将被评价叶节点的 frozen final_reward 沿选择路径回传给所有祖先节点。
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path


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
MCTSNode = _common.MCTSNode


def backpropagate(node: MCTSNode, reward: float) -> None:
    """
    Propagate frozen final_reward from a leaf node to the root.
    将冻结 final_reward 从叶节点回传到根节点。

    Only visits and total_reward are updated.
    只更新 visits 和 total_reward。
    """
    current = node

    while current is not None:
        current.visits += 1
        current.total_reward += reward
        current = current.parent