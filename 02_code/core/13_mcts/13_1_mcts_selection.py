"""
This module decides which existing child node should be followed during tree traversal.
本模块决定树搜索过程中应该沿着哪个已有子节点继续向下搜索。
"""

from __future__ import annotations
import importlib.util
import math
import random
import sys
from pathlib import Path



def _load_common_module():
    """
    EN: Load 13_0_mcts_common.py once and reuse the same module object.
    CN: 只加载一次 13_0_mcts_common.py，并复用同一个模块对象。

    EN: This avoids repeatedly reloading the shared dataclass definitions.
    CN: 这样可以避免重复加载公共 dataclass 定义。
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


def uct_score(parent: MCTSNode, child: MCTSNode, config) -> float:
    if child.visits <= 0:
        return float("inf")

    # EN: Exploitation term: average reward observed from this child.
    # CN: 利用项：该子节点已有回传 reward 的平均值。
    exploitation = child.mean_reward

    # EN: Exploration term: favors less-visited children.
    # CN: 探索项：倾向于访问次数较少的子节点。
    parent_visits = max(1, parent.visits)
    exploration = config.exploration_constant * math.sqrt(
        math.log(parent_visits + 1.0) / child.visits
    )

    return exploitation + exploration


def select_child(
    parent: MCTSNode,
    iteration: int,
    config,
    prior_gate_rng: random.Random,
    expansion_rng: random.Random,
    selection_records: list[dict],
    event_counter: list[int],
) -> MCTSNode:
    """
    Select one child under a parent node.  从一个父节点下选择一个子节点。
    Rule:
    规则：
    1. If unvisited children exist, select one unvisited child first without prior gating.
       如果存在未访问子节点，则优先选择未访问子节点，不使用先验门控。

    2. If all children have been visited, sample one Bernoulli gate for the whole sibling set.
       如果所有子节点都已访问，则对整个兄弟节点集合只抽样一次 Bernoulli 门控。

    3. gate=False: compare UCT only.
       gate=False：只比较 UCT。

    4. gate=True: compare UCT + gamma * P_success when P_success is valid.
       gate=True：当 P_success 有效时，比较 UCT + gamma * P_success。
    """
    if not parent.children:
        raise RuntimeError("Cannot select a child from a parent without children.")

    # EN: Standard MCTS rule: every child should receive at least one evaluation.
    # CN: 标准 MCTS 规则：每个子节点应至少获得一次评价机会。
    unvisited = [child for child in parent.children if child.visits == 0]

    if unvisited:
        unvisited.sort(key=lambda child: child.smiles)
        selected = expansion_rng.choice(unvisited)

        event_id = event_counter[0]
        event_counter[0] += 1

        for child in unvisited:
            selection_records.append(
                {
                    "selection_event_id": event_id,
                    "iteration": iteration,
                    "parent_node_id": parent.node_id,
                    "parent_smiles": parent.smiles,
                    "parent_depth": parent.depth,
                    "candidate_node_id": child.node_id,
                    "candidate_smiles": child.smiles,
                    "candidate_visits_before_selection": child.visits,
                    "prior_eligible": False,
                    "prior_gate_open": False,
                    "prior_valid": math.isfinite(child.p_success),
                    "p_call": config.p_call,
                    "gamma": config.gamma,
                    "uct_score": float("inf"),
                    "p_success": child.p_success,
                    "selection_score": float("inf"),
                    "selected": child.node_id == selected.node_id,
                    "selection_rule": "force_unvisited_child_exploration",
                }
            )

        return selected

    # EN: Only after all children have been visited does prior gating become eligible.
    # CN: 只有当所有子节点都访问过后，才允许启用先验门控。
    gate_open = prior_gate_rng.random() < config.p_call

    event_id = event_counter[0]
    event_counter[0] += 1

    scored_children: list[tuple[MCTSNode, float, float, bool, str]] = []

    for child in parent.children:
        uct = uct_score(parent, child, config)

        prior_value = child.p_success
        valid_prior = math.isfinite(prior_value)

        if gate_open and valid_prior:
            # EN: Prior gate is open and P_success is valid.
            # CN: 先验门控打开，且 P_success 有效。
            selection_score = uct + config.gamma * prior_value
            selection_rule = "UCT_plus_P_success"

        elif gate_open and not valid_prior:
            # EN: Gate is open, but P_success is unavailable; fall back to pure UCT.
            # CN: 先验门控打开，但 P_success 不可用，因此回退为纯 UCT。
            selection_score = uct
            selection_rule = "UCT_prior_missing_fallback"

        else:
            # EN: Prior gate is closed; use pure UCT.
            # CN: 先验门控关闭，使用纯 UCT。
            selection_score = uct
            selection_rule = "UCT_only"

        scored_children.append(
            (child, uct, selection_score, valid_prior, selection_rule)
        )

    # EN: Stable sorting makes the selection reproducible when scores tie.
    # CN: 稳定排序保证分数相同时选择结果可复现。
    scored_children.sort(
        key=lambda item: (item[2], item[0].smiles, -item[0].node_id),
        reverse=True,
    )

    selected = scored_children[0][0]

    for child, uct, selection_score, valid_prior, selection_rule in scored_children:
        selection_records.append(
            {
                "selection_event_id": event_id,
                "iteration": iteration,
                "parent_node_id": parent.node_id,
                "parent_smiles": parent.smiles,
                "parent_depth": parent.depth,
                "candidate_node_id": child.node_id,
                "candidate_smiles": child.smiles,
                "candidate_visits_before_selection": child.visits,
                "prior_eligible": True,
                "prior_gate_open": gate_open,
                "prior_valid": valid_prior,
                "p_call": config.p_call,
                "gamma": config.gamma,
                "uct_score": uct,
                "p_success": child.p_success,
                "selection_score": selection_score,
                "selected": child.node_id == selected.node_id,
                "selection_rule": selection_rule,
            }
        )

    return selected