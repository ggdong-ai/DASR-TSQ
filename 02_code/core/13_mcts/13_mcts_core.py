"""
This file assembles the four MCTS steps:Selection -> Expansion -> Direct leaf evaluation -> Backpropagation
本文件组装 MCTS 的四个步骤：选择 -> 扩展 -> 叶节点直接评价 -> 反向回传
"""

from __future__ import annotations
import copy
import importlib.util
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional


_THIS_DIR = Path(__file__).resolve().parent


def _load(module_name: str, file_name: str):
    """
    Load one split MCTS module from the current directory.
    从当前目录加载一个拆分后的 MCTS 模块。
    """
    path = _THIS_DIR / file_name

    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


_common = _load("mcts_common", "13_0_mcts_common.py")
_selection = _load("mcts_selection", "13_1_mcts_selection.py")
_expansion = _load("mcts_expansion", "13_2_mcts_expansion.py")
_evaluation = _load("mcts_evaluation", "13_3_mcts_evaluation.py")
_backprop = _load("mcts_backprop", "13_4_mcts_backprop.py")


TSQMCTSConfig = _common.TSQMCTSConfig
MCTSNode = _common.MCTSNode
MCTSResult = _common.MCTSResult
canonicalize_smiles = _common.canonicalize_smiles
safe_float = _common.safe_float

select_child = _selection.select_child

ensure_pending_candidates = _expansion.ensure_pending_candidates
expand_one_child = _expansion.expand_one_child
MoleculeMutator = _expansion.MoleculeMutator

FrozenFitnessAdapter = _evaluation.FrozenFitnessAdapter
PSuccessPrior = _evaluation.PSuccessPrior
evaluate_node = _evaluation.evaluate_node

backpropagate = _backprop.backpropagate


class TSQMCTS:
    """
    Single-root prior-throttled MCTS local search engine.
    单 root 的 prior-throttled MCTS 局部搜索引擎。
    """

    def __init__(
        self,
        config: TSQMCTSConfig,
        fitness_adapter: Optional[FrozenFitnessAdapter] = None,
        prior: Optional[PSuccessPrior] = None,
        mutator: Optional[MoleculeMutator] = None,
    ) -> None:
        config.validate()

        self.config = config

        # EN: Frozen evaluator for final_reward.
        # CN: 用于 final_reward 的冻结评分器。
        self.fitness = fitness_adapter or FrozenFitnessAdapter()

        # EN: P_success prior used only for selection.
        # CN: 只用于 selection 的 P_success 先验。
        self.prior = prior or PSuccessPrior()

        # EN: Local molecular graph-edit generator.
        # CN: 局部分子图编辑生成器。
        self.mutator = mutator or MoleculeMutator()

        # EN: Separate random streams improve reproducibility.
        # CN: 分离随机流可以提高实验可复现性。
        self._mutation_rng = random.Random(config.random_seed + 101)
        self._prior_gate_rng = random.Random(config.random_seed + 202)
        self._expansion_rng = random.Random(config.random_seed + 303)

        self._next_node_id = 0
        self._selection_event_counter = [0]
        self._seen_smiles: set[str] = set()
        self._nodes_by_id: dict[int, MCTSNode] = {}
        self._selection_records: list[dict[str, Any]] = []
        self._best_state: dict[str, Any] = {
            "best_reward": -float("inf"),
            "best_node": None,
        }

    def _reset_tree_state(self) -> None:
        """
        Reset tree-specific state before each search.
        每次 search 前重置当前搜索树状态。
        """
        self._next_node_id = 0
        self._selection_event_counter = [0]
        self._seen_smiles = set()
        self._nodes_by_id = {}
        self._selection_records = []
        self._best_state = {
            "best_reward": -float("inf"),
            "best_node": None,
        }

    def _new_node(
        self,
        smiles: str,
        parent: Optional[MCTSNode],
        mutation_action: str,
    ) -> MCTSNode:
        """
        Create a new MCTS node and compute its P_success prior.
        创建一个新的 MCTS 节点，并计算其 P_success 先验。
        """
        canonical = canonicalize_smiles(smiles)

        if canonical is None:
            raise ValueError("Cannot create node from invalid SMILES.")

        node = MCTSNode(
            node_id=self._next_node_id,
            smiles=canonical,
            parent=parent,
            depth=0 if parent is None else parent.depth + 1,
            mutation_action=mutation_action,
        )

        self._next_node_id += 1

        # EN: P_success is stored for future selection only.
        # CN: P_success 仅保存供后续 selection 使用。
        prior_result = self.prior.predict(canonical)
        node.p_success = safe_float(prior_result.get("p_success"))
        node.prior_status = str(prior_result.get("prior_status", "unknown"))

        self._nodes_by_id[node.node_id] = node

        return node

    def _tree_policy(self, root: MCTSNode, iteration: int) -> MCTSNode:
        """
        Find the next molecule that should be evaluated in this MCTS iteration.
        找到本轮 MCTS 应该评价的下一个分子节点。

        Starting from the root, the tree policy walks down the current search tree.
        它从 root 出发，沿着当前搜索树向下走。

        At each node:
        在每个节点处：

        1. If this node still has unexpanded mutation candidates, expand one candidate
           into a new child and return this new child for direct evaluation.
           如果该节点还有尚未扩展的突变候选，则扩展一个新 child，
           并返回这个新 child 进行直接评价。

        2. If all mutation candidates have already been expanded but child nodes exist,
           select one child by the selection rule and continue walking downward.
           如果该节点的突变候选已经全部扩展完，但存在 child 节点，
           则根据 selection 规则选择一个 child，并继续向下走。

        3. If the maximum depth is reached or no further move is available, return the
           current node as the leaf to be evaluated.
           如果达到最大深度，或当前节点已经无路可走，则返回当前节点作为待评价叶节点。
        """
        node = root

        while True:
            if node.depth >= self.config.max_depth:
                return node

            ensure_pending_candidates(
                node=node,
                config=self.config,
                mutator=self.mutator,
                mutation_rng=self._mutation_rng,
                seen_smiles=self._seen_smiles,
            )

            if node.pending_candidates:
                expanded = expand_one_child(
                    node=node,
                    new_node_fn=self._new_node,
                    config=self.config,
                    mutator=self.mutator,
                    mutation_rng=self._mutation_rng,
                    expansion_rng=self._expansion_rng,
                    seen_smiles=self._seen_smiles,
                )

                if expanded is not None:
                    return expanded

                continue

            if node.children:
                node = select_child(
                    parent=node,
                    iteration=iteration,
                    config=self.config,
                    prior_gate_rng=self._prior_gate_rng,
                    expansion_rng=self._expansion_rng,
                    selection_records=self._selection_records,
                    event_counter=self._selection_event_counter,
                )

                continue

            return node

    def _flatten_node_records(self, root: MCTSNode) -> list[dict[str, Any]]:
        """
        Flatten the search tree into tabular node records.
        将搜索树展平成表格形式的节点记录。
        """
        records: list[dict[str, Any]] = []
        stack = [root]

        while stack:
            node = stack.pop()

            parent_id = node.parent.node_id if node.parent is not None else None
            parent_smiles = node.parent.smiles if node.parent is not None else None
            score = node.score_record

            records.append(
                {
                    "node_id": node.node_id,
                    "parent_node_id": parent_id,
                    "smiles": node.smiles,
                    "parent_smiles": parent_smiles,
                    "depth": node.depth,
                    "mutation_action": node.mutation_action,
                    "visits": node.visits,
                    "total_reward": node.total_reward,
                    "mean_reward": node.mean_reward,
                    "p_success": node.p_success,
                    "prior_status": node.prior_status,
                    "scoring_status": score.get("scoring_status"),
                    "valid_smiles": score.get("valid_smiles"),
                    "pred_gsk3b_pic50": safe_float(score.get("pred_gsk3b_pic50")),
                    "pred_jnk3_pic50": safe_float(score.get("pred_jnk3_pic50")),
                    "dual_min_pic50": safe_float(score.get("dual_min_pic50")),
                    "dual_mean_pic50": safe_float(score.get("dual_mean_pic50")),
                    "qed": safe_float(score.get("qed")),
                    "sa_score": safe_float(score.get("sa_score")),
                    "sa_reward": safe_float(score.get("sa_reward")),
                    "dual_activity_reward": safe_float(
                        score.get("dual_activity_reward")
                    ),
                    "final_reward": safe_float(score.get("final_reward")),
                }
            )

            for child in sorted(
                node.children,
                key=lambda item: item.node_id,
                reverse=True,
            ):
                stack.append(child)

        records.sort(key=lambda record: record["node_id"])

        return records

    def _build_summary(self, root: MCTSNode) -> dict[str, Any]:
        """
        Build a JSON-serializable summary of one search.
        构建一次搜索的 JSON 可序列化摘要。
        """
        eligible_event_ids = {
            record["selection_event_id"]
            for record in self._selection_records
            if bool(record["prior_eligible"])
        }

        gate_open_event_ids = {
            record["selection_event_id"]
            for record in self._selection_records
            if bool(record["prior_eligible"]) and bool(record["prior_gate_open"])
        }

        observed_fraction = (
            len(gate_open_event_ids) / len(eligible_event_ids)
            if eligible_event_ids
            else float("nan")
        )

        best_node = self._best_state.get("best_node") or root
        best_score = best_node.score_record

        return {
            "experiment_name": self.config.experiment_name,
            "run_id": self.config.run_id,
            "root_smiles": root.smiles,
            "mcts_config": asdict(self.config),
            "n_nodes": len(self._nodes_by_id),
            "n_unique_structures": len(self._seen_smiles),
            "n_selection_record_rows": len(self._selection_records),
            "n_prior_eligible_events": len(eligible_event_ids),
            "n_prior_gate_open_events": len(gate_open_event_ids),
            "observed_prior_activation_fraction": observed_fraction,
            "best_node_id": best_node.node_id,
            "best_smiles": best_node.smiles,
            "best_final_reward": safe_float(best_score.get("final_reward"), 0.0),
            "best_pred_gsk3b_pic50": safe_float(best_score.get("pred_gsk3b_pic50")),
            "best_pred_jnk3_pic50": safe_float(best_score.get("pred_jnk3_pic50")),
            "best_dual_min_pic50": safe_float(best_score.get("dual_min_pic50")),
            "best_qed": safe_float(best_score.get("qed")),
            "best_sa_score": safe_float(best_score.get("sa_score")),
            "frozen_fitness": self.fitness.metadata(),
            "p_success_prior": self.prior.metadata(),
            "algorithm_note": (
                "p_call is the Bernoulli prior-gating probability, not a fixed score weight. "
                "Unvisited children are evaluated first. P_success is used only during eligible "
                "sibling selection and never changes final_reward."
            ),
        }

    def search(self, root_smiles: str) -> MCTSResult:
        """
        Run one complete single-root MCTS search.运行一次完整的单 root MCTS 搜索。
        """
        self._reset_tree_state()

        canonical_root = canonicalize_smiles(root_smiles)

        if canonical_root is None:
            raise ValueError(f"Invalid root SMILES: {root_smiles}")

        root = self._new_node(
            smiles=canonical_root,
            parent=None,
            mutation_action="initial_seed",
        )

        self._seen_smiles.add(root.smiles)

        # EN: Evaluate the root once before starting MCTS iterations.
        # CN: 在开始 MCTS 迭代前，先评价一次 root。
        root_reward = evaluate_node(root, self.fitness, self._best_state)
        backpropagate(root, root_reward)

        for iteration in range(1, self.config.mcts_iterations + 1):
            leaf = self._tree_policy(root, iteration)
            reward = evaluate_node(leaf, self.fitness, self._best_state)
            backpropagate(leaf, reward)

        node_records = self._flatten_node_records(root)
        summary = self._build_summary(root)

        return MCTSResult(
            root=root,
            best_node=self._best_state.get("best_node") or root,
            node_records=node_records,
            selection_records=copy.deepcopy(self._selection_records),
            summary=summary,
        )


__all__ = [
    "TSQMCTSConfig",
    "TSQMCTS",
    "MCTSResult",
    "FrozenFitnessAdapter",
    "PSuccessPrior",
    "MoleculeMutator",
    "canonicalize_smiles",
]


if __name__ == "__main__":
    print(
        "This module is an importable prior-throttled MCTS engine. "
        "Use a runner script to execute searches."
    )