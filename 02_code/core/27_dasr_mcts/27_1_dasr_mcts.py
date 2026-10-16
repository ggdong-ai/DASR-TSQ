"""
27_1_dasr_mcts.py
==================
Independent DASR-MCTS integration branch for fixed-budget online evaluation.
用于固定预算在线评价的独立 DASR-MCTS 集成分支。

The validated Step 13 files are imported but never modified. This branch changes
only the expansion-budget allocation step:

    generate up to K=32 raw mutations -> frozen pre-oracle ranker -> retain B=8

All retained candidates are subsequently evaluated by the same frozen molecular
fitness oracle used by the original MCTS. The ranker never changes final_reward.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MCTS_DIR = PROJECT_ROOT / "02_code" / "core" / "13_mcts"


def _load(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not path.exists():
        raise FileNotFoundError(f"Required module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_common = _load("mcts_common", MCTS_DIR / "13_0_mcts_common.py")
_selection = _load("mcts_selection", MCTS_DIR / "13_1_mcts_selection.py")
_expansion = _load("mcts_expansion", MCTS_DIR / "13_2_mcts_expansion.py")
_evaluation = _load("mcts_evaluation", MCTS_DIR / "13_3_mcts_evaluation.py")
_backprop = _load("mcts_backprop", MCTS_DIR / "13_4_mcts_backprop.py")
_core = _load("step13_mcts_core", MCTS_DIR / "13_mcts_core.py")
_rankers = _load(
    "dasr_frozen_sibling_rankers",
    _THIS_DIR / "27_0_frozen_sibling_rankers.py",
)

TSQMCTSConfig = _common.TSQMCTSConfig
MCTSNode = _common.MCTSNode
MutationCandidate = _common.MutationCandidate
MoleculeMutator = _expansion.MoleculeMutator
FrozenFitnessAdapter = _evaluation.FrozenFitnessAdapter
canonicalize_smiles = _common.canonicalize_smiles
safe_float = _common.safe_float
select_child = _selection.select_child
evaluate_node = _evaluation.evaluate_node
backpropagate = _backprop.backpropagate
BaseTSQMCTS = _core.TSQMCTS
FrozenSiblingRankerBank = _rankers.FrozenSiblingRankerBank

POLICY_TO_RANKER = {
    "random_8": None,
    "action_only_6plus2": "action_only",
    "absolute_child_6plus2": "absolute_child",
    "pointwise_delta_6plus2": "pointwise_delta",
    "pairwise_ranker_6plus2": "pairwise_ranker",
    "listwise_ranker_6plus2": "listwise_ranker",
    "listwise_ranker_top8": "listwise_ranker",
}
SUPPORTED_POLICIES = tuple(POLICY_TO_RANKER)


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def generic_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return "__ACYCLIC__"
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(generic, canonical=True, isomericSmiles=False)


class NullPrior:
    """Disable the legacy P_success branch-selection prior for clean isolation."""

    def predict(self, smiles: str) -> dict[str, Any]:
        return {
            "canonical_smiles": canonicalize_smiles(smiles),
            "p_success": float("nan"),
            "prior_status": "disabled_for_dasr_expansion_study",
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "reason": (
                "Step 27 isolates pre-oracle expansion ranking. Legacy P_success "
                "is disabled for every online comparator."
            ),
        }


@dataclass
class DASRMCTSConfig(TSQMCTSConfig):
    """Step 13 configuration plus fixed candidate/oracle budget controls."""

    raw_candidate_pool_size: int = 32
    oracle_budget: int = 31
    candidate_policy: str = "listwise_ranker_6plus2"
    top_ranked_count: int = 6
    random_exploration_count: int = 2
    max_tree_policy_steps: int = 2000
    max_stagnant_steps: int = 100

    def validate(self) -> None:
        super().validate()
        if self.raw_candidate_pool_size < self.max_children_per_node:
            raise ValueError(
                "raw_candidate_pool_size must be >= max_children_per_node."
            )
        if self.oracle_budget < 1:
            raise ValueError("oracle_budget must be >= 1.")
        if self.candidate_policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Unsupported candidate_policy={self.candidate_policy!r}; "
                f"expected {SUPPORTED_POLICIES}"
            )
        if self.max_tree_policy_steps < self.oracle_budget:
            raise ValueError("max_tree_policy_steps must be >= oracle_budget.")
        if self.max_stagnant_steps < 1:
            raise ValueError("max_stagnant_steps must be >= 1.")
        if self.candidate_policy == "random_8":
            return
        if self.candidate_policy == "listwise_ranker_top8":
            return
        if self.top_ranked_count < 0 or self.random_exploration_count < 0:
            raise ValueError("top/random candidate counts cannot be negative.")
        if (
            self.top_ranked_count + self.random_exploration_count
            != self.max_children_per_node
        ):
            raise ValueError(
                "For *_6plus2 policies, top_ranked_count + "
                "random_exploration_count must equal max_children_per_node."
            )


@dataclass
class DASRMCTSResult:
    root: MCTSNode
    best_node: MCTSNode
    node_records: list[dict[str, Any]]
    selection_records: list[dict[str, Any]]
    summary: dict[str, Any]
    oracle_trajectory: list[dict[str, Any]]
    candidate_selection_records: list[dict[str, Any]]


class CandidateBudgetSelector:
    """Select B candidates from a frozen raw pool without querying the oracle."""

    def __init__(
        self,
        ranker_bank: FrozenSiblingRankerBank,
        config: DASRMCTSConfig,
    ) -> None:
        self.ranker_bank = ranker_bank
        self.config = config

    def select(
        self,
        parent_smiles: str,
        raw_candidates: Sequence[MutationCandidate],
        node_id: int,
    ) -> tuple[list[MutationCandidate], dict[str, Any]]:
        policy = self.config.candidate_policy
        budget = min(self.config.max_children_per_node, len(raw_candidates))
        if budget <= 0:
            return [], {
                "candidate_policy": policy,
                "raw_candidate_count": len(raw_candidates),
                "selected_candidate_count": 0,
                "selected_top_count": 0,
                "selected_random_count": 0,
            }

        rng = random.Random(
            stable_seed(
                self.config.random_seed,
                "candidate_selection",
                policy,
                parent_smiles,
                node_id,
            )
        )

        scores: np.ndarray | None = None
        selected_top: list[MutationCandidate] = []
        selected_random: list[MutationCandidate] = []

        if policy == "random_8":
            indices = list(range(len(raw_candidates)))
            rng.shuffle(indices)
            selected_random = [raw_candidates[index] for index in indices[:budget]]
            selected = selected_random
        else:
            ranker_name = POLICY_TO_RANKER[policy]
            assert ranker_name is not None
            scores = self.ranker_bank.score_candidates(
                ranker_name,
                parent_smiles,
                raw_candidates,
            )
            ordered_indices = sorted(
                range(len(raw_candidates)),
                key=lambda index: (
                    -float(scores[index]),
                    raw_candidates[index].smiles,
                    raw_candidates[index].action_name,
                ),
            )

            if policy == "listwise_ranker_top8":
                top_count = budget
                random_count = 0
            else:
                top_count = min(self.config.top_ranked_count, budget)
                random_count = min(
                    self.config.random_exploration_count,
                    budget - top_count,
                )

            selected_top_indices = ordered_indices[:top_count]
            remainder = ordered_indices[top_count:]
            rng.shuffle(remainder)
            selected_random_indices = remainder[:random_count]

            # If a small raw pool leaves unused budget, fill it by ranking order.
            used = set(selected_top_indices) | set(selected_random_indices)
            fill_indices = [index for index in ordered_indices if index not in used]
            fill_needed = budget - len(used)
            selected_top_indices.extend(fill_indices[:fill_needed])

            selected_top = [raw_candidates[index] for index in selected_top_indices]
            selected_random = [raw_candidates[index] for index in selected_random_indices]
            selected = selected_top + selected_random

        selected_smiles = [candidate.smiles for candidate in selected]
        if len(selected_smiles) != len(set(selected_smiles)):
            raise RuntimeError("Duplicate candidates selected within one sibling pool.")
        raw_smiles = {candidate.smiles for candidate in raw_candidates}
        if not set(selected_smiles).issubset(raw_smiles):
            raise RuntimeError("Selected candidate was not present in the raw pool.")

        score_map: dict[str, float] = {}
        if scores is not None:
            score_map = {
                candidate.smiles: float(score)
                for candidate, score in zip(raw_candidates, scores)
            }

        audit = {
            "candidate_policy": policy,
            "ranker_name": POLICY_TO_RANKER[policy],
            "raw_candidate_count": len(raw_candidates),
            "selected_candidate_count": len(selected),
            "selected_top_count": len(selected_top),
            "selected_random_count": len(selected_random),
            "raw_candidate_smiles_json": json.dumps(
                [candidate.smiles for candidate in raw_candidates],
                ensure_ascii=False,
            ),
            "selected_candidate_smiles_json": json.dumps(
                selected_smiles,
                ensure_ascii=False,
            ),
            "selected_candidate_actions_json": json.dumps(
                [candidate.action_name for candidate in selected],
                ensure_ascii=False,
            ),
            "selected_scores_json": json.dumps(
                [score_map.get(candidate.smiles) for candidate in selected],
                ensure_ascii=False,
            ),
            "score_min": (
                float(np.min(scores)) if scores is not None and len(scores) else float("nan")
            ),
            "score_mean": (
                float(np.mean(scores)) if scores is not None and len(scores) else float("nan")
            ),
            "score_max": (
                float(np.max(scores)) if scores is not None and len(scores) else float("nan")
            ),
        }
        return selected, audit


class DASRMCTS(BaseTSQMCTS):
    """Step 13 MCTS with a frozen, pre-oracle candidate-budget allocator."""

    def __init__(
        self,
        config: DASRMCTSConfig,
        ranker_bank: FrozenSiblingRankerBank,
        fitness_adapter: Optional[FrozenFitnessAdapter] = None,
        mutator: Optional[MoleculeMutator] = None,
    ) -> None:
        config.validate()
        super().__init__(
            config=config,
            fitness_adapter=fitness_adapter or FrozenFitnessAdapter(),
            prior=NullPrior(),
            mutator=mutator or MoleculeMutator(),
        )
        self.config: DASRMCTSConfig = config
        self.ranker_bank = ranker_bank
        self.candidate_selector = CandidateBudgetSelector(ranker_bank, config)
        self._candidate_selection_records: list[dict[str, Any]] = []
        self._oracle_trajectory: list[dict[str, Any]] = []
        self._evaluated_rewards: dict[str, float] = {}

    def _reset_tree_state(self) -> None:
        super()._reset_tree_state()
        self._candidate_selection_records = []
        self._oracle_trajectory = []
        self._evaluated_rewards = {}

    def _prepare_pending_candidates(self, node: MCTSNode) -> None:
        if node.pending_candidates is not None:
            return
        if node.depth >= self.config.max_depth:
            node.pending_candidates = []
            return

        raw_rng = random.Random(
            stable_seed(
                self.config.random_seed,
                "raw_candidate_pool",
                node.smiles,
                node.depth,
            )
        )
        raw_candidates = self.mutator.generate_candidates(
            smiles=node.smiles,
            max_candidates=self.config.raw_candidate_pool_size,
            rng=raw_rng,
        )

        filtered: list[MutationCandidate] = []
        observed: set[str] = set()
        for candidate in raw_candidates:
            if candidate.smiles in self._seen_smiles:
                continue
            if candidate.smiles in observed:
                continue
            observed.add(candidate.smiles)
            filtered.append(candidate)

        selected, audit = self.candidate_selector.select(
            parent_smiles=node.smiles,
            raw_candidates=filtered,
            node_id=node.node_id,
        )
        audit.update(
            {
                "parent_node_id": node.node_id,
                "parent_smiles": node.smiles,
                "parent_depth": node.depth,
                "raw_candidate_pool_size_requested": self.config.raw_candidate_pool_size,
                "oracle_child_budget": self.config.max_children_per_node,
            }
        )
        self._candidate_selection_records.append(audit)
        node.pending_candidates = list(selected)

    def _tree_policy(self, root: MCTSNode, iteration: int) -> MCTSNode:
        node = root
        while True:
            if node.depth >= self.config.max_depth:
                return node

            self._prepare_pending_candidates(node)
            if node.pending_candidates:
                candidate = node.pending_candidates.pop(0)
                if candidate.smiles in self._seen_smiles:
                    continue
                child = self._new_node(
                    smiles=candidate.smiles,
                    parent=node,
                    mutation_action=candidate.action_name,
                )
                node.children.append(child)
                self._seen_smiles.add(child.smiles)
                return child

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

    def _record_oracle_call(self, node: MCTSNode, reward: float) -> None:
        self._evaluated_rewards[node.smiles] = float(reward)
        rewards = np.asarray(list(self._evaluated_rewards.values()), dtype=np.float64)
        top_n = min(10, len(rewards))
        top10_mean = float(np.mean(np.sort(rewards)[-top_n:]))
        scaffolds = {generic_scaffold(smiles) for smiles in self._evaluated_rewards}
        score = node.score_record
        self._oracle_trajectory.append(
            {
                "oracle_call": len(self._oracle_trajectory) + 1,
                "node_id": node.node_id,
                "parent_node_id": node.parent.node_id if node.parent is not None else None,
                "depth": node.depth,
                "smiles": node.smiles,
                "mutation_action": node.mutation_action,
                "final_reward": float(reward),
                "best_reward_so_far": float(np.max(rewards)),
                "top10_mean_reward_so_far": top10_mean,
                "n_unique_evaluated_molecules": len(self._evaluated_rewards),
                "n_unique_generic_scaffolds": len(scaffolds),
                "valid_smiles": bool(score.get("valid_smiles", False)),
                "scoring_status": str(score.get("scoring_status", "unknown")),
                "dual_min_pic50": safe_float(score.get("dual_min_pic50")),
                "qed": safe_float(score.get("qed")),
                "sa_score": safe_float(score.get("sa_score")),
            }
        )

    def _build_dasr_summary(
        self,
        root: MCTSNode,
        tree_policy_steps: int,
        termination_reason: str,
    ) -> dict[str, Any]:
        summary = super()._build_summary(root)
        summary.update(
            {
                "algorithm": "DASR-MCTS fixed pre-oracle expansion budget",
                "candidate_policy": self.config.candidate_policy,
                "raw_candidate_pool_size": self.config.raw_candidate_pool_size,
                "oracle_child_budget": self.config.max_children_per_node,
                "oracle_budget_requested": self.config.oracle_budget,
                "oracle_calls_used": len(self._oracle_trajectory),
                "tree_policy_steps": tree_policy_steps,
                "termination_reason": termination_reason,
                "n_candidate_selection_events": len(
                    self._candidate_selection_records
                ),
                "ranker_metadata": self.ranker_bank.metadata(),
                "isolation_note": (
                    "Legacy P_success is disabled for all Step 27 variants. "
                    "Only candidate-budget allocation changes; final_reward and "
                    "UCT/backpropagation remain frozen."
                ),
            }
        )
        return summary

    def search(self, root_smiles: str) -> DASRMCTSResult:
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

        root_reward = evaluate_node(root, self.fitness, self._best_state)
        backpropagate(root, root_reward)
        self._record_oracle_call(root, root_reward)

        tree_policy_steps = 0
        stagnant_steps = 0
        termination_reason = "oracle_budget_reached"

        while len(self._oracle_trajectory) < self.config.oracle_budget:
            tree_policy_steps += 1
            if tree_policy_steps > self.config.max_tree_policy_steps:
                termination_reason = "max_tree_policy_steps_reached"
                break

            leaf = self._tree_policy(root, tree_policy_steps)
            already_evaluated = bool(leaf.score_record)
            if already_evaluated:
                reward = safe_float(leaf.score_record.get("final_reward"), 0.0)
                if not math.isfinite(reward):
                    reward = 0.0
                backpropagate(leaf, reward)
                stagnant_steps += 1
                if stagnant_steps >= self.config.max_stagnant_steps:
                    termination_reason = "search_space_exhausted_or_stagnant"
                    break
                continue

            reward = evaluate_node(leaf, self.fitness, self._best_state)
            backpropagate(leaf, reward)
            self._record_oracle_call(leaf, reward)
            stagnant_steps = 0

        node_records = self._flatten_node_records(root)
        summary = self._build_dasr_summary(
            root,
            tree_policy_steps,
            termination_reason,
        )
        return DASRMCTSResult(
            root=root,
            best_node=self._best_state.get("best_node") or root,
            node_records=node_records,
            selection_records=copy.deepcopy(self._selection_records),
            summary=summary,
            oracle_trajectory=copy.deepcopy(self._oracle_trajectory),
            candidate_selection_records=copy.deepcopy(
                self._candidate_selection_records
            ),
        )


__all__ = [
    "DASRMCTSConfig",
    "DASRMCTS",
    "DASRMCTSResult",
    "FrozenSiblingRankerBank",
    "FrozenFitnessAdapter",
    "MoleculeMutator",
    "NullPrior",
    "SUPPORTED_POLICIES",
    "generic_scaffold",
]
