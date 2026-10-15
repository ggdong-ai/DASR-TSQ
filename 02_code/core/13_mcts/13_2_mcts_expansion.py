"""
This module generates local molecular graph edits and expands one new child node.
本模块生成局部分子图编辑候选，并扩展一个新的子节点。

This is not a reaction planner.
本模块不是反应规划器。
"""

from __future__ import annotations
import importlib.util
import random
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors


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

MAX_HEAVY_ATOMS = _common.MAX_HEAVY_ATOMS
MAX_MOLECULAR_WEIGHT = _common.MAX_MOLECULAR_WEIGHT
MutationCandidate = _common.MutationCandidate
MCTSNode = _common.MCTSNode
canonicalize_smiles = _common.canonicalize_smiles


class MoleculeMutator:
    """
    Generate sanitized local molecular mutation candidates.
    生成经过 sanitize 的局部分子突变候选。
    """

    # EN: Simple one-atom attachments.
    # CN: 简单的单原子连接操作。
    SIMPLE_ATTACHMENTS: tuple[tuple[str, str], ...] = (
        ("attach_methyl", "C"),
        ("attach_fluoro", "F"),
        ("attach_chloro", "Cl"),
        ("attach_hydroxy_oxygen", "O"),
        ("attach_amino_nitrogen", "N"),
    )

    # EN: The first atom of each fragment is used as the attachment point.
    # CN: 每个片段的第 0 个原子会作为连接点。
    AROMATIC_FRAGMENTS: tuple[tuple[str, str], ...] = (
        ("attach_phenyl", "c1ccccc1"),
        ("attach_pyridyl", "c1ccncc1"),
    )

    def __init__(
        self,
        max_molecular_weight: float = MAX_MOLECULAR_WEIGHT,
        max_heavy_atoms: int = MAX_HEAVY_ATOMS,
    ) -> None:
        self.max_molecular_weight = max_molecular_weight
        self.max_heavy_atoms = max_heavy_atoms

    def _is_candidate_valid(self, mol: Chem.Mol, parent_smiles: str) -> str | None:
        """
        Sanitize and filter a candidate molecule.
        对候选分子进行 sanitize 和过滤。

        Return canonical SMILES if valid; otherwise return None.
        若候选有效，返回 canonical SMILES；否则返回 None。
        """
        if mol is None:
            return None

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        if Descriptors.MolWt(mol) > self.max_molecular_weight:
            return None

        if mol.GetNumHeavyAtoms() > self.max_heavy_atoms:
            return None

        candidate = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        if candidate == parent_smiles:
            return None

        return candidate

    def _attach_atom(
        self,
        parent_mol: Chem.Mol,
        atom_idx: int,
        atomic_symbol: str,
        parent_smiles: str,
    ) -> str | None:
        """
        Attach one atom to the selected atom of the parent molecule.
        在父分子的指定原子上连接一个原子。
        """
        rw_mol = Chem.RWMol(parent_mol)

        try:
            new_idx = rw_mol.AddAtom(Chem.Atom(atomic_symbol))
            rw_mol.AddBond(atom_idx, new_idx, Chem.BondType.SINGLE)
            candidate_mol = rw_mol.GetMol()
        except Exception:
            return None

        return self._is_candidate_valid(candidate_mol, parent_smiles)

    def _attach_fragment(
        self,
        parent_mol: Chem.Mol,
        atom_idx: int,
        fragment_smiles: str,
        parent_smiles: str,
    ) -> str | None:
        """
        Attach an aromatic fragment to the selected atom of the parent molecule.
        在父分子的指定原子上连接一个芳香片段。

        The first atom of the fragment is used as the attachment point.
        片段的第 0 个原子作为连接点。
        """
        fragment = Chem.MolFromSmiles(fragment_smiles)

        if fragment is None:
            return None

        try:
            combined = Chem.CombineMols(parent_mol, fragment)
            rw_mol = Chem.RWMol(combined)

            # EN: Index of the first atom in the appended fragment.
            # CN: 新追加片段中第一个原子的索引。
            fragment_attach_idx = parent_mol.GetNumAtoms()

            rw_mol.AddBond(atom_idx, fragment_attach_idx, Chem.BondType.SINGLE)
            candidate_mol = rw_mol.GetMol()
        except Exception:
            return None

        return self._is_candidate_valid(candidate_mol, parent_smiles)

    def _change_bond_order(
        self,
        parent_mol: Chem.Mol,
        begin_idx: int,
        end_idx: int,
        new_bond_type: Chem.BondType,
        parent_smiles: str,
    ) -> str | None:
        """
        Try to change the bond order of a non-aromatic bond.
        尝试改变一个非芳香键的键级。
        """
        rw_mol = Chem.RWMol(parent_mol)

        try:
            rw_mol.RemoveBond(begin_idx, end_idx)
            rw_mol.AddBond(begin_idx, end_idx, new_bond_type)
            candidate_mol = rw_mol.GetMol()
        except Exception:
            return None

        return self._is_candidate_valid(candidate_mol, parent_smiles)

    def _replace_terminal_halogen(
        self,
        parent_mol: Chem.Mol,
        atom_idx: int,
        new_symbol: str,
        parent_smiles: str,
    ) -> str | None:
        """
        Replace a halogen atom with another halogen.
        将一个卤素原子替换为另一个卤素原子。
        """
        atom = parent_mol.GetAtomWithIdx(atom_idx)

        if atom.GetSymbol() not in {"F", "Cl", "Br", "I"}:
            return None

        rw_mol = Chem.RWMol(parent_mol)

        try:
            rw_mol.ReplaceAtom(atom_idx, Chem.Atom(new_symbol))
            candidate_mol = rw_mol.GetMol()
        except Exception:
            return None

        return self._is_candidate_valid(candidate_mol, parent_smiles)

    def generate_candidates(
        self,
        smiles: str,
        max_candidates: int,
        rng: random.Random,
    ) -> list[MutationCandidate]:
        """
        Generate unique local mutation candidates for a parent molecule.
        为一个父分子生成唯一的局部突变候选。
        """
        parent_smiles = canonicalize_smiles(smiles)

        if parent_smiles is None:
            return []

        parent_mol = Chem.MolFromSmiles(parent_smiles)

        if parent_mol is None:
            return []

        candidate_to_action: dict[str, str] = {}

        # EN: Try atom and fragment attachments on atoms with available valence.
        # CN: 在具有可用价态的原子上尝试原子连接和片段连接。
        for atom in parent_mol.GetAtoms():
            atom_idx = atom.GetIdx()

            if atom.GetImplicitValence() > 0 or atom.GetNumExplicitHs() > 0:
                for action_name, atomic_symbol in self.SIMPLE_ATTACHMENTS:
                    candidate = self._attach_atom(
                        parent_mol,
                        atom_idx,
                        atomic_symbol,
                        parent_smiles,
                    )
                    if candidate is not None:
                        candidate_to_action.setdefault(candidate, action_name)

                for action_name, fragment_smiles in self.AROMATIC_FRAGMENTS:
                    candidate = self._attach_fragment(
                        parent_mol,
                        atom_idx,
                        fragment_smiles,
                        parent_smiles,
                    )
                    if candidate is not None:
                        candidate_to_action.setdefault(candidate, action_name)

        # EN: Try limited bond-order changes for non-aromatic single/double bonds.
        # CN: 对非芳香单键/双键尝试有限的键级变化。
        for bond in parent_mol.GetBonds():
            if bond.GetIsAromatic():
                continue

            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            if bond.GetBondType() == Chem.BondType.SINGLE:
                candidate = self._change_bond_order(
                    parent_mol,
                    begin_idx,
                    end_idx,
                    Chem.BondType.DOUBLE,
                    parent_smiles,
                )
                if candidate is not None:
                    candidate_to_action.setdefault(candidate, "single_to_double_bond")

            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                candidate = self._change_bond_order(
                    parent_mol,
                    begin_idx,
                    end_idx,
                    Chem.BondType.SINGLE,
                    parent_smiles,
                )
                if candidate is not None:
                    candidate_to_action.setdefault(candidate, "double_to_single_bond")

        # EN: Try conservative halogen substitutions.
        # CN: 尝试保守的卤素替换。
        for atom in parent_mol.GetAtoms():
            if atom.GetSymbol() in {"F", "Cl", "Br", "I"}:
                for replacement in ("F", "Cl", "Br"):
                    candidate = self._replace_terminal_halogen(
                        parent_mol,
                        atom.GetIdx(),
                        replacement,
                        parent_smiles,
                    )
                    if candidate is not None:
                        candidate_to_action.setdefault(
                            candidate,
                            f"halogen_to_{replacement}",
                        )

        candidates = [
            MutationCandidate(smiles=smi, action_name=action)
            for smi, action in candidate_to_action.items()
        ]

        # EN: Sort first for deterministic behavior, then shuffle with a controlled RNG.
        # CN: 先排序保证确定性，再用受控随机数打乱。
        candidates.sort(key=lambda item: item.smiles)
        rng.shuffle(candidates)

        return candidates[:max_candidates]


def ensure_pending_candidates(
    node: MCTSNode,
    config,
    mutator: MoleculeMutator,
    mutation_rng: random.Random,
    seen_smiles: set[str],
) -> None:
    """
    Generate and freeze the candidate pool of a node on first use.
    在节点第一次使用时生成并冻结该节点的候选池。
    """
    if node.pending_candidates is not None:
        return

    if node.depth >= config.max_depth:
        node.pending_candidates = []
        return

    raw_candidates = mutator.generate_candidates(
        smiles=node.smiles,
        max_candidates=config.max_children_per_node * 4,
        rng=mutation_rng,
    )

    candidates: list[MutationCandidate] = []

    for candidate in raw_candidates:
        if candidate.smiles in seen_smiles:
            continue

        candidates.append(candidate)

        if len(candidates) >= config.max_children_per_node:
            break

    node.pending_candidates = candidates


def expand_one_child(
    node: MCTSNode,
    new_node_fn,
    config,
    mutator: MoleculeMutator,
    mutation_rng: random.Random,
    expansion_rng: random.Random,
    seen_smiles: set[str],
) -> MCTSNode | None:
    """
    Expand one new child from the node's pending candidate pool.
    从当前节点的候选池中扩展一个新的子节点。
    """
    ensure_pending_candidates(
        node=node,
        config=config,
        mutator=mutator,
        mutation_rng=mutation_rng,
        seen_smiles=seen_smiles,
    )

    if not node.pending_candidates:
        return None

    index = expansion_rng.randrange(len(node.pending_candidates))
    candidate = node.pending_candidates.pop(index)

    if candidate.smiles in seen_smiles:
        return None

    child = new_node_fn(
        smiles=candidate.smiles,
        parent=node,
        mutation_action=candidate.action_name,
    )

    node.children.append(child)
    seen_smiles.add(child.smiles)

    return child