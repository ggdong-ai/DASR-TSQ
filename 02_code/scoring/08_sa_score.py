from __future__ import annotations

"""
SA score uses Ertl-Schuffenhauer synthetic accessibility score

SA_score:
    1.0 = easier to synthesize
    10.0 = harder to synthesize

SA_reward:
    1.0 = easier to synthesize
    0.0 = harder to synthesize

Important:
    If myZip.pkl is missing, this module raises an error.
    It does not silently return SA=5.0.
"""

import math
import os
import pickle
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


_FSCORES: dict[int, float] | None = None
_FSCORES_PATH: Path | None = None


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def candidate_fpscore_paths() -> list[Path]:
    """Common locations for myZip.pkl"""
    project_root = get_project_root()
    script_dir = Path(__file__).resolve().parent

    candidates: list[Path] = []

    env_path = os.environ.get("SA_FPSCORES_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            script_dir / "fpscores.pkl" / "myZip.pkl",
            script_dir / "myZip.pkl",
            project_root / "resources" / "sa_score" / "fpscores.pkl" / "myZip.pkl",
            project_root / "resources" / "sa_score" / "myZip.pkl",
            project_root / "fpscores.pkl" / "myZip.pkl",
            project_root / "myZip.pkl",
        ]
    )

    return candidates


def find_fpscores_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve the fpscores file path"""
    if explicit_path is not None:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"指定的 SA fpscores 文件不存在：{path}")

    for path in candidate_fpscore_paths():
        if path.exists():
            return path

    searched = "\n".join(str(p) for p in candidate_fpscore_paths())
    raise FileNotFoundError(
        "找不到 SA-score 所需的 myZip.pkl。\n"
        "推荐放置位置：02_code/scoring/fpscores.pkl/myZip.pkl\n"
        "也可以设置环境变量 SA_FPSCORES_PATH 指向该文件。\n\n"
        f"已搜索路径：\n{searched}"
    )


def read_fragment_scores(fpscores_path: str | Path | None = None) -> dict[int, float]:
    """Load and cache fragment score dictionary"""
    global _FSCORES, _FSCORES_PATH

    resolved_path = find_fpscores_path(fpscores_path)

    if _FSCORES is not None and _FSCORES_PATH == resolved_path:
        return _FSCORES

    with open(resolved_path, "rb") as handle:
        data = pickle.load(handle)

    out_dict: dict[int, float] = {}

    for item in data:
        score = float(item[0])
        for bit_id in item[1:]:
            out_dict[int(bit_id)] = score

    if not out_dict:
        raise RuntimeError(f"SA-score fragment dictionary is empty: {resolved_path}")

    _FSCORES = out_dict
    _FSCORES_PATH = resolved_path
    return out_dict


def num_bridgeheads_and_spiro(mol: Chem.Mol) -> tuple[int, int]:
    """Return bridgehead and spiro atom counts"""
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return n_bridgehead, n_spiro


def calculate_sa_score(
    mol: Chem.Mol,
    fpscores_path: str | Path | None = None,
) -> float:
    """
    Calculate Ertl-Schuffenhauer synthetic accessibility score.
    Returns:
        SA_score in [1, 10].
        1 = easier to synthesize; 10 = harder to synthesize.
    """
    if mol is None:
        raise ValueError("calculate_sa_score received mol=None.")

    fscores = read_fragment_scores(fpscores_path)

    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()

    if not fps:
        raise ValueError("Empty Morgan fingerprint; cannot calculate SA score.")

    score1 = 0.0
    nf = 0

    for bit_id, count in fps.items():
        nf += count
        score1 += fscores.get(int(bit_id), -4.0) * count

    score1 /= nf

    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ring_info = mol.GetRingInfo()
    n_bridgeheads, n_spiro = num_bridgeheads_and_spiro(mol)
    n_macrocycles = sum(1 for ring in ring_info.AtomRings() if len(ring) > 8)

    size_penalty = n_atoms**1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0

    score2 = (
        0.0
        - size_penalty
        - stereo_penalty
        - spiro_penalty
        - bridge_penalty
        - macrocycle_penalty
    )

    score3 = 0.0
    if n_atoms > len(fps):
        score3 = math.log(float(n_atoms) / len(fps)) * 0.5

    raw_score = score1 + score2 + score3

    min_score = -4.0
    max_score = 2.5

    sa_score = 11.0 - (raw_score - min_score + 1.0) / (max_score - min_score) * 9.0

    if sa_score > 8.0:
        sa_score = 8.0 + math.log(sa_score + 1.0 - 9.0)

    if sa_score > 10.0:
        sa_score = 10.0
    elif sa_score < 1.0:
        sa_score = 1.0

    return float(sa_score)


def sa_score_to_reward(sa_score: float) -> float:
    """Convert SA_score [1,10] into SA_reward [1,0]."""
    reward = (10.0 - float(sa_score)) / 9.0
    return float(np.clip(reward, 0.0, 1.0))


def calculate_sa_reward(
    mol: Chem.Mol,
    fpscores_path: str | Path | None = None,
) -> float:
    """Calculate SA_reward directly from RDKit Mol."""
    return sa_score_to_reward(calculate_sa_score(mol, fpscores_path=fpscores_path))


def get_sa_score_metadata(fpscores_path: str | Path | None = None) -> dict[str, str]:
    """Metadata for scoring_config.json."""
    path = find_fpscores_path(fpscores_path)
    return {
        "fpscores_path": str(path),
        "sa_score_range": "1.0 easy to synthesize; 10.0 difficult to synthesize",
        "sa_reward_formula": "SA_reward = clip((10 - SA_score) / 9, 0, 1)",
    }