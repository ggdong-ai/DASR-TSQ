from __future__ import annotations

"""Prepare the frozen P0 SMILES input used by REINVENT4 Mol2Mol.

This is a protocol-preparation utility, not an optimizer.
It does not alter the frozen formal_p0.csv.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\pycharm\project\tsq_mcts_rebuild").resolve()
COMMON_V4 = PROJECT_ROOT / "02_code" / "external_comparison" / "30_72_external_benchmark_common_v4.py"
OUT_ROOT = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_reinvent4_preflight"
)
P0_SMI = OUT_ROOT / "reinvent4_formal_p0.smi"
MANIFEST = OUT_ROOT / "reinvent4_formal_p0_manifest.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    common = load_module("_reinvent4_prepare_common_v4", COMMON_V4)
    p0_path, raw, canonical = common.load_frozen_p0()

    if len(raw) != 200 or len(canonical) != 200 or len(set(canonical)) != 200:
        raise RuntimeError("Frozen P0 identity check failed.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    P0_SMI.write_text("\n".join(canonical) + "\n", encoding="utf-8")

    manifest = {
        "purpose": "REINVENT4 Mol2Mol conditioning input derived from frozen project P0",
        "source_p0_path": str(Path(p0_path).resolve()),
        "source_p0_sha256": common.FROZEN_P0_SHA256,
        "n_source_rows": len(raw),
        "n_output_rows": len(canonical),
        "n_unique_canonical": len(set(canonical)),
        "representation": "canonical SMILES; one compound per line",
        "output_path": str(P0_SMI.resolve()),
        "output_sha256": sha256_file(P0_SMI),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("REINVENT4 P0 PREPARED")
    print(f"source P0 SHA256 : {manifest['source_p0_sha256']}")
    print(f"rows             : {manifest['n_output_rows']}")
    print(f"unique canonical : {manifest['n_unique_canonical']}")
    print(f"P0 .smi          : {P0_SMI}")
    print(f"P0 .smi SHA256   : {manifest['output_sha256']}")
    print(f"manifest         : {MANIFEST}")


if __name__ == "__main__":
    main()
