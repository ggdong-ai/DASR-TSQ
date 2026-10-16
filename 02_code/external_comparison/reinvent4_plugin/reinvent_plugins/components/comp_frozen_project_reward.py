"""REINVENT4 scoring plugin for the frozen DASR-TSQ project reward.

Scientific role
---------------
Expose the project's already-frozen scalar `final_reward` to REINVENT4 without
redefining the task inside REINVENT.  The plugin also reuses the external
benchmark v4 broker so raw requests, cache hits, invalid requests, and unique
valid evaluations follow the same accounting semantics as the other external
baselines.

This is an external namespace plugin.  It must NOT be copied over an official
REINVENT4 source file and it does not alter the frozen REINVENT4 commit.
"""
from __future__ import annotations

__all__ = ["FrozenProjectReward"]

import importlib.util
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    # REINVENT collects endpoint params into lists, even for one endpoint.
    project_root: List[str]
    technical_request_cap: List[int]
    state_json: List[str]


def _load_module(name: str, path: Path):
    path = Path(path).resolve()
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".__tmp__")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
    tmp.replace(path)


@add_tag("__component")
class FrozenProjectReward:
    """Return the project's frozen final_reward as one REINVENT score endpoint."""

    def __init__(self, params: Parameters):
        lengths = {
            len(params.project_root),
            len(params.technical_request_cap),
            len(params.state_json),
        }
        if lengths != {1}:
            raise ValueError(
                "FrozenProjectReward supports exactly one endpoint; "
                "project_root, technical_request_cap and state_json must each have one value."
            )

        self.number_of_endpoints = 1
        self.smiles_type = "rdkit_smiles"

        self.project_root = Path(params.project_root[0]).resolve()
        self.state_json = Path(params.state_json[0]).resolve()
        self.archive_jsonl = self.state_json.with_name("oracle_archive.jsonl")
        self.batch_jsonl = self.state_json.with_name("oracle_batches.jsonl")

        common_path = (
            self.project_root
            / "02_code"
            / "external_comparison"
            / "30_72_external_benchmark_common_v4.py"
        )
        if not common_path.exists():
            raise FileNotFoundError(common_path)

        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        self.common = _load_module(
            "_reinvent4_external_benchmark_common_v4",
            common_path,
        )

        self.scorer = self.common.FrozenProjectScorer()
        self.broker = self.common.ExternalBenchmarkOracleBroker(
            scorer=self.scorer,
            technical_request_cap=int(params.technical_request_cap[0]),
            method="reinvent4_mol2mol",
        )

        # v4 accounting policy: prime the common frozen P0 once before
        # method-generated candidates are scored.  This primes only the shared
        # scorer/cache/accounting; it does not train or update the REINVENT agent.
        p0_path, p0_raw, p0_canonical = self.common.load_frozen_p0()
        self.broker.prime_p0(p0_raw)

        self.p0_path = Path(p0_path).resolve()
        self.p0_canonical = list(p0_canonical)
        self._archive_written = 0
        self._batch_written = 0

        # Fresh run: never append to stale smoke/formal accounting artifacts.
        for path in (self.archive_jsonl, self.batch_jsonl):
            if path.exists():
                path.unlink()

        self._flush_new_records()  # persist the 200 primed P0 records
        self._write_state()

    def _append_jsonl(self, path: Path, rows) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _flush_new_records(self) -> None:
        new_archive = self.broker.archive[self._archive_written :]
        self._append_jsonl(self.archive_jsonl, new_archive)
        self._archive_written = len(self.broker.archive)

        new_batches = self.broker.batch_log[self._batch_written :]
        self._append_jsonl(self.batch_jsonl, new_batches)
        self._batch_written = len(self.broker.batch_log)

    def _write_state(self) -> None:
        payload = {
            "adapter": "FrozenProjectReward",
            "method": "reinvent4_mol2mol",
            "project_root": str(self.project_root),
            "p0_path": str(self.p0_path),
            "p0_sha256": self.common.sha256_file(self.p0_path),
            "p0_size": len(self.p0_canonical),
            "p0_primed": True,
            "counters": self.broker.counters(),
            "n_archive_rows": len(self.broker.archive),
            "n_scoring_batches": len(self.broker.batch_log),
            "technical_cap_reached": bool(self.broker.technical_cap_reached),
            "archive_jsonl": str(self.archive_jsonl),
            "batch_jsonl": str(self.batch_jsonl),
        }
        _atomic_write_json(self.state_json, payload)

    def __call__(self, smilies: List[str]) -> ComponentResults:
        values = [str(value) for value in smilies]
        records = self.broker.score_batch_atomic(values)

        scores = np.asarray(
            [float(record["final_reward"]) for record in records],
            dtype=float,
        )

        self._flush_new_records()
        self._write_state()

        return ComponentResults([scores])
