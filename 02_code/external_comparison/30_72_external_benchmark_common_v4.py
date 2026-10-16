from __future__ import annotations

"""
Common infrastructure for external benchmark v4.

Purpose
-------
This module intentionally separates the FINAL v4 external-comparison protocol
from the historical Step-28-matched v3.1/v3.2 protocol.

The historical file ``30_0_external_matched_common.py`` is kept immutable
because completed GraphGA/MARS provenance depends on it.  This v4 module
reuses only its already-validated generic chemistry/scoring/accounting helpers.

Scientific v4 policy
--------------------
* No cross-method Step-28 matched raw-request cap is imposed.
* Each method retains its predeclared/native stopping rule.
* Raw oracle requests, unique valid evaluations, cache hits and invalid
  requests are still recorded transparently.
* A large request cap may be supplied only as a NON-BINDING technical safety
  guard.  Reaching that guard invalidates the formal run.
"""

import importlib.util
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
LEGACY_COMMON_PATH = THIS_DIR / "30_0_external_matched_common.py"

PROTOCOL_VERSION = "external_benchmark_v4"
FORMAL_SEEDS = (42, 43, 44, 45, 46)
FORMAL_METHODS = (
    "dasr_tsq",
    "graphga",
    "mars",
    "moleculargraphpareto_nsga2",
    "reinvent4",
)


def _load_legacy():
    name = "_external_matched_common_v31_frozen_for_v4_helpers"
    if name in sys.modules:
        return sys.modules[name]
    if not LEGACY_COMMON_PATH.exists():
        raise FileNotFoundError(LEGACY_COMMON_PATH)
    spec = importlib.util.spec_from_file_location(name, LEGACY_COMMON_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(LEGACY_COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_L = _load_legacy()

# Re-export ONLY generic/frozen project utilities needed by v4.
sha256_file = _L.sha256_file
load_json = _L.load_json
write_json = _L.write_json
write_json_gz = _L.write_json_gz
read_json_gz = _L.read_json_gz
load_module = _L.load_module

canonicalize_smiles = _L.canonicalize_smiles
generic_scaffold = _L.generic_scaffold
detect_smiles_column = _L.detect_smiles_column
load_frozen_p0 = _L.load_frozen_p0
FROZEN_P0_SHA256 = _L.FROZEN_P0_SHA256

FullScore = _L.FullScore
FrozenProjectScorer = _L.FrozenProjectScorer
RequestBudgetReached = _L.RequestBudgetReached

lookup_existing_scores = _L.lookup_existing_scores
normalized_objectives = _L.normalized_objectives
pareto_mask = _L.pareto_mask
hypervolume = _L.hypervolume
population_metrics = _L.population_metrics


class ExternalBenchmarkOracleBroker(_L.RequestBudgetOracleBroker):
    """v4 scorer/cache/accounting broker.

    ``technical_request_cap`` is a non-binding engineering guard, NOT a
    scientific matched budget.  A formal v4 run must fail if this guard is hit.
    """

    def __init__(
        self,
        scorer: FrozenProjectScorer,
        technical_request_cap: int,
        method: str,
    ) -> None:
        super().__init__(
            scorer=scorer,
            request_budget=int(technical_request_cap),
            method=str(method),
        )
        self.technical_request_cap = int(technical_request_cap)

    @property
    def technical_cap_reached(self) -> bool:
        return bool(self.exhausted)


# Backward-compatible v4-facing alias so adapters can use one conventional name.
RequestBudgetOracleBroker = ExternalBenchmarkOracleBroker
