from __future__ import annotations

"""
Static preflight for the post-hoc MGP + TSQ transferability analysis.

No optimization is executed and no new performance result is produced.

Checks:
- existing 5-seed formal MGP baseline is complete and internally protocol-consistent;
- frozen official MGP source commit/P0 identity remain unchanged;
- the published MGP selection/crowding source structure is still present;
- q=5 for MGP population=100 preserves the same 5% relative occupancy quota
  as formal DASR-TSQ q=10 for population=200;
- the transfer adapter and baseline runner are frozen by SHA256 for the smoke.
"""

import ast
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CODE_DIR = THIS_FILE.parent

BASE_RUNNER = CODE_DIR / "30_79_run_mgp_nsga2_v4.py"
ADAPTER = CODE_DIR / "30_81_mgp_tsq_transfer_adapter_v1.py"

MGP_ROOT = PROJECT_ROOT / "08_external" / "MolecularGraphPareto_official"
MGP_NSGA = MGP_ROOT / "nsga-ii" / "nsga2.py"
FREEZE_INFO = (
    PROJECT_ROOT
    / "08_external"
    / "_source_freeze"
    / "MolecularGraphPareto_SOURCE_FREEZE.txt"
)
BASE_PROTOCOL = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_protocol"
    / "MGP_V4_PROTOCOL.json"
)
BASE_FORMAL_ROOT = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_formal"
)

OUT_DIR = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_tsq_transfer_preflight"
)
OUT_JSON = OUT_DIR / "MGP_TSQ_TRANSFER_PREFLIGHT.json"

EXPECTED_COMMIT = "826e533b1b3995a8944e7c5cefe087806ff8c03f"
EXPECTED_P0_SHA256 = "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"
SEEDS = (42, 43, 44, 45, 46)

MGP_POPULATION = 100
TRANSFER_Q = 5
DASR_FORMAL_POPULATION = 200
DASR_FORMAL_Q = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_freeze(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def canonical_json_hash(payload: dict) -> str:
    tmp = dict(payload)
    expected = tmp.pop("protocol_sha256", None)
    raw = json.dumps(
        tmp,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    observed = hashlib.sha256(raw).hexdigest()
    if expected is not None and observed != expected:
        raise RuntimeError(
            f"Baseline MGP protocol self-hash mismatch: {observed} != {expected}"
        )
    return observed


def static_mgp_selection_checks() -> dict[str, bool]:
    text = MGP_NSGA.read_text(encoding="utf-8")
    ast.parse(text)

    checks = {
        "selection_uses_official_fronts":
            "fronts = self.calculate_fronts(molecules)" in text,
        "selection_uses_official_extrema":
            "minimal_values, maximal_values = self.extrema(molecules)" in text,
        "selection_uses_official_crowding":
            "self.crowding_distance(fronts, minimal_values, maximal_values)" in text,
        "partial_front_sorted_by_descending_crowding":
            "front.sort(key=lambda x: x.crowding_distance, reverse=True)" in text,
        "dominance_is_maximize_all":
            "np.all(fitnesses_a >= fitnesses_b)" in text
            and "np.any(fitnesses_a > fitnesses_b)" in text,
        "published_crowding_accumulation_present":
            "pair[1].crowding_distance +=" in text,
        "published_batch_branch_quirk_present":
            'if self.task == "mpo_dap_kinases" or "mpo_anipsychotics":' in text,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Unexpected frozen MGP selection source: {checks}")
    return checks


def main() -> None:
    required = [BASE_RUNNER, ADAPTER, MGP_NSGA, FREEZE_INFO, BASE_PROTOCOL]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))

    freeze = parse_freeze(FREEZE_INFO)
    if freeze.get("Commit") != EXPECTED_COMMIT:
        raise RuntimeError(
            f"MGP frozen commit mismatch: {freeze.get('Commit')} != {EXPECTED_COMMIT}"
        )

    baseline_protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    baseline_protocol_hash = canonical_json_hash(baseline_protocol)

    recorded_runner_hash = (
        baseline_protocol
        .get("project_code_sha256", {})
        .get(BASE_RUNNER.name)
    )
    current_runner_hash = sha256_file(BASE_RUNNER)
    if recorded_runner_hash and recorded_runner_hash != current_runner_hash:
        raise RuntimeError(
            "Current MGP formal runner differs from the frozen baseline protocol:\n"
            f"recorded={recorded_runner_hash}\ncurrent={current_runner_hash}"
        )

    markers = {}
    marker_protocol_hashes = set()
    for seed in SEEDS:
        marker_path = BASE_FORMAL_ROOT / f"seed_{seed}" / "TASK_COMPLETED.json"
        if not marker_path.exists():
            raise FileNotFoundError(marker_path)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("completed") is not True:
            raise RuntimeError(f"MGP seed {seed} is not completed")
        if marker.get("method") != "MolecularGraphPareto-NSGA-II":
            raise RuntimeError(f"Unexpected method in MGP seed {seed}")
        if int(marker.get("native_generations_completed", -1)) != 150:
            raise RuntimeError(f"MGP seed {seed} is not a complete 150-generation run")
        if marker.get("official_commit") != EXPECTED_COMMIT:
            raise RuntimeError(f"MGP seed {seed} official commit mismatch")
        if marker.get("p0_sha256") != EXPECTED_P0_SHA256:
            raise RuntimeError(f"MGP seed {seed} P0 mismatch")
        if marker.get("technical_safety_cap_nonbinding") is not True:
            raise RuntimeError(f"MGP seed {seed} hit technical safety cap")
        marker_protocol_hashes.add(str(marker.get("protocol_sha256")))
        markers[str(seed)] = {
            "protocol_sha256": marker.get("protocol_sha256"),
            "requests": marker.get("oracle_counters", {}).get("requests"),
            "unique_valid_evaluations": marker.get(
                "oracle_counters", {}
            ).get("unique_valid_evaluations"),
        }

    if len(marker_protocol_hashes) != 1:
        raise RuntimeError(
            f"Formal MGP seeds do not share one protocol hash: {marker_protocol_hashes}"
        )

    # The formal baseline runner itself computes canonical_json_hash(PROTOCOL_JSON).
    # The marker hash therefore must match the currently available local baseline
    # protocol before this paired transfer experiment proceeds.
    marker_protocol_hash = next(iter(marker_protocol_hashes))
    if marker_protocol_hash != baseline_protocol_hash:
        raise RuntimeError(
            "Local MGP protocol JSON does not match the protocol used by completed "
            "formal seeds. Do NOT overwrite or re-freeze anything.\n"
            f"marker={marker_protocol_hash}\nlocal={baseline_protocol_hash}"
        )

    mgp_ratio = TRANSFER_Q / MGP_POPULATION
    dasr_ratio = DASR_FORMAL_Q / DASR_FORMAL_POPULATION
    if abs(mgp_ratio - dasr_ratio) > 1e-15:
        raise RuntimeError("Relative TSQ quota mismatch")

    checks = static_mgp_selection_checks()
    adapter_mod = load_module("mgp_tsq_transfer_adapter_preflight", ADAPTER)
    if not hasattr(adapter_mod, "mgp_tsq_environmental_selection"):
        raise RuntimeError("Transfer adapter entry point missing")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "diagnostic_only": True,
        "optimization_executed": False,
        "formal_performance_result": False,
        "analysis_identity": "posthoc_mgp_plus_tsq_transferability",
        "baseline_mgp": {
            "official_commit": EXPECTED_COMMIT,
            "p0_sha256": EXPECTED_P0_SHA256,
            "population_size": MGP_POPULATION,
            "generations": 150,
            "seeds": list(SEEDS),
            "formal_seed_markers": markers,
            "formal_protocol_sha256": marker_protocol_hash,
            "formal_runner_sha256": current_runner_hash,
        },
        "transfer_intervention": {
            "changed_component_only": "environmental_selection_scaffold_quota",
            "mgp_native_front_calculation_retained": True,
            "mgp_native_crowding_distance_retained": True,
            "mgp_native_mutation_retained": True,
            "mgp_native_crossover_retained": True,
            "mgp_native_arbiter_retained": True,
            "mgp_native_objectives_retained": True,
            "max_per_scaffold": TRANSFER_Q,
            "mgp_population_size": MGP_POPULATION,
            "relative_quota": mgp_ratio,
            "formal_dasr_reference_q": DASR_FORMAL_Q,
            "formal_dasr_reference_population": DASR_FORMAL_POPULATION,
            "formal_dasr_relative_quota": dasr_ratio,
            "quota_fallback_retained": (
                "If the strict quota cannot fill the population, remaining slots "
                "are filled from quota-blocked molecules in MGP priority order."
            ),
        },
        "static_mgp_source_checks": checks,
        "adapter_sha256": sha256_file(ADAPTER),
        "preflight_script_sha256": sha256_file(THIS_FILE),
        "next_step": (
            "Run one-update seed42 smoke. Require exact native-selection identity "
            "when max_per_scaffold=100 before any formal MGP+TSQ transfer run."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT_JSON)

    print("=" * 92)
    print("MGP + TSQ transfer static preflight: PASS")
    print(f"Baseline protocol SHA : {marker_protocol_hash}")
    print(f"Baseline runner SHA   : {current_runner_hash}")
    print(f"Adapter SHA           : {payload['adapter_sha256']}")
    print(f"Relative quota        : {TRANSFER_Q}/{MGP_POPULATION} = {mgp_ratio:.3f}")
    print(f"Output                : {OUT_JSON}")
    print("=" * 92)


if __name__ == "__main__":
    main()
