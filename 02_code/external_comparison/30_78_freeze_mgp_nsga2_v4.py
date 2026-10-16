from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CODE_DIR = THIS_FILE.parent

COMMON = CODE_DIR / "30_72_external_benchmark_common_v4.py"
LEGACY_COMMON = CODE_DIR / "30_0_external_matched_common.py"
PREFLIGHT = CODE_DIR / "30_76_preflight_mgp_p0_v4.py"
SMOKE = CODE_DIR / "30_77_smoke_mgp_nsga2_v4.py"
RUNNER = CODE_DIR / "30_79_run_mgp_nsga2_v4.py"

MGP_ROOT = PROJECT_ROOT / "08_external" / "MolecularGraphPareto_official"
FREEZE_INFO = PROJECT_ROOT / "08_external" / "_source_freeze" / "MolecularGraphPareto_SOURCE_FREEZE.txt"
PREFLIGHT_JSON = PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_v4_mgp_preflight" / "MGP_P0_CHEMISTRY_PREFLIGHT.json"
SMOKE_JSON = PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_v4_mgp_runtime_smoke" / "MGP_ONE_UPDATE_RUNTIME_SMOKE.json"

OUT_DIR = PROJECT_ROOT / "06_paper" / "experiment_records" / "external_benchmark_v4_mgp_protocol"
PROTOCOL_JSON = OUT_DIR / "MGP_V4_PROTOCOL.json"
PROTOCOL_SHA = OUT_DIR / "MGP_V4_PROTOCOL_SHA256.txt"

EXPECTED_COMMIT = "826e533b1b3995a8944e7c5cefe087806ff8c03f"
EXPECTED_P0_SHA256 = "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"

SOURCE_REL = [
    "nsga-ii/nsga2.py",
    "nsga-ii/configuration/config.yaml",
    "nsga-ii/nsga2/base.py",
    "nsga-ii/nsga2/infrastructure.py",
    "nsga-ii/nsga2/mechanism.py",
    "nsga-ii/nsga2/operations.py",
    "nsga-ii/data/smarts/mutation_collection.tsv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_freeze(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def canonical_json_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main():
    for p in [COMMON, LEGACY_COMMON, PREFLIGHT, SMOKE, RUNNER, FREEZE_INFO, PREFLIGHT_JSON, SMOKE_JSON]:
        if not p.exists():
            raise FileNotFoundError(p)

    freeze = parse_freeze(FREEZE_INFO)
    if freeze.get("Commit") != EXPECTED_COMMIT:
        raise RuntimeError("MGP source commit mismatch")

    pre = load_json(PREFLIGHT_JSON)
    smoke = load_json(SMOKE_JSON)
    if pre.get("passed") is not True:
        raise RuntimeError("MGP chemistry preflight did not PASS")
    if smoke.get("passed") is not True:
        raise RuntimeError("MGP runtime smoke did not PASS")
    if smoke.get("official_commit") != EXPECTED_COMMIT:
        raise RuntimeError("Runtime smoke commit mismatch")
    if smoke.get("p0_sha256") != EXPECTED_P0_SHA256:
        raise RuntimeError("Runtime smoke P0 hash mismatch")

    source_hashes = {}
    for rel in SOURCE_REL:
        p = MGP_ROOT / rel
        if not p.exists():
            raise FileNotFoundError(p)
        source_hashes[rel] = sha256_file(p)

    payload = {
        "protocol_name": "external_benchmark_v4_mgp_nsga2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_identity": (
            "Published MolecularGraphPareto NSGA-II search mechanics with only "
            "task-specific four-objective scorer replacement and project P0 input adapter."
        ),
        "common_infrastructure_policy": {
            "formal_v4_common": COMMON.name,
            "historical_v31_common_kept_immutable": LEGACY_COMMON.name,
            "cross_method_step28_matched_cap_used_in_v4": False,
            "technical_safety_cap_only": True,
        },
        "official_source": {
            "repository": freeze.get("Repository"),
            "branch": freeze.get("Branch"),
            "commit": freeze.get("Commit"),
            "working_tree_at_freeze": freeze.get("Working tree at freeze"),
            "source_sha256": source_hashes,
            "official_source_edited": False,
        },
        "validated_inputs": {
            "p0_sha256": EXPECTED_P0_SHA256,
            "p0_supplied": 200,
            "native_arbiter_accepted_in_preflight": int(pre["native_arbiter_accepted_count"]),
            "chemistry_preflight_sha256": sha256_file(PREFLIGHT_JSON),
            "runtime_smoke_sha256": sha256_file(SMOKE_JSON),
            "runtime_smoke_passed": True,
        },
        "scientific_config": {
            "seeds": [42, 43, 44, 45, 46],
            "initial_size": 200,
            "native_population_size": 100,
            "native_batch_size": 20,
            "generations": 150,
            "arbiter_rules": ["Glaxo"],
            "search_objectives": [
                "clip((pred_gsk3b_pic50-5)/3,0,1)",
                "clip((pred_jnk3_pic50-5)/3,0,1)",
                "qed",
                "sa_reward",
            ],
            "objective_direction": "maximize_all",
            "native_mutation": True,
            "native_crossover": True,
            "native_pareto_sorting": True,
            "native_crowding_selection": True,
        },
        "published_source_behavior_retained": {
            "batch_branch_quirk_retained": True,
            "crowding_distance_accumulation_quirk_retained": True,
            "no_posthoc_algorithm_fix": True,
        },
        "resource_accounting": {
            "p0_priming_included": True,
            "raw_requests_count_before_cache": True,
            "unique_valid_evaluations_reported": True,
            "cache_hits_reported": True,
            "technical_safety_request_cap": 1000000,
            "technical_cap_is_scientific_budget": False,
            "native_stop": "150 completed generations",
        },
        "runtime_compatibility": {
            "search_core_from_official_source": True,
            "import_or_reporting_only_stubs_allowed": [
                "hydra",
                "dask",
                "pymoo",
                "pygmo",
                "omegaconf",
                "indices.faith",
            ],
            "stubs_must_not_replace_search_operators": True,
        },
        "project_code_sha256": {
            COMMON.name: sha256_file(COMMON),
            LEGACY_COMMON.name: sha256_file(LEGACY_COMMON),
            PREFLIGHT.name: sha256_file(PREFLIGHT),
            SMOKE.name: sha256_file(SMOKE),
            RUNNER.name: sha256_file(RUNNER),
        },
    }
    payload["protocol_sha256"] = canonical_json_hash(payload)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    PROTOCOL_SHA.write_text(payload["protocol_sha256"] + "\n", encoding="utf-8")

    print("=" * 90)
    print("MGP v4 FORMAL PROTOCOL FROZEN")
    print("commit          :", EXPECTED_COMMIT)
    print("P0 SHA256       :", EXPECTED_P0_SHA256)
    print("population      : 100")
    print("batch size      : 20")
    print("generations     : 150")
    print("seeds           : 42,43,44,45,46")
    print("protocol SHA256 :", payload["protocol_sha256"])
    print("output          :", PROTOCOL_JSON)
    print("=" * 90)


if __name__ == "__main__":
    main()
