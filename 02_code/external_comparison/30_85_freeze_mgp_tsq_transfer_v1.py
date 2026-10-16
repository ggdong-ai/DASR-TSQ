from __future__ import annotations

"""
Freeze the post-hoc MGP + TSQ(q=5) transferability protocol.

Run ONCE after:
- 30_82 static preflight PASS
- 30_83 one-update smoke PASS
- exact native/no-quota selection identity == True
- q=5 smoke requires no quota relaxation

Do not re-freeze after a formal seed has started.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CODE_DIR = THIS_FILE.parent

ADAPTER = CODE_DIR / "30_81_mgp_tsq_transfer_adapter_v1.py"
PREFLIGHT_SCRIPT = CODE_DIR / "30_82_preflight_mgp_tsq_transfer_v1.py"
SMOKE_SCRIPT = CODE_DIR / "30_83_smoke_mgp_tsq_transfer_v1.py"
FORMAL_RUNNER = CODE_DIR / "30_86_run_mgp_tsq_transfer_v1.py"

PREFLIGHT_JSON = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_tsq_transfer_preflight"
    / "MGP_TSQ_TRANSFER_PREFLIGHT.json"
)
SMOKE_JSON = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_tsq_transfer_smoke"
    / "MGP_TSQ_ONE_UPDATE_SMOKE.json"
)
BASE_PROTOCOL_JSON = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_protocol"
    / "MGP_V4_PROTOCOL.json"
)
BASE_RUNNER = CODE_DIR / "30_79_run_mgp_nsga2_v4.py"
FREEZE_INFO = (
    PROJECT_ROOT
    / "08_external"
    / "_source_freeze"
    / "MolecularGraphPareto_SOURCE_FREEZE.txt"
)

OUT_DIR = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_mgp_tsq_transfer_protocol"
)
OUT_JSON = OUT_DIR / "MGP_TSQ_TRANSFER_PROTOCOL.json"
OUT_SHA = OUT_DIR / "MGP_TSQ_TRANSFER_PROTOCOL.sha256.txt"

EXPECTED_COMMIT = "826e533b1b3995a8944e7c5cefe087806ff8c03f"
EXPECTED_P0_SHA256 = "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"
SEEDS = [42, 43, 44, 45, 46]

POPULATION_SIZE = 100
BATCH_SIZE = 20
GENERATIONS = 150
TRANSFER_Q = 5
RELATIVE_QUOTA = 0.05
TECHNICAL_REQUEST_CAP = 1_000_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_protocol_hash(payload: dict) -> str:
    tmp = dict(payload)
    tmp.pop("protocol_sha256", None)
    raw = json.dumps(
        tmp,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_freeze(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> None:
    required = [
        ADAPTER,
        PREFLIGHT_SCRIPT,
        SMOKE_SCRIPT,
        FORMAL_RUNNER,
        PREFLIGHT_JSON,
        SMOKE_JSON,
        BASE_PROTOCOL_JSON,
        BASE_RUNNER,
        FREEZE_INFO,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))

    if OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        observed = canonical_protocol_hash(existing)
        expected = existing.get("protocol_sha256")
        if expected == observed:
            print("=" * 92)
            print("MGP+TSQ transfer protocol already exists and self-hash is valid.")
            print(f"Protocol SHA256: {observed}")
            print("Do NOT re-freeze. Use the existing protocol.")
            print("=" * 92)
            return
        raise RuntimeError(
            "Existing transfer protocol has an invalid self-hash. "
            "Do not overwrite it blindly."
        )

    pre = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE_JSON.read_text(encoding="utf-8"))
    base_protocol = json.loads(BASE_PROTOCOL_JSON.read_text(encoding="utf-8"))
    freeze = parse_freeze(FREEZE_INFO)

    if pre.get("passed") is not True:
        raise RuntimeError("Static preflight did not PASS")
    if smoke.get("passed") is not True:
        raise RuntimeError("One-update smoke did not PASS")
    if smoke.get("formal_ready_without_review") is not True:
        raise RuntimeError("Smoke is not formal-ready without review")
    if (
        smoke.get("adapter_noquota_equivalence", {})
        .get("exact_native_order_identity") is not True
    ):
        raise RuntimeError("No-quota adapter does not exactly reproduce native MGP order")
    q5_diag = smoke.get("transfer_q5_one_update", {}).get(
        "selection_diagnostics", {}
    )
    if q5_diag.get("quota_relaxation_used") is not False:
        raise RuntimeError("q=5 smoke required quota relaxation")
    if int(q5_diag.get("final_max_scaffold_occupancy", -1)) > TRANSFER_Q:
        raise RuntimeError("q=5 smoke exceeded the intended scaffold quota")

    if freeze.get("Commit") != EXPECTED_COMMIT:
        raise RuntimeError("Frozen MGP source commit mismatch")
    if smoke.get("baseline_mgp_commit") != EXPECTED_COMMIT:
        raise RuntimeError("Smoke MGP commit mismatch")
    if smoke.get("p0_sha256") != EXPECTED_P0_SHA256:
        raise RuntimeError("Smoke P0 hash mismatch")

    pre_base = pre.get("baseline_mgp", {})
    if pre_base.get("official_commit") != EXPECTED_COMMIT:
        raise RuntimeError("Preflight MGP commit mismatch")
    if pre_base.get("p0_sha256") != EXPECTED_P0_SHA256:
        raise RuntimeError("Preflight P0 hash mismatch")
    if int(pre_base.get("population_size", -1)) != POPULATION_SIZE:
        raise RuntimeError("Preflight population size mismatch")
    if int(pre_base.get("generations", -1)) != GENERATIONS:
        raise RuntimeError("Preflight generation count mismatch")

    current_base_runner_sha = sha256_file(BASE_RUNNER)
    if pre_base.get("formal_runner_sha256") != current_base_runner_sha:
        raise RuntimeError(
            "Baseline MGP runner changed after transfer preflight."
        )

    base_protocol_self_hash = dict(base_protocol).get("protocol_sha256")
    base_without_hash = dict(base_protocol)
    base_without_hash.pop("protocol_sha256", None)
    base_observed_hash = hashlib.sha256(
        json.dumps(
            base_without_hash,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if base_protocol_self_hash != base_observed_hash:
        raise RuntimeError("Baseline MGP protocol self-hash mismatch")
    if pre_base.get("formal_protocol_sha256") != base_observed_hash:
        raise RuntimeError("Preflight baseline protocol hash mismatch")

    payload = {
        "protocol_name": "external_benchmark_v4_mgp_plus_tsq_transfer_q5",
        "analysis_status": "posthoc_transferability_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Does the TSQ scaffold-occupancy mechanism retain its intended "
            "population-diversity control when transferred to a strong, independent "
            "published Pareto optimizer (MolecularGraphPareto NSGA-II)?"
        ),
        "baseline_identity": {
            "method": "MolecularGraphPareto-NSGA-II",
            "official_commit": EXPECTED_COMMIT,
            "p0_sha256": EXPECTED_P0_SHA256,
            "baseline_protocol_sha256": base_observed_hash,
            "baseline_protocol_file_sha256": sha256_file(BASE_PROTOCOL_JSON),
            "baseline_runner_sha256": current_base_runner_sha,
            "baseline_formal_seeds_completed": SEEDS,
        },
        "transfer_intervention": {
            "name": "TSQ scaffold-occupancy quota",
            "changed_component_only": "environmental_selection",
            "max_per_scaffold_q": TRANSFER_Q,
            "population_size": POPULATION_SIZE,
            "relative_occupancy_quota": RELATIVE_QUOTA,
            "rationale": (
                "q=5 for N=100 preserves the same nominal 5% scaffold-occupancy "
                "quota as the formal DASR-TSQ setting q=10 for N=200."
            ),
            "fallback_semantics": (
                "Retain the original TSQ fallback: if the strict quota cannot fill "
                "the target population, fill remaining slots from quota-blocked "
                "candidates in MGP priority order. Relaxation is recorded per generation."
            ),
        },
        "retained_mgp_mechanics": {
            "native_initialization": True,
            "native_arbiter": True,
            "native_mutation": True,
            "native_crossover": True,
            "native_parent_sampling": True,
            "native_pareto_front_calculation": True,
            "native_crowding_distance": True,
            "published_source_quirks_retained": True,
            "same_four_objective_frozen_scorer": True,
        },
        "scientific_config": {
            "seeds": SEEDS,
            "initial_size": 200,
            "native_population_size": POPULATION_SIZE,
            "native_batch_size": BATCH_SIZE,
            "generations": GENERATIONS,
            "arbiter_rules": ["Glaxo"],
            "search_objectives": [
                "clip((pred_gsk3b_pic50-5)/3,0,1)",
                "clip((pred_jnk3_pic50-5)/3,0,1)",
                "qed",
                "sa_reward",
            ],
            "objective_direction": "maximize_all",
        },
        "resource_accounting": {
            "p0_priming_included": True,
            "raw_requests_count_before_cache": True,
            "unique_valid_evaluations_reported": True,
            "cache_hits_reported": True,
            "invalid_requests_reported": True,
            "technical_safety_request_cap": TECHNICAL_REQUEST_CAP,
            "technical_cap_is_scientific_budget": False,
            "stop_rule": "150 completed generations",
        },
        "validation_evidence": {
            "preflight_json_sha256": sha256_file(PREFLIGHT_JSON),
            "smoke_json_sha256": sha256_file(SMOKE_JSON),
            "smoke_exact_native_noquota_order_identity": True,
            "smoke_q5_strict_selected_count": int(
                q5_diag.get("strict_selected_count", -1)
            ),
            "smoke_q5_quota_relaxed_count": int(
                q5_diag.get("quota_relaxed_count", -1)
            ),
            "smoke_q5_unique_scaffolds": int(
                q5_diag.get("final_unique_scaffolds", -1)
            ),
            "smoke_q5_max_scaffold_occupancy": int(
                q5_diag.get("final_max_scaffold_occupancy", -1)
            ),
        },
        "code_sha256": {
            ADAPTER.name: sha256_file(ADAPTER),
            PREFLIGHT_SCRIPT.name: sha256_file(PREFLIGHT_SCRIPT),
            SMOKE_SCRIPT.name: sha256_file(SMOKE_SCRIPT),
            FORMAL_RUNNER.name: sha256_file(FORMAL_RUNNER),
            THIS_FILE.name: sha256_file(THIS_FILE),
        },
        "interpretation_guardrails": [
            "This is a post-hoc transferability analysis motivated by the completed MGP baseline.",
            "Do not describe it as a prespecified primary external benchmark.",
            "Do not tune q after observing formal transfer results.",
            "Primary paired interpretation is MGP versus MGP+TSQ under the same seed and native MGP horizon.",
            "Report quota relaxation if it occurs; do not silently discard such generations.",
        ],
    }
    payload["protocol_sha256"] = canonical_protocol_hash(payload)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUT_SHA.write_text(payload["protocol_sha256"] + "\n", encoding="utf-8")

    print("=" * 92)
    print("MGP + TSQ(q=5) transfer protocol FROZEN")
    print(f"Protocol SHA256 : {payload['protocol_sha256']}")
    print(f"Adapter SHA256  : {payload['code_sha256'][ADAPTER.name]}")
    print(f"Formal runner   : {payload['code_sha256'][FORMAL_RUNNER.name]}")
    print(f"Output          : {OUT_JSON}")
    print("Do not run this freeze script again after formal runs begin.")
    print("=" * 92)


if __name__ == "__main__":
    main()
