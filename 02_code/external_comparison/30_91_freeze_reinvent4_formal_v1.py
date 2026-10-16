from __future__ import annotations

"""Freeze REINVENT4 Mol2Mol formal benchmark protocol v1.

Run exactly once before the first formal REINVENT4 seed.

This script does NOT run optimization.  It verifies the already completed
preflight/smoke provenance and writes an immutable formal protocol record.
If an identical freeze already exists, it verifies and exits.  It refuses to
overwrite a non-identical formal protocol.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(r"D:\pycharm\project\tsq_mcts_rebuild").resolve()
CODE_DIR = PROJECT_ROOT / "02_code" / "external_comparison"
REINVENT_ROOT = PROJECT_ROOT / "08_external" / "REINVENT4_official"

PREFLIGHT_ROOT = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_reinvent4_preflight"
)
FORMAL_PROTOCOL_ROOT = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_reinvent4_protocol"
)

P0_MANIFEST = PREFLIGHT_ROOT / "reinvent4_formal_p0_manifest.json"
SCORER_ENV_REPORT = PREFLIGHT_ROOT / "scorer_cross_environment_report.json"
P0_SMI = PREFLIGHT_ROOT / "reinvent4_formal_p0.smi"

PRIOR = REINVENT_ROOT / "priors" / "mol2mol_medium_similarity.prior"
PLUGIN = (
    CODE_DIR / "reinvent4_plugin" / "reinvent_plugins"
    / "components" / "comp_frozen_project_reward.py"
)
FORMAL_SEED42_TOML = CODE_DIR / "reinvent4_formal_seed42_v1.toml"

BASE_MATCHED_PROTOCOL = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_matched_step28_v3_1"
    / "external_matched_protocol.json"
)

PROTOCOL_JSON = FORMAL_PROTOCOL_ROOT / "REINVENT4_FORMAL_PROTOCOL.json"
FREEZE_MARKER = FORMAL_PROTOCOL_ROOT / "REINVENT4_FORMAL_PROTOCOL_FROZEN.json"

EXPECTED_AUDITED_COMMIT = "ee0d56f4a07472bbb622cd0858184d06f11bff5d"
EXPECTED_P0_SOURCE_SHA256 = "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc"
EXPECTED_P0_SMI_SHA256 = "b0fcc657ebd306ac326699b0599bab57f5a91825396ffa3d8c6ca3b1ebb71fcd"
EXPECTED_PRIOR_MD5 = "78b75e478bfd7dcccee8ece010bc0b07"
EXPECTED_PRIOR_SHA256 = "a0f40a2fb4baedaf51ae199bb38a61a8c8d26f03062977eaf06e34079e9f22ca"
EXPECTED_PLUGIN_SHA256 = "c1a9b112b943ca955d0312491ac476c1bc4aaeba6f386d2a3643caa280c12a7d"

EXPECTED_CRITICAL_SOURCE_SHA256 = {
    "reinvent/runmodes/samplers/mol2mol.py":
        "cb8aed2a9253861ed8cc9170a0ef5c810bb198758a8f530d783cc7e45979fcdd",
    "reinvent/runmodes/RL/learning.py":
        "9c853cf8b2ebdba16e66ceadb2e5dcbc5138703dd852ffdd68e0a0738eac5d3b",
    "reinvent/models/model_factory/sample_batch.py":
        "5fd5aefeae474320d1a4717a4594969f793813874172149170cdc7bce68caea0",
    "reinvent/runmodes/RL/distance_penalty.py":
        "0bfc3b25e3456cec923cdbed66d1e012d6ef93d92c53c366efb374f58a56ca2e",
    "reinvent/runmodes/RL/run_staged_learning.py":
        "ccfe97a7229302a783acf07f853eaf28b690bd5ea64cad07d851b2f666995dd5",
    "reinvent/utils/prior_registry.py":
        "2b4bab5b03429b53c6eb83014ed1787230e698c24679897d9e932889d001fd0c",
}

FORMAL_SEEDS = [42, 43, 44, 45, 46]
FORMAL_MAX_STEPS = 100


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(payload: dict) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    required = [
        P0_MANIFEST, SCORER_ENV_REPORT, P0_SMI, PRIOR, PLUGIN,
        FORMAL_SEED42_TOML, BASE_MATCHED_PROTOCOL,
    ]
    for path in required:
        require(path.exists(), f"Missing required file: {path}")

    p0_manifest = load_json(P0_MANIFEST)
    require(
        p0_manifest.get("source_p0_sha256") == EXPECTED_P0_SOURCE_SHA256,
        "Frozen P0 source SHA mismatch",
    )
    require(
        p0_manifest.get("output_sha256") == EXPECTED_P0_SMI_SHA256,
        "Frozen REINVENT4 P0 .smi SHA mismatch",
    )
    require(
        int(p0_manifest.get("n_output_rows", -1)) == 200
        and int(p0_manifest.get("n_unique_canonical", -1)) == 200,
        "Frozen REINVENT4 P0 must be 200/200 unique canonical molecules",
    )
    require(digest(P0_SMI) == EXPECTED_P0_SMI_SHA256, "P0 .smi bytes changed")

    scorer_report = load_json(SCORER_ENV_REPORT)
    require(scorer_report.get("passed") is True, "Cross-environment scorer report is not PASS")
    require(int(scorer_report.get("n_molecules", -1)) == 200, "Scorer report did not test 200 molecules")
    require(int(scorer_report.get("n_exact_mismatches", -1)) == 0, "Scorer exact mismatch")
    require(int(scorer_report.get("n_numeric_mismatches", -1)) == 0, "Scorer numeric mismatch")
    require(float(scorer_report.get("atol", 999)) == 1e-10, "Unexpected scorer tolerance")

    prior_md5 = digest(PRIOR, "md5")
    prior_sha256 = digest(PRIOR, "sha256")
    require(prior_md5 == EXPECTED_PRIOR_MD5, f"Official Zenodo prior MD5 mismatch: {prior_md5}")
    require(prior_sha256 == EXPECTED_PRIOR_SHA256, f"Prior SHA256 mismatch: {prior_sha256}")
    require(digest(PLUGIN) == EXPECTED_PLUGIN_SHA256, "Frozen scoring plugin changed")

    source_hashes = {}
    for rel, expected in EXPECTED_CRITICAL_SOURCE_SHA256.items():
        path = REINVENT_ROOT / Path(rel)
        require(path.exists(), f"Missing critical REINVENT4 source file: {rel}")
        observed = digest(path)
        require(observed == expected, f"Critical source hash mismatch: {rel}")
        source_hashes[rel] = observed

    matched = load_json(BASE_MATCHED_PROTOCOL)
    caps = {
        str(seed): int(matched["matched_request_caps"][str(seed)]["request_cap"])
        for seed in FORMAL_SEEDS
    }
    require(caps == {
        "42": 1040200,
        "43": 1040200,
        "44": 1040143,
        "45": 1040200,
        "46": 1040200,
    }, f"Unexpected matched request caps: {caps}")

    # Scientific choices below are frozen BEFORE viewing any formal REINVENT4 result.
    protocol_core = {
        "protocol_version": "reinvent4_mol2mol_formal_v1",
        "scientific_status": "PRE-PERFORMANCE FORMAL FREEZE",
        "performance_used_for_tuning": False,
        "formal_seeds": FORMAL_SEEDS,
        "implementation": {
            "software": "REINVENT4",
            "installed_version": "4.8.24",
            "audited_source_commit": EXPECTED_AUDITED_COMMIT,
            "source_identity_mode": "critical_file_hash_fallback_for_archive_extraction",
            "critical_source_sha256": source_hashes,
        },
        "prior": {
            "name": "mol2mol_medium_similarity.prior",
            "source": "official REINVENT4 priors Zenodo artifact",
            "zenodo_md5": EXPECTED_PRIOR_MD5,
            "local_md5": prior_md5,
            "local_sha256": prior_sha256,
            "internal_metadata_hash_note": (
                "REINVENT4 4.8.24 reports the legacy prior internal xxhash metadata "
                "as invalid, but the exact downloaded artifact is independently "
                "verified against the official Zenodo MD5 and local SHA256. "
                "The prior is not modified or re-saved."
            ),
        },
        "initialization": {
            "p0_size": 200,
            "p0_source_sha256": EXPECTED_P0_SOURCE_SHA256,
            "p0_smi_sha256": EXPECTED_P0_SMI_SHA256,
            "conditioning_semantics": "200 conditioning SMILES x batch_size=1 = 200 native Mol2Mol output slots per RL step",
        },
        "scorer": {
            "component": "FrozenProjectReward",
            "plugin_sha256": EXPECTED_PLUGIN_SHA256,
            "cross_environment_identity_report_sha256": digest(SCORER_ENV_REPORT),
            "cross_environment_n": 200,
            "cross_environment_atol": 1e-10,
            "cross_environment_exact_mismatches": 0,
            "cross_environment_numeric_mismatches": 0,
        },
        "formal_parameters": {
            "generator": "Mol2Mol",
            "prior_variant": "medium_similarity",
            "sample_strategy": "multinomial",
            "batch_size": 1,
            "randomize_smiles": False,
            "isomeric_smiles": False,
            "temperature": 1.0,
            "distance_threshold": 100,
            "learning_strategy": "dap",
            "sigma": 128,
            "learning_rate": 0.0001,
            "diversity_filter": {
                "type": "IdenticalMurckoScaffold",
                "bucket_size": 25,
                "minscore": 0.4,
                "minsimilarity": 0.4,
                "penalty_multiplier": 0.5,
            },
            "max_steps": FORMAL_MAX_STEPS,
            "min_steps": FORMAL_MAX_STEPS,
            "max_score": 1.0,
            "termination": "simple",
            "horizon_rationale": (
                "Fixed 100-step horizon predeclared before formal REINVENT4 results; "
                "the official REINVENT4 staged-learning example uses max_steps=100. "
                "No post-hoc horizon tuning is permitted."
            ),
        },
        "resource_accounting": {
            "primary_resource": "requests entering FrozenProjectReward",
            "p0_prime_requests": 200,
            "cache_hits_count_toward_requests": True,
            "invalid_or_duplicate_sequences_filtered_by_REINVENT_before_the_component": (
                "record separately as generator-level validity/duplication; they do "
                "not count as oracle requests because no FrozenProjectReward call occurs"
            ),
            "seed_specific_nonbinding_ceiling_from_step28": caps,
            "technical_cap_must_not_be_reached": True,
            "actual_requests_and_unique_evaluations_must_be_reported": True,
        },
        "final_evaluation_policy": {
            "final_native_batch": (
                "Use all 200 native Mol2Mol output slots from the final RL step, "
                "retaining duplicate slots and invalid slots."
            ),
            "invalid_slot_policy": (
                "Invalid/unsanitizable final-step slots receive zero reward/objectives "
                "for population-level summary metrics and have no scaffold."
            ),
            "duplicate_slot_policy": (
                "Retain duplicates as separate population slots for mean/Top-k/Tier-1 "
                "population summaries; scaffold occupancy is computed over the same "
                "200 slots."
            ),
            "archive_policy": (
                "The full unique valid oracle archive is retained separately and is "
                "not substituted for the native final batch."
            ),
            "reason": (
                "REINVENT4 is a generative RL method rather than a persistent "
                "population optimizer; the last native 200-slot generation is the "
                "predeclared final-state analogue and avoids post-hoc top-k archive selection."
            ),
        },
        "trajectory_policy": {
            "unit": "native RL step",
            "x_axis": "cumulative FrozenProjectReward requests including P0 prime",
            "generated_slots_per_step": 200,
            "record_generator_validity_and_duplicate_rates": True,
        },
        "matched_protocol_provenance": {
            "path": str(BASE_MATCHED_PROTOCOL),
            "sha256": digest(BASE_MATCHED_PROTOCOL),
        },
        "seed42_formal_toml": {
            "path": str(FORMAL_SEED42_TOML),
            "sha256": digest(FORMAL_SEED42_TOML),
        },
    }

    protocol_sha = canonical_hash(protocol_core)
    protocol = dict(protocol_core)
    protocol["protocol_sha256"] = protocol_sha

    marker = {
        "frozen": True,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "performance_used_for_tuning": False,
        "formal_seed42_authorized": True,
        "protocol_file": str(PROTOCOL_JSON),
    }

    FORMAL_PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)

    if PROTOCOL_JSON.exists() or FREEZE_MARKER.exists():
        require(PROTOCOL_JSON.exists() and FREEZE_MARKER.exists(),
                "Partial existing REINVENT4 formal freeze found; do not overwrite.")
        existing_protocol = load_json(PROTOCOL_JSON)
        existing_marker = load_json(FREEZE_MARKER)
        require(existing_protocol == protocol,
                "Existing formal protocol differs from the requested freeze. Refusing overwrite.")
        require(existing_marker.get("protocol_sha256") == protocol_sha,
                "Existing freeze marker protocol SHA mismatch.")
        print("REINVENT4 FORMAL PROTOCOL ALREADY FROZEN: IDENTICAL")
    else:
        PROTOCOL_JSON.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        FREEZE_MARKER.write_text(
            json.dumps(marker, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("REINVENT4 FORMAL PROTOCOL FROZEN")

    print("protocol SHA256 :", protocol_sha)
    print("prior MD5       :", prior_md5)
    print("prior SHA256    :", prior_sha256)
    print("P0 SHA256       :", EXPECTED_P0_SOURCE_SHA256)
    print("P0 .smi SHA256  :", EXPECTED_P0_SMI_SHA256)
    print("plugin SHA256   :", EXPECTED_PLUGIN_SHA256)
    print("formal steps    :", FORMAL_MAX_STEPS)
    print("randomize       : False")
    print("seed42 cap      :", caps["42"])
    print("seed42 TOML SHA :", digest(FORMAL_SEED42_TOML))
    print("formal_seed42_authorized = True")


if __name__ == "__main__":
    main()
