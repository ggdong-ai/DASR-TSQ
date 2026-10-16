from __future__ import annotations

"""Verify/materialize REINVENT4 formal execution configs for seeds 43-46.

v2 fixes a bookkeeping-only bug in v1:
the v1 manifest included a transient per-file status string ("CREATED" vs
"ALREADY EXISTS: IDENTICAL") in the object later compared on reruns.  Thus a
perfectly valid second invocation could report "Existing execution manifest
differs" even though all scientific configs were byte-identical.

v2 treats status as non-scientific runtime metadata.  It never changes the
already-frozen scientific protocol and never overwrites a non-identical config.
It also accepts and verifies an existing v1 manifest produced on first
materialization.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(r"D:\pycharm\project\tsq_mcts_rebuild").resolve()
CODE_DIR = PROJECT_ROOT / "02_code" / "external_comparison"
PROTOCOL_DIR = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_reinvent4_protocol"
)
SEED42_DIR = (
    PROJECT_ROOT / "06_paper" / "experiment_records"
    / "external_benchmark_v4_reinvent4_formal" / "seed_42"
)

PROTOCOL_JSON = PROTOCOL_DIR / "REINVENT4_FORMAL_PROTOCOL.json"
FREEZE_MARKER = PROTOCOL_DIR / "REINVENT4_FORMAL_PROTOCOL_FROZEN.json"
SEED42_TASK = SEED42_DIR / "TASK_COMPLETED.json"
BASE_TOML = CODE_DIR / "reinvent4_formal_seed42_v1.toml"

# Keep using the existing manifest path.  v2 verifies an existing v1 manifest
# rather than replacing it merely because a transient status string changed.
MANIFEST = PROTOCOL_DIR / "REINVENT4_SEEDS43_46_EXECUTION_MANIFEST.json"

EXPECTED_PROTOCOL_SHA = "592929f79e673058fe3d308519df9378dea28c268e28778a3836e84928ff9379"
EXPECTED_BASE_TOML_SHA = "05506b83d417ddcae7af1f613ef2693dd0e06d6c7248c25a8b435988d30ff96f"

SEEDS = [43, 44, 45, 46]
CAPS = {
    43: 1_040_200,
    44: 1_040_143,
    45: 1_040_200,
    46: 1_040_200,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stable_entry(seed: int, target: Path, cap: int, config_sha: str) -> dict:
    return {
        "seed": seed,
        "technical_request_cap": cap,
        "config_path": str(target),
        "config_sha256": config_sha,
    }


def main() -> None:
    for path in (PROTOCOL_JSON, FREEZE_MARKER, SEED42_TASK, BASE_TOML):
        require(path.exists(), f"Missing required file: {path}")

    protocol = load_json(PROTOCOL_JSON)
    freeze = load_json(FREEZE_MARKER)
    task42 = load_json(SEED42_TASK)

    require(protocol.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA,
            "Frozen formal protocol SHA mismatch")
    require(freeze.get("frozen") is True, "Formal protocol is not frozen")
    require(freeze.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA,
            "Freeze marker protocol SHA mismatch")
    require(protocol.get("formal_seeds") == [42, 43, 44, 45, 46],
            "Unexpected formal seed list")
    require(protocol.get("performance_used_for_tuning") is False,
            "Frozen protocol says performance was used for tuning")

    require(task42.get("completed") is True and task42.get("audited") is True,
            "seed42 is not formally completed/audited")
    require(task42.get("audit_version") == "v2",
            "seed42 must pass the corrected v2 audit")
    require(task42.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA,
            "seed42 TASK_COMPLETED protocol SHA mismatch")
    require(task42.get("scientific_protocol_changed_by_audit_fix") is False,
            "seed42 audit unexpectedly reports a scientific-protocol change")

    observed_base_sha = sha256_file(BASE_TOML)
    require(observed_base_sha == EXPECTED_BASE_TOML_SHA,
            f"Frozen seed42 TOML changed: {observed_base_sha}")

    frozen_caps = protocol["resource_accounting"]["seed_specific_nonbinding_ceiling_from_step28"]
    require(
        {int(k): int(v) for k, v in frozen_caps.items()} ==
        {42: 1040200, 43: 1040200, 44: 1040143, 45: 1040200, 46: 1040200},
        f"Frozen request ceilings changed: {frozen_caps}",
    )

    base = BASE_TOML.read_text(encoding="utf-8")
    require(base.count("seed_42") >= 5, "Unexpected seed42 path structure in base TOML")
    require(base.count("technical_request_cap = 1040200") == 1,
            "Unexpected technical_request_cap occurrence in base TOML")
    require("min_steps = 100" in base and "max_steps = 100" in base,
            "Frozen 100-step horizon missing from base TOML")
    require("randomize_smiles = false" in base,
            "Frozen randomize_smiles=false missing from base TOML")

    stable_entries = {}
    display_status = {}

    for seed in SEEDS:
        cap = CAPS[seed]
        expected_text = base.replace("seed_42", f"seed_{seed}")
        if cap != 1_040_200:
            expected_text = expected_text.replace(
                "technical_request_cap = 1040200",
                f"technical_request_cap = {cap}",
            )

        target = CODE_DIR / f"reinvent4_formal_seed{seed}_v1.toml"
        if target.exists():
            require(
                target.read_text(encoding="utf-8") == expected_text,
                f"Refusing to use non-identical existing config: {target}",
            )
            display_status[str(seed)] = "EXISTS: IDENTICAL"
        else:
            target.write_text(expected_text, encoding="utf-8", newline="\n")
            display_status[str(seed)] = "CREATED"

        stable_entries[str(seed)] = stable_entry(
            seed, target, cap, sha256_file(target)
        )

    seed42_task_sha = sha256_file(SEED42_TASK)

    # Verify an existing v1 manifest using only stable/scientific fields.
    if MANIFEST.exists():
        existing = load_json(MANIFEST)
        require(existing.get("formal_protocol_sha256") == EXPECTED_PROTOCOL_SHA,
                "Existing execution manifest formal protocol SHA mismatch")
        require(existing.get("base_seed42_toml_sha256") == observed_base_sha,
                "Existing execution manifest base TOML SHA mismatch")
        require(existing.get("seed42_task_completed_sha256") == seed42_task_sha,
                "Existing execution manifest seed42 TASK_COMPLETED SHA mismatch")
        require(existing.get("scientific_protocol_changed") is False,
                "Existing execution manifest says scientific protocol changed")
        require(existing.get("performance_based_parameter_change") is False,
                "Existing execution manifest says parameters changed based on performance")

        old_entries = existing.get("entries", {})
        for seed in SEEDS:
            old = old_entries.get(str(seed))
            require(isinstance(old, dict), f"Existing manifest missing seed {seed}")
            exp = stable_entries[str(seed)]
            for key in ("seed", "technical_request_cap", "config_path", "config_sha256"):
                require(old.get(key) == exp[key],
                        f"Existing manifest seed {seed} differs in stable field {key}")
        manifest_mode = "EXISTING v1 MANIFEST VERIFIED"
        existing_manifest_sha = existing.get("manifest_sha256", "not-recorded")
    else:
        # New v2 manifests contain only stable content in their canonical hash.
        core = {
            "manifest_version": "reinvent4_seeds43_46_execution_materialization_v2",
            "scientific_protocol_changed": False,
            "performance_based_parameter_change": False,
            "formal_protocol_sha256": EXPECTED_PROTOCOL_SHA,
            "base_seed42_toml_sha256": observed_base_sha,
            "seed42_task_completed_sha256": seed42_task_sha,
            "entries": stable_entries,
        }
        manifest_sha = hashlib.sha256(
            json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest = dict(core)
        manifest["manifest_sha256"] = manifest_sha
        manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
        MANIFEST.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest_mode = "NEW v2 MANIFEST CREATED"
        existing_manifest_sha = manifest_sha

    print("REINVENT4 seeds43-46 execution configs VERIFIED")
    print("formal protocol SHA :", EXPECTED_PROTOCOL_SHA)
    print("base seed42 TOML SHA:", observed_base_sha)
    for seed in SEEDS:
        e = stable_entries[str(seed)]
        print(
            f"seed {seed}: cap={e['technical_request_cap']} "
            f"sha256={e['config_sha256']} [{display_status[str(seed)]}]"
        )
    print("manifest mode       :", manifest_mode)
    print("manifest SHA field  :", existing_manifest_sha)
    print("formal_seeds43_46_authorized = True")


if __name__ == "__main__":
    main()
