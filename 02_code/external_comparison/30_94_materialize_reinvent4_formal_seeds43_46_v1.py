from __future__ import annotations

"""Materialize REINVENT4 formal execution configs for seeds 43-46.

This is an execution-materialization step only.  It does NOT change the
pre-performance scientific protocol.  Each config is deterministically derived
from the already-frozen seed42 TOML by changing only:
  1) seed-specific output paths, and
  2) the seed-specific non-binding technical request ceiling already frozen in
     REINVENT4_FORMAL_PROTOCOL.json.

The random seed itself is passed on the REINVENT4 CLI, as it was for seed42.
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

    entries = {}
    for seed in SEEDS:
        cap = CAPS[seed]
        text = base.replace("seed_42", f"seed_{seed}")
        if cap != 1_040_200:
            text = text.replace(
                "technical_request_cap = 1040200",
                f"technical_request_cap = {cap}",
            )

        target = CODE_DIR / f"reinvent4_formal_seed{seed}_v1.toml"
        if target.exists():
            require(
                target.read_text(encoding="utf-8") == text,
                f"Refusing to overwrite non-identical existing config: {target}",
            )
            status = "ALREADY EXISTS: IDENTICAL"
        else:
            target.write_text(text, encoding="utf-8", newline="\n")
            status = "CREATED"

        entries[str(seed)] = {
            "seed": seed,
            "technical_request_cap": cap,
            "config_path": str(target),
            "config_sha256": sha256_file(target),
            "derivation": (
                "byte-for-byte frozen seed42 TOML with only seed_42 path tokens "
                f"changed to seed_{seed}"
                + (
                    f" and technical_request_cap changed to frozen seed{seed} ceiling {cap}"
                    if cap != 1_040_200 else ""
                )
            ),
            "status": status,
        }

    manifest_core = {
        "manifest_version": "reinvent4_seeds43_46_execution_materialization_v1",
        "scientific_protocol_changed": False,
        "performance_based_parameter_change": False,
        "formal_protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "base_seed42_toml_sha256": observed_base_sha,
        "seed42_task_completed_sha256": sha256_file(SEED42_TASK),
        "entries": entries,
    }
    manifest_sha = hashlib.sha256(
        json.dumps(
            manifest_core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    manifest = dict(manifest_core)
    manifest["manifest_sha256"] = manifest_sha
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()

    if MANIFEST.exists():
        existing = load_json(MANIFEST)
        existing_core = {
            k: existing[k]
            for k in manifest_core
        }
        require(existing_core == manifest_core,
                "Existing execution manifest differs; refusing overwrite")
        require(existing.get("manifest_sha256") == manifest_sha,
                "Existing execution manifest SHA field mismatch")
        print("REINVENT4 seeds43-46 execution manifest ALREADY EXISTS: IDENTICAL")
    else:
        MANIFEST.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("REINVENT4 seeds43-46 execution configs MATERIALIZED")

    print("formal protocol SHA :", EXPECTED_PROTOCOL_SHA)
    print("base seed42 TOML SHA:", observed_base_sha)
    for seed in SEEDS:
        e = entries[str(seed)]
        print(
            f"seed {seed}: cap={e['technical_request_cap']} "
            f"sha256={e['config_sha256']} [{e['status']}]"
        )
    print("manifest SHA256     :", manifest_sha)
    print("formal_seeds43_46_authorized = True")


if __name__ == "__main__":
    main()
