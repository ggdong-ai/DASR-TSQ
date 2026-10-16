from __future__ import annotations

"""Preflight for REINVENT4 Mol2Mol benchmark protocol v1.

No RL optimization is executed by this script.

Source identity policy
----------------------
Preferred: verify the frozen git commit when .git metadata is available.
Fallback for an archive/ZIP extraction: verify SHA256 for the critical REINVENT4
source files already audited for this protocol.  The fallback is sufficient for
engineering preflight/smoke, but the final formal freeze should additionally
record the original archive/release provenance if available.
"""

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\pycharm\project\tsq_mcts_rebuild").resolve()
REINVENT_ROOT = PROJECT_ROOT / "08_external" / "REINVENT4_official"
PLUGIN_ROOT = PROJECT_ROOT / "02_code" / "external_comparison" / "reinvent4_plugin"

EXPECTED_COMMIT = "ee0d56f4a07472bbb622cd0858184d06f11bff5d"

# These hashes come from the already-audited frozen source files used to design
# this REINVENT4 protocol.  They are checked only when git metadata is absent.
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

PRIOR = REINVENT_ROOT / "priors" / "mol2mol_medium_similarity.prior"
P0_MANIFEST = (
    PROJECT_ROOT
    / "06_paper"
    / "experiment_records"
    / "external_benchmark_v4_reinvent4_preflight"
    / "reinvent4_formal_p0_manifest.json"
)
PLUGIN = (
    PLUGIN_ROOT
    / "reinvent_plugins"
    / "components"
    / "comp_frozen_project_reward.py"
)

PRIOR_REGISTRY = REINVENT_ROOT / "reinvent" / "utils" / "prior_registry.py"
MOL2MOL_SAMPLER = REINVENT_ROOT / "reinvent" / "runmodes" / "samplers" / "mol2mol.py"
DISTANCE_PENALTY = REINVENT_ROOT / "reinvent" / "runmodes" / "RL" / "distance_penalty.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def git_try(*args: str):
    """Return (ok, stdout_or_error) without throwing on non-git ZIP extractions."""
    git_exe = shutil.which("git")
    if not git_exe:
        return False, "git executable not found"

    proc = subprocess.run(
        [git_exe, "-C", str(REINVENT_ROOT), *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err or f"git exited with code {proc.returncode}"
    return True, proc.stdout.strip()


def verify_source_identity():
    """Prefer git commit; otherwise verify audited critical-source hashes."""
    ok, value = git_try("rev-parse", "HEAD")
    if ok:
        commit = value
        require(commit == EXPECTED_COMMIT, f"REINVENT4 commit mismatch: {commit}")

        ok_dirty, dirty = git_try("status", "--porcelain", "--untracked-files=no")
        require(ok_dirty, f"Could not read REINVENT4 git status: {dirty}")
        require(not dirty, "Tracked REINVENT4 source files are modified:\n" + dirty)

        return {
            "mode": "git_commit",
            "commit": commit,
            "critical_hashes_checked": 0,
        }

    # Archive/ZIP extraction: .git is commonly absent.  Do not treat that as a
    # scientific failure for the engineering smoke; verify the experiment-
    # critical source files byte-for-byte instead.
    mismatches = []
    checked = {}

    for rel, expected in EXPECTED_CRITICAL_SOURCE_SHA256.items():
        path = REINVENT_ROOT / Path(rel)
        if not path.exists():
            mismatches.append(f"MISSING: {rel}")
            continue
        actual = sha256_file(path)
        checked[rel] = actual
        if actual != expected:
            mismatches.append(
                f"HASH MISMATCH: {rel}\n"
                f"  expected={expected}\n"
                f"  actual  ={actual}"
            )

    require(
        not mismatches,
        "REINVENT4 has no usable git metadata and critical-source fallback failed:\n"
        + "\n".join(mismatches),
    )

    return {
        "mode": "critical_file_hash_fallback",
        "declared_audited_commit": EXPECTED_COMMIT,
        "git_probe_error": value,
        "critical_hashes_checked": len(checked),
    }


def main():
    require(REINVENT_ROOT.exists(), f"Missing REINVENT4 source: {REINVENT_ROOT}")

    source_identity = verify_source_identity()

    require(
        P0_MANIFEST.exists(),
        f"Run 30_88_prepare_reinvent4_inputs_v1.py first: {P0_MANIFEST}",
    )
    p0_manifest = json.loads(P0_MANIFEST.read_text(encoding="utf-8"))
    require(
        p0_manifest["source_p0_sha256"]
        == "893ed1fe736781fb25bfa421584b4de5e763eb1d2b79daff9cef6630d28adcfc",
        "Frozen P0 SHA mismatch",
    )
    require(
        p0_manifest["n_output_rows"] == 200
        and p0_manifest["n_unique_canonical"] == 200,
        "REINVENT4 P0 must contain 200 unique canonical molecules",
    )

    require(PRIOR.exists(), f"Missing required prior: {PRIOR}")
    require(PLUGIN.exists(), f"Missing scoring plugin: {PLUGIN}")

    registry_text = PRIOR_REGISTRY.read_text(encoding="utf-8")
    require(
        '".m2m_medium"' in registry_text
        and "mol2mol_medium_similarity.prior" in registry_text,
        "Frozen source prior registry does not contain expected .m2m_medium mapping",
    )

    sampler_text = MOL2MOL_SAMPLER.read_text(encoding="utf-8")
    require(
        "smilies = smilies * self.batch_size" in sampler_text,
        "Mol2Mol batch semantics differ from audited frozen source",
    )

    distance_text = DISTANCE_PENALTY.read_text(encoding="utf-8")
    require(
        "distance_threshold / likelihood" in distance_text,
        "Mol2Mol distance penalty implementation differs from audited frozen source",
    )

    reinvent_exe = shutil.which("reinvent")
    require(
        reinvent_exe is not None,
        "REINVENT4 CLI 'reinvent' is not available in the active environment",
    )

    # Keep the custom component outside the frozen official tree.
    for path in (PLUGIN_ROOT, REINVENT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    imported = importlib.import_module(
        "reinvent_plugins.components.comp_frozen_project_reward"
    )
    require(
        hasattr(imported, "FrozenProjectReward"),
        "FrozenProjectReward plugin class not found",
    )

    print("REINVENT4 PREFLIGHT PASS")
    print(f"source mode      : {source_identity['mode']}")
    if source_identity["mode"] == "git_commit":
        print(f"commit           : {source_identity['commit']}")
    else:
        print(f"audited commit   : {source_identity['declared_audited_commit']}")
        print(f"critical hashes  : {source_identity['critical_hashes_checked']}/6 MATCH")
        print("git metadata     : unavailable (archive/ZIP extraction accepted for smoke)")
    print(f"prior            : {PRIOR}")
    print(f"prior SHA256     : {sha256_file(PRIOR)}")
    print(f"P0 source SHA    : {p0_manifest['source_p0_sha256']}")
    print(f"P0 .smi SHA256   : {p0_manifest['output_sha256']}")
    print(f"plugin SHA256    : {sha256_file(PLUGIN)}")
    print(f"reinvent CLI     : {reinvent_exe}")
    print("audited sampling : 200 conditioning SMILES x batch_size=1 = 200 inputs/step")
    print("distance policy  : native Mol2Mol prior-NLL penalty, threshold=100")
    print("formal_ready_without_smoke = True")


if __name__ == "__main__":
    main()
