from __future__ import annotations

"""Run one GraphGA seed under the final Step-28-matched raw-request cap."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import importlib.util

THIS_DIR = Path(__file__).resolve().parent
COMMON_PATH = THIS_DIR / "30_0_external_matched_common.py"


def load_common():
    name = "external_matched_common_v31_graphga"
    spec = importlib.util.spec_from_file_location(name, COMMON_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(COMMON_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = load_common()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=C.FORMAL_SEEDS)
    ap.add_argument("--protocol_dir", type=Path, default=C.DEFAULT_PROTOCOL_DIR)
    ap.add_argument("--output_root", type=Path, default=C.DEFAULT_RESULT_ROOT / "graphga")
    args = ap.parse_args()

    protocol, marker = C.load_protocol(args.protocol_dir)
    for filename in ("30_0_external_matched_common.py", "30_2_run_graphga_matched.py"):
        C.verify_active_code_hash(protocol, filename)

    seed = int(args.seed)
    cap = C.request_cap_for_seed(protocol, seed)
    out = args.output_root.resolve() / f"seed_{seed}"
    if C.validate_completed_task(out, "graphga", seed, marker["protocol_sha256"]):
        print(f"GRAPHGA seed {seed}: ALREADY COMPLETE / HASH PASS")
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Incomplete/nonempty GraphGA output exists: {out}\n"
            "Inspect it manually; this runner never overwrites formal output."
        )
    out.mkdir(parents=True, exist_ok=True)

    repo = Path(protocol["graphga"]["repo"]).resolve()
    C.remove_python_cache_artifacts(repo)
    if C.git_commit(repo) != C.EXPECTED_GRAPHGA_COMMIT or C.git_is_dirty(repo):
        raise RuntimeError("GraphGA frozen repository commit/cleanliness mismatch.")
    for path, expected in (
        (repo / "molopt" / "graph_ga" / "run.py", C.EXPECTED_GRAPHGA_RUN_SHA256),
        (repo / "molopt" / "base.py", C.EXPECTED_GRAPHGA_BASE_SHA256),
        (repo / "molopt" / "graph_ga" / "hparams_default.yaml", C.EXPECTED_GRAPHGA_HPARAMS_SHA256),
    ):
        if C.sha256_file(path) != expected:
            raise RuntimeError(f"GraphGA frozen source/config changed: {path}")

    sys.path.insert(0, str(repo))
    from molopt.graph_ga import GraphGA

    _, raw_p0, can_p0 = C.load_frozen_p0(Path(args.protocol_dir) / "formal_p0.csv")
    scorer = C.FrozenProjectScorer()
    broker = C.RequestBudgetOracleBroker(scorer, cap, "graphga")
    broker.prime_p0(raw_p0)

    defaults = protocol["graphga"]["official_defaults"]
    optimizer = C.build_graphga_runtime_without_tdc_constructor(
        GraphGA,
        broker=broker,
        smi_file=Path(args.protocol_dir) / "formal_p0.smi",
        initial_smiles=raw_p0,
        n_jobs=1,
        max_oracle_calls=cap,
        freq_log=100,
        output_dir=out / "molopt_native",
    )

    boundary_stop = False
    returned_normally = False
    try:
        C.run_official_graphga_search(
            optimizer,
            scorer=scorer,
            seed=seed,
            patience=int(protocol["graphga"]["patience"]),
            population_size=200,
            offspring_size=int(defaults["offspring_size"]),
            mutation_rate=float(defaults["mutation_rate"]),
        )
        returned_normally = True
    except C.RequestBudgetReached as exc:
        boundary_stop = True
        print("[GraphGA] expected matched-request boundary stop:", exc)

    snapshots = C.graphga_population_snapshots(broker, can_p0, population_size=200)
    final_smiles = list(snapshots[-1]["canonical_smiles"])
    archive = broker.archive_dataframe()
    counters = broker.counters()
    validation = C.validate_common_archive(archive, can_p0, counters["requests"])
    final_population = C.scores_for_population(final_smiles, archive, expected_size=200)
    final_metrics = C.population_metrics(final_population, compute_hypervolume=True)

    actual = int(counters["requests"])
    native_early = bool(returned_normally and actual < cap)
    exact_cap = bool(actual == cap)
    request_cap_boundary = bool(boundary_stop or exact_cap)
    if boundary_stop and actual >= cap:
        # Possible only for scalar-like exact stop; GraphGA normally undershoots because of atomic batches.
        request_cap_boundary = True
    if not boundary_stop and not returned_normally:
        raise RuntimeError("GraphGA ended without an identifiable stop reason.")

    archive_path = out / "archive_unique.csv.gz"
    final_path = out / "final_population.csv"
    snapshots_path = out / "population_snapshots.json.gz"
    batch_log_path = out / "oracle_batch_log.json.gz"
    metadata_path = out / "scorer_broker_metadata.json"
    broker.save_archive(archive_path)
    final_population.to_csv(final_path, index=False)
    C.write_json_gz(snapshots_path, snapshots)
    broker.save_batch_log(batch_log_path)
    C.write_json(metadata_path, broker.metadata())

    completion = {
        "completed": True,
        "created_utc": utc_now(),
        "method": "graphga",
        "seed": seed,
        "protocol_sha256": marker["protocol_sha256"],
        "request_cap": cap,
        "actual_requests": actual,
        "request_utilization": actual / cap,
        "request_cap_boundary_stop": request_cap_boundary,
        "native_early_stop_before_request_cap": native_early,
        "atomic_batch_boundary": True,
        "counters": counters,
        "n_completed_native_updates": int(sum(s.get("stage") == "generation_update" for s in snapshots)),
        "final_population_source": "official GraphGA candidate-pool scores sorted exactly as frozen run.py, top 200",
        "archive_validation": validation,
        "final_metrics": final_metrics,
        "graphga_commit": C.git_commit(repo),
        "graphga_run_py_sha256": C.sha256_file(repo / "molopt" / "graph_ga" / "run.py"),
        "population_size": 200,
        "offspring_size": int(defaults["offspring_size"]),
        "mutation_rate": float(defaults["mutation_rate"]),
        "patience": int(protocol["graphga"]["patience"]),
        "n_jobs": 1,
        "archive_sha256": C.sha256_file(archive_path),
        "final_population_sha256": C.sha256_file(final_path),
        "population_snapshots_sha256": C.sha256_file(snapshots_path),
        "oracle_batch_log_sha256": C.sha256_file(batch_log_path),
        "scorer_broker_metadata_sha256": C.sha256_file(metadata_path),
        "performance_used_for_tuning": False,
    }
    C.write_json(out / "TASK_COMPLETED.json", completion)

    print("=" * 88)
    print(f"GRAPHGA MATCHED seed {seed}: TECHNICAL PASS")
    print("=" * 88)
    print(f"request cap     : {cap:,}")
    print(f"actual requests : {actual:,} ({actual/cap:.2%})")
    print("request boundary:", request_cap_boundary)
    print("native early stop:", native_early)
    print("completed updates:", completion["n_completed_native_updates"])
    print("Output:", out)
    print("Do NOT tune GraphGA after observing this formal result.")


if __name__ == "__main__":
    main()
