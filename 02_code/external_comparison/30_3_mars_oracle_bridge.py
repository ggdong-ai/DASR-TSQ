from __future__ import annotations

"""JSONL frozen-project oracle server for MARS (runs in the project .venv)."""

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import importlib.util

THIS_DIR = Path(__file__).resolve().parent
COMMON_PATH = THIS_DIR / "30_0_external_matched_common.py"


def load_common():
    name = "external_matched_common_v31_mars_bridge"
    spec = importlib.util.spec_from_file_location(name, COMMON_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(COMMON_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = load_common()


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True, choices=C.FORMAL_SEEDS)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--batch_log", type=Path, required=True)
    args = ap.parse_args()

    protocol, marker = C.load_protocol(args.protocol_dir)
    for filename in ("30_0_external_matched_common.py", "30_3_mars_oracle_bridge.py"):
        C.verify_active_code_hash(protocol, filename)
    cap = C.request_cap_for_seed(protocol, int(args.seed))
    _, raw_p0, _ = C.load_frozen_p0(Path(args.protocol_dir) / "formal_p0.csv")

    # Keep stdout as a strict JSONL channel.  Any scorer/RDKit diagnostics are
    # redirected to stderr, which the legacy-side launcher stores separately.
    with contextlib.redirect_stdout(sys.stderr):
        scorer = C.FrozenProjectScorer()
        broker = C.RequestBudgetOracleBroker(scorer, cap, "mars")
        broker.prime_p0(raw_p0)

    emit(
        {
            "status": "ready",
            "protocol_sha256": marker["protocol_sha256"],
            "seed": int(args.seed),
            "request_cap": cap,
            "p0_count": 200,
            "counters": broker.counters(),
        }
    )

    for line in sys.stdin:
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "hello":
                emit(
                    {
                        "status": "ok",
                        "protocol_sha256": marker["protocol_sha256"],
                        "seed": int(args.seed),
                        "request_cap": cap,
                        "counters": broker.counters(),
                    }
                )
            elif cmd == "score_batch":
                smiles = [str(x) for x in req.get("smiles", [])]
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        records = broker.score_batch_atomic(smiles)
                    emit({"status": "ok", "scores": records, "counters": broker.counters()})
                except C.RequestBudgetReached as exc:
                    emit(
                        {
                            "status": "budget_reached",
                            "message": str(exc),
                            "counters": broker.counters(),
                        }
                    )
            elif cmd == "counters":
                emit({"status": "ok", "counters": broker.counters()})
            elif cmd == "metadata":
                with contextlib.redirect_stdout(sys.stderr):
                    metadata = broker.metadata()
                emit({"status": "ok", "metadata": metadata})
            elif cmd == "save":
                with contextlib.redirect_stdout(sys.stderr):
                    broker.save_archive(args.archive)
                    broker.save_batch_log(args.batch_log)
                emit(
                    {
                        "status": "ok",
                        "archive": str(args.archive.resolve()),
                        "batch_log": str(args.batch_log.resolve()),
                        "counters": broker.counters(),
                    }
                )
            elif cmd == "shutdown":
                with contextlib.redirect_stdout(sys.stderr):
                    broker.save_archive(args.archive)
                    broker.save_batch_log(args.batch_log)
                emit({"status": "ok", "counters": broker.counters()})
                return
            else:
                emit({"status": "error", "error": f"unknown command: {cmd}"})
        except Exception as exc:
            emit({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


if __name__ == "__main__":
    main()
