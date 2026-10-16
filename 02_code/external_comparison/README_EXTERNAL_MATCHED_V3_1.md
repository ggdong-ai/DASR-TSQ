# Final external comparison: Step-28 matched raw-request benchmark v3.1

Project root assumed by the launchers:

`D:\pycharm\project\tsq_mcts_rebuild`

This package supersedes the old **10K-global-unique v1/v2** external benchmark and the earlier draft v3 package. Old result/protocol directories are retained only as provenance and are never read as formal performance by v3.1.

## Scientific resource rule

For each paired seed 42–46, the external comparator receives a **maximum raw oracle-request cap equal to the final `total_oracle_requests_cumulative` of the completed DASR–TSQ Step-28 run with the same seed**.

A request is counted before canonicalization and cache lookup. Therefore:

- frozen P0 priming counts;
- cache-hit requests count;
- invalid requests count;
- model evaluations/cache hits/invalids/unique valid molecules are reported separately.

Known Step-28 caps are approximately 1.04 million requests per seed; the freeze script extracts them from the authoritative completed Step-28 files and does not hard-code them as experimental outcomes.

For batch-based GraphGA and MARS, the budget boundary is **atomic**: if the next native scoring batch cannot fit entirely, it is rejected before consuming any request. For scalar-stream MNopt, the common oracle stops exactly at the request cap and the final population is the last completed 200-offspring environmental update.

Native early stopping/max-step rules are preserved. Thus the comparison is under the same **maximum available request budget**; a method that terminates natively before the cap is not artificially extended. Its actual request utilization is explicitly reported.

## Method identity

- DASR–TSQ: completed formal Step 28 is reused; it is **not rerun**.
- GraphGA: frozen official `mol-opt` GraphGA search mechanics; population size is 200 only to accept the exact common P0, while offspring size 70 and mutation rate 0.067 remain frozen.
- MARS: frozen official MARS search mechanics, `Sampler_SA`, native frozen 1000-step maximum and early stopping preserved; common project oracle replaces native property scorers.
- MNopt: **paper-faithful reimplementation** only. Do not describe it as the authors' exact official executable implementation. Frozen reimplementation core and source snapshots are retained unchanged; common project oracle replaces native property scorers.

## Active files — continuous sequence

| No. | File | Purpose / environment |
|---|---|---|
| 30_0 | `30_0_external_matched_common.py` | common scorer/request broker/metrics |
| 30_1 | `30_1_freeze_external_matched_protocol.py` | zero-optimization protocol freeze, project `.venv` |
| 30_2 | `30_2_run_graphga_matched.py` | GraphGA one-seed runner, project `.venv` |
| 30_3 | `30_3_mars_oracle_bridge.py` | MARS project-oracle JSONL server, project `.venv` |
| 30_4 | `30_4_run_mars_matched.py` | MARS one-seed runner, `mars_legacy` |
| 30_5 | `30_5_mnopt_source_snapshot.py` | frozen MNopt source-snapshot verifier |
| 30_6 | `30_6_medchem_source_snapshot.py` | frozen medchem source-snapshot verifier |
| 30_7 | `30_7_mnopt_reimpl_core.py` | unchanged pre-performance MNopt core |
| 30_8 | `30_8_mnopt_oracle_server.py` | MNopt project-oracle JSONL server, project `.venv` |
| 30_9 | `30_9_run_mnopt_matched.py` | MNopt one-seed runner, `mnopt_legacy` |
| 30_10 | `30_10_analyze_external_matched.py` | common final analysis, project `.venv` |
| 30_11 | `30_11_FREEZE_PROTOCOL.bat` | click: freeze only |
| 30_12 | `30_12_GRAPHGA_5SEEDS.bat` | click: GraphGA seeds 42–46 |
| 30_13 | `30_13_MARS_5SEEDS.bat` | click: MARS seeds 42–46 |
| 30_14 | `30_14_MNOPT_5SEEDS.bat` | click: MNopt seeds 42–46 |
| 30_15 | `30_15_ANALYZE.bat` | click: final analysis |
| 30_16 | `30_16_RUN_ALL.bat` | click: freeze → three comparator batches → analysis |

## Installation

1. Extract the zip to any temporary folder.
2. Double-click `INSTALL_AND_CLEAN.bat`.
3. It copies the final 30_0–30_16 files into:
   `D:\pycharm\project\tsq_mcts_rebuild\02_code\external_comparison\`
4. It syntax-checks the installed Python files **before** removing known obsolete v1/v2 launchers/runners that are not required for frozen provenance.
5. It does **not** delete old result/protocol directories or `08_external` source repositories.
6. Frozen MNopt provenance Python files referenced by existing method-freeze records are intentionally retained; they are **not** part of the active v3.1 run chain. Historical method-freeze builders such as `35_0`, `35_2b`, `35_2c`, `35_3` are also retained as evidence.

## Running

### Safest first action

Double-click:

`30_11_FREEZE_PROTOCOL.bat`

It must end with:

`EXTERNAL MATCHED STEP-28 V3.1 PROTOCOL FREEZE: PASS`

and:

`NO external optimization was run.`

The freeze verifies exact ordered P0 identity across all five Step-28 reference seeds, current scorer reproduction, GraphGA source hashes, MARS frozen source/method configuration, MNopt frozen core/snapshots, and the five same-seed request caps.

### One-click complete run

After installation you may directly double-click:

`30_16_RUN_ALL.bat`

It is fail-closed and sequential:

1. protocol freeze/hash verification;
2. GraphGA seeds 42–46;
3. MARS seeds 42–46;
4. MNopt seeds 42–46;
5. common analysis.

No methods run in parallel. If a stage fails, later stages are not started. A previously completed seed is hash-checked and skipped; an incomplete/nonempty formal seed directory is never overwritten silently.

You may instead run `30_12`–`30_15` one at a time.

## Output locations

Frozen protocol:

`06_paper\experiment_records\external_benchmark_matched_step28_v3_1\`

New formal external results:

`05_results\external_comparison\matched_step28_v3_1\`

Final analysis:

`05_results\external_comparison\matched_step28_v3_1\analysis\`

Key analysis files:

- `external_run_metrics.csv`
- `external_summary_mean_sd.csv`
- `external_paired_effects_vs_dasr.csv`
- `external_resource_checkpoints.csv`
- `external_population_trajectory.csv.gz`
- `EXTERNAL_ANALYSIS_COMPLETED.json`

## Analysis boundary

The primary comparison object is the **last completed native population of 200 molecules** for each method/seed. Primary endpoints are Top-10 mean final reward, 4-D hypervolume, Tier-1 count, and unique generic Bemis–Murcko scaffolds.

Resource efficiency additionally uses a pre-frozen **piecewise-constant last-completed-population trajectory** from request 200. If a comparator stops natively before its matched cap, its last completed population is carried forward only for the resource-AUC calculation. Actual request utilization remains reported separately.

With five paired seeds, report effect sizes, bootstrap CIs and direction consistency. Do not make a conventional exact two-sided `p < 0.05` claim from five paired seeds.

## Do not do after freeze/formal outcomes

Do not change P0, scorer/oracles, seeds, request caps, GraphGA offspring/mutation/patience, MARS frozen configuration, MNopt frozen configuration, endpoint definitions, or analysis rules after observing formal performance. Engineering/runtime bugs may be repaired only if the scientific configuration and previously completed valid tasks remain unchanged and auditable.
