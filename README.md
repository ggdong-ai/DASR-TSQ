# DASR–TSQ

**Decision-Aligned Sibling Ranking and Topology Scaffold Quotas for Multiobjective Molecular Optimization**

## Overview

DASR–TSQ combines two mechanisms for multiobjective molecular optimization:

- **DASR (decision-aligned sibling ranking):** prioritizes sibling candidates for limited oracle evaluation in local Monte Carlo tree search (MCTS).
- **TSQ (topology scaffold quotas):** limits the occupancy of generic Bemis–Murcko scaffolds during NSGA-II environmental selection.

The study considers four objectives: predicted GSK3β activity, predicted JNK3 activity, quantitative estimate of drug-likeness (QED), and synthetic accessibility (SA). Reported target activities are **QSAR predictions**, not experimentally measured activities of the generated molecules.

## Repository contents

| Path | Contents |
| --- | --- |
| `01_data/raw/` and `01_data/clean/` | ChEMBL-derived input tables and processed modeling/candidate datasets |
| `01_data/splits/paper_cv/` | Random and scaffold cross-validation splits and split diagnostics |
| `01_data/sibling_ranking_p0_safe_v2/` | Frozen sibling-ranking train/validation/test datasets and validation records |
| `02_code/core/13_mcts/` | MCTS components |
| `02_code/core/27_dasr_mcts/` | Frozen sibling rankers, DASR–MCTS integration, and formal local-search entry point |
| `02_code/core/28_dasr_tsq/` | DASR–TSQ evolution engine and formal Step 28 v4.2 entry point |
| `02_code/core/29_tsq_sensitivity/` | TSQ quota-sensitivity implementation and entry point |
| `02_code/models/` and `02_code/scoring/` | QSAR/ranker training, oracle prediction, reward and scoring utilities |
| `02_code/external_comparison/` | Baseline adapters, launch/preflight/audit scripts, frozen REINVENT4 TOML configurations and supporting documentation |
| `03_models/` | Saved QSAR cross-validation models, final activity oracles, sibling rankers, and auxiliary model files |
| `05_results/qsar/` | QSAR metrics and prediction records |
| `05_results/dasr_tsq_end_to_end/` | Frozen Step 28 run records, per-generation populations, and formal analysis outputs |
| `05_results/external_comparison/` | Available GraphGA and MARS matched-benchmark records |
| `requirements-lock.txt` | Package snapshot from the development environment |

Directory inclusion is not, by itself, evidence that every experimental workflow has been independently reproduced from a clean installation.

## Main implementation entry points

The repository contains the following formal experiment scripts:

- **Local DASR–MCTS:** `02_code/core/27_dasr_mcts/27_4_run_dasr_mcts_formal.py`
- **End-to-end DASR–TSQ:** `02_code/core/28_dasr_tsq/28_4_run_end_to_end_formal_v4_2.py`
- **TSQ quota sensitivity:** `02_code/core/29_tsq_sensitivity/29_1_run_tsq_quota_sensitivity.py`
- **Final QSAR oracles:** `02_code/models/06_train_final_oracles.py` and `02_code/models/07_test_final_oracles.py`
- **Sibling ranker training:** `02_code/models/24_train_sibling_rankers.py`

These are *file locations*, not verified command-line invocations. Exact working directories, required input paths, command-line options, dependency versions, and fresh-environment execution are still to be validated before publication.

## Frozen Step 28 experiments

Formal end-to-end results are organized under:

`05_results/dasr_tsq_end_to_end/step28_formal_v4_2/tasks/`

The recorded task folders cover seeds **42–46** for each of the following configurations:

| Configuration | Task-directory prefix |
| --- | --- |
| Baseline (random selection; no TSQ) | `step28_baseline_random_no_tsq_seed` |
| TSQ only (random selection; quota 10) | `step28_tsq_only_random_tsq10_seed` |
| DASR only (listwise ranking; no TSQ) | `step28_dasr_only_listwise_no_tsq_seed` |
| DASR–TSQ (listwise ranking; quota 10) | `step28_dasr_tsq_listwise_tsq10_seed` |
| Absolute-child selection; no TSQ | `step28_absolute_only_no_tsq_seed` |
| Absolute-child selection; quota 10 | `step28_absolute_tsq_tsq10_seed` |

Each recorded task directory includes `history.csv` and `final_population.csv`, together with additional run records and/or population snapshots. The four Baseline/TSQ-only/DASR-only/DASR–TSQ configurations constitute the primary factorial comparison; the two absolute-child configurations are additional controls.

Selected analysis outputs are in:

`05_results/dasr_tsq_end_to_end/step28_formal_analysis_v4_2/`

In particular:

- `formal_variant_summary.csv`: variant-level summaries
- `formal_paired_seed_effects.csv`: paired-seed effects
- `formal_generation_trajectory.csv.gz`: generation-level trajectory data
- `formal_run_level_metrics.csv`: run-level metrics

## External comparisons

The project includes adapters and/or experiment configuration scripts for **GraphGA**, **MARS**, **MGP–NSGA-II**, and **REINVENT4–Mol2Mol** under `02_code/external_comparison/`. Some matched GraphGA and MARS run outputs are present under `05_results/external_comparison/matched_step28_v3_1/`.

These external methods have their own software dependencies and upstream implementations. Consult the experiment scripts and `02_code/external_comparison/README_EXTERNAL_MATCHED_V3_1.md` for available implementation records. Upstream source versions, task-specific modifications, access requirements, and the availability of every baseline's frozen result files must be checked before claiming complete baseline reproduction.

## Data, models, and reproducibility status

- Source training and evaluation datasets, saved models, formal optimization records, and selected statistical summaries are organized as indicated above.
- **Data S1 and Data S2** are machine-readable datasets prepared for the manuscript's Supporting Information. Files bearing those submission names were **not identified in the supplied tracked-file manifest**; they should be added or linked explicitly before final release if repository distribution is intended.
- The manuscript-specific Figure 4 plotting script and a verified, end-to-end set of reproduction commands were **not identified/validated from the supplied file inventory**.
- `requirements-lock.txt` records package versions, but its portability and sufficiency for a clean-environment installation have not yet been independently checked.

**Current status:** Pre-submission reproducibility repository. Code, input files and frozen result records have been collected; clean-install verification, executable reproduction instructions, completeness of publication datasets/figures, and final release permissions remain to be confirmed.

## Citation and access

Manuscript in preparation for the *Journal of Chemical Information and Modeling*. Please cite the final published article when available. Repository access, licensing, and the immutable manuscript-associated release will be specified after author approval and final validation.
