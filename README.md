# DASR–TSQ

**Decision-Aligned Sibling Ranking and Topology Scaffold Quotas for Multiobjective Molecular Optimization**

## Overview

DASR–TSQ is a hierarchical molecular optimization framework that combines two components:

- **DASR:** Decision-aligned sibling ranking prioritizes candidate evaluation during local Monte Carlo tree search (MCTS) under a limited oracle budget.
- **TSQ:** Topology scaffold quotas constrain the occupancy of generic Bemis–Murcko scaffolds during NSGA-II environmental selection.

The framework is evaluated on four-objective molecular optimization involving predicted GSK3β and JNK3 activity, quantitative estimate of drug-likeness (QED), and synthetic accessibility.

## Repository Structure

- `02_code/`: Algorithm implementations and experimental scripts.
- `03_models/`: Selected trained models and associated configurations.
- `requirements-lock.txt`: Python environment package snapshot.

Additional analysis scripts, experimental configurations, and supporting datasets will be incorporated as the repository is finalized.

## Experimental Design

The study includes:

- QSAR surrogate model development and validation.
- DASR training and independent sibling-ranking evaluation.
- Fixed-budget local MCTS comparisons.
- Four-configuration DASR–TSQ factorial ablation.
- TSQ scaffold-quota sensitivity analysis.
- Comparisons with external molecular optimization baselines.

The formal evolutionary experiments use five random seeds (42–46).

## Reproducibility

The repository is under preparation. Installation instructions, verified execution commands, frozen experimental configurations, and result-reproduction instructions will be added following validation.

## Data and Software Availability

The manuscript is accompanied by Supporting Information and two machine-readable molecular datasets (Data S1 and Data S2). Their repository locations and access instructions will be documented when the release is finalized.

## Status

Private repository under preparation for manuscript submission to the *Journal of Chemical Information and Modeling*.