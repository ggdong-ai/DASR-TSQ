from __future__ import annotations

"""
MGP + TSQ transfer-selection adapter, v1.

Scientific purpose
------------------
Apply ONLY the TSQ scaffold-occupancy quota to the published
MolecularGraphPareto (MGP) NSGA-II environmental-selection stage.

Everything upstream remains MGP-native:
- official mutation
- official crossover
- official Arbiter
- official four-objective fitness vectors
- official Pareto-front calculation
- official crowding-distance implementation, including its published
  accumulation behavior

The quota is applied after MGP has assigned Pareto rank/crowding priority.

Important:
- This module does NOT import/reimplement the project's NSGA-II sorter.
- This avoids silently replacing MGP's Pareto/crowding implementation.
- If the quota is non-binding (max_per_scaffold >= population size), this
  adapter is designed to reproduce MGP's native selection ORDER exactly.
- If the strict quota cannot fill the population, the adapter retains the
  original TSQ fallback principle: fill remaining slots from quota-blocked
  candidates in MGP priority order, without duplicating molecule objects.
"""

from collections import Counter
from typing import Any, Callable, Sequence


def _stable_crowding_order(front: Sequence[Any]) -> list[Any]:
    """MGP-native partial-front priority: descending crowding, stable ties."""
    return sorted(
        list(front),
        key=lambda molecule: molecule.crowding_distance,
        reverse=True,
    )


def _front_fits_quota(
    front: Sequence[Any],
    scaffold_counts: dict[str, int],
    max_per_scaffold: int,
    scaffold_fn: Callable[[str], str],
) -> bool:
    proposed = Counter(scaffold_fn(m.smiles) for m in front)
    for scaffold, increment in proposed.items():
        if scaffold_counts.get(scaffold, 0) + int(increment) > max_per_scaffold:
            return False
    return True


def mgp_tsq_environmental_selection(
    engine: Any,
    molecules: Sequence[Any],
    scaffold_fn: Callable[[str], str],
    max_per_scaffold: int,
    population_size: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """
    MGP-native NSGA-II priority + TSQ scaffold quota.

    The MGP engine itself performs:
        calculate_fronts -> extrema -> crowding_distance

    Selection logic:
      1) Traverse Pareto fronts in native rank order.
      2) When a complete front can be accepted without making the quota
         relevant, preserve its exact native order.
      3) When capacity truncation or the quota makes subset selection
         necessary, use MGP's own descending crowding-distance priority.
      4) Strictly respect max_per_scaffold during the first pass.
      5) If strict quota cannot fill the population, relax the quota only
         for the remaining slots, using previously blocked candidates in
         MGP priority order. This mirrors the original TSQ fallback principle.

    No molecule is rescored here.
    """
    if max_per_scaffold < 1:
        raise ValueError("max_per_scaffold must be >= 1")

    target = int(
        engine.population_size if population_size is None else population_size
    )
    if target < 1:
        raise ValueError("population_size must be >= 1")

    pool = list(molecules)
    if not pool:
        return [], {
            "n_input": 0,
            "population_target": target,
            "max_per_scaffold": int(max_per_scaffold),
            "relative_quota": float(max_per_scaffold / target),
            "front_sizes": [],
            "strict_selected_count": 0,
            "quota_relaxed_count": 0,
            "blocked_by_quota_count": 0,
            "strict_unique_scaffolds": 0,
            "strict_max_scaffold_occupancy": 0,
            "final_unique_scaffolds": 0,
            "final_max_scaffold_occupancy": 0,
            "quota_binding": False,
            "quota_relaxation_used": False,
            "strict_scaffold_counts": {},
            "final_scaffold_counts": {},
        }

    # Crucially, retain the official MGP Pareto/crowding implementation.
    fronts = engine.calculate_fronts(pool)
    minimal_values, maximal_values = engine.extrema(pool)
    engine.crowding_distance(fronts, minimal_values, maximal_values)

    selected: list[Any] = []
    selected_ids: set[int] = set()
    scaffold_counts: dict[str, int] = {}
    blocked: list[Any] = []
    blocked_ids: set[int] = set()

    def add_strict(molecule: Any) -> None:
        scaffold = scaffold_fn(molecule.smiles)
        selected.append(molecule)
        selected_ids.add(id(molecule))
        scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1

    def block(molecule: Any) -> None:
        key = id(molecule)
        if key not in selected_ids and key not in blocked_ids:
            blocked.append(molecule)
            blocked_ids.add(key)

    # Strict quota pass.
    for front in fronts:
        if len(selected) >= target:
            break

        remaining = target - len(selected)

        # This branch is deliberately identical to native MGP behavior when
        # the whole front fits and the quota does not bind: append front in
        # its original calculate_fronts order.
        whole_front_native = (
            len(front) <= remaining
            and _front_fits_quota(
                front,
                scaffold_counts,
                max_per_scaffold,
                scaffold_fn,
            )
        )

        if whole_front_native:
            for molecule in front:
                add_strict(molecule)
            continue

        # Native MGP uses descending crowding distance when a front must be
        # truncated. We also use that priority when TSQ itself forces a subset.
        ordered_front = _stable_crowding_order(front)

        for molecule in ordered_front:
            if len(selected) >= target:
                break

            scaffold = scaffold_fn(molecule.smiles)
            current = scaffold_counts.get(scaffold, 0)

            if current < max_per_scaffold:
                add_strict(molecule)
            else:
                block(molecule)

        # If target was not reached, keep any quota-blocked members available
        # for the fallback phase. Members skipped only because target was
        # already full are irrelevant.
        if len(selected) < target:
            for molecule in ordered_front:
                scaffold = scaffold_fn(molecule.smiles)
                if (
                    id(molecule) not in selected_ids
                    and scaffold_counts.get(scaffold, 0) >= max_per_scaffold
                ):
                    block(molecule)

    strict_selected_count = len(selected)
    strict_counts = dict(scaffold_counts)

    # Original TSQ semantics include a no-duplication fallback when the strict
    # quota cannot fill the requested population.
    if len(selected) < target:
        for molecule in blocked:
            if len(selected) >= target:
                break
            if id(molecule) in selected_ids:
                continue
            scaffold = scaffold_fn(molecule.smiles)
            selected.append(molecule)
            selected_ids.add(id(molecule))
            scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1

    diagnostics = {
        "n_input": len(pool),
        "population_target": target,
        "max_per_scaffold": int(max_per_scaffold),
        "relative_quota": float(max_per_scaffold / target),
        "front_sizes": [len(front) for front in fronts],
        "strict_selected_count": strict_selected_count,
        "quota_relaxed_count": len(selected) - strict_selected_count,
        "blocked_by_quota_count": len(blocked),
        "strict_unique_scaffolds": len(strict_counts),
        "strict_max_scaffold_occupancy": (
            max(strict_counts.values()) if strict_counts else 0
        ),
        "final_unique_scaffolds": len(scaffold_counts),
        "final_max_scaffold_occupancy": (
            max(scaffold_counts.values()) if scaffold_counts else 0
        ),
        "quota_binding": bool(blocked),
        "quota_relaxation_used": len(selected) > strict_selected_count,
        "strict_scaffold_counts": strict_counts,
        "final_scaffold_counts": dict(scaffold_counts),
    }

    return selected, diagnostics
