"""Network graph helpers for MetaInsight data structures.

These helpers mirror key parts of the R ``sub_networks`` logic:
- graph construction from treatment comparisons,
- study filtering for treatment sets,
- subnetwork discovery,
- loops / nodesplitting pre-checks.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, TypeVar

import pandas as pd
import warnings
import re


def _find_data_shape(data: pd.DataFrame) -> str:
    return "long" if "T" in data.columns else "wide"


T = TypeVar("T")


def find_all_treatments(data: pd.DataFrame, treatment_ids: pd.DataFrame | None = None, study: str | None = None) -> List[T]:
    """Return treatment labels/IDs present in the data.

    If ``treatment_ids`` is supplied, treatment IDs are converted back to labels.
    """

    study_data = data if study is None else data[data["Study"] == study]
    if study_data.empty:
        return []

    if "T" in study_data.columns:
        values = study_data["T"].dropna().tolist()
    else:
        t_columns = [col for col in study_data.columns if re.match(r"^T(\.[0-9]+)?$", str(col))]
        if not t_columns:
            return []
        values = study_data[t_columns].stack().dropna().tolist()

    if treatment_ids is None:
        return list(dict.fromkeys(values))

    lookup = {int(row.Number): str(row.Label) for row in treatment_ids.itertuples()}
    labeled_values: List[str] = []
    for value in values:
        if pd.isna(value):
            continue
        try:
            label = lookup[int(value)]
            labeled_values.append(label)
        except (ValueError, TypeError, KeyError):
            labeled_values.append(str(value))

    return list(dict.fromkeys(labeled_values))


def find_studies_including_treatments(data: pd.DataFrame, treatments: Iterable, all_or_any: str) -> List[str]:
    """Return studies that include the requested treatment set.

    Parameters
    ----------
    data:
        Long or wide treatment table.
    treatments:
        Iterable of treatment identifiers/names.
    all_or_any:
        Either ``"all"`` or ``"any"``.
    """

    treatment_lookup = {str(t) for t in treatments}
    if not treatment_lookup:
        return []

    matching: List[str] = []
    if "T" in data.columns:
        for study, study_data in data.groupby("Study", sort=False):
            present = {str(t) for t in study_data["T"].dropna().tolist()}
            if all_or_any == "any":
                if present.intersection(treatment_lookup):
                    matching.append(str(study))
            elif all_or_any == "all":
                if treatment_lookup.issubset(present):
                    matching.append(str(study))
    else:
        t_cols = [col for col in data.columns if re.match(r"^T(\.[0-9]+)?$", str(col))]
        for _, row in data.iterrows():
            study = str(row["Study"])
            present = {str(row[col]) for col in t_cols if pd.notna(row.get(col))}
            if all_or_any == "any":
                if present.intersection(treatment_lookup):
                    matching.append(study)
            elif all_or_any == "all":
                if treatment_lookup.issubset(present):
                    matching.append(study)

    return matching


def create_graph(data: pd.DataFrame) -> Dict[str, set[str]]:
    """Construct an undirected treatment comparison graph.

    Nodes are treatment labels from the dataset and edges connect treatments compared
    in the same study.
    """

    graph: Dict[str, set[str]] = defaultdict(set)
    for _, row in data.groupby("Study", sort=False):
        treatments = list(dict.fromkeys(str(t) for t in find_all_treatments(row) if pd.notna(t)))
        if len(treatments) < 2:
            if treatments:
                graph.setdefault(treatments[0], set())
            continue

        for left, right in combinations(treatments, 2):
            graph[left].add(right)
            graph[right].add(left)

    for treatment in find_all_treatments(data):
        graph.setdefault(str(treatment), set())

    return graph


def _connected_components(graph: Dict[str, set[str]]) -> List[set[str]]:
    seen: set[str] = set()
    components: List[set[str]] = []

    for node in graph:
        if node in seen:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(graph.get(current, set()) - seen)
        components.append(component)

    return components


def identify_subnetworks(
    data: pd.DataFrame,
    treatment_ids: pd.DataFrame,
    reference_treatment_name: str | None = None,
    subnet_name_prefix: str = "subnet_",
) -> Dict[str, Dict[str, List[int]]]:
    """Find connected subnetworks and their study membership.

    Parameters
    ----------
    data:
        Study-level network data where treatments are either labels or numeric ids.
    treatment_ids:
        Table with ``Number`` and ``Label`` columns.
    reference_treatment_name:
        Name of the reference treatment to anchor ``subnet_1``.
    subnet_name_prefix:
        Prefix for returned subnet names.

    Returns
    -------
    Dict keyed by subnet name, each item containing:
    - ``treatments`` (numeric treatment IDs)
    - ``studies`` (study names)
    """

    if treatment_ids is None or treatment_ids.empty:
        return {}

    id_to_label = {int(row.Number): str(row.Label) for row in treatment_ids.itertuples()}

    # Determine whether study data appears numeric (already replaced treatment ids)
    # or still uses treatment labels.
    all_observed = [value for value in find_all_treatments(data) if pd.notna(value)]
    if not all_observed:
        return {}

    numeric_observations = []
    for value in all_observed[:50]:
        try:
            int(float(value))
            numeric_observations.append(True)
        except (TypeError, ValueError):
            numeric_observations.append(False)

    data_is_numeric = all(numeric_observations)

    labeled_data = data.copy()
    if data_is_numeric:
        if "T" in labeled_data.columns:
            try:
                labels = pd.to_numeric(labeled_data["T"], errors="coerce")
                labeled_data["T"] = labels.apply(lambda value: id_to_label.get(int(value)) if pd.notna(value) else value)
            except (TypeError, ValueError):
                labeled_data["T"] = pd.Series([pd.NA] * len(labeled_data), index=labeled_data.index)
        else:
            t_cols = [c for c in labeled_data.columns if re.match(r"^T(\.[0-9]+)?$", str(c))]
            for col in t_cols:
                labels = pd.to_numeric(labeled_data[col], errors="coerce")
                labeled_data[col] = labels.apply(lambda value: id_to_label.get(int(value)) if pd.notna(value) else value)

    data_treatments = set(find_all_treatments(labeled_data))
    if not data_treatments:
        return {}

    if reference_treatment_name is None:
        reference_label = str(treatment_ids.iloc[0]["Label"])
    else:
        reference_label = str(reference_treatment_name)
        if reference_label not in data_treatments:
            warnings.warn(
                f"Reference treatment '{reference_label}' cannot be found in the data. "
                f"Using '{treatment_ids.iloc[0]['Label']}' as reference treatment instead"
            )
            reference_label = str(treatment_ids.iloc[0]["Label"])

    if reference_label not in data_treatments:
        observed = [tid for tid, label in id_to_label.items() if label in data_treatments]
        if observed:
            fallback = min(observed)
            fallback_label = str(id_to_label[fallback])
            warnings.warn(
                f"Reference treatment '{treatment_ids.iloc[0]['Label']}' cannot be found in the treatment list. "
                f"Using '{fallback_label}' as reference treatment instead"
            )
            reference_label = fallback_label

    graph = create_graph(labeled_data)
    components = _connected_components(graph)

    treatment_to_component: dict[str, int] = {}
    for comp_index, component in enumerate(components):
        for treatment in component:
            treatment_to_component[treatment] = comp_index

    component_treatments: Dict[int, List[int]] = {}
    for tid, label in id_to_label.items():
        comp = treatment_to_component.get(label)
        if comp is None:
            continue
        component_treatments.setdefault(comp, []).append(tid)

    subnets: dict[str, dict[str, list]] = {}
    reference_found = False
    for comp in sorted(component_treatments):
        treated_ids = component_treatments[comp]
        studies = find_studies_including_treatments(
            labeled_data,
            [id_to_label[tid] for tid in treated_ids],
            all_or_any="any",
        )
        if not studies:
            continue

        has_reference = str(reference_label) in [id_to_label[tid] for tid in treated_ids]
        subnet_index = (
            1
            if has_reference
            else len(subnets) + (2 if not reference_found else 1)
        )
        subnet_name = f"{subnet_name_prefix}{subnet_index}"

        subnets[subnet_name] = {
            "treatments": sorted(treated_ids),
            "studies": studies,
        }
        if has_reference:
            reference_found = True

    return dict(sorted(subnets.items(), key=lambda item: item[0]))


def _graph_has_cycle(graph: Dict[str, set[str]]) -> bool:
    visited: set[str] = set()

    def _dfs(node: str, parent: str | None) -> bool:
        visited.add(node)
        for neighbour in graph.get(node, set()):
            if neighbour == parent:
                continue
            if neighbour in visited:
                return True
            if _dfs(neighbour, node):
                return True
        return False

    for node in graph:
        if node not in visited:
            if _dfs(node, None):
                return True
    return False


def _all_simple_paths(graph: Dict[str, set[str]], start: str, end: str) -> List[List[str]]:
    # DFS over paths; graph is tiny in unit tests
    paths: List[List[str]] = []
    stack: List[tuple[str, List[str], set[str]]] = [(start, [start], {start})]

    while stack:
        node, path, seen = stack.pop()
        for neighbour in sorted(graph.get(node, set())):
            if neighbour == end and len(path) > 1:
                paths.append(path + [neighbour])
                continue
            if neighbour in seen:
                continue
            stack.append((neighbour, path + [neighbour], seen | {neighbour}))

    return paths


def is_nodesplittable(data: pd.DataFrame, treatments: Iterable[str]) -> Dict[str, object]:
    """Identify if a network has splittable nodes.

    Returns a dict with keys ``is_nodesplittable`` and ``reason``.
    """

    graph = create_graph(data)
    if not _graph_has_cycle(graph):
        return {
            "is_nodesplittable": False,
            "reason": "There are no loops in the network.",
        }

    treatment_list = list(treatments)
    if len(treatment_list) == 0:
        return {
            "is_nodesplittable": False,
            "reason": "In all loops, heterogeneity and inconsistency cannot be distinguished.",
        }

    for i, treatment1 in enumerate(treatment_list):
        for treatment2 in treatment_list[i + 1 :]:
            adjacent_to_treatment1 = graph.get(str(treatment1), set())
            if str(treatment2) not in adjacent_to_treatment1:
                continue

            for path in _all_simple_paths(graph, str(treatment1), str(treatment2)):
                if len(path) <= 2:
                    continue
                supporting_studies = []
                for comparison_index in range(len(path) - 1):
                    studies = find_studies_including_treatments(
                        data,
                        [path[comparison_index], path[comparison_index + 1]],
                        "all",
                    )
                    supporting_studies.append(studies)

                closing_studies = find_studies_including_treatments(
                    data,
                    [path[-1], path[0]],
                    "all",
                )
                supporting_studies.append(closing_studies)

                if len(supporting_studies) == len(set(tuple(sorted(studies)) for studies in supporting_studies)):
                    return {"is_nodesplittable": True, "reason": None}

    return {
        "is_nodesplittable": False,
        "reason": "In all loops, heterogeneity and inconsistency cannot be distinguished.",
    }
