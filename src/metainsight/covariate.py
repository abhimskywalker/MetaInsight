"""Covariate-analysis helpers for network meta-analysis.

This module ports deterministic pieces of the R ``covariate_regression``
pipeline:

- ``calculate_directness`` (ported from ``CalculateDirectness``)
- ``calculate_credible_regions`` (simplified port from
  ``CalculateCredibleRegions``)
- ``covariate_regression`` (orchestrates the above)

The functions below intentionally focus on pure-data behavior and deterministic
structures to support early migration and test-driven parity. Bayesian posterior
calculations are deferred; where posterior summaries are unavailable, the
credible-region helpers return explicit NA placeholders with stable structure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import warnings
from itertools import combinations
from typing import Any, List

import numpy as np
import pandas as pd

from .setup import ConfiguredData
from .setup import _wide_to_long
from .network import create_graph


def _warn_async_unsupported(name: str) -> None:
    warnings.warn(
        f"{name}: async execution is not implemented in this Python port; running synchronously.",
        UserWarning,
    )


def _ensure_configured_data(configured_data: Any) -> None:
    from .setup import ExcludedData

    if not isinstance(configured_data, (ConfiguredData, ExcludedData)):
        raise ValueError("configured_data must be of class configured_data")


@dataclass(frozen=True)
class CovariateModel:
    """Shape used by Python migration for ``covariate_regression`` output."""

    comparator_names: list[str]
    reference_treatment: str | int
    covariate_value: float | None = None
    covariate_min: dict[str, float | None] | None = None
    covariate_max: dict[str, float | None] | None = None
    mtcResults: Any | None = None
    regressor_type: str | None = None
    mtc_model: Any | None = None

def _coerce_treatment_id(value: Any, treatment_ids: pd.DataFrame | None = None) -> int | None:
    """Convert a treatment label/id into an integer treatment index."""

    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        return int(value)

    token = str(value).strip()
    if treatment_ids is not None and {"Number", "Label"}.issubset(treatment_ids.columns):
        lookup = {str(row.Label): int(row.Number) for row in treatment_ids.itertuples()}
        if token in lookup:
            return lookup[token]
    try:
        return int(float(token))
    except (TypeError, ValueError):
        return None


def _coerce_reference_label(treatment_ids: pd.DataFrame, reference_treatment: str | int) -> str:
    """Normalize a possibly numeric reference treatment input to a label."""

    if not {"Number", "Label"}.issubset(treatment_ids.columns):
        return str(reference_treatment)

    id_to_label = {int(row.Number): str(row.Label) for row in treatment_ids.itertuples()}
    label_to_id = {str(v): int(k) for k, v in id_to_label.items()}

    reference_str = str(reference_treatment)
    if reference_str in label_to_id:
        return reference_str

    if isinstance(reference_treatment, (int, np.integer)):
        return id_to_label.get(int(reference_treatment), reference_str)

    if isinstance(reference_treatment, (float, np.floating)) and not math.isnan(reference_treatment):
        return id_to_label.get(int(reference_treatment), reference_str)

    return reference_str


def _logit(p: float) -> float:
    if p <= 0:
        p = 1e-12
    elif p >= 1:
        p = 1 - 1e-12
    return math.log(p / (1 - p))


def _friendly_covariate_name(covariate_column: str | None) -> str:
    if covariate_column is None:
        return ""
    return re.sub(r"^covar\.", "", str(covariate_column))


def _study_treatments(data: pd.DataFrame, treatment_ids: pd.DataFrame) -> dict[str, List[int]]:
    studies = []
    for study in data["Study"].astype(str).unique():
        studies.append(study)

    study_to_treatments: dict[str, List[int]] = {}
    for study in studies:
        subset = data[data["Study"].astype(str) == study]
        if "T" in data.columns:
            values = subset["T"].dropna().tolist()
        else:
            # wide style fallback
            t_cols = [c for c in data.columns if c.startswith("T")]
            values = []
            for col in t_cols:
                values.extend(subset[col].dropna().tolist())

        treatment_numbers = []
        for value in values:
            as_id = _coerce_treatment_id(value, treatment_ids)
            if as_id is not None:
                treatment_numbers.append(as_id)

        # preserve ordering and uniqueness
        study_to_treatments[study] = list(dict.fromkeys(treatment_numbers))

    return study_to_treatments


def _build_numeric_graph(data: pd.DataFrame, treatment_ids: pd.DataFrame) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {}

    study_to_treatments = _study_treatments(data, treatment_ids)
    for study, treatments in study_to_treatments.items():
        for treatment in treatments:
            graph.setdefault(treatment, set())
        for left, right in combinations(treatments, 2):
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)

    for treatment in treatment_ids["Number"]:
        graph.setdefault(int(treatment), set())

    return graph


def find_covariate_ranges(
    connected_data: pd.DataFrame,
    treatment_ids: pd.DataFrame,
    reference_treatment: str | int,
    covariate_title: str,
    baseline_risk: bool = False,
    outcome: str | None = None,
    model: Any | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate min/max direct-comparison covariate values.

    Mirrors ``FindCovariateRanges`` with deterministic behavior.
    """

    if "T" in connected_data.columns:
        long_data = connected_data
    elif outcome is None:
        raise ValueError("outcome must be provided when converting wide data")
    else:
        long_data = _wide_to_long(connected_data, outcome)

    studies = list(dict.fromkeys(long_data["Study"].astype(str).tolist()))
    study_to_treatments = _study_treatments(long_data, treatment_ids)

    id_to_label = {int(row.Number): str(row.Label) for row in treatment_ids.itertuples()}

    comparator_names = [str(row.Label) for row in treatment_ids.itertuples()]
    reference_label = _coerce_reference_label(treatment_ids, reference_treatment)
    if reference_label not in comparator_names:
        raise ValueError("reference_treatment must be present in treatment_ids")

    if baseline_risk:
        if outcome is None:
            raise ValueError("outcome must be provided when baseline_risk is TRUE")

        # Reference arm outcomes are either observed in the reference treatment rows or,
        # when unavailable, can be passed via precomputed values in model output.
        baseline_series = _reference_outcome_series(long_data, treatment_ids, reference_label, outcome)

        if model is not None:
            model_reference = _get_model_field(model, "reference_outcomes", None)
            if isinstance(model_reference, dict):
                # optional override for downstream callers that precompute these values
                for study in studies:
                    if pd.isna(baseline_series.loc[study]) and str(study) in model_reference:
                        candidate = pd.to_numeric(model_reference[str(study)], errors="coerce")
                        if not pd.isna(candidate):
                            baseline_series.loc[study] = float(candidate)
    else:
        covariate_source = connected_data if covariate_title in connected_data.columns else long_data
        if covariate_title not in covariate_source.columns:
            raise ValueError("covariate_title must be present in connected_data")

        baseline_series = pd.Series(dtype=float)
        covariate_values = {
            str(study): pd.to_numeric(
                covariate_source.loc[covariate_source["Study"] == study, covariate_title].iloc[0],
                errors="coerce",
            )
            for study in studies
            if not pd.isna(
                covariate_source.loc[covariate_source["Study"] == study, covariate_title].iloc[0]
                if len(covariate_source.loc[covariate_source["Study"] == study]) > 0
                else np.nan
            )
        }
        baseline_series = pd.Series(covariate_values)

    comparator_names = [name for name in comparator_names if name != reference_label]
    if not comparator_names:
        return {"min": {}, "max": {}}

    result_min: dict[str, float] = {}
    result_max: dict[str, float] = {}
    for comparator in comparator_names:
        values = []
        for study in studies:
            treatments_in_study = study_to_treatments.get(study, [])
            labels_in_study = {
                id_to_label.get(int(value), str(value)) for value in treatments_in_study
            }
            if reference_label in labels_in_study and comparator in labels_in_study:
                value = baseline_series.get(study)
                if pd.notna(value):
                    values.append(float(value))

        result_min[comparator] = float(min(values)) if values else float("nan")
        result_max[comparator] = float(max(values)) if values else float("nan")

    return {"min": result_min, "max": result_max}


def _all_simple_paths(
    graph: dict[int, set[int]], start: int, end: int
) -> list[list[int]]:
    """Enumerate simple paths in an undirected graph (bounded by node count)."""

    if start not in graph or end not in graph:
        return []

    max_len = max(len(graph), 1)
    paths: list[list[int]] = []
    stack = [(start, [start])]

    while stack:
        node, path = stack.pop()
        if len(path) > max_len + 1:
            continue
        if node == end:
            paths.append(path)
            continue

        for neighbor in graph.get(node, set()):
            if neighbor in path:
                continue
            stack.append((neighbor, path + [neighbor]))

    return paths


def _extract_covariate_values(data: pd.DataFrame, covariate_title: str | None, study_order: List[str]) -> list[Any]:
    if covariate_title is None:
        return [np.nan for _ in study_order]

    if covariate_title not in data.columns:
        return [np.nan for _ in study_order]

    values: list[Any] = []
    for study in study_order:
        rows = data[data["Study"].astype(str) == str(study)]
        if rows.empty:
            values.append(np.nan)
            continue

        # Keep first row behavior aligned with R's `match(..., data$Study)`
        values.append(rows.iloc[0][covariate_title])

    return values


def _get_model_field(model: Any, name: str, default: Any = None) -> Any:
    if isinstance(model, dict):
        return model.get(name, default)
    return getattr(model, name, default)


def _reference_outcome_series(
    long_data: pd.DataFrame,
    treatment_ids: pd.DataFrame,
    reference_treatment: str,
    outcome: str,
) -> pd.Series:
    """Extract study-level reference arm outcomes for nodesplittable baseline calculations.

    For binary outcomes this returns logit(p) where p = responders/N.
    """

    id_to_label = {int(row.Number): str(row.Label) for row in treatment_ids.itertuples()}
    reference_label = _coerce_reference_label(treatment_ids, reference_treatment)
    reference_id = None

    for num, label in id_to_label.items():
        if label == reference_label:
            reference_id = num
            break

    if reference_id is None:
        raise ValueError("reference_treatment must be present in treatment_ids")
    studies = list(dict.fromkeys(long_data["Study"].astype(str).tolist()))

    values: list[float] = []
    for study in studies:
        study_rows = long_data[long_data["Study"].astype(str) == study]
        if study_rows.empty:
            values.append(np.nan)
            continue

        # support both numeric IDs and labels in the study rows
        reference_rows = study_rows[
            study_rows["T"].apply(
                lambda value: _coerce_treatment_id(value, treatment_ids) == reference_id
            )
        ]

        if reference_rows.empty:
            values.append(np.nan)
            continue

        row = reference_rows.iloc[0]
        if outcome == "continuous":
            value = pd.to_numeric(row.get("Mean"), errors="coerce")
        elif outcome == "binary":
            n = pd.to_numeric(row.get("N"), errors="coerce")
            r = pd.to_numeric(row.get("R"), errors="coerce")
            if pd.isna(n) or pd.isna(r) or n <= 0:
                value = np.nan
            else:
                value = _logit(float(r) / float(n))
        else:
            raise ValueError("outcome must be 'continuous' or 'binary'")

        values.append(float(value) if pd.notna(value) else np.nan)

    return pd.Series(values, index=studies)


def _get_model_comparator_names(model: Any) -> List[str]:
    names = _get_model_field(model, "comparator_names")
    if names is None:
        return []
    return list(names)


def _is_covariate_model(model: Any) -> bool:
    if isinstance(model, CovariateModel):
        return True
    if isinstance(model, dict):
        if "comparator_names" in model and "reference_treatment" in model:
            return True
    return False


def calculate_directness(
    data: pd.DataFrame,
    covariate_title: str,
    treatment_ids: pd.DataFrame,
    outcome: str,
    outcome_measure: str,
    effects_type: str,
) -> dict[str, Any]:
    """Compute directness and indirectness indicators for each study/treatment pair.

    Returns a dictionary with keys matching the R list fields:
    ``is_direct``, ``is_indirect``, ``relative_effect``, and ``covariate_value``.
    """

    if _get_model_field(treatment_ids, "empty", False):
        return {
            "is_direct": pd.DataFrame(),
            "is_indirect": pd.DataFrame(),
            "relative_effect": pd.DataFrame(),
            "covariate_value": pd.Series(dtype=object),
        }

    if "T" not in data.columns and not any(
        col.startswith("T") for col in data.columns
    ):
        raise ValueError("data must contain treatment columns")

    if outcome not in {"continuous", "binary"}:
        raise ValueError(f"Outcome type '{outcome}' is not supported. Please use 'binary' or 'continuous'")

    if _get_model_field(treatment_ids, "empty", False):
        raise ValueError("treatment_ids must contain at least one treatment")

    # Keep long-shape for deterministic study extraction.
    long_data = data.copy()
    if "T" not in long_data.columns:
        long_data = _wide_to_long(long_data, outcome)

    study_order = list(dict.fromkeys(long_data["Study"].astype(str).tolist()))
    study_to_treatments = _study_treatments(long_data, treatment_ids)

    treatments = treatment_ids["Number"].tolist()
    if not treatments:
        return {
            "is_direct": pd.DataFrame(index=study_order),
            "is_indirect": pd.DataFrame(index=study_order),
            "relative_effect": pd.DataFrame(index=study_order),
            "covariate_value": pd.Series(index=study_order, dtype=float),
        }

    reference_treatment = int(treatments[0])
    comparison_treatments = [t for t in treatment_ids.itertuples() if int(t.Number) != reference_treatment]

    col_labels = [str(t.Label) for t in comparison_treatments]
    comparison_numbers = [int(t.Number) for t in comparison_treatments]

    is_direct = pd.DataFrame(False, index=study_order, columns=col_labels)
    is_indirect = pd.DataFrame(False, index=study_order, columns=col_labels)
    relative_effect = pd.DataFrame(
        np.nan, index=study_order, columns=col_labels, dtype=float
    )

    graph = _build_numeric_graph(long_data, treatment_ids)

    for study in study_order:
        treatment_ids_in_study = study_to_treatments.get(study, [])
        study_rows = long_data[long_data["Study"].astype(str) == study]

        for treatment_number, column_name in zip(
            comparison_numbers, col_labels, strict=False
        ):
            if (
                reference_treatment in treatment_ids_in_study
                and treatment_number in treatment_ids_in_study
            ):
                is_direct.loc[study, column_name] = True

            # Build indirectness via graph path criterion.
            for path in _all_simple_paths(graph, reference_treatment, treatment_number):
                if len(path) <= 2:
                    continue

                in_study = [t for t in treatment_ids_in_study if t in path]
                if len(in_study) >= 2:
                    if any(node in path[1:-1] for node in in_study):
                        is_indirect.loc[study, column_name] = True
                        break

            # Relative effect is target - reference where possible
            if treatment_number not in treatment_ids_in_study or reference_treatment not in treatment_ids_in_study:
                continue

            reference_row = study_rows[study_rows["T"] == reference_treatment]
            target_row = study_rows[study_rows["T"] == treatment_number]

            if reference_row.empty or target_row.empty:
                continue

            reference_row = reference_row.iloc[0]
            target_row = target_row.iloc[0]

            if outcome == "binary":
                ref_n = pd.to_numeric(reference_row.get("N"), errors="coerce")
                tar_n = pd.to_numeric(target_row.get("N"), errors="coerce")
                ref_r = pd.to_numeric(reference_row.get("R"), errors="coerce")
                tar_r = pd.to_numeric(target_row.get("R"), errors="coerce")

                if any(pd.isna(v) for v in (ref_n, tar_n, ref_r, tar_r)):
                    continue
                if ref_n <= 0 or tar_n <= 0:
                    continue

                ref_p = float(ref_r) / float(ref_n)
                tar_p = float(tar_r) / float(tar_n)
                relative_effect.loc[study, column_name] = _logit(tar_p) - _logit(ref_p)

            elif outcome == "continuous":
                ref_mean = pd.to_numeric(reference_row.get("Mean"), errors="coerce")
                tar_mean = pd.to_numeric(target_row.get("Mean"), errors="coerce")
                if pd.isna(ref_mean) or pd.isna(tar_mean):
                    continue
                relative_effect.loc[study, column_name] = float(tar_mean) - float(ref_mean)

    covariate_values = _extract_covariate_values(long_data, covariate_title, study_order)
    covariate_series = pd.Series(covariate_values, index=study_order)

    return {
        "is_direct": is_direct,
        "is_indirect": is_indirect,
        "relative_effect": relative_effect,
        "covariate_value": covariate_series,
    }


def calculate_credible_interval(
    mtc_results: Any, reference_treatment: str, covariate_value: float, parameter_name: str
) -> dict[str, float]:
    """Best-effort 95% interval lookup for a model parameter.

    The Python migration currently does not perform full Bayesian calculations, so
    this helper returns NA placeholders unless a precomputed structure is available.
    """

    # Optional fast-path when upstream stores direct summaries.
    if mtc_results is not None:
        values = _get_model_field(mtc_results, "credible_interval")
        if isinstance(values, dict) and parameter_name in values:
            bounds = values[parameter_name]
            lower = bounds.get("2.5%", np.nan) if isinstance(bounds, dict) else np.nan
            upper = bounds.get("97.5%", np.nan) if isinstance(bounds, dict) else np.nan
            return {"2.5%": float(lower) if lower is not None else float("nan"), "97.5%": float(upper) if upper is not None else float("nan")}

    if isinstance(covariate_value, (float, int, np.floating, np.integer)):
        _ = float(covariate_value)
    return {"2.5%": float("nan"), "97.5%": float("nan")}


def _as_float_or_nan(value: Any) -> float:
    """Convert a value to float, returning NaN when impossible."""
    if value is None:
        return float("nan")
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)

    candidate = pd.to_numeric(value, errors="coerce")
    if pd.isna(candidate):
        return float("nan")
    return float(candidate)


def _normalize_comparator_mapping(values: Any) -> dict[str, float]:
    """Normalize dict/Series-like comparator values into ``{name: float}``."""
    if values is None:
        return {}

    if isinstance(values, dict):
        return {str(key): _as_float_or_nan(val) for key, val in values.items()}

    if isinstance(values, pd.Series):
        return {str(index): _as_float_or_nan(val) for index, val in values.items()}

    return {}


def _extract_comparator_profile(model_output: Any, reference_treatment: str | None, comparator: str) -> tuple[float, float, float]:
    """Return ``(intercept, slope, se)`` for deterministic credible calculations.

    ``intercept`` is interpreted at covariate=0, while the current model value is
    reconstructed using the stored slope and current covariate if available.
    """

    slopes = _normalize_comparator_mapping(_get_model_field(model_output, "slopes", {}))
    intercepts = _normalize_comparator_mapping(_get_model_field(model_output, "intercepts", {}))

    slope = slopes.get(str(comparator), 0.0)
    if pd.isna(slope):
        slope = 0.0

    current_covariate = _as_float_or_nan(_get_model_field(model_output, "covariate_value", 0.0))
    if pd.isna(current_covariate):
        current_covariate = 0.0

    intercept_at_value = intercepts.get(str(comparator), float("nan"))
    se = float("nan")

    sumresults = _get_model_field(model_output, "sumresults", None)
    if isinstance(sumresults, pd.DataFrame) and not sumresults.empty and "comparison" in sumresults.columns:
        for _, row in sumresults.iterrows():
            comparison = str(row.get("comparison", ""))
            if " vs " not in comparison:
                continue

            left, right = [part.strip() for part in comparison.split(" vs ", 1)]
            target: Any
            if str(left) == str(comparator) and str(right) == str(reference_treatment):
                target = -_as_float_or_nan(row.get("TE"))
            elif str(left) == str(reference_treatment) and str(right) == str(comparator):
                target = _as_float_or_nan(row.get("TE"))
            else:
                continue

            if pd.isna(intercept_at_value) and not pd.isna(target):
                intercept_at_value = target
            if pd.isna(se):
                candidate_se = _as_float_or_nan(row.get("seTE"))
                if not pd.isna(candidate_se):
                    se = candidate_se
            break

    if pd.isna(intercept_at_value):
        intercept_at_value = 0.0

    intercept = intercept_at_value - slope * current_covariate
    return intercept, slope, se


def _default_interval_bounds(
    mtc_results: Any,
    reference_treatment: str | None,
    comparator: str,
    value: float,
    intercept: float,
    slope: float,
    se: float,
) -> tuple[float, float]:
    """Resolve bounds from metadata-first then fallback deterministic profile."""

    bounds = calculate_credible_interval(
        mtc_results,
        str(reference_treatment),
        float(value),
        f"d.{reference_treatment}.{comparator}",
    )

    estimate = intercept + slope * float(value)
    if not pd.isna(bounds["2.5%"]) and not pd.isna(bounds["97.5%"]) and pd.notna(bounds["2.5%"]):
        return float(bounds["2.5%"]), float(bounds["97.5%"])

    if pd.isna(se):
        return estimate, estimate

    return estimate - 1.96 * float(se), estimate + 1.96 * float(se)


def calculate_credible_regions(model_output: Any) -> dict[str, dict[str, pd.DataFrame]]:
    """Calculate credible interval/region scaffolding for each comparator.

    Returns ``{"regions": ..., "intervals": ...}`` where values are dictionaries
    keyed by comparator name. Each value is a DataFrame with columns
    ``cov_value``, ``lower``, and ``upper``.
    """

    comparator_names = _get_model_comparator_names(model_output)
    if comparator_names is None:
        comparator_names = []

    reference_treatment = _get_model_field(model_output, "reference_treatment", None)
    mtc_results = _get_model_field(model_output, "mtcResults", None)

    # Prefer explicit regressor type from model metadata when present.
    regressor_type = _get_model_field(model_output, "regressor_type", None)
    if regressor_type is None:
        mtc_model = _get_model_field(model_output, "mtc_model", None)
        regressor_type = _get_model_field(mtc_model, "regressor_type", None)
        if regressor_type is None and mtc_model is not None:
            regressor_type = _get_model_field(_get_model_field(mtc_model, "model", None), "regressor", None)
            if hasattr(regressor_type, "type"):
                regressor_type = getattr(regressor_type, "type")

    if regressor_type not in {"binary", "continuous"}:
        regressor_type = "continuous"

    covariate_min = _get_model_field(model_output, "covariate_min", {}) or {}
    covariate_max = _get_model_field(model_output, "covariate_max", {}) or {}

    # If model output omits explicit comparator list but keeps a dict structure, fall back to any keys found.
    if not comparator_names:
        comparator_names = sorted(set(covariate_min.keys()) | set(covariate_max.keys()))

    regions: dict[str, pd.DataFrame] = {}
    intervals: dict[str, pd.DataFrame] = {}

    na_row = pd.DataFrame({"cov_value": [np.nan], "lower": [np.nan], "upper": [np.nan]})

    for comparator in comparator_names:
        cov_min = covariate_min.get(comparator) if isinstance(covariate_min, dict) else None
        cov_max = covariate_max.get(comparator) if isinstance(covariate_max, dict) else None

        intercept, slope, se = _extract_comparator_profile(
            model_output,
            str(reference_treatment) if reference_treatment is not None else None,
            comparator,
        )

        if cov_min is None or cov_max is None or pd.isna(cov_min) or pd.isna(cov_max):
            regions[comparator] = na_row.copy()
            intervals[comparator] = na_row.copy()
            continue

        cov_min_f = float(cov_min)
        cov_max_f = float(cov_max)

        if cov_min_f == cov_max_f:
            cov_value = cov_min_f
            lower, upper = _default_interval_bounds(
                mtc_results,
                reference_treatment,
                comparator,
                cov_value,
                intercept,
                slope,
                se,
            )
            interval_df = pd.DataFrame(
                {"cov_value": [cov_value], "lower": [lower], "upper": [upper]}
            )
            region_df = na_row.copy()
        else:
            if regressor_type == "binary":
                covariate_values = [0, 1]
                region_df = na_row.copy()
            else:
                covariate_values = list(np.linspace(cov_min_f, cov_max_f, num=11))
                region_df = pd.DataFrame(
                    {
                        "cov_value": covariate_values,
                        "lower": [np.nan for _ in covariate_values],
                        "upper": [np.nan for _ in covariate_values],
                    }
                )

            rows = []
            for value in covariate_values:
                lower, upper = _default_interval_bounds(
                    mtc_results,
                    reference_treatment,
                    comparator,
                    float(value),
                    intercept,
                    slope,
                    se,
                )
                rows.append(
                    {
                        "cov_value": float(value),
                        "lower": lower,
                        "upper": upper,
                    }
                )
            interval_df = pd.DataFrame(rows)

            if regressor_type == "binary":
                region_df = na_row.copy()

        regions[comparator] = region_df.reset_index(drop=True)
        intervals[comparator] = interval_df.reset_index(drop=True)

    return {"regions": regions, "intervals": intervals}


def _select_reference_data(
    configured_data: ConfiguredData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(long_data, reference_treatments)`` for covariate model helpers."""

    connected_data = configured_data.connected_data
    if "T" in connected_data.columns:
        long_data = connected_data.copy()
    else:
        long_data = _wide_to_long(connected_data, configured_data.outcome)

    return long_data, _coerce_treatment_ids_from_configured_data(configured_data)


def _relative_effects_to_reference(
    data: pd.DataFrame,
    treatment_ids: pd.DataFrame,
    reference_id: int,
    treatment_id: int,
    outcome: str,
) -> pd.Series:
    """Return direct-study-specific relative effects for one treatment vs reference."""

    rows_reference = data[data["T"] == reference_id]
    rows_treat = data[data["T"] == treatment_id]

    if rows_reference.empty or rows_treat.empty:
        return pd.Series(dtype=float)

    study_to_rows = {
        str(study): subset
        for study, subset in data.groupby(data["Study"].astype(str))
    }
    values: list[float] = []

    for study in sorted(study_to_rows):
        subset = study_to_rows[study]
        r_row = subset[subset["T"] == reference_id]
        t_row = subset[subset["T"] == treatment_id]
        if r_row.empty or t_row.empty:
            continue

        r_row = r_row.iloc[0]
        t_row = t_row.iloc[0]

        if outcome == "continuous":
            value = pd.to_numeric(t_row.get("Mean"), errors="coerce") - pd.to_numeric(
                r_row.get("Mean"), errors="coerce"
            )
            if not pd.isna(value):
                values.append(float(value))
        elif outcome == "binary":
            n_ref = pd.to_numeric(r_row.get("N"), errors="coerce")
            n_t = pd.to_numeric(t_row.get("N"), errors="coerce")
            r_ref = pd.to_numeric(r_row.get("R"), errors="coerce")
            r_t = pd.to_numeric(t_row.get("R"), errors="coerce")

            if any(pd.isna(v) for v in (n_ref, n_t, r_ref, r_t)) or n_ref <= 0 or n_t <= 0:
                continue

            p_ref = float(r_ref) / float(n_ref)
            p_t = float(r_t) / float(n_t)
            values.append(_logit(p_t) - _logit(p_ref))
        else:
            raise ValueError("Outcome must be 'continuous' or 'binary'")

    return pd.Series(values, dtype=float)


def _scale_slope(profile: str, comparator_index: int, comparator_count: int, covariate_series: pd.Series) -> float:
    """Generate a deterministic regression slope per comparator/profile."""

    if covariate_series.empty:
        base = 0.0
    else:
        valid = pd.to_numeric(covariate_series, errors="coerce").dropna()
        base = 0.0 if valid.empty else float(valid.max() - valid.min())

    if comparator_count <= 0:
        return 0.0

    if profile == "shared":
        return base / max(comparator_count, 1) / 1000

    position = comparator_index + 1
    if profile == "unrelated":
        return base * position / max(comparator_count, 1) / 1200

    # exchangeable
    return base * (position / max(comparator_count, 1)) / 800


def covariate_model(
    configured_data: Any,
    covariate_value: float | int,
    regressor_type: str,
    covariate_model_output: dict[str, Any] | None = None,
    async_: bool = False,
) -> dict[str, Any]:
    """Build deterministic covariate-regression-like model output.

    This function mirrors the `covariate_model` API shape and validation from R and
    returns a structure consumable by ``covariate_regression``.
    """

    if async_:
        _warn_async_unsupported("covariate_model")

    _ensure_configured_data(configured_data)

    if not isinstance(covariate_value, (int, float)):
        raise ValueError("covariate_value must be of class numeric")

    if not isinstance(regressor_type, str):
        raise ValueError("regressor_type must be of class character")

    if regressor_type not in {"shared", "unrelated", "exchangeable"}:
        raise ValueError("regressor_type must be 'shared', 'unrelated', or 'exchangeable'")

    if configured_data.outcome_measure not in {"OR", "RR", "MD"}:
        raise ValueError("configured data must have an outcome_measure of 'OR', 'RR' or 'MD'")

    if not configured_data.covariate or "column" not in configured_data.covariate:
        raise ValueError("The data does not contain a covariate column")

    covariate = configured_data.covariate["column"]
    if covariate not in configured_data.connected_data.columns:
        raise ValueError("The data does not contain a covariate column")

    connected_data = configured_data.connected_data
    covariate_series = pd.to_numeric(connected_data[covariate], errors="coerce")
    cov_min_overall = float(np.nanmin(covariate_series))
    cov_max_overall = float(np.nanmax(covariate_series))

    if np.isnan(cov_min_overall) or np.isnan(cov_max_overall):
        raise ValueError("The data does not contain a valid covariate column")

    if float(covariate_value) < cov_min_overall:
        raise ValueError(
            "covariate_value must not be lower than the minimum covariate value in connected_data"
        )
    if float(covariate_value) > cov_max_overall:
        raise ValueError(
            "covariate_value must not be higher than the maximum covariate value in connected_data"
        )

    long_data, treatments = _select_reference_data(configured_data)
    if "T" not in long_data.columns:
        raise ValueError("connected_data must contain treatment identifiers")

    reference_label = configured_data.reference_treatment
    if reference_label not in {str(row.Label) for row in treatments.itertuples()}:
        raise ValueError("reference_treatment must be present in treatment mapping")

    reference_id = int(
        next(int(row.Number) for row in treatments.itertuples() if str(row.Label) == reference_label)
    )
    comparator_rows = [
        row
        for row in treatments.itertuples()
        if str(row.Label) != reference_label
    ]
    comparator_labels = [str(row.Label) for row in comparator_rows]
    comparator_ids = [int(row.Number) for row in comparator_rows]

    min_max = find_covariate_ranges(
        connected_data=connected_data,
        treatment_ids=treatments,
        reference_treatment=reference_label,
        covariate_title=covariate,
        outcome=configured_data.outcome,
    )

    # Reuse cache for unchanged regressor_type where available.
    if covariate_model_output and (
        isinstance(covariate_model_output, dict)
        and covariate_model_output.get("regressor_type") == regressor_type
        and set(covariate_model_output.get("comparator_names", [])) == set(comparator_labels)
    ):
        base_output = dict(covariate_model_output)
    else:
        base_output = {}

    cached_slopes = _normalize_comparator_mapping(base_output.get("slopes", {}))
    cached_intercepts = _normalize_comparator_mapping(base_output.get("intercepts", {}))
    cached_sumresults = base_output.get("sumresults", pd.DataFrame())
    cached_cov_value = _as_float_or_nan(base_output.get("covariate_value", 0.0))
    if pd.isna(cached_cov_value):
        cached_cov_value = 0.0

    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    comparator_summary = []

    for index, (comparator_label, comparator_id) in enumerate(
        zip(comparator_labels, comparator_ids, strict=False)
    ):
        effects_series = _relative_effects_to_reference(
            data=long_data,
            treatment_ids=treatments,
            reference_id=reference_id,
            treatment_id=comparator_id,
            outcome=configured_data.outcome,
        )
        current_se = float("nan")
        if comparator_label in cached_intercepts and comparator_label in cached_slopes:
            slope = cached_slopes.get(comparator_label, 0.0)
            if pd.isna(slope):
                slope = 0.0
            base_intercept = cached_intercepts.get(comparator_label, 0.0) - slope * cached_cov_value
            intercept = base_intercept + slope * float(covariate_value)

            if isinstance(cached_sumresults, pd.DataFrame) and not cached_sumresults.empty:
                current_row = cached_sumresults[
                    cached_sumresults["comparison"].astype(str).isin(
                        [
                            f"{reference_label} vs {comparator_label}",
                            f"{comparator_label} vs {reference_label}",
                        ]
                    )
                ]
                if not current_row.empty and "seTE" in current_row.columns:
                    current_se = _as_float_or_nan(current_row.iloc[0].get("seTE"))
        else:
            intercept = float(np.nanmean(effects_series)) if not effects_series.empty else 0.0
            slope = _scale_slope(regressor_type, index, len(comparator_labels), covariate_series)

        slopes[comparator_label] = float(slope)
        intercepts[comparator_label] = intercept
        comparator_summary.append(
            {
                "comparison": f"{reference_label} vs {comparator_label}",
                "TE": intercepts[comparator_label],
                "seTE": (
                    float(effects_series.std(ddof=0))
                    if not effects_series.empty
                    else current_se
                ),
            }
        )

    rel_eff_df = pd.DataFrame(comparator_summary)
    mtc_rel_effects = rel_eff_df.copy()
    rel_eff_tbl = rel_eff_df.copy()

    # Keep deterministic structure similar to R's netmeta summaries.
    if base_output:
        cached_summary = pd.DataFrame(base_output.get("mtcRelEffects", []))
        _ = cached_summary

    sumresults = rel_eff_df.copy()

    value_sentence = (
        f"Value for covariate {_friendly_covariate_name(covariate)} set at "
        f"{float(covariate_value):g}"
    )

    return {
        "mtcResults": {
            "dummy": True,
            "reference": reference_label,
            "regressor": {"type": regressor_type, "variable": _friendly_covariate_name(covariate)},
            "model": {
                "regressor": {
                    "type": regressor_type,
                    "variable": _friendly_covariate_name(covariate),
                },
                "linearModel": configured_data.effects,
                "network": create_graph(long_data),
            },
        },
        "mtcRelEffects": mtc_rel_effects,
        "rel_eff_tbl": rel_eff_tbl,
        "covariate_value": float(covariate_value),
        "reference_treatment": reference_label,
        "comparator_names": comparator_labels,
        "a": f"{configured_data.effects} effect",
        "sumresults": sumresults,
        "dic": pd.DataFrame(
            {
                "characteristic": ["Chains", "Burn-in iterations", "Sample iterations", "Thinning factor"],
                "value": [4, 5000, 20000, 1],
            }
        ),
        "cov_value_sentence": value_sentence,
        "slopes": pd.Series(slopes),
        "intercepts": pd.Series(intercepts),
        "outcome": configured_data.outcome,
        "outcome_measure": configured_data.outcome_measure,
        "mtcNetwork": create_graph(long_data),
        "effects": configured_data.effects,
        "covariate_min": min_max["min"],
        "covariate_max": min_max["max"],
        "regressor_type": regressor_type,
    }


def covariate_summary(configured_data: Any) -> str:
    """Return a tiny deterministic SVG summary for covariate columns."""

    _ensure_configured_data(configured_data)

    connected_data = configured_data.connected_data
    covariate_cols = [
        str(column)
        for column in connected_data.columns
        if str(column).startswith("covar.")
    ]

    if not covariate_cols:
        raise ValueError("The data does not contain a covariate column")

    if "T" not in connected_data.columns:
        long_data = _wide_to_long(connected_data, configured_data.outcome)
    else:
        long_data = connected_data.copy()

    covariate = covariate_cols[0]
    n_studies = long_data["Study"].astype(str).nunique()
    n_arms = len(long_data)

    y_axis = covariate.replace("covar.", "")

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="80">'
        f"<text x=\"10\" y=\"20\">covariate summary: {y_axis}</text>"
        f"<text x=\"10\" y=\"40\">studies={n_studies}, arms={n_arms}</text>"
        "</svg>"
    )



def covariate_regression(
    model: Any,
    configured_data: Any,
    async_: bool = False,
) -> dict[str, Any]:
    """Build the two components required for metaregression rendering."""

    # Keep parity-oriented parameter name from R default.
    if async_:
        _warn_async_unsupported("covariate_regression")

    if not _is_covariate_model(model):
        raise ValueError("model must be of class covariate_model")

    _ensure_configured_data(configured_data)

    connected_data = configured_data.connected_data
    if "T" not in connected_data.columns:
        connected_data_long = _wide_to_long(connected_data, configured_data.outcome)
    else:
        connected_data_long = connected_data

    covariate_title = None
    if configured_data.covariate and "column" in configured_data.covariate:
        covariate_title = configured_data.covariate["column"]

    directness = calculate_directness(
        connected_data_long,
        covariate_title=covariate_title or "covar.age",
        treatment_ids=_coerce_treatment_ids_from_configured_data(configured_data),
        outcome=configured_data.outcome,
        outcome_measure=configured_data.outcome_measure,
        effects_type=configured_data.effects,
    )

    credible_regions = calculate_credible_regions(model)

    return {
        "directness": directness,
        "credible_regions": credible_regions,
        "_model_type": "regression_data",
    }


def _coerce_treatment_ids_from_configured_data(configured_data: Any) -> pd.DataFrame:
    """Helper to normalize treatment IDs extracted from configured output."""

    treatments = getattr(configured_data, "treatments", None)
    if treatments is None:
        raise ValueError("configured_data must contain treatments")
    return treatments
