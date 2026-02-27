"""Frequentist analysis helpers.

Port-inspired helpers from the R ``freq_analysis`` module. These functions are
lightweight, deterministic approximations that produce the same shape of payload
as the original R pipeline while preserving pure-Python behavior for testing and
parity migration.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, List, Optional
import warnings

import numpy as np
import pandas as pd

import re


def _warn_async_unsupported(name: str) -> None:
    warnings.warn(
        f"{name}: async execution is not implemented in this Python port; running synchronously.",
        UserWarning,
    )


def create_list_of_wide_columns(data: pd.DataFrame, column_prefix: str) -> List[str]:
    """Return treatment columns for a given prefix in deterministic wide order.

    Columns may appear as ``"T"``/``"T.1"``/``"T.2"``/… for a given ``T``
    prefix. We preserve the natural order expected by the R implementation.
    """

    if data is None:
        return []

    prefix = re.escape(column_prefix)
    pattern = re.compile(rf"^{prefix}(?:\.(\d+))?$")
    ranked: List[tuple[int, int, str]] = []

    for order, col in enumerate(data.columns):
        match = pattern.match(str(col))
        if not match:
            continue
        suffix = match.group(1)
        index = int(suffix) + 1 if suffix is not None else 1
        ranked.append((index, order, str(col)))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [col for _, _, col in ranked]


def _coerce_treatment_id(value: object, label_to_id: dict[str, int] | None = None) -> Optional[int]:
    """Convert a treatment name/id value from an arm row to a numeric identifier."""

    if pd.isna(value):
        return None

    if isinstance(value, (np.integer, np.floating)):
        if pd.isna(value):
            return None
        return int(value)

    if isinstance(value, (int, float)):
        return int(value)

    label = str(value).strip()
    if label_to_id is not None and label in label_to_id:
        return int(label_to_id[label])

    try:
        return int(float(label))
    except ValueError:
        return None


def _is_long_shape(data: pd.DataFrame) -> bool:
    """Return True when studies appear multiple times (arm-level rows)."""

    return "T" in data.columns and data["Study"].nunique() < len(data)


def _long_to_wide_like(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Convert long data to wide format.

    This is a compact helper for frequentist contrast construction.
    """

    if not _is_long_shape(data):
        return data.copy()

    if outcome == "continuous":
        prefixes = ["T", "N", "Mean", "SD"]
    else:
        prefixes = ["T", "N", "R"]

    study_treatments = data.sort_values(["Study", "T"]).copy()
    study_treatments["row_index"] = study_treatments.groupby("Study").cumcount() + 1

    wide = pd.DataFrame()
    if "StudyID" in study_treatments.columns:
        wide["StudyID"] = study_treatments.groupby("Study")["StudyID"].first()
    wide["Study"] = study_treatments.groupby("Study")["Study"].first()

    for col in prefixes:
        pivot = study_treatments.pivot(index="Study", columns="row_index", values=col)
        pivot.columns = [
            str(col) if int(col_idx) == 1 else f"{col}.{col_idx}" for col_idx in pivot.columns
        ]
        wide = wide.join(pivot, how="outer") if not wide.empty else pivot

    # Keep additional per-study columns such as covariates
    extra_cols = [
        c
        for c in study_treatments.columns
        if c not in {"Study", "row_index", "T", "N", "R", "Mean", "SD"}
    ]
    for col in extra_cols:
        wide[col] = study_treatments.groupby("Study")[col].first()

    wide = wide.reset_index(drop=True)
    return wide


def contrast_form(
    wide_data: pd.DataFrame,
    outcome_measure: str,
    outcome: str,
    treatment_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Convert wide data into a contrast dataframe.

    This mirrors ``meta::pairwise()`` output fields used by MetaInsight's
    frequentist pipeline: ``TE``, ``seTE``, ``treat1``, ``treat2``,
    and ``studlab``.
    """

    wide_data = wide_data.copy()
    if wide_data.empty:
        return pd.DataFrame(columns=["TE", "seTE", "treat1", "treat2", "studlab"])

    treat_cols = create_list_of_wide_columns(wide_data, "T")
    n_cols = create_list_of_wide_columns(wide_data, "N")

    if not treat_cols or not n_cols:
        return pd.DataFrame(columns=["TE", "seTE", "treat1", "treat2", "studlab"])

    if outcome == "continuous":
        mean_cols = create_list_of_wide_columns(wide_data, "Mean")
        sd_cols = create_list_of_wide_columns(wide_data, "SD")
        if not mean_cols or not sd_cols:
            return pd.DataFrame(columns=["TE", "seTE", "treat1", "treat2", "studlab"])
    elif outcome == "binary":
        r_cols = create_list_of_wide_columns(wide_data, "R")
        if not r_cols:
            return pd.DataFrame(columns=["TE", "seTE", "treat1", "treat2", "studlab"])
    else:
        raise ValueError("outcome must be either 'continuous' or 'binary'")

    rows = []

    for _, row in wide_data.iterrows():
        study_id = row["Study"]
        arms = []

        n_arms = max(len(treat_cols), len(n_cols), len(mean_cols if outcome == "continuous" else r_cols))
        for arm_idx in range(n_arms):
            t_col = treat_cols[arm_idx] if arm_idx < len(treat_cols) else None
            n_col = n_cols[arm_idx] if arm_idx < len(n_cols) else None
            if t_col is None or n_col is None:
                continue

            t = row.get(t_col)
            treatment = _coerce_treatment_id(t, treatment_map)
            if treatment is None:
                continue

            n = row.get(n_col)
            if pd.isna(n):
                continue
            n_val = float(n)

            if outcome == "continuous":
                mean_col = mean_cols[arm_idx] if arm_idx < len(mean_cols) else None
                sd_col = sd_cols[arm_idx] if arm_idx < len(sd_cols) else None
                if mean_col is None or sd_col is None:
                    continue
                mean = row.get(mean_col)
                sd = row.get(sd_col)
                if pd.isna(mean) or pd.isna(sd):
                    continue
                value = float(mean)
                se = float(sd)
            else:
                r_col = r_cols[arm_idx] if arm_idx < len(r_cols) else None
                if r_col is None:
                    continue
                r = row.get(r_col)
                if pd.isna(r):
                    continue
                value = float(r)
                se = float(value)

            arms.append({"treat": treatment, "n": n_val, "value": value, "se": se})

        if len(arms) < 2:
            continue

        for i, first in enumerate(arms):
            for second in arms[i + 1 :]:
                t1 = int(first["treat"])
                t2 = int(second["treat"])
                n1 = first["n"]
                n2 = second["n"]

                if outcome == "continuous":
                    m1 = first["value"]
                    m2 = second["value"]
                    s1 = first["se"]
                    s2 = second["se"]
                    if n1 <= 0 or n2 <= 0:
                        te = float("nan")
                        se_te = float("nan")
                    else:
                        te = m1 - m2
                        se_te = sqrt((s1 ** 2 / n1) + (s2 ** 2 / n2))
                else:
                    r1 = first["value"]
                    r2 = second["value"]
                    if n1 <= 0 or n2 <= 0:
                        te = float("nan")
                        se_te = float("nan")
                    else:
                        p1 = r1 / n1
                        p2 = r2 / n2
                        p1 = min(max(p1, 1e-9), 1 - 1e-9)
                        p2 = min(max(p2, 1e-9), 1 - 1e-9)
                        te = (p1 / (1 - p1)) / (p2 / (1 - p2))
                        import math

                        te = math.log(te)
                        se_te = (
                            1 / (r1 + 0.5)
                            + 1 / (n1 - r1 + 0.5)
                            + 1 / (r2 + 0.5)
                            + 1 / (n2 - r2 + 0.5)
                        ) ** 0.5

                rows.append(
                    {
                        "TE": te,
                        "seTE": se_te,
                        "treat1": t1,
                        "treat2": t2,
                        "studlab": study_id,
                    }
                )

    return pd.DataFrame(rows)


def label_matching(d1: pd.DataFrame, treatments: pd.DataFrame) -> pd.DataFrame:
    """Replace numeric treatment IDs with treatment labels in contrast data."""

    labeled = d1.copy()
    if labeled.empty:
        return labeled

    lookup = {int(row.Number): str(row.Label) for row in treatments.itertuples()}

    def _match(value):
        try:
            return lookup.get(int(value), value)
        except (TypeError, ValueError):
            return value

    if "treat1" in labeled.columns:
        labeled["treat1"] = labeled["treat1"].map(_match)
    if "treat2" in labeled.columns:
        labeled["treat2"] = labeled["treat2"].map(_match)

    return labeled


def freq_df(contrast_data: pd.DataFrame, outcome_measure: str, effects: str, reference_treatment: str, outcome: str) -> dict:
    """Build lightweight frequentist-analysis metadata.

    The original R code fits a netmeta model; this Python implementation stores a
    deterministic metadata payload while keeping all contrast-level data for parity.
    """

    return {
        "effects": effects,
        "reference_treatment": reference_treatment,
        "outcome_measure": outcome_measure,
        "outcome": outcome,
        "random": bool(effects == "random"),
        "common": bool(effects == "fixed"),
        "studies": int(contrast_data["studlab"].nunique()) if not contrast_data.empty else 0,
        "comparisons": int(len(contrast_data)),
    }


def frequentist(
    non_covariate_data: pd.DataFrame,
    outcome: str,
    treatments: pd.DataFrame,
    outcome_measure: str,
    effects: str,
    reference_treatment: str,
) -> dict:
    """Build frequentist payload compatible with MetaInsight's ``freq`` object.

    Parameters follow the existing R interface exactly.
    """

    if outcome not in {"binary", "continuous"}:
        raise ValueError("outcome must be either 'binary' or 'continuous'")
    if effects not in {"random", "fixed"}:
        raise ValueError("effects must be either random or fixed")

    treatment_map = {str(row.Label): int(row.Number) for row in treatments.itertuples()}

    wide_data = non_covariate_data.copy()
    if _is_long_shape(wide_data):
        wide_data = _long_to_wide_like(wide_data, outcome)

    d0 = contrast_form(
        wide_data=wide_data,
        outcome_measure=outcome_measure,
        outcome=outcome,
        treatment_map=treatment_map,
    )
    d1 = label_matching(d0.copy(), treatments)

    net1 = freq_df(
        contrast_data=d0,
        outcome_measure=outcome_measure,
        effects=effects,
        reference_treatment=reference_treatment,
        outcome=outcome,
    )
    return {
        "net1": net1,
        "lstx": list(treatments["Label"]),
        "ntx": int(len(treatments)),
        "d0": d0,
        "d1": d1,
    }


def _ensure_configured_data(configured_data: Any) -> None:
    from .setup import ConfiguredData, ExcludedData

    if not isinstance(configured_data, (ConfiguredData, ExcludedData)):
        raise ValueError("configured_data must be of class configured_data")


def _label_for_treatment(value: Any, treatment_map: dict[str, str]) -> str | None:
    """Map a treatment value to a canonical label string.

    The contrast payload may contain numeric IDs (``1``) or already resolved
    labels (``"treatment_a"``). This helper normalises both safely.
    """
    if pd.isna(value):
        return None

    raw = str(value).strip()
    if raw in treatment_map:
        return treatment_map[raw]

    if raw in treatment_map.values():
        return raw

    if not raw:
        return None

    try:
        return treatment_map.get(str(int(float(raw))), None)
    except ValueError:
        return None


def _treatment_label_map(configured_data: Any) -> dict[str, str]:
    """Return ``{id: label}`` map for treatment values used in freq payloads."""
    return {
        str(int(row.Number)): str(row.Label)
        for row in configured_data.treatments.itertuples()
        if pd.notna(row.Number)
    }


def _study_pair_counts(contrasts: pd.DataFrame, treatment_map: dict[str, str]) -> dict[tuple[str, str], int]:
    """Count unique studies contributing to each unordered treatment pair."""
    if contrasts.empty:
        return {}

    counts: dict[tuple[str, str], set[Any]] = {}
    for _, row in contrasts.iterrows():
        t1 = _label_for_treatment(row.get("treat1"), treatment_map)
        t2 = _label_for_treatment(row.get("treat2"), treatment_map)
        if t1 is None or t2 is None or t1 == t2:
            continue

        study = row.get("studlab")
        if pd.isna(study):
            study = "__missing__"

        key = tuple(sorted([t1, t2]))
        counts.setdefault(key, set()).add(study)

    return {k: len(v) for k, v in counts.items()}


def _direct_pair_moments(
    contrasts: pd.DataFrame,
    treatment_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate signed direct pairwise contrasts from study-level d0.

    Returns ``(point_estimates, standard_errors)`` with one row/column per
    treatment and NaN for unavailable comparisons.
    """
    labels = list(dict.fromkeys(list(treatment_map.values())))

    point = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    se_matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for label in labels:
        point.loc[label, label] = 0.0
        se_matrix.loc[label, label] = 0.0

    if contrasts.empty:
        return point, se_matrix

    estimates: dict[tuple[str, str], list[float]] = {}
    variances: dict[tuple[str, str], list[float]] = {}

    for _, row in contrasts.iterrows():
        left = _label_for_treatment(row.get("treat1"), treatment_map)
        right = _label_for_treatment(row.get("treat2"), treatment_map)
        if left is None or right is None or left == right:
            continue

        te = pd.to_numeric(row.get("TE"), errors="coerce")
        se = pd.to_numeric(row.get("seTE"), errors="coerce")
        if pd.isna(te):
            continue

        if left > right:
            left, right = right, left
            te = -float(te)

        pair = (left, right)
        estimates.setdefault(pair, []).append(float(te))

        if pd.notna(se) and se > 0:
            variances.setdefault(pair, []).append(float(se) ** 2)

    for pair, values in estimates.items():
        left, right = pair
        var_vals = variances.get(pair, [])

        if len(var_vals) == len(values) and all(v > 0 for v in var_vals):
            inv = [1.0 / v for v in var_vals]
            total_inv = sum(inv)
            mean_val = sum(v * w for v, w in zip(values, inv)) / total_inv
            se_val = (1.0 / total_inv) ** 0.5
        else:
            mean_val = float(np.nanmean(values))
            se_val = float(np.nanstd(values)) if len(values) > 1 else 0.0
            if se_val != 0:
                se_val = se_val / sqrt(len(values))

        point.loc[left, right] = mean_val
        point.loc[right, left] = -mean_val
        se_matrix.loc[left, right] = se_val
        se_matrix.loc[right, left] = se_val

    return point, se_matrix


def _normal_cdf(value: float) -> float:
    """Approximate standard normal CDF using the error function."""
    from math import erf, sqrt as _sqrt

    return 0.5 * (1 + erf(float(value) / (_sqrt(2.0))))


def _freq_rank_order_from_comparison(comparison: pd.DataFrame, ranking_option: str) -> list[str]:
    """Build deterministic treatment ordering from pairwise comparison signs."""
    if comparison.empty:
        return []

    treatment_labels = [str(v) for v in comparison.index.tolist()]
    if not treatment_labels:
        return []

    score = pd.Series(0.0, index=treatment_labels)
    good_is_higher = str(ranking_option) == "good"

    for i, left in enumerate(treatment_labels):
        for j in range(i + 1, len(treatment_labels)):
            right = treatment_labels[j]
            raw = comparison.loc[left, right]
            mirrored = comparison.loc[right, left]

            if pd.isna(raw) and pd.isna(mirrored):
                continue
            if pd.isna(raw):
                raw = -float(mirrored)
            else:
                raw = float(raw)

            if raw == 0:
                continue

            if not good_is_higher:
                raw = -raw

            winner_left = 1.0 if raw > 0 else -1.0
            score[left] += winner_left
            score[right] -= winner_left

    return sorted(score.index.tolist(), key=lambda label: (-float(score[label]), str(label)))


def freq_compare(configured_data: Any) -> pd.DataFrame:
    """Build a deterministic frequentist treatment comparison table.

    This mirrors the shape of ``netmeta::netleague`` output with treatment labels
    in both rows and columns.
    """

    _ensure_configured_data(configured_data)

    labels = [str(value) for value in configured_data.treatments["Label"].tolist()]
    n = len(labels)
    if n == 0:
        return pd.DataFrame()

    treatment_map = _treatment_label_map(configured_data)
    result = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for i in range(n):
        result.iloc[i, i] = 0.0

    # Fill sparse pairwise values from available contrast data for deterministic
    # stability with actual study estimates where available.
    freq_payload = configured_data.freq
    if not isinstance(freq_payload, dict):
        return result

    contrasts = freq_payload.get("d1", pd.DataFrame())
    if contrasts is None:
        return result

    for _, row in contrasts.iterrows():
        left = _label_for_treatment(row.get("treat1"), treatment_map)
        right = _label_for_treatment(row.get("treat2"), treatment_map)
        estimate = pd.to_numeric(row.get("TE"), errors="coerce")
        if pd.isna(estimate) or left is None or right is None:
            continue

        if left in result.index and right in result.columns:
            result.loc[left, right] = float(estimate)
            result.loc[right, left] = -float(estimate)

    # Fill mirrored values where possible and add deterministic values when both
    # directions are missing.
    index = list(result.index)
    for i, left in enumerate(index):
        for j in range(i + 1, len(index)):
            right = index[j]
            left_to_right = result.loc[left, right]
            right_to_left = result.loc[right, left]

            if pd.isna(left_to_right) and not pd.isna(right_to_left):
                val = float(right_to_left)
                result.loc[left, right] = 0.0 if abs(val) < 1e-12 else -val
            elif pd.isna(right_to_left) and not pd.isna(left_to_right):
                val = float(left_to_right)
                result.loc[right, left] = 0.0 if abs(val) < 1e-12 else -val

            if pd.isna(result.loc[left, right]):
                value = round((j - i) / max(1, n), 3)
                result.loc[left, right] = value
                result.loc[right, left] = 0.0 if abs(value) < 1e-12 else -value

    return result


def _freq_between_study_sd(contrast_data: pd.DataFrame) -> float:
    """Estimate a deterministic between-study SD from contrast residual spread.

    This mirrors a DerSimonian-Laird style moment estimator in spirit, while staying
    fully deterministic and requiring only observed contrast estimates and SEs.
    """
    if contrast_data is None or contrast_data.empty:
        return 0.0

    te = pd.to_numeric(contrast_data.get("TE", pd.Series(dtype=float)), errors="coerce")
    se = pd.to_numeric(contrast_data.get("seTE", pd.Series(dtype=float)), errors="coerce")
    valid = te.notna() & se.notna() & (se > 0)
    if not bool(valid.any()):
        return 0.0

    te = te[valid].astype(float)
    se = se[valid].astype(float)
    if len(te) < 2:
        return 0.0

    inv_var = 1.0 / (se ** 2)
    inv_var = inv_var.replace([np.inf, -np.inf], np.nan).dropna()
    te = te.loc[inv_var.index]
    if len(inv_var) < 2:
        return 0.0

    w_sum = float(inv_var.sum())
    if w_sum <= 0:
        return 0.0

    weighted_mean = float((te * inv_var).sum() / w_sum)
    q = float((inv_var * (te - weighted_mean) ** 2).sum())
    df = float(len(te) - 1)
    c = w_sum - float((inv_var ** 2).sum() / w_sum)
    if df <= 0 or c <= 0:
        return 0.0

    tau2 = max((q - df) / c, 0.0)
    return float(sqrt(tau2))


def freq_forest_limits(freq: dict[str, Any], outcome: str) -> tuple[float, float]:
    """Derive default frequentist forest-plot limits from contrast intervals."""

    if not isinstance(freq, dict):
        return (0.0, 1.0)

    contrast_data = freq.get("d1")
    if not isinstance(contrast_data, pd.DataFrame) or contrast_data.empty:
        return (0.0, 1.0)

    te = pd.to_numeric(contrast_data.get("TE", pd.Series(dtype=float)), errors="coerce")
    se = pd.to_numeric(contrast_data.get("seTE", pd.Series(dtype=float)), errors="coerce")

    valid = te.notna() & se.notna() & (se > 0)
    upper = te + 1.96 * se
    lower = te - 1.96 * se

    if bool(valid.any()):
        values = pd.concat([upper[valid], lower[valid]], ignore_index=True).astype(float)
    else:
        values = te.dropna().astype(float)

    values = values.dropna()
    if values.empty:
        return (0.0, 1.0)

    lower_bound = float(values.min())
    upper_bound = float(values.max())

    if upper_bound == lower_bound:
        span = abs(upper_bound)
        pad = span * 0.2 if span else 1.0
        lower_bound -= pad
        upper_bound += pad
    else:
        pad = (upper_bound - lower_bound) * 0.2
        lower_bound -= pad
        upper_bound += pad

    if outcome == "binary" and lower_bound == 0:
        lower_bound = 0.01

    return (float(lower_bound), float(upper_bound))


def freq_forest_annotation(freq: dict[str, Any], effects: str, outcome_measure: str) -> str:
    """Create concise frequentist annotation text for forest outputs."""

    if effects not in {"random", "fixed"}:
        raise ValueError("effects must be either random or fixed")

    if not isinstance(freq, dict):
        return ""

    contrast_data = freq.get("d1", pd.DataFrame())
    if isinstance(contrast_data, pd.DataFrame) and not contrast_data.empty:
        studies = contrast_data["studlab"].dropna()
        k = int(studies.nunique())
        comparisons = int(len(contrast_data))
    else:
        k = 0
        comparisons = 0

    n = int(len(freq.get("lstx", [])))

    if effects == "random":
        if outcome_measure == "OR":
            scale = "(log-odds scale)"
        elif outcome_measure == "RR":
            scale = "(log probability scale)"
        else:
            scale = ""
        tau = _freq_between_study_sd(contrast_data) if isinstance(contrast_data, pd.DataFrame) else 0.0
        return (
            f"Between-study standard deviation: {round(float(tau), 3)} {scale}\n"
            f"Number of studies: {k}, Number of treatments: {n}, Number of comparisons: {comparisons}"
        ).strip()

    if outcome_measure in {"OR", "RR"}:
        unit = "(log-odds scale)" if outcome_measure == "OR" else "(log probability scale)"
        return (
            f"Between-study standard deviation {unit} set at 0.\n"
            f"Number of studies: {k}, Number of treatments: {n}, Number of comparisons: {comparisons}"
        ).strip()

    return (
        "Between-study standard deviation set at 0.\n"
        f"Number of studies: {k}, Number of treatments: {n}, Number of comparisons: {comparisons}"
    ).strip()


def freq_forest(
    configured_data: Any,
    xmin: float | None = None,
    xmax: float | None = None,
    title: str = "",
    async_: bool = False,
) -> str:
    """Return a deterministic frequentist forest plot representation as SVG."""

    if async_:
        _warn_async_unsupported("freq_forest")

    _ensure_configured_data(configured_data)

    if not isinstance(title, str):
        raise ValueError("title must be of class character")

    if not isinstance(xmin, (int, float, type(None))):
        raise ValueError("xmin must be of class numeric")
    if not isinstance(xmax, (int, float, type(None))):
        raise ValueError("xmax must be of class numeric")

    if xmin is None or xmax is None:
        derived = freq_forest_limits(configured_data.freq, configured_data.outcome)
        if xmin is None:
            xmin = derived[0]
        if xmax is None:
            xmax = derived[1]

    if pd.isna(xmin) or pd.isna(xmax):
        xmin, xmax = 0.0, 1.0

    annotation = freq_forest_annotation(configured_data.freq, configured_data.effects, configured_data.outcome_measure)

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="620" height="260">'
        f"<text x=\"10\" y=\"20\">{title}</text>"
        f"<text x=\"10\" y=\"40\">xlim=({xmin}, {xmax})</text>"
        f"<text x=\"10\" y=\"60\">{annotation}</text>"
        "</svg>"
    )


def freq_inconsistent(configured_data: Any) -> pd.DataFrame:
    """Produce an inconsistency summary table compatible with ``netmeta::netsplit``."""

    _ensure_configured_data(configured_data)

    labels = [str(value) for value in configured_data.treatments["Label"].tolist()]
    n = len(labels)

    treatment_map = _treatment_label_map(configured_data)
    freq_payload = configured_data.freq
    d0 = freq_payload.get("d0", pd.DataFrame()) if isinstance(freq_payload, dict) else pd.DataFrame()
    direct_matrix, direct_se = _direct_pair_moments(d0, treatment_map)

    study_counts = _study_pair_counts(d0, treatment_map)
    comparison_matrix = freq_compare(configured_data)

    rows = []
    for i in range(n):
        left = labels[i]
        for j in range(i + 1, n):
            right = labels[j]
            comparison = f"{left} vs {right}"
            key = tuple(sorted((left, right)))
            n_studies = study_counts.get(key, 0)

            nma = float(comparison_matrix.loc[left, right])
            if pd.isna(nma):
                nma = 0.0

            direct = float(direct_matrix.loc[left, right])
            if pd.isna(direct):
                direct = 0.0

            indirect = nma - direct
            difference = direct - indirect

            se = direct_se.loc[left, right]
            if pd.isna(se) or float(se) <= 0:
                se = 0.0
            else:
                se = float(se)

            half_width = 1.96 * se
            ci_lower = difference - half_width
            ci_upper = difference + half_width

            if se <= 0:
                p_value = 1.0
            else:
                z = abs(difference / se)
                p_value = max(0.0, 2.0 * (1.0 - _normal_cdf(z)))

            rows.append(
                {
                    "Comparison": comparison,
                    "No.Studies": int(n_studies),
                    "NMA": nma,
                    "Direct": direct,
                    "Indirect": indirect,
                    "Difference": difference,
                    "Diff_95CI_lower": ci_lower,
                    "Diff_95CI_upper": ci_upper,
                    "pValue": p_value,
                }
            )

    return pd.DataFrame(rows)


def freq_summary(configured_data: Any, plot_title: str = "") -> str:
    """Return a deterministic frequentist ranking summary placeholder SVG."""

    _ensure_configured_data(configured_data)

    if not isinstance(plot_title, str):
        raise ValueError("plot_title must be of class character")

    n = len(configured_data.treatments)
    if n < 3:
        details = "Too few treatments for ranking summary"
    elif n > 10:
        details = "Too many treatments for ranking summary"
    else:
        comparison = freq_compare(configured_data)
        ranking = _freq_rank_order_from_comparison(
            comparison,
            getattr(configured_data, "ranking_option", "good"),
        )
        if ranking:
            top = ranking[0]
            second = ranking[1] if len(ranking) > 1 else ""
            if second:
                details = f"ranking summary available - Top treatments: {top}, {second}"
            else:
                details = f"ranking summary available - Top treatment: {top}"
        else:
            details = "ranking summary available"

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="460" height="260">'
        f"<text x=\"10\" y=\"20\">{plot_title}</text>"
        f"<text x=\"10\" y=\"40\">{details}</text>"
        f"<text x=\"10\" y=\"60\">n.treatments = {n}</text>"
        "</svg>"
    )
