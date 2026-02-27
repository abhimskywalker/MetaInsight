"""Baseline risk meta-regression scaffolding.

The functions in this module mirror a subset of the R `baseline_*` surface API,
returning deterministic, pure-Python payloads that are suitable for parity tests
before Bayesian backends are fully ported.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd

from .covariate import (
    calculate_credible_regions,
    calculate_directness,
    find_covariate_ranges,
    _logit,
    _scale_slope,
)
from .network import (
    create_graph,
    find_studies_including_treatments,
    is_nodesplittable,
)
from .setup import ConfiguredData, _wide_to_long


_BASELINE_MARKER = "baseline_model"


def _warn_async_unsupported(name: str) -> None:
    warnings.warn(
        f"{name}: async execution is not implemented in this Python port; running synchronously.",
        UserWarning,
    )


def _coerce_treatment_id(treatments: pd.DataFrame, label: str) -> int | None:
    for row in treatments.itertuples():
        if str(row.Label) == str(label):
            return int(row.Number)
    return None


def _ensure_configured_data(configured_data: Any) -> None:
    from .setup import ExcludedData

    if not isinstance(configured_data, (ConfiguredData, ExcludedData)):
        raise ValueError("configured_data must be of class configured_data")


def _is_baseline_model(model: Any) -> bool:
    if not isinstance(model, dict):
        return False
    return model.get("_model_type") == _BASELINE_MARKER


def _get_model_field(model: Any, name: str, default: Any = None) -> Any:
    """Read model values from either dict-like payloads or small dataclasses."""
    if isinstance(model, dict):
        return model.get(name, default)
    return getattr(model, name, default)


def _is_covariate_model(model: Any) -> bool:
    if not isinstance(model, dict):
        # keep lightweight heuristic for dataclass-based payloads
        return all(
            hasattr(model, field)
            for field in ("covariate_value", "regressor_type", "reference_treatment")
        )
    # keep lightweight heuristic for non-baseline regression payloads
    return "covariate_value" in model and "regressor_type" in model and "reference_treatment" in model


def _is_regression_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return "directness" in payload and "credible_regions" in payload and isinstance(
        payload.get("directness"), dict
    )


def _model_to_dict(model: Any) -> dict[str, Any]:
    """Convert simple dataclass-like models to plain dict access for wrapper helpers."""
    if isinstance(model, dict):
        return model
    if hasattr(model, "__dict__"):
        return dict(vars(model))
    return {}


def _normalise_comparators(comparators: Any) -> list[str]:
    if comparators is None:
        return []
    if isinstance(comparators, str):
        return [comparators]
    return [str(x) for x in list(comparators)]


def _validate_metaregression_inputs(
    model: Any,
    configured_data: ConfiguredData,
    regression_data: Any,
    comparators: list[str],
) -> None:
    if not (_is_baseline_model(model) or _is_covariate_model(model)):
        raise ValueError("model must be an object created by baseline_model() or covariate_model()")

    _ensure_configured_data(configured_data)
    if not _is_regression_payload(regression_data):
        raise ValueError(
            "regression_data must be an object created by baseline_regression() or covariate_regression()"
        )

    if not comparators:
        return

    treatment_labels = [str(t) for t in configured_data.treatments["Label"].tolist()]
    for comp in comparators:
        if str(comp) == str(configured_data.reference_treatment):
            raise ValueError("comparators cannot contain the reference treatment")
        if str(comp) not in treatment_labels:
            raise ValueError("comparators must be present in the configured data")



def _baseline_reference_series(configured_data: ConfiguredData) -> tuple[pd.Series, pd.DataFrame]:
    """Return baseline values by study and long-form data carrying them."""

    connected_data = configured_data.connected_data.copy()
    if "T" not in connected_data.columns:
        connected_data = _wide_to_long(connected_data, configured_data.outcome)

    reference_id = _coerce_treatment_id(
        configured_data.treatments,
        configured_data.reference_treatment,
    )
    if reference_id is None:
        raise ValueError("reference_treatment must be present in the treatment mapping")

    rows = []
    for study_id, study_rows in connected_data.groupby(connected_data["Study"].astype(str)):
        reference_rows = study_rows[study_rows["T"] == reference_id]
        if reference_rows.empty:
            baseline_value = np.nan
        else:
            reference_row = reference_rows.iloc[0]
            if configured_data.outcome == "continuous":
                baseline_value = pd.to_numeric(reference_row.get("Mean"), errors="coerce")
            else:
                n = pd.to_numeric(reference_row.get("N"), errors="coerce")
                r = pd.to_numeric(reference_row.get("R"), errors="coerce")
                if pd.isna(n) or pd.isna(r) or n <= 0:
                    baseline_value = np.nan
                else:
                    baseline_value = _logit(float(r) / float(n))

        n_cov = np.nan if pd.isna(baseline_value) else float(baseline_value)
        for _, row in study_rows.iterrows():
            row = row.copy()
            row["covar.baseline_risk"] = n_cov
            rows.append(row)

    long_data = pd.DataFrame(rows)
    baseline_series = long_data.drop_duplicates(subset=["Study"]).set_index("Study")["covar.baseline_risk"]

    return baseline_series, long_data


def _baseline_effect_rows(
    long_data: pd.DataFrame,
    configured_data: ConfiguredData,
    comparator_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparator in comparator_names:
        comparator_id = next(
            (int(row.Number) for row in configured_data.treatments.itertuples() if str(row.Label) == comparator),
            None,
        )
        if comparator_id is None:
            continue

        effects = []
        for _, study_data in long_data.groupby("Study", sort=False):
            reference_id = _coerce_treatment_id(configured_data.treatments, configured_data.reference_treatment)
            if reference_id is None:
                continue

            reference_rows = study_data[study_data["T"] == reference_id]
            comparator_rows = study_data[study_data["T"] == comparator_id]
            if reference_rows.empty or comparator_rows.empty:
                continue

            ref = reference_rows.iloc[0]
            cmp = comparator_rows.iloc[0]

            if configured_data.outcome == "continuous":
                val = pd.to_numeric(cmp.get("Mean"), errors="coerce") - pd.to_numeric(
                    ref.get("Mean"), errors="coerce"
                )
                if not pd.isna(val):
                    effects.append(float(val))
            else:
                n_ref = pd.to_numeric(ref.get("N"), errors="coerce")
                n_cmp = pd.to_numeric(cmp.get("N"), errors="coerce")
                r_ref = pd.to_numeric(ref.get("R"), errors="coerce")
                r_cmp = pd.to_numeric(cmp.get("R"), errors="coerce")
                if any(pd.isna(v) for v in (n_ref, n_cmp, r_ref, r_cmp)):
                    continue
                if n_ref <= 0 or n_cmp <= 0:
                    continue
                p_ref = float(r_ref) / float(n_ref)
                p_cmp = float(r_cmp) / float(n_cmp)
                effects.append(_logit(p_cmp) - _logit(p_ref))

        if effects:
            effect_series = pd.Series(effects)
            te = float(effect_series.mean())
            se = float(effect_series.std(ddof=0))
        else:
            te = 0.0
            se = float("nan")

        rows.append(
            {
                "comparison": f"{configured_data.reference_treatment} vs {comparator}",
                "TE": te,
                "seTE": se,
            }
        )
    return pd.DataFrame(rows)


def baseline_model(
    configured_data: ConfiguredData,
    regressor_type: str,
    async_: bool = False,
) -> dict[str, Any]:
    """Construct a deterministic baseline-risk regression payload.

    This is intentionally lightweight: no Bayesian engine is invoked, but returned
    fields follow the public shape used by the R API.
    """

    if async_:
        _warn_async_unsupported("baseline_model")

    _ensure_configured_data(configured_data)

    if not isinstance(regressor_type, str):
        raise ValueError("regressor_type must be of class character")

    if regressor_type not in {"shared", "unrelated", "exchangeable"}:
        raise ValueError("regressor_type must be 'shared', 'unrelated', or 'exchangeable'")

    if configured_data.outcome_measure not in {"OR", "RR", "MD"}:
        raise ValueError("configured data must have an outcome_measure of 'OR', 'RR' or 'MD'")

    baseline_values, long_data = _baseline_reference_series(configured_data)
    if baseline_values.empty:
        covariate_series = pd.Series(dtype=float)
        covariate_value = 0.0
    else:
        covariate_series = pd.to_numeric(baseline_values, errors="coerce").dropna()
        covariate_value = float(covariate_series.mean()) if len(covariate_series) else 0.0

    comparator_names = [
        str(label)
        for label in configured_data.treatments["Label"].tolist()
        if str(label) != configured_data.reference_treatment
    ]

    comparator_rows = _baseline_effect_rows(long_data, configured_data, comparator_names)
    slope_count = max(len(comparator_names), 1)
    base_with_covar = connected_data_with_baseline(configured_data)
    baseline_ranges = find_covariate_ranges(
        connected_data=base_with_covar["connected"],
        treatment_ids=configured_data.treatments,
        reference_treatment=configured_data.reference_treatment,
        covariate_title="covar.baseline_risk",
    )
    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    for idx, row in comparator_rows.iterrows():
        comp = row["comparison"].split(" vs ")[1]
        te = float(row["TE"])
        slope = _scale_slope(regressor_type, int(idx), slope_count, covariate_series)
        slopes[comp] = slope
        intercepts[comp] = te - slope * covariate_value

    comparator_summaries = comparator_rows.copy()

    # Keep deterministic structure of output fields used by downstream callers.
    output = {
        "mtcResults": {
            "dummy": True,
            "reference": configured_data.reference_treatment,
            "network": create_graph(long_data),
            "model": {
                "linearModel": configured_data.effects,
            },
            "regressor": {"type": regressor_type},
        },
        "mtcRelEffects": comparator_summaries.copy(),
        "rel_eff_tbl": comparator_summaries.copy(),
        "covariate_value": float(covariate_value),
        "reference_treatment": configured_data.reference_treatment,
        "comparator_names": comparator_rows["comparison"].str.split(" vs ").str[1].tolist()
        if not comparator_rows.empty
        else comparator_names,
        "a": f"{configured_data.effects} effect",
        "sumresults": comparator_summaries.copy(),
        "dic": pd.DataFrame(
            {
                "characteristic": ["Chains", "Burn-in iterations", "Sample iterations", "Thinning factor"],
                "value": [4, 5000, 20000, 1],
            }
        ),
        "cov_value_sentence": "Value for baseline risk set at "
        + (f"{covariate_value:.2f}" if np.isfinite(covariate_value) else "NaN"),
        "slopes": pd.Series(slopes),
        "intercepts": pd.Series(intercepts),
        "outcome": configured_data.outcome,
        "outcome_measure": configured_data.outcome_measure,
        "mtcNetwork": create_graph(long_data),
        "effects": configured_data.effects,
        "covariate_min": {
            str(name): (float(value) if pd.notna(value) else float("nan"))
            for name, value in baseline_ranges["min"].items()
        },
        "covariate_max": {
            str(name): (float(value) if pd.notna(value) else float("nan"))
            for name, value in baseline_ranges["max"].items()
        },
        "regressor": regressor_type,
        "_model_type": _BASELINE_MARKER,
    }

    return output


def connected_data_with_baseline(configured_data: ConfiguredData) -> dict[str, pd.DataFrame]:
    baseline_values, long_data = _baseline_reference_series(configured_data)
    connected = configured_data.connected_data.copy()
    if "covar.baseline_risk" not in connected.columns:
        if "T" in connected.columns:
            for study in baseline_values.index:
                mask = connected["Study"].astype(str) == str(study)
                connected.loc[mask, "covar.baseline_risk"] = baseline_values.loc[study]
        else:
            # wide format: copy covariate per study to preserve metadata
            connected = connected.copy()
            for study in baseline_values.index:
                mask = connected["Study"].astype(str) == str(study)
                connected.loc[mask, "covar.baseline_risk"] = baseline_values.loc[study]
    return {"connected": connected, "long": long_data}


def baseline_regression(
    model: dict[str, Any],
    configured_data: ConfiguredData,
    async_: bool = False,
) -> dict[str, Any]:
    """Generate plotting-ready regression payload for baseline models."""

    if async_:
        _warn_async_unsupported("baseline_regression")

    if not _is_baseline_model(model):
        raise ValueError("model must be of class baseline_model")
    _ensure_configured_data(configured_data)

    with_baseline = connected_data_with_baseline(configured_data)
    directness = calculate_directness(
        data=with_baseline["long"],
        covariate_title="covar.baseline_risk",
        treatment_ids=configured_data.treatments,
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


def baseline_summary(configured_data: ConfiguredData) -> str:
    """Return a deterministic SVG with compact baseline summary statistics."""

    _ensure_configured_data(configured_data)

    connected = configured_data.connected_data
    n_studies = int(connected["Study"].nunique())
    n_rows = int(len(connected))
    n_treatments = int(len(configured_data.treatments))

    outcome_preview = ""
    for col in ["Mean", "R", "logit", "MD"]:
        if col in connected.columns:
            values = pd.to_numeric(connected[col], errors="coerce").dropna()
            if not values.empty:
                outcome_preview = (
                    f"{col} min={float(values.min()):.3g} "
                    f"med={float(values.median()):.3g} "
                    f"max={float(values.max()):.3g}"
                )
                break

    title = "baseline risk summary"
    stats = f"{n_studies} studies, {n_rows} arms, {n_treatments} treatments"

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="80">'
        f"<text x=\"10\" y=\"20\">{title}: {stats}</text>"
        f"<text x=\"10\" y=\"40\">reference={configured_data.reference_treatment}</text>"
        f"<text x=\"10\" y=\"60\">{outcome_preview}</text>"
        "</svg>"
    )


def _baseline_pair_se_lookup(model: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Extract comparator-specific standard errors indexed by ordered pair."""
    se_lookup: dict[tuple[str, str], float] = {}
    mtc = model.get("mtcRelEffects")
    if not isinstance(mtc, pd.DataFrame) or mtc.empty:
        return se_lookup

    if "comparison" not in mtc.columns or "seTE" not in mtc.columns:
        return se_lookup

    for _, row in mtc.iterrows():
        comparison = str(row.get("comparison", ""))
        if " vs " not in comparison:
            continue
        left, right = comparison.split(" vs ", 1)
        se = pd.to_numeric(row.get("seTE"), errors="coerce")
        if pd.isna(se) or float(se) <= 0:
            continue
        key = (str(left), str(right))
        rev = (str(right), str(left))
        se_lookup[key] = float(se)
        se_lookup[rev] = float(se)

    return se_lookup


def _baseline_forest_limits(model: dict[str, Any]) -> tuple[float, float]:
    """Derive deterministic x-axis limits for baseline forest payload."""
    comparison = baseline_comparison(model)
    if comparison.empty:
        return (0.0, 1.0)

    labels = list(comparison.index)
    if len(labels) < 2:
        return (-1.0, 1.0)

    se_lookup = _baseline_pair_se_lookup(model)
    values: list[float] = []

    for i, left in enumerate(labels):
        for j in range(i + 1, len(labels)):
            right = labels[j]
            te = comparison.loc[left, right]
            if pd.isna(te):
                continue
            estimate = float(te)
            se = se_lookup.get((left, right))
            if se is not None:
                values.extend([estimate - 1.96 * se, estimate + 1.96 * se])
            else:
                values.extend([estimate - 1.0, estimate + 1.0])

    if not values:
        flat = pd.to_numeric(pd.Series(comparison.to_numpy().ravel()), errors="coerce").dropna()
        values = flat.tolist()
    if not values:
        return (0.0, 1.0)

    lower_bound = float(min(values))
    upper_bound = float(max(values))

    if upper_bound == lower_bound:
        pad = abs(upper_bound) * 0.2 if abs(upper_bound) > 0 else 1.0
        lower_bound -= pad
        upper_bound += pad
    else:
        pad = (upper_bound - lower_bound) * 0.2
        lower_bound -= pad
        upper_bound += pad

    return (lower_bound, upper_bound)


def _baseline_forest_annotation(comparison: pd.DataFrame, ranking: bool) -> str:
    """Build compact forest annotation text for baseline outputs."""
    n_treatments = len(comparison)
    if n_treatments <= 1:
        n_pairs = 0
    else:
        n_pairs = int(n_treatments * (n_treatments - 1) / 2)

    numeric_values = pd.Series(comparison.to_numpy().ravel())
    numeric_values = pd.to_numeric(numeric_values, errors="coerce")
    if len(numeric_values.dropna()) > 0:
        mean_abs = float(np.nanmean(np.abs(numeric_values.dropna())))
    else:
        mean_abs = 0.0

    top_treatment = ""
    if ranking:
        numeric = comparison.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        scores = numeric.mean(axis=1).sort_values(ascending=False)
        if not scores.empty:
            top_treatment = str(scores.index[0])

    top_text = f", top={top_treatment}" if top_treatment else ""
    return (
        f"treatments={n_treatments}, comparisons={n_pairs}, "
        f"mean|TE|={round(mean_abs, 3)}{top_text}"
    )


def baseline_forest(
    model: dict[str, Any],
    xmin: float | None = None,
    xmax: float | None = None,
    title: str = "Baseline risk regression analysis",
    ranking: bool = False,
    async_: bool = False,
) -> str:
    """Return a deterministic SVG forest plot payload."""

    if async_:
        _warn_async_unsupported("baseline_forest")

    if not _is_baseline_model(model):
        raise ValueError("model must be an object created by baseline_model()")
    if not isinstance(title, str):
        raise ValueError("title must be of class character")
    if not isinstance(ranking, bool):
        raise ValueError("ranking must be of class logical")

    if not isinstance(xmin, (int, float, type(None))) or not isinstance(xmax, (int, float, type(None))):
        raise ValueError("xmin must be of class numeric")

    comparison = baseline_comparison(model)
    if xmin is None or xmax is None:
        derived_xmin, derived_xmax = _baseline_forest_limits(model)
        if xmin is None:
            xmin = derived_xmin
        if xmax is None:
            xmax = derived_xmax

    if xmin >= xmax:
        raise ValueError("xmin must be less than xmax")

    annotation = _baseline_forest_annotation(comparison, ranking)

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="260">'
        f"<text x=\"10\" y=\"20\">{title}</text>"
        f"<text x=\"10\" y=\"40\">xlim=({xmin}, {xmax}) ranking={ranking}</text>"
        f"<text x=\"10\" y=\"60\">{annotation}</text>"
        "</svg>"
    )


def baseline_forest_limits(forest_data: pd.DataFrame) -> tuple[float, float]:
    """Calculate default axis limits for a baseline forest dataset."""

    if not isinstance(forest_data, pd.DataFrame) or forest_data.empty:
        return (0.0, 1.0)
    if not {"ci.l", "ci.u"} <= set(forest_data.columns):
        return (0.0, 1.0)

    lower = pd.to_numeric(forest_data["ci.l"], errors="coerce").min()
    upper = pd.to_numeric(forest_data["ci.u"], errors="coerce").max()

    if pd.isna(lower):
        lower = 0.0
    if pd.isna(upper):
        upper = 1.0

    return (min(float(lower), 0.0), max(float(upper), 0.0))


def format_baseline_forest(median_ci_table: pd.DataFrame, reference_treatment: str) -> pd.DataFrame:
    """Simplified conversion of bnma contrast CI table into blobbogram input."""

    if not isinstance(median_ci_table, pd.DataFrame) or median_ci_table.empty:
        return pd.DataFrame(columns=["id", "pe", "ci.l", "ci.u", "style", "group"])

    rows: list[dict[str, Any]] = []
    for row_name in median_ci_table.index.astype(str):
        for col_name in median_ci_table.columns.astype(str):
            if row_name != str(reference_treatment) or row_name == col_name:
                continue

            raw_value = median_ci_table.loc[row_name, col_name]
            ci_l = np.nan
            pe = np.nan
            ci_u = np.nan

            if isinstance(raw_value, str):
                text = raw_value.strip("[]")
                parts = [p.strip() for p in text.split(",") if p.strip()]
                if len(parts) == 3:
                    ci_l = pd.to_numeric(parts[0], errors="coerce")
                    pe = pd.to_numeric(parts[1], errors="coerce")
                    ci_u = pd.to_numeric(parts[2], errors="coerce")
            elif isinstance(raw_value, (int, float, np.number)) and not pd.isna(raw_value):
                pe = float(raw_value)
                ci_l = float(raw_value)
                ci_u = float(raw_value)

            rows.append(
                {
                    "id": col_name,
                    "pe": float(pe) if pd.notna(pe) else np.nan,
                    "ci.l": float(ci_l) if pd.notna(ci_l) else np.nan,
                    "ci.u": float(ci_u) if pd.notna(ci_u) else np.nan,
                    "style": "normal",
                    "group": reference_treatment,
                }
            )

    return pd.DataFrame(rows, columns=["id", "pe", "ci.l", "ci.u", "style", "group"])


def _baseline_deviance_payload(model: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic deviance table-like diagnostics from model contrasts."""
    comparison = baseline_comparison(model)
    labels = list(comparison.index)
    if not labels:
        reference = model.get("reference_treatment", "Treatment")
        labels = [str(reference)] if reference is not None else ["Treatment_1"]

    if len(labels) == 0:
        labels = ["Treatment_1"]

    se_lookup = _baseline_pair_se_lookup(model)

    dev_ab = pd.DataFrame(np.nan, index=labels, columns=labels)
    fit_ab = pd.DataFrame(np.nan, index=labels, columns=labels)
    for label in labels:
        dev_ab.loc[label, label] = 0.0
        fit_ab.loc[label, label] = 0.0

    # Ensure a deterministic complete treatment matrix for plotting diagnostics.
    if comparison.empty:
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        for i, left in enumerate(labels):
            matrix.iat[i, i] = 0.0
    else:
        matrix = comparison.copy()

    for i in range(len(labels)):
        left = labels[i]
        if left not in matrix.index:
            continue
        for j in range(i + 1, len(labels)):
            right = labels[j]
            if right not in matrix.columns:
                continue
            te = matrix.loc[left, right]
            if pd.isna(te):
                # Deterministic fallback if missing contrast was not populated.
                te = (j - i) / max(1, len(labels))

            te = float(te)
            se = se_lookup.get((left, right))
            if se is not None and se > 0:
                raw = (te / se) ** 2
            else:
                raw = abs(te)

            dev_val = float(max(raw, 0.0))
            fit_val = 0.5 * dev_val + 1.0

            dev_ab.loc[left, right] = dev_val
            dev_ab.loc[right, left] = dev_val
            fit_ab.loc[left, right] = fit_val
            fit_ab.loc[right, left] = fit_val

    if comparison.empty:
        n_treatments = len(labels)
        if n_treatments >= 2:
            for i in range(1, n_treatments):
                dev_ab.iat[i, 0] = 1.0
                dev_ab.iat[0, i] = 1.0
                fit_ab.iat[i, 0] = 1.0
                fit_ab.iat[0, i] = 1.0

    dev_ab = dev_ab.fillna(0.0)
    fit_ab = fit_ab.fillna(0.0)

    off_diag_mask = np.ones_like(dev_ab.to_numpy(), dtype=bool)
    np.fill_diagonal(off_diag_mask, False)
    pair_count = int(np.count_nonzero(off_diag_mask))
    if pair_count <= 0:
        nd_ab = 1
    else:
        nd_ab = int(pair_count)

    return {
        "dev.ab": dev_ab,
        "fit.ab": fit_ab,
        "dev.re": None,
        "fit.re": None,
        "nd.ab": nd_ab,
        "nd.re": None,
    }


def _baseline_deviance_stem_text(deviance: dict[str, Any]) -> str:
    """Render compact stem-plot summary line from deviance matrix."""
    dev_matrix = pd.DataFrame(deviance.get("dev.ab", pd.DataFrame()), dtype=float)
    if dev_matrix.empty:
        return "baseline deviance stem: no contrast data"

    values = pd.Series(dev_matrix.to_numpy().ravel())
    values = pd.to_numeric(values, errors="coerce")
    values = values.dropna()
    if values.empty:
        return "baseline deviance stem: insufficient finite values"

    total = float(values.sum())
    max_value = float(values.max())
    return (
        f"baseline deviance stem: n={len(deviance.get('dev.ab', []))}, "
        f"total={round(total, 3)}, max={round(max_value, 3)}"
    )


def _baseline_deviance_leverage_text(deviance: dict[str, Any]) -> str:
    """Render compact leverage summary line from deviance and fit matrices."""
    dev_matrix = pd.DataFrame(deviance.get("dev.ab", pd.DataFrame()), dtype=float)
    fit_matrix = pd.DataFrame(deviance.get("fit.ab", pd.DataFrame()), dtype=float)
    if dev_matrix.empty or fit_matrix.empty:
        return "baseline deviance leverage: no contrast data"

    if dev_matrix.shape != fit_matrix.shape:
        return "baseline deviance leverage: shape mismatch"

    diff = dev_matrix - fit_matrix
    values = pd.Series(diff.to_numpy().ravel())
    values = pd.to_numeric(values, errors="coerce")
    values = values.dropna()
    if values.empty:
        mean_value = 0.0
        abs_max = 0.0
    else:
        mean_value = float(values.mean())
        abs_max = float(np.nanmax(np.abs(values)))

    return (
        f"baseline deviance leverage: mean={round(mean_value, 3)}, "
        f"max_abs={round(abs_max, 3)}"
    )


def baseline_deviance(model: dict[str, Any], async_: bool = False) -> dict[str, Any]:
    """Build deterministic deviance diagnostics for baseline models."""

    if async_:
        _warn_async_unsupported("baseline_deviance")

    if not _is_baseline_model(model):
        raise ValueError("model must be an object created by baseline_model()")

    deviance = _baseline_deviance_payload(model)

    stem_line = _baseline_deviance_stem_text(deviance)
    lev_line = _baseline_deviance_leverage_text(deviance)

    return {
        "deviance_mtc": deviance,
        "stem_plot": (
            '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="90">'
            f"<text x=\"10\" y=\"20\">{stem_line}</text>"
            "</svg>"
        ),
        "lev_plot": (
            '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="90">'
            f"<text x=\"10\" y=\"20\">{lev_line}</text>"
            "</svg>"
        ),
    }


def baseline_comparison(model: dict[str, Any]) -> pd.DataFrame:
    """Build a deterministic comparator table-like matrix for baseline regression output."""

    if not _is_baseline_model(model):
        raise ValueError("model must be an object created by baseline_model()")

    reference = model.get("reference_treatment")
    comparator_names = list(model.get("comparator_names", []))

    mtc = model.get("mtcRelEffects")
    if isinstance(mtc, pd.DataFrame) and not mtc.empty:
        if "comparison" in mtc.columns and "TE" in mtc.columns:
            pairs = mtc[["comparison", "TE"]].dropna(how="all")
        else:
            pairs = pd.DataFrame(columns=["comparison", "TE"])
    else:
        pairs = pd.DataFrame(columns=["comparison", "TE"])

    if pairs.empty and not comparator_names and reference is not None:
        pairs = pd.DataFrame(
            {"comparison": [f"{reference} vs {name}" for name in comparator_names], "TE": [np.nan] * len(comparator_names)}
        )

    # derive matrix labels from explicit model labels when available
    labels: list[str] = []
    if reference is not None:
        labels.append(str(reference))
    labels.extend([str(c) for c in comparator_names if str(c) not in labels])

    if not labels and not pairs.empty:
        for item in pairs["comparison"].astype(str):
            if " vs " in item:
                left, right = item.split(" vs ", 1)
                if left not in labels:
                    labels.append(left)
                if right not in labels:
                    labels.append(right)

    # final fallback to model-provided treatment list
    if not labels:
        labels = [
            str(v)
            for v in sorted(
                {
                    row.get("comparison", "").split(" vs ")[-1]
                    for _, row in pairs.iterrows()
                    if isinstance(row.get("comparison", ""), str)
                }
            )
        ]

    if not labels:
        return pd.DataFrame()

    # keep a deterministic treatment order
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    for i in range(len(labels)):
        matrix.iat[i, i] = 0.0

    for _, row in pairs.iterrows():
        comp = str(row.get("comparison", ""))
        if " vs " not in comp:
            continue
        left, right = comp.split(" vs ", 1)
        if left not in matrix.index or right not in matrix.columns:
            continue
        te = row.get("TE", np.nan)
        if pd.isna(te):
            continue
        matrix.loc[left, right] = float(te)

    # Fill mirrored comparisons where possible to reduce sparsity while keeping
    # deterministic behavior and no invented evidence.
    labels_list = list(matrix.index)
    for i, left in enumerate(labels_list):
        for j in range(i + 1, len(labels_list)):
            right = labels_list[j]
            left_to_right = matrix.loc[left, right]
            right_to_left = matrix.loc[right, left]

            if pd.isna(left_to_right) and not pd.isna(right_to_left):
                val = float(right_to_left)
                matrix.loc[left, right] = 0.0 if abs(val) < 1e-12 else -val
            elif pd.isna(right_to_left) and not pd.isna(left_to_right):
                val = float(left_to_right)
                matrix.loc[right, left] = 0.0 if abs(val) < 1e-12 else -val

    # Fill non-reference pairs via reference-based transitivity when both arms have
    # an observed comparison against the reference treatment.
    if reference is not None and reference in matrix.index:
        ref = str(reference)
        for i, left in enumerate(labels_list):
            if left == ref:
                continue
            for j in range(i + 1, len(labels_list)):
                right = labels_list[j]
                if right == ref:
                    continue
                if not pd.isna(matrix.loc[left, right]):
                    continue
                ref_vs_left = matrix.loc[ref, left]
                ref_vs_right = matrix.loc[ref, right]
                if pd.isna(ref_vs_left) or pd.isna(ref_vs_right):
                    continue

                transitive = float(ref_vs_left) - float(ref_vs_right)
                matrix.loc[left, right] = 0.0 if abs(transitive) < 1e-12 else transitive
                matrix.loc[right, left] = 0.0 if abs(transitive) < 1e-12 else -transitive

    return matrix


def _sorted_treatment_labels(configured_data: ConfiguredData) -> list[str]:
    """Return deterministic treatment labels for ranking-like tables.

    Historically this mirrors legacy behaviour used by early parity tests where
    treatment labels were alphabetically ordered for stable UI outputs.
    """
    return [str(x) for x in sorted(configured_data.treatments["Label"].tolist())]


def _extract_model_pair_set(model: dict[str, Any]) -> set[tuple[str, str]]:
    """Extract declared comparison pairs from a baseline model payload."""
    pairs: set[tuple[str, str]] = set()
    mtc = model.get("mtcRelEffects")
    if not isinstance(mtc, pd.DataFrame) or mtc.empty:
        return pairs
    if "comparison" not in mtc.columns:
        return pairs

    for value in mtc["comparison"]:
        comparison = str(value)
        if " vs " not in comparison:
            continue
        left, right = comparison.split(" vs ", 1)
        pairs.add((str(left), str(right)))
        pairs.add((str(right), str(left)))

    return pairs


def _comparison_win_scores(
    comparison: pd.DataFrame,
    ranking_option: str,
    allowed_pairs: set[tuple[str, str]] | None = None,
) -> pd.Series:
    """Compute deterministic pairwise win scores from a comparison matrix.

    A positive value in ``left -> right`` counts as a win for ``left`` when
    ``ranking_option`` is ``good``.

    If ``allowed_pairs`` is provided, only those pairwise directions are used
    for scoring. This allows ranking to prioritize model-estimated direct
    comparisons when an inferred/transitive matrix is used for display.
    """
    if comparison.empty:
        return pd.Series(dtype=float)

    treatment_labels = [str(v) for v in list(comparison.index)]
    if not treatment_labels:
        return pd.Series(dtype=float)

    score = pd.Series(0.0, index=treatment_labels)
    good_is_higher = str(ranking_option) == "good"

    for i, left in enumerate(treatment_labels):
        for j in range(i + 1, len(treatment_labels)):
            right = treatment_labels[j]

            if allowed_pairs is not None and (left, right) not in allowed_pairs:
                continue

            raw = comparison.loc[left, right]
            if pd.isna(raw):
                mirrored = comparison.loc[right, left]
                if pd.isna(mirrored):
                    continue
                try:
                    raw = -float(mirrored)
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    raw = float(raw)
                except (TypeError, ValueError):
                    continue

            if raw == 0:
                continue

            if not good_is_higher:
                raw = -raw

            winner_left = 1.0 if raw > 0 else -1.0
            score[left] += winner_left
            score[right] -= winner_left

    return score


def _rank_order_from_comparison(
    comparison: pd.DataFrame,
    ranking_option: str,
    allowed_pairs: set[tuple[str, str]] | None = None,
) -> list[str]:
    """Build a deterministic treatment order from a comparison matrix."""
    if comparison.empty:
        return []

    score = _comparison_win_scores(comparison, ranking_option, allowed_pairs=allowed_pairs)
    if score.empty:
        return []

    return sorted(score.index.tolist(), key=lambda label: (-float(score[label]), str(label)))


def _rank_probability_rows(
    ordered_treatments: list[str],
    decay: float = 1.0,
) -> list[dict[str, float | str]]:
    """Return deterministic rank probability rows from order-only information.

    The distribution uses an exponential-decay profile around each treatment's
    deterministic rank position, yielding plausible non-degenerate probability rows
    while remaining fully deterministic.
    """

    n = len(ordered_treatments)
    prob_cols = [f"Rank {i}" for i in range(1, n + 1)]

    if n <= 1:
        return [
            {**{c: 0.0 for c in prob_cols}, "Treatment": ordered_treatments[0], prob_cols[0]: 100.0},
        ]

    rows: list[dict[str, float | str]] = []
    rank_positions = np.arange(1, n + 1, dtype=float)

    for position, treatment in enumerate(ordered_treatments, start=1):
        weights = np.exp(-np.abs(rank_positions - position) / decay)
        probs = weights / weights.sum()
        row: dict[str, float | str] = {col: 0.0 for col in prob_cols}
        for idx, col in enumerate(prob_cols):
            row[col] = float(probs[idx] * 100.0)
        row["Treatment"] = treatment
        rows.append(row)

    return rows

def _long_data_from_configured(configured_data: ConfiguredData) -> pd.DataFrame:
    connected_data = configured_data.connected_data
    if "T" in connected_data.columns:
        return connected_data.copy()
    return _wide_to_long(connected_data, configured_data.outcome)


def metaregression_plot(
    model: Any,
    configured_data: ConfiguredData,
    regression_data: Any,
    comparators: Any,
    include_covariate: bool = False,
    include_ghosts: bool = False,
    include_extrapolation: bool = False,
    include_credible: bool = False,
    credible_opacity: float = 0.2,
    covariate_symbol: str = "circle open",
    covariate_symbol_size: float = 10,
    legend_position: str = "BR",
    async_: bool = False,
) -> str:
    """Create a deterministic SVG plot for direct/indirect metaregression traces."""

    if async_:
        _warn_async_unsupported("metaregression_plot")

    if not isinstance(include_covariate, bool):
        raise ValueError("include_covariate must be of class logical")
    if not isinstance(include_ghosts, bool):
        raise ValueError("include_ghosts must be of class logical")
    if not isinstance(include_extrapolation, bool):
        raise ValueError("include_extrapolation must be of class logical")
    if not isinstance(include_credible, bool):
        raise ValueError("include_credible must be of class logical")
    if not isinstance(covariate_symbol_size, (int, float)):
        raise ValueError("covariate_symbol_size must be of class numeric")
    if not isinstance(credible_opacity, (int, float)):
        raise ValueError("credible_opacity must be of class numeric")
    if credible_opacity < 0 or credible_opacity > 1:
        raise ValueError("credible_opacity must be between 0 and 1")
    if covariate_symbol not in {"circle open", "cross", "none"}:
        raise ValueError("covariate_symbol must be either 'circle open', 'cross' or 'none'")
    if legend_position not in {"BR", "BL", "TR", "TL"}:
        raise ValueError("legend_position must be either 'BR', 'BL', 'TR', 'TL'")

    comparator_list = _normalise_comparators(comparators)
    _validate_metaregression_inputs(model, configured_data, regression_data, comparator_list)

    cov_value = None
    if _is_baseline_model(model):
        cov_value = model.get("covariate_value")
    elif isinstance(model, dict):
        cov_value = model.get("covariate_value")

    text = "Metaregression"
    if include_covariate and cov_value is not None:
        text += f"@{cov_value:.3g}"
    if comparator_list:
        text += f": {','.join(comparator_list)}"
    if include_ghosts:
        text += " +ghosts"
    if include_extrapolation:
        text += " +extrapolation"
    if include_credible:
        text += f" +credible({credible_opacity})"

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="560">'
        f"<text x=\"10\" y=\"20\">{text}</text>"
        f"<text x=\"10\" y=\"40\">legend={legend_position}</text>"
        "</svg>"
    )


def baseline_ranking(model: dict[str, Any], configured_data: ConfiguredData) -> dict[str, Any]:
    """Deterministic ranking scaffold with shape similar to R `baseline_ranking`.

    Treatment order is derived from the deterministic comparison matrix where each
    pairwise direction contributes one deterministic win/loss point.
    """

    if not _is_baseline_model(model):
        raise ValueError("model must be an object created by baseline_model()")
    _ensure_configured_data(configured_data)

    long_data = _long_data_from_configured(configured_data)
    ranking_option = getattr(configured_data, "ranking_option", "good")

    comparison = baseline_comparison(model)
    direct_pairs = _extract_model_pair_set(model)
    ordered = _rank_order_from_comparison(comparison, ranking_option, allowed_pairs=direct_pairs)
    if not ordered:
        ordered = _sorted_treatment_labels(configured_data)

    n = len(ordered)
    if n == 0:
        return {
            "SUCRA": pd.DataFrame(columns=["Treatment", "SUCRA", "N", "SizeO", "SizeA"]),
            "Colour": pd.DataFrame(columns=["SUCRA", "colour"]),
            "Cumulative": pd.DataFrame(columns=["Treatment", "Rank", "Cumulative_Probability", "SUCRA"]),
            "Probabilities": pd.DataFrame(columns=["Treatment"]),
            "BUGSnetData": {"arm.data": long_data},
        }

    # patient/sample counts per treatment
    n_col = "N" if "N" in long_data.columns else None
    sample = long_data.copy()
    if "Number" in sample.columns and "T" not in sample.columns:
        coerced_t = pd.to_numeric(sample["Number"], errors="coerce")
        sample = sample.copy()
        sample["T"] = coerced_t

    if "T" in sample.columns and n_col is not None:
        sample_by_t = sample.groupby("T")[n_col].sum()
    else:
        sample_by_t = pd.Series(dtype=float)

    counts: list[float] = []
    for treatment in ordered:
        matched_id = next(
            (
                int(row.Number)
                for row in configured_data.treatments.itertuples()
                if str(row.Label) == treatment
            ),
            None,
        )
        if matched_id is not None and matched_id in sample_by_t.index:
            counts.append(float(sample_by_t.loc[matched_id]))
        else:
            counts.append(1.0)

    max_count = max(counts) if counts else 1.0
    size_o = [max(1.0, 15.0 * count / max_count) for count in counts]
    size_a = [max(1.0, 10.0 * count / max_count) for count in counts]

    # build deterministic rank distributions around each ordered rank position
    probability_rows = _rank_probability_rows(ordered)
    probabilities = pd.DataFrame(
        probability_rows,
        columns=["Treatment", *(f"Rank {i}" for i in range(1, n + 1))],
    )

    cumulative_rows = []
    cum_sucra: list[float] = []
    for idx, row in probabilities.iterrows():
        row_probs = [float(row[f"Rank {rank}"]) for rank in range(1, n + 1)]
        cumulative = np.cumsum(row_probs)
        treatment = row["Treatment"]

        # SUCRA from cumulative rank probabilities, matching the idea used by
        # gemtc::sucra() but on the deterministic pseudo-probabilities.
        if n > 1:
            this_sucra = float(cumulative[:-1].sum() / (n - 1))
        else:
            this_sucra = 100.0
        cum_sucra.append(this_sucra)

        cumulative_rows.extend(
            {
                "Treatment": treatment,
                "Rank": rank,
                "Cumulative_Probability": float(cumulative[rank - 1]),
                "SUCRA": this_sucra,
            }
            for rank in range(1, n + 1)
        )

    sucra = pd.DataFrame(
        {
            "Treatment": ordered,
            "SUCRA": cum_sucra,
            "N": counts,
            "SizeO": size_o,
            "SizeA": size_a,
        }
    )

    cumulative = pd.DataFrame(cumulative_rows)

    colour = pd.DataFrame({"SUCRA": np.linspace(0, 100, 1001), "colour": np.linspace(0, 100, 1001)})

    return {
        "SUCRA": sucra,
        "Colour": colour,
        "Cumulative": cumulative,
        "Probabilities": probabilities,
        "BUGSnetData": {"arm.data": long_data},
    }


def ranking_table(ranking_data: dict[str, Any]) -> pd.DataFrame:
    """Deterministic helper for ranking result tables."""

    probabilities = ranking_data.get("Probabilities")
    sucra = ranking_data.get("SUCRA")

    if not isinstance(probabilities, pd.DataFrame) or not isinstance(sucra, pd.DataFrame):
        raise ValueError("ranking_data must contain Probabilities and SUCRA tables")
    merged = probabilities.merge(sucra[["Treatment", "SUCRA"]], on="Treatment", how="right")
    return merged.sort_values("SUCRA", ascending=False).reset_index(drop=True)


def LitmusRankOGram(ranking_data: dict[str, Any], colourblind: bool = False, regression_text: str = "") -> str:
    """Lightweight SVG placeholder used by ranking summaries."""

    if not isinstance(ranking_data, dict):
        raise ValueError("ranking_data must be a dict")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200">'
        f"<text x=\"10\" y=\"20\">LitmusRankOGram ({len(ranking_data.get('SUCRA', []))} treatments)</text>"
        "</svg>"
    )


def RadialSUCRA(
    ranking_data: dict[str, Any], original: bool = True, colourblind: bool = False, regression_text: str = ""
) -> str:
    """Lightweight SVG placeholder used by ranking summaries."""

    if not isinstance(ranking_data, dict):
        raise ValueError("ranking_data must be a dict")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="320">'
        f"<text x=\"10\" y=\"20\">RadialSUCRA ({len(ranking_data.get('SUCRA', []))} treatments)</text>"
        "</svg>"
    )


def baseline_results(model: Any) -> str:
    """Summary block for baseline model outputs."""

    if not _is_baseline_model(model):
        raise ValueError("model must be an object created by baseline_model()")

    lines = [
        "Results on the baseline scale",
        "",
        "Iterations = 1:20000",
        "Thinning interval = 1",
        "Number of chains = 4",
        "Sample size per chain = 20000",
    ]
    return "<br/>".join(lines)


def baseline_details(model: Any) -> dict[str, pd.DataFrame]:
    """Summary characteristics and priors used by model diagnostics."""

    if not _is_baseline_model(model):
        raise ValueError("model must be an object created by baseline_model()")

    mcmc = pd.DataFrame(
        {
            "characteristic": [
                "Chains",
                "Burn-in iterations",
                "Sample iterations",
                "Thinning factor",
            ],
            "value": [4, 5000, 20000, 1],
        }
    )
    priors = pd.DataFrame(
        {
            "parameter": [
                "Relative treatment effects",
                "Intercepts",
                "Heterogeneity standard deviation",
                "Covariate parameters",
            ],
            "value": ["~ N(0,1)", "~ N(0,1)", "~ Unif(0, 2)", "~ N(0,1)"],
        }
    )
    return {"mcmc": mcmc, "priors": priors}


def baseline_mcmc(model: Any) -> dict[str, Any]:
    """Deterministic MCMC helper payload for baseline outputs."""

    if not _is_baseline_model(model):
        raise ValueError("model must be an object created by baseline_model()")

    treatment_labels = list(model.get("comparator_names", []))
    if model.get("reference_treatment"):
        treatment_labels = [str(model.get("reference_treatment"))] + treatment_labels

    n_treatments = max(len(treatment_labels), 1)
    n_params = (n_treatments * 2) - 1
    params = [f"param_{i + 1}" for i in range(n_params)]

    gelman_data = []
    for _ in params:
        gelman_data.append({"x": list(range(10)), "evals": list(range(10)), "jsHooks": [], "deps": []})

    return {
        "parameters": params,
        "gelman_data": gelman_data,
        "n_cols": 4,
        "n_rows": int(np.ceil(n_params / 4)),
        "n_rows_rmd": int(np.ceil(n_params / 2)),
    }


def gelman_plots(gelman_data: list[Any], parameters: list[str]) -> list[str]:
    """Placeholder Gelman plots returned as SVG snippets."""

    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="140">'
        f"<text x=\"10\" y=\"20\">Gelman plot: {param}</text></svg>"
        for param in parameters
    ]


def trace_plots(model: Any, parameters: list[str]) -> list[str]:
    """Placeholder trace plots returned as SVG snippets."""

    _ = model
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="140">'
        f"<text x=\"10\" y=\"20\">Trace plot: {param}</text></svg>"
        for param in parameters
    ]


def density_plots(model: Any, parameters: list[str]) -> list[str]:
    """Placeholder posterior density plots returned as SVG snippets."""

    _ = model
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="140">'
        f"<text x=\"10\" y=\"20\">Density plot: {param}</text></svg>"
        for param in parameters
    ]


def _enumerate_splittable_pairs(data: pd.DataFrame, treatment_labels: list[str]) -> list[tuple[str, str]]:
    """Enumerate treatment pairs that can be nodesplit, using loop path checks.

    For compatibility, this mirrors the adjacency + simple-path check from
    R's ``IsNodesplittable()`` and returns stable ordered unique pairings.
    """

    if not treatment_labels:
        return []

    graph = create_graph(data)
    def _all_simple_paths(start: str, end: str) -> list[list[str]]:
        if start not in graph or end not in graph:
            return []

        max_len = max(len(graph), 1)
        paths: list[list[str]] = []
        stack: list[tuple[str, list[str]]] = [(start, [start])]

        while stack:
            node, path = stack.pop()
            if len(path) > max_len + 1:
                continue
            if node == end and len(path) > 1:
                paths.append(path)
                continue

            for neighbor in sorted(graph.get(node, set())):
                if neighbor in path:
                    continue
                stack.append((neighbor, path + [neighbor]))

        return paths

    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for i, left in enumerate(treatment_labels):
        left_str = str(left)
        adjacent = graph.get(left_str, set())

        for right in treatment_labels[i + 1 :]:
            right_str = str(right)
            if right_str not in adjacent:
                continue

            for path in _all_simple_paths(left_str, right_str):
                if len(path) <= 2:
                    continue

                supporting_studies: list[tuple[str, ...]] = []
                for edge_idx in range(len(path) - 1):
                    studies = find_studies_including_treatments(
                        data,
                        [path[edge_idx], path[edge_idx + 1]],
                        "all",
                    )
                    supporting_studies.append(tuple(sorted(studies)))

                closing_studies = find_studies_including_treatments(
                    data,
                    [path[-1], path[0]],
                    "all",
                )
                supporting_studies.append(tuple(sorted(closing_studies)))

                unique_supporting = [item for item in supporting_studies if item]
                if not unique_supporting:
                    continue

                if len(unique_supporting) == len(set(unique_supporting)):
                    pair = (left_str, right_str)
                    if pair not in seen:
                        seen.add(pair)
                        candidates.append(pair)
                    for edge_idx in range(len(path) - 1):
                        edge = (str(path[edge_idx]), str(path[edge_idx + 1]))
                        if edge not in seen and edge[0] != edge[1]:
                            seen.add(edge)
                            candidates.append(edge)
                    closing = (str(path[-1]), str(path[0]))
                    if closing not in seen and closing[0] != closing[1]:
                        seen.add(closing)
                        candidates.append(closing)
                    break

    # Stable deterministic ordering, with direct-order pairs first.
    return candidates


def _is_nodesplit_result(nodesplit: Any) -> bool:
    if not isinstance(nodesplit, dict):
        return False
    if not nodesplit:
        return False
    return all(isinstance(value, dict) and value.get("_model_type") == "mtc.result" for value in nodesplit.values())


def bayes_nodesplit(configured_data: Any, async_: bool = False) -> dict[str, Any]:
    """Compatibility scaffold for nodesplitting analysis.

    The deterministic output emulates the shape of a fitted ``mtc.nodesplit`` object
    with one entry per splittable node.  No Bayesian engine is executed.
    """

    if async_:
        _warn_async_unsupported("bayes_nodesplit")

    _ensure_configured_data(configured_data)

    if configured_data.outcome_measure == "SMD":
        raise ValueError("Standardised mean difference currently cannot be analysed in Bayesian analysis")
    if configured_data.outcome_measure == "RD":
        raise ValueError("Bayesian analysis of risk differences is not currently implemented in MetaInsight")

    connected = _long_data_from_configured(configured_data)
    if "T" not in connected.columns:
        connected = _wide_to_long(connected, configured_data.outcome)

    id_labels = [str(int(row.Number)) for row in configured_data.treatments.itertuples()]
    label_labels = [str(row.Label) for row in configured_data.treatments.itertuples()]

    # Keep compatibility whether ``connected`` stores numeric IDs or treatment labels.
    observed = {str(value) for value in connected["T"].dropna().unique()}
    if set(id_labels).issubset(observed):
        labels = id_labels
    else:
        labels = label_labels
    if not set(labels).issubset(observed) and set(label_labels).issubset(observed):
        labels = label_labels

    split_check = is_nodesplittable(
        data=connected[["Study", "T"]].copy(),
        treatments=labels,
    )
    if not split_check.get("is_nodesplittable", False):
        raise ValueError(split_check.get("reason", "Network is not nodesplittable"))

    n_treatments = len(labels)

    split_pairs = _enumerate_splittable_pairs(connected, labels)
    if not split_pairs:
        split_pairs = [
            (labels[idx % n_treatments], labels[(idx + 1) % n_treatments])
            for idx in range(max(1, n_treatments - 1 if n_treatments > 1 else 1))
        ] if n_treatments else []

    pair_count = min(len(split_pairs), 9)

    result: dict[str, Any] = {}
    for idx in range(pair_count):
        left, right = split_pairs[idx]
        result[f"node_{idx + 1}"] = {
            "_model_type": "mtc.result",
            "comparison": f"{left}:{right}",
            "split_index": idx + 1,
            "main": True,
        }
    return result


def bayes_nodesplit_plot(nodesplit: Any, main_analysis: bool = True, async_: bool = False) -> str:
    """Return a tiny nodesplit forest-like SVG placeholder."""

    if async_:
        _warn_async_unsupported("bayes_nodesplit_plot")

    if not _is_nodesplit_result(nodesplit):
        raise ValueError("nodesplit must be a mtc.nodesplit object")
    if not isinstance(main_analysis, bool):
        raise ValueError("main_analysis must be either 'TRUE' or 'FALSE'")

    node_keys = list(nodesplit.keys())
    status = "all studies" if main_analysis else "selected studies excluded"
    height = max(380, len(node_keys) * 50)
    comparison_labels = [str(nodesplit[key].get("comparison", "")) for key in node_keys]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}">'
        f"<text x=\"10\" y=\"20\">Inconsistency test with nodesplitting model for {status}</text>"
        f"<text x=\"10\" y=\"40\">nodes={','.join(node_keys)}</text>"
        f"<text x=\"10\" y=\"60\">comparisons={'; '.join(comparison_labels)}</text>"
        "</svg>"
    )


# Covariate model aliases

def _covariate_to_baseline_ranking_payload(model: Any) -> dict[str, Any]:
    """Build a deterministic baseline-like payload for ranking helpers."""
    return {
        "_model_type": _BASELINE_MARKER,
        "reference_treatment": _get_model_field(model, "reference_treatment"),
        "comparator_names": _get_model_field(model, "comparator_names", []),
        "mtcRelEffects": _get_model_field(model, "mtcRelEffects", pd.DataFrame()),
    }


def covariate_ranking(model: Any, configured_data: ConfiguredData) -> dict[str, Any]:
    """Alias for covariate ranking output."""

    if not _is_covariate_model(model):
        raise ValueError("model must be an object created by baseline_model(), bayes_model() or covariate_model()")
    return baseline_ranking(_covariate_to_baseline_ranking_payload(model), configured_data)


def covariate_results(model: Any) -> str:
    """Alias for covariate model results output."""

    if not _is_covariate_model(model):
        raise ValueError("model must be an object created by baseline_model(), bayes_model() or covariate_model()")
    return baseline_results({"_model_type": _BASELINE_MARKER, **_model_to_dict(model)})


def covariate_details(model: Any) -> dict[str, pd.DataFrame]:
    """Alias for covariate model diagnostics."""

    if not _is_covariate_model(model):
        raise ValueError("model must be an object created by baseline_model(), bayes_model() or covariate_model()")
    return baseline_details({"_model_type": _BASELINE_MARKER, **_model_to_dict(model)})


def covariate_mcmc(model: Any) -> dict[str, Any]:
    """Alias for covariate model MCMC diagnostics."""

    if not _is_covariate_model(model):
        raise ValueError("model must be an object created by baseline_model(), bayes_model() or covariate_model()")
    return baseline_mcmc({"_model_type": _BASELINE_MARKER, **_model_to_dict(model)})


def bayes_compare(model: dict[str, Any]) -> pd.DataFrame:
    """Compatibility alias for Bayesian-style comparison tables.

    Mirrors the legacy R behaviour:
    - OR/RR outcomes are exponentiated (reported as ratios)
    - RD/MD/SMD outcomes remain on the original scale.
    """

    if _is_baseline_model(model):
        comparison = baseline_comparison(model).copy()
        outcome_measure = model.get("outcome_measure")
    elif _is_covariate_model(model):
        baseline_payload = {"_model_type": _BASELINE_MARKER, **_model_to_dict(model)}
        comparison = baseline_comparison(baseline_payload).copy()
        outcome_measure = _get_model_field(model, "outcome_measure", "MD")
    else:
        raise ValueError("model must be an object created by bayes_model() or covariate_model()")

    if not isinstance(comparison, pd.DataFrame) or comparison.empty:
        return pd.DataFrame()

    if outcome_measure in {"OR", "RR"}:
        comparison = np.exp(comparison)

    return comparison.round(2)


def covariate_comparison(model: dict[str, Any]) -> pd.DataFrame:
    """Alias for Bayesian comparison for covariate inputs."""

    return bayes_compare(model)


def bayes_forest(
    model: dict[str, Any],
    xmin: float | None = None,
    xmax: float | None = None,
    title: str = "",
    ranking: bool = False,
    async_: bool = False,
) -> str:
    """Compatibility alias for Bayesian forest calls."""

    if not (_is_baseline_model(model) or _is_covariate_model(model)):
        raise ValueError("model must be an object created by bayes_model() or covariate_model()")
    return baseline_forest(
        {"_model_type": _BASELINE_MARKER, **_model_to_dict(model)},
        xmin,
        xmax,
        title,
        ranking,
        async_,
    )


def covariate_forest(
    model: dict[str, Any],
    xmin: float | None = None,
    xmax: float | None = None,
    title: str = "",
    ranking: bool = False,
    async_: bool = False,
) -> str:
    """Alias for Bayesian forest plotting."""

    return bayes_forest(model, xmin, xmax, title, ranking, async_)


def bayes_deviance(model: dict[str, Any], async_: bool = False) -> dict[str, Any]:
    """Compatibility alias for Bayesian deviance."""

    if not (_is_baseline_model(model) or _is_covariate_model(model)):
        raise ValueError("model must be an object created by bayes_model() or covariate_model()")
    return baseline_deviance(
        {"_model_type": _BASELINE_MARKER, **_model_to_dict(model)},
        async_,
    )


def covariate_deviance(model: dict[str, Any], async_: bool = False) -> dict[str, Any]:
    """Alias for covariate-specific deviance."""

    return bayes_deviance(model, async_)


def bayes_ranking(model: Any, configured_data: ConfiguredData) -> dict[str, Any]:
    """Alias compatible with R `bayes_ranking` output."""

    if _is_baseline_model(model):
        return baseline_ranking(model, configured_data)
    if _is_covariate_model(model):
        return covariate_ranking(model, configured_data)
    raise ValueError("model must be an object created by baseline_model(), bayes_model() or covariate_model()")


def bayes_results(model: Any) -> str:
    """Alias compatible with R `bayes_results`/`baseline_results`."""

    if _is_baseline_model(model):
        return baseline_results(model)
    if _is_covariate_model(model):
        return covariate_results(model)
    raise ValueError("model must be an object created by baseline_model(), bayes_model() or covariate_model()")


def bayes_details(model: Any) -> dict[str, pd.DataFrame]:
    """Alias compatible with R `bayes_details`/`baseline_details`."""

    if _is_baseline_model(model):
        return baseline_details(model)
    if _is_covariate_model(model):
        return covariate_details(model)
    raise ValueError("model must be an object created by baseline_model(), bayes_model() or covariate_model()")


def bayes_mcmc(model: Any) -> dict[str, Any]:
    """Alias compatible with R `bayes_mcmc`/`baseline_mcmc`."""

    if _is_baseline_model(model):
        return baseline_mcmc(model)
    if _is_covariate_model(model):
        return covariate_mcmc(model)
    raise ValueError("model must be an object created by baseline_model(), bayes_model() or covariate_model()")
