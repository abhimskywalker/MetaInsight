"""Input loading, validation and setup configuration helpers.

This module ports key setup routines from the R package:
- ``setup_load``
- ``validate_uploaded_data``
- ``setup_configure``
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, List

import pandas as pd
import re


@dataclass(frozen=True)
class ValidationResult:
    valid: bool = True
    message: str = "Data is valid"


@dataclass(frozen=True)
class LoadedData:
    """Result from :func:`setup_load`."""

    is_data_valid: bool
    is_data_uploaded: bool
    data: pd.DataFrame
    treatments: pd.DataFrame | None
    outcome: str


@dataclass(frozen=True)
class ConfiguredData:
    """Result from :func:`setup_configure`."""

    wrangled_data: pd.DataFrame
    treatments: pd.DataFrame
    reference_treatment: str
    disconnected_indices: List[int]
    connected_data: pd.DataFrame
    non_covariate_data: pd.DataFrame
    covariate: dict
    bugsnet: pd.DataFrame
    freq: dict
    outcome: str
    outcome_measure: str
    effects: str
    ranking_option: str
    seed: int | float


@dataclass(frozen=True)
class ExcludedData:
    """Result from :func:`setup_exclude`."""

    treatments: pd.DataFrame
    reference_treatment: str
    connected_data: pd.DataFrame
    covariate: dict
    bugsnet: pd.DataFrame
    freq: dict
    outcome: str
    outcome_measure: str
    effects: str
    ranking_option: str
    seed: int | float


@dataclass(frozen=True)
class _ColumnDefinition:
    name: str
    required: bool
    type_name: str
    expected_type: str
    pattern: str
    replacement: str
    number_group: str | None


_BINARY: List[_ColumnDefinition] = [
    _ColumnDefinition(
        name="Study",
        required=True,
        type_name="character (text)",
        expected_type="string",
        pattern=r"^Study(\.[0-9]+)?$",
        replacement=r"Study\1",
        number_group=None,
    ),
    _ColumnDefinition(
        name="T",
        required=True,
        type_name="character (text)",
        expected_type="string",
        pattern=r"^T(\.[0-9]+)?$",
        replacement=r"T\1",
        number_group=r"\1",
    ),
    _ColumnDefinition(
        name="N",
        required=True,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^N(\.[0-9]+)?$",
        replacement=r"N\1",
        number_group=r"\1",
    ),
    _ColumnDefinition(
        name="R",
        required=True,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^R(\.[0-9]+)?$",
        replacement=r"R\1",
        number_group=r"\1",
    ),
    _ColumnDefinition(
        name="covar.*",
        required=False,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^covar\.(.+)$",
        replacement=r"covar.\1",
        number_group=None,
    ),
    _ColumnDefinition(
        name="rob",
        required=False,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^rob(\..+)?$",
        replacement=r"rob\1",
        number_group=None,
    ),
    _ColumnDefinition(
        name="indirectness",
        required=False,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^indirectness$",
        replacement="indirectness",
        number_group=None,
    ),
]


_CONTINUOUS: List[_ColumnDefinition] = [
    _ColumnDefinition(
        name="Study",
        required=True,
        type_name="character (text)",
        expected_type="string",
        pattern=r"^Study(\.[0-9]+)?$",
        replacement=r"Study\1",
        number_group=None,
    ),
    _ColumnDefinition(
        name="T",
        required=True,
        type_name="character (text)",
        expected_type="string",
        pattern=r"^T(\.[0-9]+)?$",
        replacement=r"T\1",
        number_group=r"\1",
    ),
    _ColumnDefinition(
        name="N",
        required=True,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^N(\.[0-9]+)?$",
        replacement=r"N\1",
        number_group=r"\1",
    ),
    _ColumnDefinition(
        name="Mean",
        required=True,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^Mean(\.[0-9]+)?$",
        replacement=r"Mean\1",
        number_group=r"\1",
    ),
    _ColumnDefinition(
        name="SD",
        required=True,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^SD(\.[0-9]+)?$",
        replacement=r"SD\1",
        number_group=r"\1",
    ),
    _ColumnDefinition(
        name="covar.*",
        required=False,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^covar\.(.+)$",
        replacement=r"covar.\1",
        number_group=None,
    ),
    _ColumnDefinition(
        name="rob",
        required=False,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^rob(\..+)?$",
        replacement=r"rob\1",
        number_group=None,
    ),
    _ColumnDefinition(
        name="indirectness",
        required=False,
        type_name="numeric",
        expected_type="numeric",
        pattern=r"^indirectness$",
        replacement="indirectness",
        number_group=None,
    ),
]


_VALID_OUTCOMES = {"binary", "continuous"}


def _get_schema(outcome: str) -> List[_ColumnDefinition]:
    if outcome == "binary":
        return _BINARY
    if outcome == "continuous":
        return _CONTINUOUS
    raise ValueError(f"Outcome {outcome} is not recognised. Please use 'binary' or 'continuous'")


def _find_matching_columns(columns: Iterable[str], pattern: str) -> List[str]:
    regex = re.compile(pattern)
    return [column for column in columns if regex.search(str(column))]


def _tidy_string_item(value: object) -> object:
    if not isinstance(value, str):
        return value
    tidy = re.sub(r"\s+", " ", value.strip())
    return pd.NA if tidy == "" else tidy


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and blank strings in all string columns."""

    cleaned = data.copy()
    object_cols = cleaned.select_dtypes(include="object").columns
    cleaned[object_cols] = cleaned[object_cols].apply(lambda series: series.map(_tidy_string_item))
    return cleaned


def _clean_identifier(value: object) -> object:
    if not isinstance(value, str):
        return value
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned


def _canonicalize_column_name(columns: Iterable[str], outcome: str) -> List[str]:
    definitions = _get_schema(outcome)
    canonical = []
    for column in columns:
        canonical_name = str(column)
        for definition in definitions:
            if re.search(definition.pattern, str(column), flags=re.IGNORECASE):
                canonical_name = re.sub(
                    definition.pattern,
                    definition.replacement,
                    str(column),
                    flags=re.IGNORECASE,
                )
                break
        canonical.append(canonical_name)

    return canonical


def _find_all_treatments(data: pd.DataFrame) -> List[str]:
    if "T" in data.columns:
        treatment_series = data["T"]
        values = treatment_series.dropna().astype(str)
    else:
        treatment_cols = [
            col for col in data.columns if re.match(r"^T(\.[0-9]+)?$", str(col))
        ]
        if not treatment_cols:
            return []
        ordered: list[str] = []
        for col in treatment_cols:
            ordered.extend(data[col].dropna().astype(str).tolist())
        values = pd.Series(ordered)

    # Preserve original order and uniqueness.
    return list(dict.fromkeys(values.tolist()))


def _create_treatment_ids(all_treatments: List[str]) -> pd.DataFrame:
    treatment_ids = pd.DataFrame({"Number": list(range(1, len(all_treatments) + 1)), "Label": all_treatments})
    treatment_ids["Label"] = treatment_ids["Label"].astype(str)
    return treatment_ids


def _create_treatment_frame(all_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"Number": list(range(1, len(all_names) + 1)), "Label": all_names})


def _upgrade_treatment_frame(data: pd.DataFrame, treatment_map: dict[int, str]) -> pd.DataFrame:
    upgraded = data.copy()

    def _convert(value: object) -> object:
        if pd.isna(value):
            return pd.NA

        try:
            treatment_id = int(float(value))
        except (TypeError, ValueError):
            return pd.NA

        return treatment_map.get(treatment_id, pd.NA)

    for column in [col for col in upgraded.columns if re.match(r"^T(\.[0-9]+)?$", str(col))]:
        upgraded[column] = upgraded[column].apply(_convert)

    return upgraded.drop(columns=["StudyID"], errors="ignore")


def _max_treatment_id_from_data(data: pd.DataFrame) -> int:
    treatment_cols = [c for c in data.columns if re.match(r"^T(\.[0-9]+)?$", str(c))]
    if not treatment_cols:
        return 0

    treatment_values = pd.concat([data[col].dropna() for col in treatment_cols], ignore_index=True)

    values: List[int] = []
    for value in treatment_values.tolist():
        try:
            values.append(int(float(value)))
        except (TypeError, ValueError):
            continue

    return max(values) if values else 0


def _reorder_treatments(treatments: pd.DataFrame, reference_treatment: str) -> pd.DataFrame:
    labels = list(treatments["Label"])
    if reference_treatment not in labels:
        raise ValueError("reference_treatment must be present in the loaded data")

    reordered = [
        reference_treatment
    ] + [label for label in labels if label != reference_treatment]

    out = treatments.copy()
    out["Label"] = pd.Categorical(out["Label"], categories=reordered, ordered=True)
    out = out.sort_values("Label").reset_index(drop=True)
    out["Number"] = range(1, len(out) + 1)
    return out


def _reorder_treatments_with_reference(treatments: pd.DataFrame, reference_treatment: str) -> pd.DataFrame:
    labels = list(treatments["Label"])
    if reference_treatment in labels:
        ordered = [reference_treatment] + [label for label in labels if label != reference_treatment]
    else:
        ordered = labels

    out = treatments.copy()
    out["Label"] = pd.Categorical(out["Label"], categories=ordered, ordered=True)
    out = out.sort_values("Label").reset_index(drop=True)
    out["Number"] = range(1, len(out) + 1)
    return out


def _reinstate_treatment_ids(data: pd.DataFrame, treatment_names: pd.DataFrame) -> pd.DataFrame:
    mapping = {int(row.Number): row.Label for row in treatment_names.itertuples(index=False)}
    out = data.copy()

    def _restore(value: object) -> object:
        if pd.isna(value):
            return pd.NA

        try:
            key = int(float(value))
        except (TypeError, ValueError):
            return pd.NA
        return mapping.get(key, pd.NA)

    if "T" in out.columns:
        out["T"] = out["T"].apply(_restore)
        out["T"] = out["T"].astype(object)
    else:
        for column in [col for col in out.columns if re.match(r"^T(\.[0-9]+)?$", str(col))]:
            out[column] = out[column].apply(_restore)
            out[column] = out[column].astype(object)

    return out


def _find_missing_columns(data: pd.DataFrame, required_columns: List[_ColumnDefinition]) -> List[str]:
    missing = []
    for definition in required_columns:
        if not _find_matching_columns(data.columns, definition.pattern):
            missing.append(definition.name)
    return missing


def _validate_matching_wide_columns(data: pd.DataFrame, numbered_columns: List[_ColumnDefinition]) -> bool:
    wide_numbers: dict[str, List[int]] = {}

    for definition in numbered_columns:
        matched = _find_matching_columns(data.columns, definition.pattern)
        suffixes: List[int] = []
        for column in matched:
            match = re.search(r"\.(\d+)$", str(column))
            if match is None:
                suffixes.append(-1)
            else:
                suffixes.append(int(match.group(1)))
        wide_numbers[definition.name] = suffixes

    if not wide_numbers:
        return True

    all_are_long = all(all(value == -1 for value in values) for values in wide_numbers.values())
    if all_are_long:
        return True

    any_long = any(all(value == -1 for value in values) for values in wide_numbers.values())
    if any_long:
        return False

    lengths = [len(values) for values in wide_numbers.values()]
    if len(set(lengths)) > 1:
        return False

    for values in wide_numbers.values():
        ordered = sorted(values)
        if ordered != list(range(1, len(ordered) + 1)):
            return False

    return True


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_string_series(series: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    return all(isinstance(value, str) or pd.isna(value) for value in series)


def _validate_column_types(data: pd.DataFrame, outcome_columns: List[_ColumnDefinition]) -> ValidationResult:
    mistyped_columns: List[str] = []

    for definition in outcome_columns:
        matching_columns = _find_matching_columns(data.columns, definition.pattern)
        for column_name in matching_columns:
            if definition.expected_type == "string" and not _is_string_series(data[column_name]):
                mistyped_columns.append(f"{column_name} should be of type {definition.type_name}")
            elif definition.expected_type == "numeric" and not _is_numeric_series(data[column_name]):
                mistyped_columns.append(f"{column_name} should be of type {definition.type_name}")

    if mistyped_columns:
        return ValidationResult(
            valid=False,
            message=(
                "Some columns have incorrect data types: "
                + ", ".join(mistyped_columns)
            ),
        )

    return ValidationResult()


def _validate_quality_columns(data: pd.DataFrame) -> ValidationResult:
    rob_indirectness = [
        col
        for col in data.columns
        if re.match(r"^rob(\..+)?$", str(col)) or re.match(r"^indirectness$", str(col))
    ]

    if not rob_indirectness:
        return ValidationResult()

    n_rob_individual = sum(
        bool(re.match(r"^rob\.", str(col)) and "." in str(col)) for col in rob_indirectness
    )
    if n_rob_individual > 10:
        return ValidationResult(
            valid=False,
            message="A maximum of 10 individual risk of bias variables are allowed.",
        )

    for var in rob_indirectness:
        grouped = data[["Study", var]].dropna(subset=[var]).groupby("Study")[var]

        studies_with_multiple = [
            str(study)
            for study, values in grouped.unique().items()
            if len(values) > 1
        ]
        if studies_with_multiple:
            return ValidationResult(
                valid=False,
                message=(
                    "Some studies do not have the same "
                    f"{var} value for every arm: {', '.join(studies_with_multiple)}."
                ),
            )

        studies_with_wrong_values = [
            str(study)
            for study, values in grouped.unique().items()
            if any(v not in {1, 2, 3} for v in values if pd.notna(v))
        ]
        if studies_with_wrong_values:
            return ValidationResult(
                valid=False,
                message=(
                    f"Some studies have values for {var} that are not 1, 2 or 3: "
                    + ", ".join(studies_with_wrong_values)
                ),
            )

    return ValidationResult()


def _validate_covariates(data: pd.DataFrame, covariate_col: str) -> ValidationResult:
    if not _is_numeric_series(data[covariate_col]):
        return ValidationResult(
            valid=False,
            message="One or more covariate values are non-numerical.",
        )

    by_study = data[["Study", covariate_col]].groupby("Study")[covariate_col].unique()

    for study, values in by_study.items():
        if any(pd.isna(v) for v in values):
            return ValidationResult(
                valid=False,
                message=(
                    "Some studies do not define covariate values for all arms: "
                    f"{study}"
                ),
            )

        if len(values) > 1:
            return ValidationResult(
                valid=False,
                message=(
                    "Some studies contain inconsistent covariate values between arms: "
                    f"{study}"
                ),
            )

    if len(pd.unique(data[covariate_col].dropna())) <= 1:
        return ValidationResult(
            valid=False,
            message="Cannot analyse covariate with no variation.",
        )

    return ValidationResult()


def _find_single_arm_studies(data: pd.DataFrame) -> List[str]:
    if "T" in data.columns:
        counts = data.groupby("Study")["T"].count()
    else:
        treatment_cols = [
            col for col in data.columns if re.match(r"^T(\.[0-9]+)?$", str(col))
        ]
        if not treatment_cols:
            return []
        counts = (
            data[treatment_cols]
            .notna()
            .sum(axis=1)
            .rename(data["Study"])
            .groupby(level=0)
            .max()
        )

    return [str(study) for study, count in counts.items() if count < 2]


def _validate_data_shape_and_count(data: pd.DataFrame) -> ValidationResult:
    single_arm_studies = _find_single_arm_studies(data)
    if single_arm_studies:
        return ValidationResult(
            valid=False,
            message="Some studies have single arms: " + ", ".join(single_arm_studies),
        )
    return ValidationResult()


def validate_uploaded_data(data: pd.DataFrame, outcome: str) -> ValidationResult:
    """Validate loaded data against the same checks as the R implementation."""

    if data is None or len(data) == 0:
        return ValidationResult(valid=False, message="File is empty")

    outcome_schema = _get_schema(outcome)
    cleaned = clean_data(data)
    required_columns = [definition for definition in outcome_schema if definition.required]

    missing = _find_missing_columns(cleaned, required_columns)
    if missing:
        return ValidationResult(
            valid=False,
            message=(
                f"Missing columns for {outcome} data: {', '.join(missing)}"
            ),
        )

    numbered_columns = [definition for definition in required_columns if definition.number_group is not None]
    if not _validate_matching_wide_columns(cleaned, numbered_columns):
        names = ", ".join(definition.name for definition in numbered_columns)
        return ValidationResult(
            valid=False,
            message=(
                "For wide format data, numbered columns ("
                + names
                + ") must all have matching sequential indices, starting from 1"
            ),
        )

    result = _validate_column_types(cleaned, outcome_schema)
    if not result.valid:
        return result

    result = _validate_data_shape_and_count(cleaned)
    if not result.valid:
        return result

    result = _validate_quality_columns(cleaned)
    if not result.valid:
        return result

    covariates = [col for col in cleaned.columns if re.match(r"^covar\.", str(col))]
    for covariate in covariates:
        result = _validate_covariates(cleaned[["Study", covariate]].copy(), covariate)
        if not result.valid:
            return result

    return ValidationResult()


def _find_data_shape(data: pd.DataFrame) -> str:
    return "long" if "T" in data.columns else "wide"


def _find_covariate_columns(data: pd.DataFrame) -> List[str]:
    return [col for col in data.columns if re.match(r"^covar\.", str(col))]


def _find_rob_columns(data: pd.DataFrame) -> List[str]:
    return [
        col
        for col in data.columns
        if re.match(r"^rob(\..+)?$", str(col)) or re.match(r"^indirectness$", str(col))
    ]


def _replace_treatment_ids(data: pd.DataFrame, treatment_ids: pd.DataFrame) -> pd.DataFrame:
    """Replace treatment labels with treatment ids from a mapping table."""

    replaced = data.copy()
    lookup = {row.Label: row.Number for row in treatment_ids.itertuples()}
    if "T" in data.columns:
        replaced["T"] = replaced["T"].astype(str).map(lookup)
    else:
        for col in [c for c in data.columns if re.match(r"^T(\.[0-9]+)?$", str(c))]:
            replaced[col] = replaced[col].astype(str).map(lookup)

    # Keep columns numeric (ids are ints)
    if "T" in replaced.columns:
        replaced["T"] = pd.to_numeric(replaced["T"], errors="coerce")
    for col in [c for c in replaced.columns if re.match(r"^T(\.[0-9]+)?$", str(c))]:
        replaced[col] = pd.to_numeric(replaced[col], errors="coerce")

    return replaced


def _add_study_ids(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    study_names = list(dict.fromkeys(out["Study"].astype(str)))
    out["StudyID"] = out["Study"].astype(str).map({name: idx + 1 for idx, name in enumerate(study_names)})
    return out


def _sort_by_studyid_then_treatment(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    if _find_data_shape(data) == "long":
        sorted_data = data.sort_values(["StudyID", "T"]).reset_index(drop=True)
    else:
        # convert to long -> sort -> back to wide for deterministic ordering
        sorted_long = _wide_to_long(data, outcome).sort_values(["StudyID", "T"]).reset_index(drop=True)
        sorted_data = _long_to_wide(sorted_long, outcome)
    return sorted_data


def _reorder_columns(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    ordered: List[str] = ["StudyID", "Study"]

    if "T.1" in data.columns or "N.1" in data.columns or "Mean.1" in data.columns:
        if outcome == "continuous":
            ordered.extend(["T.1", "N.1", "Mean.1", "SD.1"])
        else:
            ordered.extend(["T.1", "R.1", "N.1"])
    else:
        if outcome == "continuous":
            ordered.extend(["T", "N", "Mean", "SD"])
        else:
            ordered.extend(["T", "R", "N"])

    base_cols = set(ordered)
    extra_cols = [c for c in data.columns if c not in base_cols]

    rob_cols = [c for c in extra_cols if c in _find_rob_columns(data)]
    covar_cols = [c for c in extra_cols if c in _find_covariate_columns(data)]
    others = [c for c in extra_cols if c not in rob_cols + covar_cols]
    ordered.extend(others)

    # For compatibility with existing behaviour, include quality columns after core metrics.
    if outcome == "continuous":
        ordered.extend(["rob", "indirectness"])
        ordered.extend([c for c in rob_cols if c not in {"rob", "indirectness"}])
    else:
        ordered.extend(["rob", "indirectness"])
        ordered.extend([c for c in rob_cols if c not in {"rob", "indirectness"}])

    ordered.extend(covar_cols)

    # add any additional columns that are not in the known ordering (for example per-study columns)
    remaining = [c for c in data.columns if c not in ordered]
    ordered.extend(remaining)

    # de-duplicate while preserving order
    final_order: List[str] = []
    for column in ordered:
        if column in final_order:
            continue
        if column in data.columns:
            final_order.append(column)

    return data[final_order]


def _wide_to_long(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    outcome = outcome.lower()
    if _find_data_shape(data) == "long":
        return data.copy()

    if outcome == "continuous":
        prefixes = ["T", "N", "Mean", "SD"]
    else:
        prefixes = ["T", "R", "N"]

    cols_by_study = {
        prefix: [col for col in data.columns if re.match(rf"^{prefix}(\.[0-9]+)?$", str(col))]
        for prefix in prefixes
    }

    if "StudyID" in data.columns:
        study_id_map = {
            str(row["Study"]): str(row["StudyID"])
            for _, row in data.iterrows()
            if pd.notna(row.get("StudyID"))
        }
    else:
        study_id_map = {
            str(study): idx + 1
            for idx, study in enumerate(dict.fromkeys(data["Study"].astype(str)))
        }

    max_arms = 0
    for values in cols_by_study.values():
        if not values:
            continue
        max_arms = max(max_arms, len(values))

    long_rows = []
    for _, row in data.iterrows():
        for arm in range(1, max_arms + 1):
            arm_suffix = f".{arm}"
            extracted = {
                "Study": row.get("Study"),
                "StudyID": study_id_map.get(str(row.get("Study")), pd.NA),
            }

            for prefix in prefixes:
                exact = f"{prefix}{arm_suffix}"
                if exact in row and pd.notna(row.get(exact)):
                    extracted[prefix] = row.get(exact)
                elif arm == 1 and prefix in row:
                    # Handle legacy wide format without suffixed first arm.
                    extracted[prefix] = row.get(prefix)
                else:
                    extracted[prefix] = pd.NA

            # Carry through quality/covariate columns if present
            for column in data.columns:
                if column.startswith("Study") or re.match(r"^[TNRMSD]|^rob", str(column)):
                    continue
                if column == "StudyID":
                    continue
                extracted[column] = row[column]

            # If there is a meaningful row, append it.
            if extracted.get(prefixes[0]) is not pd.NA and pd.notna(extracted.get(prefixes[0])):
                long_rows.append(extracted)

    long_data = pd.DataFrame(long_rows)

    # Reorder required columns to keep stable.
    cols = ["Study", "StudyID", "T"]
    if outcome == "continuous":
        cols.extend(["Mean", "SD", "N"])
    else:
        cols.extend(["R", "N"])
    long_data = long_data[cols + [c for c in long_data.columns if c not in cols]]

    return long_data


def _long_to_wide(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    if _find_data_shape(data) != "long":
        return data.copy()

    study_treatments = data.sort_values(["Study", "T"]).copy()
    study_treatments["row_index"] = study_treatments.groupby("Study").cumcount() + 1

    if outcome == "continuous":
        measure_cols = ["N", "Mean", "SD"]
    else:
        measure_cols = ["R", "N"]

    studies = list(study_treatments["Study"].drop_duplicates())
    wide = pd.DataFrame({"Study": studies})
    if "StudyID" in study_treatments.columns:
        study_id_map = study_treatments.groupby("Study")["StudyID"].first()
        wide["StudyID"] = wide["Study"].map(study_id_map)

    for col in measure_cols:
        pivot = study_treatments.pivot(index="Study", columns="row_index", values=col)
        pivot.columns = [f"{col}.{int(col_idx)}" for col_idx in pivot.columns]
        wide = wide.merge(pivot.reset_index(), on="Study", how="left")

    # Treatment columns are mapped separately.
    t_pivot = study_treatments.pivot(index="Study", columns="row_index", values="T")
    t_pivot.columns = [f"T.{int(idx)}" for idx in t_pivot.columns]
    wide = wide.merge(t_pivot.reset_index(), on="Study", how="left")

    # Preserve covariate/other auxiliary columns from any study row.
    extra_cols = [
        col for col in study_treatments.columns
        if col not in {"Study", "StudyID", "T", "row_index", "N", "Mean", "SD", "R"}
    ]
    for col in extra_cols:
        wide[col] = wide["Study"].map(study_treatments.groupby("Study")[col].first())

    return _reorder_columns(wide, outcome)


def _find_study_treatments(data: pd.DataFrame, study: str) -> List[int]:
    study_rows = data[data["Study"] == study]
    if "T" in data.columns:
        values = study_rows["T"].dropna().tolist()
    else:
        t_cols = [c for c in data.columns if re.match(r"^T(\.[0-9]+)?$", str(c))]
        values = []
        for c in t_cols:
            row_value = study_rows[c].iloc[0]
            if pd.notna(row_value):
                values.append(int(row_value))
    # Remove missing
    return sorted({int(v) for v in values if pd.notna(v)})


def _build_components(study_to_treatments: dict[str, List[int]], max_treatment: int) -> dict[int, int]:
    parent = {t: t for t in range(1, max_treatment + 1)}

    def _find_root(node: int) -> int:
        while parent[node] != node:
            node = parent[node]
        return node

    def _union(a: int, b: int) -> None:
        ra = _find_root(a)
        rb = _find_root(b)
        if ra != rb:
            parent[rb] = ra

    for treatments in study_to_treatments.values():
        if len(treatments) < 2:
            continue
        for left, right in combinations(treatments, 2):
            _union(left, right)

    components = {node: _find_root(node) for node in parent.keys()}
    # Make component indices contiguous for easier interpretation.
    comp_map: dict[int, int] = {}
    next_index = 1
    for node in sorted(components):
        comp = components[node]
        if comp not in comp_map:
            comp_map[comp] = next_index
            next_index += 1
        components[node] = comp_map[comp]
    return components


def _identify_subnetworks(data: pd.DataFrame, treatments: pd.DataFrame, reference_treatment_name: str) -> tuple[list[str], List[int]]:
    # Returns (connected_studies, disconnected_indices)
    study_to_treatments = {
        study: _find_study_treatments(data, study)
        for study in data["Study"].astype(str).unique()
    }

    treatment_to_id = {int(row.Number): row.Label for row in treatments.itertuples()}

    component_map = _build_components(study_to_treatments, max(treatment_to_id.keys() or [0]))

    # Find component of reference treatment.
    ref_row = treatments[treatments["Label"] == reference_treatment_name]
    if ref_row.empty:
        raise ValueError("reference_treatment must be present in the loaded data")

    reference_id = int(ref_row["Number"].iloc[0])
    reference_component = component_map.get(reference_id, 1)

    connected_studies: List[str] = []
    for study, ids in study_to_treatments.items():
        if not ids:
            continue
        if any(component_map[t] == reference_component for t in ids if t in component_map):
            connected_studies.append(study)

    unique_studies = list(data["Study"].astype(str).unique())
    disconnected_indices = [idx + 1 for idx, name in enumerate(unique_studies) if name not in set(connected_studies)]
    return connected_studies, disconnected_indices


def _remove_covariates(data: pd.DataFrame) -> pd.DataFrame:
    covar_cols = _find_covariate_columns(data)
    if not covar_cols:
        return data.copy()
    cols = [c for c in data.columns if c not in covar_cols]
    return data[cols].copy()


def _infer_covariate_type(series: pd.Series) -> str:
    values = set(pd.to_numeric(series, errors="coerce").dropna().unique())
    if values == {0.0, 1.0}:
        return "binary"
    return "continuous"


def _build_bugsnet(data: pd.DataFrame, outcome: str, treatment_ids: pd.DataFrame) -> pd.DataFrame:
    base = data.copy()
    # Keep only connected/cleaned observations and ensure long format.
    if _find_data_shape(base) == "wide":
        base = _wide_to_long(base, outcome)

    base = base.sort_values(["StudyID", "T"], ascending=[True, False]).reset_index(drop=True)

    # label map
    labels = {int(row.Number): row.Label for row in treatment_ids.itertuples()}
    base = base.copy()
    if "T" in base.columns:
        base["Treatment"] = base["T"].map(labels)

    # add standard error for continuous outcomes
    if outcome == "continuous" and "SD" in base.columns and "N" in base.columns:
        base["se"] = base["SD"].astype(float) / (base["N"].astype(float).pow(0.5))

    # Keep a minimal tidy subset similar to R output.
    keep_cols = [c for c in ["StudyID", "Study", "Treatment", "N", "Mean", "SD", "R", "se", "covar.age"] if c in base.columns]
    return base[keep_cols].copy()


def _long_from_data(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    if _find_data_shape(data) == "long":
        return data.copy()
    return _wide_to_long(data, outcome)


def _build_contrast_data(non_cov: pd.DataFrame, treatment_ids: pd.DataFrame, outcome: str, outcome_measure: str) -> pd.DataFrame:
    long = _long_from_data(non_cov, outcome)
    if "T" not in long.columns:
        return pd.DataFrame()

    # one row per pairwise comparison
    rows = []
    id_to_label = {int(row.Number): row.Label for row in treatment_ids.itertuples()}

    for _, study_df in long.groupby("Study", sort=False):
        row_data = study_df.to_dict("records")
        if len(row_data) < 2:
            continue

        for first, second in combinations(row_data, 2):
            t1 = int(first["T"])
            t2 = int(second["T"])
            study = str(first["Study"])
            if outcome == "continuous":
                n1 = float(first["N"]) if pd.notna(first.get("N")) else float("nan")
                n2 = float(second["N"]) if pd.notna(second.get("N")) else float("nan")
                m1 = float(first["Mean"]) if pd.notna(first.get("Mean")) else float("nan")
                m2 = float(second["Mean"]) if pd.notna(second.get("Mean")) else float("nan")
                v1 = float(first["SD"]) if pd.notna(first.get("SD")) else float("nan")
                v2 = float(second["SD"]) if pd.notna(second.get("SD")) else float("nan")
                te = m1 - m2
                se = ((v1 ** 2 / n1) + (v2 ** 2 / n2)) ** 0.5 if n1 > 0 and n2 > 0 else float("nan")
            else:
                n1 = float(first["N"]) if pd.notna(first.get("N")) else float("nan")
                n2 = float(second["N"]) if pd.notna(second.get("N")) else float("nan")
                r1 = float(first["R"]) if pd.notna(first.get("R")) else float("nan")
                r2 = float(second["R"]) if pd.notna(second.get("R")) else float("nan")
                p1 = r1 / n1 if n1 else 0.0
                p2 = r2 / n2 if n2 else 0.0
                p1 = max(min(p1, 1 - 1e-9), 1e-9)
                p2 = max(min(p2, 1 - 1e-9), 1e-9)
                te = (p1 / (1 - p1)) / (p2 / (1 - p2))
                te = __import__("math").log(te)
                se = (
                    1 / (r1 + 0.5)
                    + 1 / (r2 + 0.5)
                    + 1 / ((n1 - r1) + 0.5)
                    + 1 / ((n2 - r2) + 0.5)
                ) ** 0.5

            rows.append(
                {
                    "TE": te,
                    "seTE": se,
                    "treat1": t1,
                    "treat2": t2,
                    "treat1_name": id_to_label.get(t1),
                    "treat2_name": id_to_label.get(t2),
                    "studlab": study,
                    "outcome_measure": outcome_measure,
                }
            )

    return pd.DataFrame(rows)


def _build_freq_payload(non_cov: pd.DataFrame, outcome: str, treatments: pd.DataFrame, outcome_measure: str, effects: str, reference_treatment: str) -> dict:
    d0 = _build_contrast_data(non_cov, treatments, outcome, outcome_measure)

    # label-matched contrast table
    d1 = d0.copy()
    if not d1.empty:
        d1["treat1"] = d1["treat1_name"]
        d1["treat2"] = d1["treat2_name"]

    return {
        "net1": {
            "effects": effects,
            "reference_treatment": reference_treatment,
            "outcome": outcome,
        },
        "lstx": treatments["Label"].tolist(),
        "ntx": int(len(treatments)),
        "d0": d0,
        "d1": d1,
    }


def setup_upgrade(
    data_path: str,
    treatments: str,
    logger: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Upgrade legacy treatment IDs in a legacy-format dataset to labels."""

    if not isinstance(data_path, (str, Path)):
        raise TypeError("data_path must be of class character")

    if not isinstance(treatments, str):
        raise TypeError("treatments must be of class character")

    if not re.fullmatch(r"^[a-zA-Z_]+(,[a-zA-Z_]+)*$", treatments):
        raise ValueError("The treatment names must only contain words separated by commas")

    path = Path(data_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("data_path must link to either a .csv file")

    if not path.exists():
        raise FileNotFoundError("The specified file does not exist")

    data = pd.read_csv(path)
    unnamed = [col for col in data.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        data = data.drop(columns=unnamed)
    data = clean_data(data)

    treatment_names = [name.strip() for name in treatments.split(",")]
    if any(name == "" for name in treatment_names):
        raise ValueError("The treatment names must only contain words separated by commas")

    input_treatment_count = len(treatment_names)
    data_treatment_count = _max_treatment_id_from_data(data)

    if input_treatment_count != data_treatment_count:
        message = (
            f"Your input data contains {input_treatment_count} treatments "
            f"but your treatment list contains {data_treatment_count} treatments"
        )
        if logger is None:
            raise ValueError(message)
        logger(message)

    treatment_frame = _create_treatment_frame(treatment_names)
    mapping = {int(row.Number): str(row.Label) for row in treatment_frame.itertuples(index=False)}
    upgraded = _upgrade_treatment_frame(data, mapping)

    if logger is not None:
        logger("Your data has successfully been upgraded.")

    return upgraded


def setup_load(
    data_path: str | Path | None = None,
    outcome: str = "continuous",
    logger: Callable[[str], None] | None = None,
) -> LoadedData:
    """Load study-level MetaInsight data from a file and validate format."""

    if not isinstance(outcome, str):
        raise TypeError("outcome must be of class character")

    if outcome not in _VALID_OUTCOMES:
        raise ValueError("outcome must be either binary or continuous")

    if data_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        filename = "continuous_long.csv" if outcome == "continuous" else "binary_long.csv"
        path = repo_root / "inst" / "extdata" / filename
        is_uploaded = False
    else:
        if not isinstance(data_path, (str, Path)):
            raise TypeError("data_path must be of class character")

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError("The specified file does not exist")

        suffix = path.suffix.lower()
        if suffix not in {".csv", ".xlsx"}:
            raise ValueError("data_path must link to either a .csv or .xlsx file")
        is_uploaded = True

    if path.suffix.lower() == ".csv":
        raw_data = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        raw_data = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format for study-level data.")

    cleaned = clean_data(raw_data)
    cleaned.columns = _canonicalize_column_name(cleaned.columns, outcome)

    validation = validate_uploaded_data(cleaned, outcome)
    if not validation.valid:
        message = (
            "Uploaded data was invalid because: "
            f"{validation.message}. "
            "Please check you data file and ensure that you have the correct outcome type selected."
        )
        if logger is None:
            raise ValueError(message)
        logger(message)
        return LoadedData(
            is_data_valid=False,
            is_data_uploaded=is_uploaded,
            data=cleaned,
            treatments=None,
            outcome=outcome,
        )

    treatments = _create_treatment_ids(_find_all_treatments(cleaned))

    return LoadedData(
        is_data_uploaded=is_uploaded,
        is_data_valid=True,
        data=cleaned,
        treatments=treatments,
        outcome=outcome,
    )


def setup_configure(
    loaded_data: LoadedData,
    reference_treatment: str,
    effects: str,
    outcome_measure: str,
    ranking_option: str,
    seed: int | float,
    logger: Callable[[str], None] | None = None,
) -> ConfiguredData:
    """Convert validated loaded data into connected analysis data structures."""

    if not isinstance(loaded_data, LoadedData):
        raise TypeError("loaded_data must be of class loaded_data")
    if not isinstance(reference_treatment, str):
        raise TypeError("reference_treatment must be of class character")
    if not isinstance(effects, str):
        raise TypeError("effects must be of class character")
    if not isinstance(outcome_measure, str):
        raise TypeError("outcome_measure must be of class character")
    if not isinstance(ranking_option, str):
        raise TypeError("ranking_option must be of class character")
    if not isinstance(seed, (int, float)):
        raise TypeError("seed must be of class numeric")

    if not loaded_data.is_data_valid:
        if logger is not None:
            logger("loaded_data must contain valid data")
        raise ValueError("loaded_data does not contain valid data")

    if loaded_data.treatments is None or loaded_data.treatments.empty:
        raise ValueError("loaded_data must contain treatment information")

    if loaded_data.outcome == "binary" and outcome_measure not in {"OR", "RR", "RD"}:
        raise ValueError("When outcome is binary, outcome_measure must be either OR, RR or RD")
    if loaded_data.outcome == "continuous" and outcome_measure not in {"MD", "SMD"}:
        raise ValueError("When outcome is continuous, outcome_measure must be either MD or SMD")

    if effects not in {"random", "fixed"}:
        raise ValueError("effects must be either random or fixed")

    if ranking_option not in {"good", "bad"}:
        raise ValueError("ranking_option must be either good or bad")

    reference_clean = str(_clean_identifier(reference_treatment))

    treatments = loaded_data.treatments.copy().reset_index(drop=True)
    treatments["Label"] = treatments["Label"].astype(str)
    raw_to_clean = {str(label): _clean_identifier(label) for label in treatments["Label"]}
    clean_to_raw = {clean: raw for raw, clean in raw_to_clean.items()}

    if reference_clean not in clean_to_raw:
        raise ValueError("reference_treatment must be present in the loaded data")

    reference_raw = clean_to_raw[reference_clean]
    ordered_treatment_labels = [reference_raw] + [
        label for label in treatments["Label"]
        if _clean_identifier(label) != reference_clean
    ]

    raw_treatments = pd.DataFrame(
        {
            "Number": list(range(1, len(ordered_treatment_labels) + 1)),
            "Label": ordered_treatment_labels,
        }
    )
    reordered_treatments = raw_treatments.copy()
    reordered_treatments["Label"] = reordered_treatments["Label"].map(_clean_identifier)

    wrangled = _replace_treatment_ids(loaded_data.data, raw_treatments)
    wrangled = _add_study_ids(wrangled)
    wrangled = _reorder_columns(wrangled, loaded_data.outcome)
    wrangled = _sort_by_studyid_then_treatment(wrangled, loaded_data.outcome)

    connected_studies, disconnected_indices = _identify_subnetworks(
        wrangled,
        reordered_treatments,
        reference_clean,
    )
    connected_data = wrangled[wrangled["Study"].isin(connected_studies)].copy()

    if disconnected_indices and logger is not None:
        logger(
            "The uploaded data comprises a disconnected network. "
            f"Only the subnetwork containing the reference treatment ({reference_clean}) "
            "will be displayed and disconnected studies are shown in the logger."
        )
        disconnected_names = [study for study in wrangled["Study"].astype(str).unique() if study not in set(connected_studies)]
        logger("Disconnected studies: " + ",".join(disconnected_names))

    non_covariate_data = _remove_covariates(connected_data)

    covariate = {}
    covars = _find_covariate_columns(connected_data)
    if covars:
        covariate_column = covars[0]
        covariate_type = _infer_covariate_type(connected_data[covariate_column])
        covariate = {
            "column": covariate_column,
            "name": str(covariate_column).replace("covar.", ""),
            "type": covariate_type,
        }

    bugsnet = _build_bugsnet(connected_data, loaded_data.outcome, reordered_treatments)
    freq = _build_freq_payload(
        non_cov=non_covariate_data,
        outcome=loaded_data.outcome,
        treatments=reordered_treatments,
        outcome_measure=outcome_measure,
        effects=effects,
        reference_treatment=reference_clean,
    )

    return ConfiguredData(
        wrangled_data=wrangled.reset_index(drop=True),
        treatments=reordered_treatments.reset_index(drop=True),
        reference_treatment=reference_clean,
        disconnected_indices=disconnected_indices,
        connected_data=connected_data.reset_index(drop=True),
        non_covariate_data=non_covariate_data.reset_index(drop=True),
        covariate=covariate,
        bugsnet=bugsnet,
        freq=freq,
        outcome=loaded_data.outcome,
        outcome_measure=outcome_measure,
        effects=effects,
        ranking_option=ranking_option,
        seed=seed,
    )


def setup_exclude(
    configured_data: ConfiguredData,
    exclusions: list[str] | None = None,
    logger: Callable[[str], None] | None = None,
) -> ExcludedData:
    """Return a subset of connected studies after excluding selected studies."""

    if not isinstance(configured_data, ConfiguredData):
        raise TypeError("configured_data must be of class configured_data")

    if exclusions is None:
        exclusions = []
    elif not isinstance(exclusions, list):
        raise TypeError("exclusions must be of class character")
    elif any(not isinstance(study, str) for study in exclusions):
        raise TypeError("exclusions must be of class character")

    study_labels = set(configured_data.connected_data["Study"].astype(str).tolist())
    invalid = [s for s in exclusions if s not in study_labels]
    if invalid:
        raise ValueError("exclusions must in the present in the loaded data")

    if exclusions:
        reduced = configured_data.connected_data[~configured_data.connected_data["Study"].astype(str).isin(exclusions)].copy()
    else:
        reduced = configured_data.connected_data.copy()

    if reduced.empty:
        raise ValueError("You have excluded all the studies")

    labeled = _reinstate_treatment_ids(reduced, configured_data.treatments)

    treatments_present = configured_data.treatments[
        configured_data.treatments["Label"].isin(set(labeled["T"].dropna().astype(str)))
    ]

    reference_treatment = configured_data.reference_treatment
    if reference_treatment not in treatments_present["Label"].tolist():
        # Reference treatment is no longer represented; choose the first remaining treatment.
        if len(treatments_present) == 0:
            raise ValueError("You have excluded all the studies")
        reference_treatment = str(treatments_present["Label"].iloc[0])
        if logger is not None:
            logger(f"Reference treatment has been changed to {reference_treatment}")

    treatments = _reorder_treatments_with_reference(
        treatments_present.reset_index(drop=True),
        reference_treatment,
    )

    connected_treatments = _replace_treatment_ids(labeled, treatments)

    if "Study" in connected_treatments.columns:
        study_labels_in_data = set(connected_treatments["Study"].astype(str))
        if study_labels_in_data:
            filtered = connected_treatments[connected_treatments["Study"].astype(str).isin(study_labels_in_data)]
        else:
            filtered = connected_treatments
    else:
        filtered = connected_treatments

    # Ensure data remains connected to reference treatment.
    connected_studies, _ = _identify_subnetworks(filtered, treatments, reference_treatment)
    connected_data = filtered[filtered["Study"].astype(str).isin(connected_studies)]

    if connected_data.empty:
        raise ValueError("You have excluded all the studies")

    if set(labeled["Study"].astype(str)) != set(connected_data["Study"].astype(str)):
        if logger is not None:
            disconnected = sorted(study_labels - set(connected_data["Study"].astype(str)))
            if disconnected:
                logger("The uploaded data comprises a disconnected network. "
                       "Only the subnetwork containing the reference treatment "
                       f"({reference_treatment}) will be displayed and disconnected studies are shown in the logger.")
                logger("Disconnected studies: " + ",".join(disconnected))

    non_covariate_data = _remove_covariates(connected_data)

    covariate = {}
    covars = _find_covariate_columns(connected_data)
    if covars:
        covariate_column = covars[0]
        covariate = {
            "column": covariate_column,
            "name": str(covariate_column).replace("covar.", ""),
            "type": _infer_covariate_type(connected_data[covariate_column]),
        }

    bugsnet = _build_bugsnet(connected_data, configured_data.outcome, treatments)
    freq = _build_freq_payload(
        non_cov=non_covariate_data,
        outcome=configured_data.outcome,
        treatments=treatments,
        outcome_measure=configured_data.outcome_measure,
        effects=configured_data.effects,
        reference_treatment=reference_treatment,
    )

    return ExcludedData(
        treatments=treatments.reset_index(drop=True),
        reference_treatment=reference_treatment,
        connected_data=connected_data.reset_index(drop=True),
        covariate=covariate,
        bugsnet=bugsnet,
        freq=freq,
        outcome=configured_data.outcome,
        outcome_measure=configured_data.outcome_measure,
        effects=configured_data.effects,
        ranking_option=configured_data.ranking_option,
        seed=configured_data.seed,
    )
