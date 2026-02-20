"""Input loading and validation helpers for MetaInsight.

This module is a Python port of the `setup_load` and related validation
helpers from the original R package.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    cleaned[object_cols] = cleaned[object_cols].applymap(_tidy_string_item)
    return cleaned


def _canonicalize_column_name(columns: Iterable[str], outcome: str) -> List[str]:
    definitions = _get_schema(outcome)
    canonical = []
    for column in columns:
        canonical_name = column
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
        values = data[treatment_cols].stack(dropna=True).dropna().astype(str)

    # Preserve original order and uniqueness
    return list(dict.fromkeys(values.tolist()))


def _create_treatment_ids(all_treatments: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"Number": list(range(1, len(all_treatments) + 1)), "Label": all_treatments})


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
                # No suffix -> could be long format.
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

    n_rob_individual = sum(bool(re.match(r"^rob\.", str(col)) and "." in str(col)) for col in rob_indirectness)
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
