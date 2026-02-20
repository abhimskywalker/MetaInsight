"""Core data and analysis scaffolding for the Python port of MetaInsight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd


DEFAULT_OUTCOME_COLUMNS = {
    "study_id": ["study", "study_id", "trial"],
    "treatment": ["treat", "treatment", "arm"],
    "baseline": ["baseline", "control", "comparator"],
    "estimate": ["estimate", "effect", "mean", "log_odds"],
    "se": ["se", "std_error", "standard_error"],
}


@dataclass(frozen=True)
class MetaInsightConfig:
    """Validated configuration for a MetaInsight analysis run."""

    outcome: str
    outcome_measure: Literal["MD", "SMD", "OR", "RR", "HR"]
    reference_treatment: str
    model_type: str = "random"
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", self.metadata or {})


@dataclass(frozen=True)
class MetaInsightBundle:
    """Container with raw study-level data and derived treatment info."""

    data: pd.DataFrame
    treatment_df: pd.DataFrame
    config: MetaInsightConfig

    @property
    def study_count(self) -> int:
        return int(self.data["study_id"].nunique())

    @property
    def treatment_count(self) -> int:
        return int(self.treatment_df["treatment"].nunique())


def _infer_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Infer one canonical column name from candidate alternatives."""

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return ""


def _coerce_and_validate_data(data: pd.DataFrame) -> pd.DataFrame:
    required = [
        _infer_column(data, DEFAULT_OUTCOME_COLUMNS["study_id"]),
        _infer_column(data, DEFAULT_OUTCOME_COLUMNS["treatment"]),
        _infer_column(data, DEFAULT_OUTCOME_COLUMNS["estimate"]),
        _infer_column(data, DEFAULT_OUTCOME_COLUMNS["se"]),
    ]

    missing = [col for col in required if not col]
    if missing:
        raise ValueError(
            "Missing required columns. Expected columns equivalent to: "
            f"{', '.join(required)}."
        )

    df = data.copy()
    # Standardize canonical names used by the package API.
    rename_map = {
        required[0]: "study_id",
        required[1]: "treatment",
        required[2]: "estimate",
        required[3]: "se",
    }
    df = df.rename(columns=rename_map)
    df["study_id"] = df["study_id"].astype(str)
    df["treatment"] = df["treatment"].astype(str)
    numeric_cols = ["estimate", "se"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid = df[df["se"] <= 0]
    if not invalid.empty:
        raise ValueError("All standard errors must be strictly positive.")

    missing_rows = df[["study_id", "estimate", "se"]].isna().any(axis=1)
    if missing_rows.any():
        df = df[~missing_rows]

    if df.empty:
        raise ValueError("No complete rows available after validation.")

    return df[["study_id", "treatment", "estimate", "se"]]


def load_data(path_or_data: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load study-level NMA input data from a file path or DataFrame."""

    if isinstance(path_or_data, pd.DataFrame):
        return _coerce_and_validate_data(path_or_data)

    data_path = Path(path_or_data)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find file: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        raw = pd.read_csv(data_path)
    elif suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(data_path)
    elif suffix in {".tsv"}:
        raw = pd.read_csv(data_path, sep="\t")
    elif suffix in {".parquet", ".pq"}:
        raw = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file format for study-level data.")

    return _coerce_and_validate_data(raw)


def setup_configure(
    data: pd.DataFrame,
    outcome: str,
    outcome_measure: Literal["MD", "SMD", "OR", "RR", "HR"] = "MD",
    reference_treatment: str = "Placebo",
    model_type: str = "random",
    **metadata: Any,
) -> MetaInsightBundle:
    """Validate configuration and build a normalized data bundle."""

    if not outcome:
        raise ValueError("Outcome name cannot be empty")
    if model_type not in {"random", "fixed"}:
        raise ValueError("model_type must be either 'random' or 'fixed'")
    if not data["treatment"].eq(reference_treatment).any():
        raise ValueError("Reference treatment is not present in treatment column")

    treatment_df = data[["treatment"]].drop_duplicates().copy().reset_index(drop=True)
    config = MetaInsightConfig(
        outcome=outcome,
        outcome_measure=outcome_measure,
        reference_treatment=reference_treatment,
        model_type=model_type,
        metadata=dict(metadata),
    )

    return MetaInsightBundle(data=data.reset_index(drop=True), treatment_df=treatment_df, config=config)
