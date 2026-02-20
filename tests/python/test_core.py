from pathlib import Path

import pandas as pd
import pytest

from metainsight.core import _coerce_and_validate_data, load_data, setup_configure


def test_load_data_from_dataframe() -> None:
    df = pd.DataFrame(
        {
            "study": ["s1", "s2", "s3"],
            "treatment": ["A", "B", "A"],
            "estimate": [0.2, 0.1, -0.4],
            "se": [0.3, 0.2, 0.25],
        }
    )

    normalized = load_data(df)

    assert list(normalized.columns) == ["study_id", "treatment", "estimate", "se"]
    assert normalized.shape == (3, 4)


def test_coerce_raises_when_missing_columns() -> None:
    df = pd.DataFrame({"study": ["s1"], "estimate": [1.0], "se": [0.2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        _coerce_and_validate_data(df)


def test_setup_configure_builds_bundle() -> None:
    df = pd.DataFrame(
        {
            "study_id": ["s1", "s1", "s2", "s3"],
            "treatment": ["Placebo", "Drug", "Placebo", "Drug"],
            "estimate": [0.1, -0.2, 0.0, 0.4],
            "se": [0.5, 0.3, 0.2, 0.6],
        }
    )

    bundle = setup_configure(
        df,
        outcome="Continuous",
        outcome_measure="MD",
        reference_treatment="Placebo",
        model_type="random",
    )

    assert bundle.study_count == 3
    assert bundle.treatment_count == 2


def test_setup_configure_reference_validation(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "study": ["s1", "s2"],
            "treatment": ["A", "A"],
            "estimate": [0.2, 0.1],
            "se": [0.2, 0.3],
        }
    )

    normalized = load_data(df)
    with pytest.raises(ValueError, match="Reference treatment"):
        setup_configure(
            normalized,
            outcome="Binary",
            outcome_measure="OR",
            reference_treatment="Missing",
        )
