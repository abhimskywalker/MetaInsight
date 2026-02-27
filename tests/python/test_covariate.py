from __future__ import annotations

from pathlib import Path
from math import log

import numpy as np
import pandas as pd
import pytest

from metainsight import (
    CovariateModel,
    ConfiguredData,
    calculate_credible_regions,
    calculate_directness,
    covariate_regression,
    setup_configure,
    setup_load,
)


_TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "testthat" / "data"


def _to_treatment_ids() -> pd.DataFrame:
    return pd.DataFrame({"Number": [1, 2, 3, 4], "Label": ["A", "B", "C", "D"]})


def _logit(p: float) -> float:
    return log(p / (1 - p))


def _load_directness_fixture() -> pd.DataFrame:
    return pd.read_csv(_TEST_DATA_DIR / "Test_directness.csv")


def _configure_continuous_with_cov() -> ConfiguredData:
    loaded = setup_load(
        data_path=_TEST_DATA_DIR / "Cont_long_continuous_cov.csv",
        outcome="continuous",
    )
    return setup_configure(
        loaded_data=loaded,
        reference_treatment="the Great",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=123,
    )



def test_calculate_directness_gathers_covariate_values() -> None:
    data = _load_directness_fixture()
    treatment_ids = _to_treatment_ids()
    data = data.copy()
    data["T"] = data["T"].map({"A": 1, "B": 2, "C": 3, "D": 4})

    contributions = calculate_directness(
        data=data,
        covariate_title="covar.age",
        treatment_ids=treatment_ids,
        outcome="binary",
        outcome_measure="OR",
        effects_type="random",
    )

    studies = list(data["Study"].unique())
    expected_cov = pd.Series(data["covar.age"].tolist(), index=data["Study"]).groupby("Study").first()
    expected_cov = expected_cov.reindex(studies)

    assert list(contributions["covariate_value"].index) == studies
    assert contributions["covariate_value"].tolist() == expected_cov.tolist()


def test_calculate_directness_correctly_fills_direct_and_indirect_matrices() -> None:
    data = _load_directness_fixture()
    treatment_ids = _to_treatment_ids()
    data = data.copy()
    data["T"] = data["T"].map({"A": 1, "B": 2, "C": 3, "D": 4})

    contributions = calculate_directness(
        data=data,
        covariate_title="covar.age",
        treatment_ids=treatment_ids,
        outcome="binary",
        outcome_measure="OR",
        effects_type="random",
    )

    studies = list(data["Study"].unique())

    expected_is_direct = pd.DataFrame(
        data=np.array(
            [
                [True, False, False],
                [True, True, False],
                [False, True, False],
                [False, False, True],
                [False, False, False],
            ],
            dtype=bool,
        ),
        index=studies,
        columns=["B", "C", "D"],
    )

    expected_is_indirect = pd.DataFrame(
        data=np.array(
            [
                [False, True, False],
                [True, True, False],
                [True, False, False],
                [False, False, False],
                [True, True, False],
            ],
            dtype=bool,
        ),
        index=studies,
        columns=["B", "C", "D"],
    )

    pd.testing.assert_frame_equal(contributions["is_direct"], expected_is_direct)
    pd.testing.assert_frame_equal(contributions["is_indirect"], expected_is_indirect)


def test_calculate_directness_relative_effects_match_pairwise_contrast_sign() -> None:
    data = _load_directness_fixture()
    treatment_ids = _to_treatment_ids()
    data = data.copy()
    data["T"] = data["T"].map({"A": 1, "B": 2, "C": 3, "D": 4})

    contributions = calculate_directness(
        data=data,
        covariate_title="covar.age",
        treatment_ids=treatment_ids,
        outcome="binary",
        outcome_measure="OR",
        effects_type="random",
    )

    studies = list(data["Study"].unique())
    expected = pd.DataFrame(
        data=np.nan,
        index=studies,
        columns=["B", "C", "D"],
        dtype=float,
    )

    expected.loc["StudyAB", "B"] = _logit(85 / 114) - _logit(39 / 113)
    expected.loc["StudyABC", "B"] = _logit(59 / 66) - _logit(36 / 66)
    expected.loc["StudyABC", "C"] = _logit(53 / 66) - _logit(36 / 66)
    expected.loc["StudyAC", "C"] = _logit(78 / 89) - _logit(36 / 88)
    expected.loc["StudyAD", "D"] = _logit(126 / 134) - _logit(76 / 132)

    pd.testing.assert_frame_equal(contributions["relative_effect"], expected)


def test_calculate_credible_regions_structure() -> None:
    model = {
        "comparator_names": ["B", "C", "D"],
        "reference_treatment": "A",
        "covariate_min": {"B": 1.0, "C": 2.0, "D": 3.0},
        "covariate_max": {"B": 1.0, "C": 3.0, "D": 4.0},
        "regressor_type": "continuous",
    }

    result = calculate_credible_regions(model)

    assert set(result.keys()) == {"regions", "intervals"}
    assert set(result["regions"].keys()) == {"B", "C", "D"}
    assert set(result["intervals"].keys()) == {"B", "C", "D"}

    assert len(result["regions"]["B"]) == 1
    assert len(result["intervals"]["B"]) == 1
    assert not pd.isna(result["intervals"]["B"].iloc[0]["lower"])

    assert len(result["regions"]["C"]) == 11
    assert len(result["intervals"]["C"]) == 11
    assert (result["intervals"]["C"]["lower"] == result["intervals"]["C"]["upper"]).all()


def test_covariate_regression_returns_directness_and_credible_regions() -> None:
    configured = _configure_continuous_with_cov()
    model = CovariateModel(
        comparator_names=[label for label in configured.treatments["Label"].tolist() if label != configured.reference_treatment],
        reference_treatment=configured.reference_treatment,
        covariate_min={name: 1.0 for name in configured.treatments["Label"] if name != configured.reference_treatment},
        covariate_max={name: 2.0 for name in configured.treatments["Label"] if name != configured.reference_treatment},
        regressor_type="continuous",
    )

    result = covariate_regression(model, configured)

    assert set(result.keys()) == {"directness", "credible_regions", "_model_type"}

    directness = result["directness"]
    assert set(directness.keys()) == {"is_direct", "is_indirect", "relative_effect", "covariate_value"}

    credible_regions = result["credible_regions"]
    assert set(credible_regions.keys()) == {"regions", "intervals"}
    assert isinstance(credible_regions["regions"], dict)
    assert isinstance(credible_regions["intervals"], dict)


def test_covariate_regression_type_errors() -> None:
    configured = _configure_continuous_with_cov()
    model = CovariateModel(
        comparator_names=["A", "B"],
        reference_treatment="A",
        covariate_min={"B": 1.0},
        covariate_max={"B": 1.0},
    )

    with pytest.raises(ValueError, match="model must be of class covariate_model"):
        covariate_regression("bad", configured)

    with pytest.raises(ValueError, match="configured_data must be of class configured_data"):
        covariate_regression(model, "bad")


def test_covariate_async_flags_run_synchronously_with_warning() -> None:
    configured = _configure_continuous_with_cov()
    model = CovariateModel(
        comparator_names=[label for label in configured.treatments["Label"] if label != configured.reference_treatment],
        reference_treatment=configured.reference_treatment,
        covariate_min={label: 1.0 for label in configured.treatments["Label"] if label != configured.reference_treatment},
        covariate_max={label: 2.0 for label in configured.treatments["Label"] if label != configured.reference_treatment},
        regressor_type="continuous",
    )

    import warnings

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = covariate_regression(model, configured, async_=True)

    assert set(result.keys()) == {"directness", "credible_regions", "_model_type"} or set(result.keys()) == {"directness", "credible_regions"}
    assert any("async" in str(msg.message).lower() for msg in captured)

