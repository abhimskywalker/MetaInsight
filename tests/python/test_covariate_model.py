from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from metainsight import (
    bayes_details,
    bayes_mcmc,
    bayes_ranking,
    bayes_results,
    covariate_comparison,
    covariate_deviance,
    covariate_details,
    covariate_forest,
    covariate_mcmc,
    covariate_model,
    covariate_ranking,
    covariate_results,
    covariate_summary,
    find_covariate_ranges,
    setup_configure,
    setup_load,
)

_TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "testthat" / "data"


def _configure_with_cov() -> object:
    loaded = setup_load(
        data_path=_TEST_DATA_DIR / "Contribution_continuous_long_continuous_cov.csv",
        outcome="continuous",
    )
    configured = setup_configure(
        loaded,
        reference_treatment="Paracetamol",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=123,
    )
    return configured


def _assert_range_dict(actual: dict[str, float], expected: dict[str, float]) -> None:
    assert set(actual.keys()) == set(expected.keys())
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if pd.isna(expected_value):
            assert pd.isna(actual_value)
        else:
            assert actual_value == expected_value


def test_find_covariate_ranges_continuous_long() -> None:
    configured = _configure_with_cov()
    ranges = find_covariate_ranges(
        connected_data=configured.connected_data,
        treatment_ids=configured.treatments,
        reference_treatment=configured.reference_treatment,
        covariate_title=configured.covariate["column"],
        outcome=configured.outcome,
    )

    _assert_range_dict(
        ranges["min"],
        {
            "Ibuprofen": 98,
            "A_stiff_drink": 95,
            "Sleep": 97,
            "Exercise": float("nan"),
        },
    )
    _assert_range_dict(
        ranges["max"],
        {
            "Ibuprofen": 99,
            "A_stiff_drink": 99,
            "Sleep": 98,
            "Exercise": float("nan"),
        },
    )


def test_find_covariate_ranges_baseline_risk_continuous() -> None:
    connected_data = pd.DataFrame(
        {
            "Study": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "T": ["1", "2", "1", "3", "2", "3"],
            "N": [100, 100, 40, 40, 50, 50],
            "Mean": [1.0, 2.0, 1.5, 2.5, 3.0, 4.0],
            "SD": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        }
    )
    treatment_ids = pd.DataFrame(
        {
            "Number": [1, 2, 3],
            "Label": ["A", "B", "C"],
        }
    )

    ranges = find_covariate_ranges(
        connected_data=connected_data,
        treatment_ids=treatment_ids,
        reference_treatment="A",
        covariate_title="ignored",
        baseline_risk=True,
        outcome="continuous",
    )

    _assert_range_dict(
        ranges["min"],
        {
            "B": 1.0,
            "C": 1.5,
        },
    )
    _assert_range_dict(
        ranges["max"],
        {
            "B": 1.0,
            "C": 1.5,
        },
    )

    # int reference IDs should resolve to labels via treatment mapping
    ranges_with_numeric_ref = find_covariate_ranges(
        connected_data=connected_data,
        treatment_ids=treatment_ids,
        reference_treatment=1,
        covariate_title="ignored",
        baseline_risk=True,
        outcome="continuous",
    )

    _assert_range_dict(
        ranges_with_numeric_ref["min"],
        {
            "B": 1.0,
            "C": 1.5,
        },
    )
    _assert_range_dict(
        ranges_with_numeric_ref["max"],
        {
            "B": 1.0,
            "C": 1.5,
        },
    )


def test_find_covariate_ranges_continuous_wide() -> None:
    loaded = setup_load(
        data_path=_TEST_DATA_DIR / "Contribution_continuous_wide_continuous_cov.csv",
        outcome="continuous",
    )
    configured = setup_configure(
        loaded,
        reference_treatment="Paracetamol",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=123,
    )

    ranges = find_covariate_ranges(
        connected_data=configured.connected_data,
        treatment_ids=configured.treatments,
        reference_treatment=configured.reference_treatment,
        covariate_title="covar.age",
        outcome=configured.outcome,
    )

    _assert_range_dict(
        ranges["min"],
        {
            "Ibuprofen": 98,
            "A_stiff_drink": 95,
            "Sleep": 97,
            "Exercise": float("nan"),
        },
    )
    _assert_range_dict(
        ranges["max"],
        {
            "Ibuprofen": 99,
            "A_stiff_drink": 99,
            "Sleep": 98,
            "Exercise": float("nan"),
        },
    )


def test_covariate_model_output_structure() -> None:
    configured = _configure_with_cov()
    result = covariate_model(configured, 98, "shared")

    expected_keys = {
        "mtcResults",
        "mtcRelEffects",
        "rel_eff_tbl",
        "covariate_value",
        "reference_treatment",
        "comparator_names",
        "a",
        "sumresults",
        "dic",
        "cov_value_sentence",
        "slopes",
        "intercepts",
        "outcome",
        "outcome_measure",
        "mtcNetwork",
        "effects",
        "covariate_min",
        "covariate_max",
    }
    assert expected_keys.issubset(set(result))

    assert result["reference_treatment"] == "Paracetamol"
    assert result["outcome"] == "continuous"
    assert result["outcome_measure"] == "MD"
    assert result["effects"] == "random"
    assert result["a"] == "random effect"
    assert result["covariate_value"] == 98.0
    assert result["cov_value_sentence"] == "Value for covariate age set at 98"
    assert list(result["comparator_names"]) == [
        "Ibuprofen",
        "A_stiff_drink",
        "Sleep",
        "Exercise",
    ]

    assert isinstance(result["slopes"], pd.Series)
    assert isinstance(result["intercepts"], pd.Series)
    assert list(result["slopes"].index) == list(result["comparator_names"])
    assert list(result["intercepts"].index) == list(result["comparator_names"])

    _assert_range_dict(
        result["covariate_min"],
        {
            "Ibuprofen": 98,
            "A_stiff_drink": 95,
            "Sleep": 97,
            "Exercise": float("nan"),
        },
    )
    _assert_range_dict(
        result["covariate_max"],
        {
            "Ibuprofen": 99,
            "A_stiff_drink": 99,
            "Sleep": 98,
            "Exercise": float("nan"),
        },
    )


def test_covariate_model_respects_covariate_value_and_regressor_type() -> None:
    configured = _configure_with_cov()
    baseline = covariate_model(configured, 98, "shared")
    updated = covariate_model(configured, 99, "shared", covariate_model_output=baseline)

    assert updated["covariate_value"] == 99.0
    assert updated["cov_value_sentence"] == "Value for covariate age set at 99"
    assert not baseline["sumresults"].equals(updated["sumresults"])

    cached = covariate_model(configured, 98, "shared", covariate_model_output=updated)
    for comparator in baseline["comparator_names"]:
        slope = float(updated["slopes"][comparator])
        expected = float(updated["intercepts"][comparator]) - 1.0 * slope
        assert float(cached["intercepts"][comparator]) == pytest.approx(expected)

    different_type = covariate_model(configured, 99, "unrelated", covariate_model_output=baseline)
    assert different_type["regressor_type"] == "unrelated"
    assert not updated["slopes"].equals(different_type["slopes"])


def test_covariate_summary_outputs_svg() -> None:
    configured = _configure_with_cov()
    svg = covariate_summary(configured)
    assert svg.startswith("<svg")


def test_covariate_summary_validation_error() -> None:
    loaded = setup_load(data_path=_TEST_DATA_DIR / "Cont_long.csv", outcome="continuous")
    configured = setup_configure(
        loaded,
        reference_treatment="the Great",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=123,
    )

    with pytest.raises(ValueError, match="The data does not contain a covariate column"):
        covariate_summary(configured)


def test_covariate_model_validation_errors() -> None:
    configured = _configure_with_cov()

    with pytest.raises(ValueError, match="configured_data must be of class configured_data"):
        covariate_model("not-configured", 98, "shared")

    with pytest.raises(ValueError, match="covariate_value must be of class numeric"):
        covariate_model(configured, "98", "shared")

    with pytest.raises(ValueError, match="regressor_type must be of class character"):
        covariate_model(configured, 98, 123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="regressor_type must be "):
        covariate_model(configured, 98, "invalid")

    with pytest.raises(ValueError, match="configured data must have an outcome_measure of 'OR', 'RR' or 'MD'"):
        bad_outcome = replace(configured, outcome_measure="SMD")
        covariate_model(bad_outcome, 98, "shared")

    with pytest.raises(ValueError, match="must not be lower than the minimum"):
        covariate_model(configured, 1, "shared")

    with pytest.raises(ValueError, match="must not be higher than the maximum"):
        covariate_model(configured, 1_000, "shared")

    no_cov_loaded = setup_load(data_path=_TEST_DATA_DIR / "Cont_long.csv", outcome="continuous")
    no_cov_config = setup_configure(
        no_cov_loaded,
        reference_treatment="the Great",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=123,
    )
    with pytest.raises(ValueError, match="The data does not contain a covariate column"):
        covariate_model(no_cov_config, 99, "shared")


def test_covariate_aliases_for_ranking_and_diagnostics() -> None:
    configured = _configure_with_cov()
    model = covariate_model(configured, 98, "shared")

    ranking = covariate_ranking(model, configured)
    assert set(ranking.keys()) == {
        "SUCRA",
        "Colour",
        "Cumulative",
        "Probabilities",
        "BUGSnetData",
    }

    results = covariate_results(model)
    assert "<br/>" in results

    details = covariate_details(model)
    assert set(details.keys()) == {"mcmc", "priors"}

    mcmc = covariate_mcmc(model)
    assert set(mcmc.keys()) == {"parameters", "gelman_data", "n_cols", "n_rows", "n_rows_rmd"}

    # Compatibility helper calls
    assert isinstance(covariate_summary(configured), str)
    assert covariate_forest(model).startswith("<svg")
    assert covariate_deviance(model)
    comparison = covariate_comparison(model)
    assert comparison.shape[0] == len(configured.treatments)
    assert comparison.shape[1] == len(configured.treatments)

    # ensure alias output is consistent shape style
    assert ranking["SUCRA"].shape[0] == len(model["comparator_names"]) + 1

    ordered_labels = [str(label) for label in configured.treatments["Label"]]
    assert ranking["SUCRA"]["Treatment"].tolist() != sorted(ordered_labels)

    bayes_rank = bayes_ranking(model, configured)
    assert set(bayes_rank.keys()) == set(ranking.keys())
    pd.testing.assert_frame_equal(bayes_rank["SUCRA"], ranking["SUCRA"])

    assert bayes_results(model) == results
    pd.testing.assert_frame_equal(bayes_details(model)["mcmc"], details["mcmc"])
    pd.testing.assert_frame_equal(bayes_details(model)["priors"], details["priors"])
    assert bayes_mcmc(model) == mcmc

    with pytest.raises(ValueError, match=r"model must be an object created by baseline_model\(\), bayes_model\(\) or covariate_model\(\)"):
        bayes_ranking("bad", configured)
    with pytest.raises(ValueError, match=r"model must be an object created by baseline_model\(\), bayes_model\(\) or covariate_model\(\)"):
        bayes_results("bad")
    with pytest.raises(ValueError, match=r"model must be an object created by baseline_model\(\), bayes_model\(\) or covariate_model\(\)"):
        bayes_details("bad")
    with pytest.raises(ValueError, match=r"model must be an object created by baseline_model\(\), bayes_model\(\) or covariate_model\(\)"):
        bayes_mcmc("bad")
