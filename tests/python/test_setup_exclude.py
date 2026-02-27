from pathlib import Path

import pytest

from metainsight import ConfiguredData, ExcludedData, setup_configure, setup_exclude, setup_load


_TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "testthat" / "data"


def _configured_continuous_with_cov() -> ConfiguredData:
    # using the same source data that includes covariates and a reference that can change
    loaded = setup_load(
        data_path=_TEST_DATA_DIR / "Cont_long_continuous_cov.csv",
        outcome="continuous",
    )
    return setup_configure(
        loaded,
        reference_treatment="the Little",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=2024,
    )


def test_setup_exclude_checks_input_types_and_values() -> None:
    configured = _configured_continuous_with_cov()

    with pytest.raises(TypeError, match="configured_data must be of class configured_data"):
        setup_exclude("not_configured", ["Leo"])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="exclusions must be of class character"):
        setup_exclude(configured, [123])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="exclusions must in the present in the loaded data"):
        setup_exclude(configured, ["Study1000"])



def test_setup_exclude_returns_expected_structure() -> None:
    configured = _configured_continuous_with_cov()
    result = setup_exclude(configured, ["Leo"])

    assert isinstance(result, ExcludedData)
    assert isinstance(result.treatments, type(configured.treatments))
    assert isinstance(result.reference_treatment, str)
    assert isinstance(result.connected_data, type(configured.connected_data))
    assert isinstance(result.covariate, dict)
    assert isinstance(result.bugsnet, type(configured.bugsnet))
    assert isinstance(result.freq, dict)
    assert isinstance(result.outcome, str)
    assert isinstance(result.outcome_measure, str)
    assert isinstance(result.effects, str)
    assert isinstance(result.ranking_option, str)
    assert isinstance(result.seed, (int, float))

    # dataset has a covariate column and should preserve structured metadata
    assert "column" in result.covariate
    assert result.covariate["column"] == "covar.age"
    assert result.covariate["name"] == "age"
    assert result.covariate["type"] in {"continuous", "binary"}


def test_setup_exclude_removes_selected_studies() -> None:
    configured = _configured_continuous_with_cov()
    base_studies = set(configured.connected_data["Study"].astype(str).unique())
    result = setup_exclude(configured, ["Leo", "Constantine"])

    assert "Leo" not in set(result.connected_data["Study"].astype(str).unique())
    assert "Constantine" not in set(result.connected_data["Study"].astype(str).unique())
    assert "Justinian" in set(result.connected_data["Study"].astype(str).unique())
    assert len(result.connected_data["Study"].unique()) == len(base_studies) - 2
    assert all(study not in set(result.bugsnet["Study"]) for study in {"Leo", "Constantine"})
    assert all(study not in set(result.freq["d0"]["studlab"]) for study in {"Leo", "Constantine"})
    assert len(set(result.freq["d0"]["studlab"])) == len(base_studies) - 2


def test_setup_exclude_changes_reference_when_necessary() -> None:
    configured = _configured_continuous_with_cov()
    logs: list[str] = []

    result = setup_exclude(
        configured,
        ["Leo"],
        logger=logs.append,
    )

    # excluding Leo removes the original reference treatment, so first connected treatment changes
    assert result.reference_treatment == "the_Great"
    assert configured.reference_treatment == "the_Little"
    assert logs
    assert "has been changed to the_Great" in logs[0]


def test_setup_exclude_no_exclusions_is_noop() -> None:
    configured = _configured_continuous_with_cov()
    result = setup_exclude(configured, None)

    # setup_exclude rebuilds non-covariate payloads, so covariate-only columns may differ.
    assert {"Study", "Treatment", "N", "Mean"}.issubset(set(result.bugsnet.columns))
    assert {"Study", "Treatment", "N", "Mean"}.issubset(set(configured.bugsnet.columns))
    assert list(result.treatments["Label"]) == list(configured.treatments["Label"])
    assert result.reference_treatment == configured.reference_treatment
    assert set(result.connected_data["Study"]) == set(configured.connected_data["Study"])


def test_setup_exclude_rejects_empty_dataset() -> None:
    configured = _configured_continuous_with_cov()
    all_studies = sorted(set(configured.connected_data["Study"]))

    with pytest.raises(ValueError, match="You have excluded all the studies"):
        setup_exclude(configured, all_studies)
