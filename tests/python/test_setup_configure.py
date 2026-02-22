from pathlib import Path

import pandas as pd
import pytest

from metainsight.setup import ConfiguredData, LoadedData, setup_configure, setup_load


_TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "testthat" / "data"


def _load_binary() -> LoadedData:
    return setup_load(data_path=_TEST_DATA_DIR / "Binary_long.csv", outcome="binary")


def _load_continuous() -> LoadedData:
    return setup_load(data_path=_TEST_DATA_DIR / "Cont_long.csv", outcome="continuous")


def _load_continuous_with_cov() -> LoadedData:
    return setup_load(data_path=_TEST_DATA_DIR / "Cont_long_continuous_cov.csv", outcome="continuous")


def test_setup_configure_checks_input_types():
    loaded = _load_continuous()

    with pytest.raises(TypeError, match="loaded_data must be of class loaded_data"):
        setup_configure(
            loaded_data="not_loaded",
            reference_treatment="Placebo",
            effects="fixed",
            outcome_measure="MD",
            ranking_option="good",
            seed=999,
        )

    with pytest.raises(TypeError, match="reference_treatment must be of class character"):
        setup_configure(
            loaded,
            reference_treatment=123,
            effects="fixed",
            outcome_measure="MD",
            ranking_option="good",
            seed=999,
        )

    with pytest.raises(TypeError, match="effects must be of class character"):
        setup_configure(
            loaded,
            reference_treatment="Placebo",
            effects=123,
            outcome_measure="MD",
            ranking_option="good",
            seed=999,
        )

    with pytest.raises(TypeError, match="outcome_measure must be of class character"):
        setup_configure(
            loaded,
            reference_treatment="Placebo",
            effects="fixed",
            outcome_measure=123,
            ranking_option="good",
            seed=999,
        )

    with pytest.raises(TypeError, match="ranking_option must be of class character"):
        setup_configure(
            loaded,
            reference_treatment="Placebo",
            effects="fixed",
            outcome_measure="MD",
            ranking_option=123,
            seed=999,
        )

    with pytest.raises(TypeError, match="seed must be of class numeric"):
        setup_configure(
            loaded,
            reference_treatment="Placebo",
            effects="fixed",
            outcome_measure="MD",
            ranking_option="good",
            seed="999",
        )


def test_setup_configure_checks_parameter_values():
    loaded = _load_continuous_with_cov()

    with pytest.raises(ValueError, match="reference_treatment must be present in the loaded data"):
        setup_configure(
            loaded,
            reference_treatment="not-present",
            effects="fixed",
            outcome_measure="MD",
            ranking_option="good",
            seed=99,
        )

    with pytest.raises(ValueError, match="outcome_measure must be either MD or SMD"):
        setup_configure(
            loaded,
            reference_treatment="the Great",
            effects="fixed",
            outcome_measure="INVALID",
            ranking_option="good",
            seed=99,
        )

    binary = _load_binary()
    with pytest.raises(ValueError, match="outcome_measure must be either OR, RR or RD"):
        setup_configure(
            binary,
            reference_treatment="Placebo",
            effects="fixed",
            outcome_measure="INVALID",
            ranking_option="good",
            seed=99,
        )

    with pytest.raises(ValueError, match="ranking_option must be either good or bad"):
        setup_configure(
            loaded,
            reference_treatment="the Great",
            effects="fixed",
            outcome_measure="MD",
            ranking_option="not_good",
            seed=99,
        )


def test_setup_configure_returns_expected_structure_binary_and_continuous():
    loaded_binary = _load_binary()
    result_binary = setup_configure(
        loaded_binary,
        reference_treatment="the Great",
        effects="random",
        outcome_measure="OR",
        ranking_option="good",
        seed=999,
    )

    assert isinstance(result_binary, ConfiguredData)
    assert isinstance(result_binary.wrangled_data, pd.DataFrame)
    assert isinstance(result_binary.treatments, pd.DataFrame)
    assert isinstance(result_binary.reference_treatment, str)
    assert isinstance(result_binary.connected_data, pd.DataFrame)
    assert isinstance(result_binary.non_covariate_data, pd.DataFrame)
    assert isinstance(result_binary.covariate, dict)
    assert isinstance(result_binary.bugsnet, pd.DataFrame)
    assert isinstance(result_binary.freq, dict)
    assert result_binary.outcome == "binary"
    assert result_binary.outcome_measure == "OR"
    assert result_binary.effects == "random"
    assert result_binary.ranking_option == "good"
    assert isinstance(result_binary.disconnected_indices, list)
    assert isinstance(result_binary.seed, (int, float))

    loaded_cont = _load_continuous()
    result_cont = setup_configure(
        loaded_cont,
        reference_treatment="the Great",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=999,
    )

    assert isinstance(result_cont, ConfiguredData)
    assert isinstance(result_cont.wrangled_data, pd.DataFrame)
    assert result_cont.outcome == "continuous"
    assert result_cont.reference_treatment == "the_Great"
    assert len(result_cont.disconnected_indices) == 0


def test_setup_configure_wrangled_data_properties_continuous():
    loaded = _load_continuous()
    result = setup_configure(
        loaded,
        reference_treatment="the Great",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=999,
    )

    assert "StudyID" in result.wrangled_data.columns
    assert result.wrangled_data["StudyID"].nunique() == result.connected_data["StudyID"].nunique()
    assert len(result.connected_data) >= 2 * result.wrangled_data["StudyID"].nunique()
    assert set(result.wrangled_data["T"].dropna().unique()).issubset(set(range(1, 7)))
    assert len(result.wrangled_data) == len(loaded.data)


def test_setup_configure_disconnected_logger_message():
    loaded = setup_load(data_path=_TEST_DATA_DIR / "continuous_long_disconnected.csv", outcome="continuous")
    messages = []

    result = setup_configure(
        loaded,
        reference_treatment="A",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=2024,
        logger=messages.append,
    )

    assert result.disconnected_indices
    assert any("disconnected network" in msg for msg in messages)
    assert all("Disconnected studies" in msg for msg in messages if "Disconnected studies" in msg)


def test_setup_configure_covariate_detection():
    result_with_cov = setup_configure(
        _load_continuous_with_cov(),
        reference_treatment="the Great",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=999,
    )

    assert result_with_cov.covariate["column"] == "covar.age"
    assert result_with_cov.covariate["name"] == "age"
    assert result_with_cov.covariate["type"] == "continuous"

    result_without_cov = setup_configure(
        _load_continuous(),
        reference_treatment="the Great",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=999,
    )
    assert result_without_cov.covariate == {}
