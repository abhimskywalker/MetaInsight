from pathlib import Path

from dataclasses import replace

import pandas as pd
import pytest
import warnings

from metainsight import bayes_nodesplit, bayes_nodesplit_plot
from metainsight.setup import setup_configure, setup_load


_TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "testthat" / "data"


def _configure_nodesplit() -> object:
    loaded = setup_load(
        data_path=_TEST_DATA_DIR / "Cont_nodesplit.csv",
        outcome="continuous",
    )
    return setup_configure(
        loaded,
        reference_treatment="Placebo",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=123,
    )


def test_bayes_nodesplit_compatibility_shape() -> None:
    configured = _configure_nodesplit()

    result = bayes_nodesplit(configured)
    assert isinstance(result, dict)
    assert len(result) == 9
    assert all(isinstance(value, dict) for value in result.values())
    assert all(value.get("_model_type") == "mtc.result" for value in result.values())

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        async_result = bayes_nodesplit(configured, async_=True)
    assert len(async_result) == 9
    assert any("async" in str(msg.message).lower() for msg in captured)
    assert all(value.get("comparison", "").count(":") == 1 for value in async_result.values())

    plot = bayes_nodesplit_plot(result)
    assert plot.startswith("<svg")
    assert "all studies" in plot

    plot_sub = bayes_nodesplit_plot(result, main_analysis=False)
    assert plot_sub.startswith("<svg")
    assert "selected studies excluded" in plot_sub

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        async_plot = bayes_nodesplit_plot(result, async_=True)
    assert async_plot.startswith("<svg")
    assert any("async" in str(msg.message).lower() for msg in captured)


def test_bayes_nodesplit_not_nodesplittable(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "Study": ["S1", "S1", "S2", "S2"],
            "T": ["A", "B", "A", "C"],
            "N": [10, 10, 12, 12],
            "Mean": [1.0, 1.1, 1.2, 1.1],
            "SD": [0.5, 0.6, 0.4, 0.3],
        }
    )

    path = tmp_path / "not_nodesplit.csv"
    frame.to_csv(path, index=False)

    loaded = setup_load(data_path=path, outcome="continuous")
    configured = setup_configure(
        loaded,
        reference_treatment="A",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=123,
    )

    with pytest.raises(ValueError, match=r"There are no loops in the network\.?$"):
        bayes_nodesplit(configured)


def test_bayes_nodesplit_validation() -> None:
    configured = _configure_nodesplit()
    result = bayes_nodesplit(configured)

    with pytest.raises(ValueError, match="configured_data must be of class configured_data"):
        bayes_nodesplit("bad")
    with pytest.raises(ValueError, match="nodesplit must be a mtc.nodesplit object"):
        bayes_nodesplit_plot("bad")
    with pytest.raises(ValueError, match="main_analysis must be either 'TRUE' or 'FALSE'"):
        bayes_nodesplit_plot(result, main_analysis="TRUE")


def test_bayes_nodesplit_unsupported_measure() -> None:
    configured = _configure_nodesplit()
    with pytest.raises(ValueError, match="Standardised mean difference currently cannot be analysed in Bayesian analysis"):
        bayes_nodesplit(replace(configured, outcome_measure="SMD"))
    with pytest.raises(ValueError, match="Bayesian analysis of risk differences is not currently implemented in MetaInsight"):
        bayes_nodesplit(replace(configured, outcome_measure="RD"))
