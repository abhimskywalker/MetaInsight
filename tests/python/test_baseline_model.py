from pathlib import Path
import dataclasses

import numpy as np
import pandas as pd
import pytest
import warnings
from metainsight import (
    LitmusRankOGram,
    RadialSUCRA,
    baseline_comparison,
    baseline_deviance,
    baseline_details,
    baseline_forest,
    baseline_forest_limits,
    baseline_mcmc,
    baseline_model,
    baseline_ranking,
    baseline_regression,
    baseline_results,
    baseline_summary,
    setup_exclude,
    bayes_forest,
    bayes_deviance,
    bayes_compare,
    format_baseline_forest,
    gelman_plots,
    metaregression_plot,
    ranking_table,
    setup_configure,
    setup_load,
)

_TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "testthat" / "data"



def _configure_continuous() -> object:
    loaded = setup_load(data_path=_TEST_DATA_DIR / "Cont_long.csv", outcome="continuous")
    configured = setup_configure(
        loaded_data=loaded,
        reference_treatment="the Great",
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=999,
    )
    return configured


def _exclude_continuous() -> object:
    configured = _configure_continuous()
    return setup_exclude(configured, exclusions=["Leo", "Constantine"])


def _configure_binary() -> object:
    loaded = setup_load(data_path=_TEST_DATA_DIR / "Binary_long.csv", outcome="binary")
    configured = setup_configure(
        loaded_data=loaded,
        reference_treatment="the Great",
        effects="random",
        outcome_measure="OR",
        ranking_option="good",
        seed=123,
    )
    return configured


def test_baseline_model_output_structure_continuous() -> None:
    configured = _configure_continuous()
    result = baseline_model(configured, "shared")

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
        "regressor",
        "_model_type",
    }

    assert expected_keys.issubset(set(result))
    assert result["_model_type"] == "baseline_model"
    assert result["reference_treatment"] == configured.reference_treatment
    assert result["outcome"] == "continuous"
    assert result["outcome_measure"] == "MD"
    assert result["regressor"] == "shared"
    assert result["a"] == "random effect"

    expected_comparators = [
        str(name)
        for name in configured.treatments["Label"]
        if str(name) != configured.reference_treatment
    ]
    assert result["comparator_names"] == expected_comparators
    assert sorted(list(result["slopes"].index)) == sorted(expected_comparators)


def test_baseline_model_output_structure_binary() -> None:
    configured = _configure_binary()
    result = baseline_model(configured, "unrelated")

    assert result["outcome"] == "binary"
    assert result["outcome_measure"] == "OR"
    assert result["regressor"] == "unrelated"
    assert "Value for baseline risk set at" in result["cov_value_sentence"]
    assert isinstance(result["covariate_min"], dict)
    assert isinstance(result["covariate_max"], dict)


def test_baseline_regression_and_summary_and_forest() -> None:
    configured = _configure_continuous()
    model = baseline_model(configured, "shared")

    regression = baseline_regression(model, configured)
    assert set(regression.keys()) == {"directness", "credible_regions", "_model_type"}

    directness = regression["directness"]
    assert set(directness.keys()) == {"is_direct", "is_indirect", "relative_effect", "covariate_value"}

    credible = regression["credible_regions"]
    assert set(credible.keys()) == {"regions", "intervals"}

    summary_svg = baseline_summary(configured)
    forest_svg = baseline_forest(model)
    assert summary_svg.startswith("<svg")
    assert "baseline risk summary" in summary_svg
    assert "reference=" in summary_svg
    assert forest_svg.startswith("<svg")

    excluded = _exclude_continuous()
    assert baseline_summary(excluded).startswith("<svg")
    assert "baseline risk summary" in baseline_summary(excluded)

    dev = baseline_deviance(model)
    assert set(dev.keys()) == {"deviance_mtc", "stem_plot", "lev_plot"}
    deviance_payload = dev["deviance_mtc"]
    assert isinstance(deviance_payload, dict)
    assert isinstance(deviance_payload.get("dev.ab"), pd.DataFrame)
    assert isinstance(deviance_payload.get("fit.ab"), pd.DataFrame)
    assert deviance_payload["nd.ab"] >= 1
    assert "baseline deviance stem" in dev["stem_plot"]
    assert "baseline deviance leverage" in dev["lev_plot"]

    comparison = baseline_comparison(model)
    assert comparison.shape[0] == len(configured.treatments)
    assert comparison.shape[1] == len(configured.treatments)
    assert configured.reference_treatment in comparison.index
    assert configured.reference_treatment in comparison.columns

    other = str(configured.treatments["Label"].iloc[1])
    assert comparison.loc[configured.reference_treatment, other] == -comparison.loc[other, configured.reference_treatment]

    # transitive values are deterministically filled when reference contrasts are known
    second = str(configured.treatments["Label"].iloc[2])
    if pd.notna(comparison.loc[configured.reference_treatment, second]):
        derived = comparison.loc[configured.reference_treatment, other] - comparison.loc[configured.reference_treatment, second]
        assert comparison.loc[other, second] == derived
        assert comparison.loc[second, other] == -derived

    # forest-data formatting helper
    median_ci = pd.DataFrame(
        {
            configured.reference_treatment: ["",
             ""],
            configured.treatments["Label"].iloc[1]: ["[0.0, 0.1, 0.2]", "[0, 0.2, 0.4]"],
        },
        index=[configured.reference_treatment, configured.treatments["Label"].iloc[2]],
    )
    formatted = format_baseline_forest(median_ci, configured.reference_treatment)
    assert set(formatted.columns) == {"id", "pe", "ci.l", "ci.u", "style", "group"}
    assert configured.reference_treatment in formatted["group"].unique()
    xlimits = baseline_forest_limits(formatted)
    assert xlimits[0] <= 0 <= xlimits[1]


def test_baseline_forest_respects_xlim_and_validation() -> None:
    configured = _configure_continuous()
    model = baseline_model(configured, "shared")

    svg_default = baseline_forest(model)
    svg_min_only = baseline_forest(model, xmin=-1)
    svg_max_only = baseline_forest(model, xmax=2)
    svg_custom = baseline_forest(model, xmin=-2, xmax=2)
    svg_bayes = bayes_forest(model, title="bayes")
    dev_bayes = bayes_deviance(model)
    comp_bayes = bayes_compare(model)

    assert isinstance(dev_bayes, dict) and "deviance_mtc" in dev_bayes
    assert comp_bayes.shape == baseline_comparison(model).shape
    assert svg_bayes.startswith("<svg")

    assert svg_default.startswith("<svg")
    assert "treatments=" in svg_default
    assert svg_min_only.startswith("<svg")
    assert svg_max_only.startswith("<svg")
    assert svg_custom.startswith("<svg")
    assert svg_min_only != svg_custom
    assert svg_max_only != svg_custom

    with pytest.raises(ValueError, match="title must be of class character"):
        baseline_forest(model, title=123)
    with pytest.raises(ValueError, match="xmin must be of class numeric"):
        baseline_forest(model, xmin="bad", xmax=1)
    with pytest.raises(ValueError, match="xmin must be less than xmax"):
        baseline_forest(model, xmin=3, xmax=3)


def test_baseline_compare_and_ranking_outputs() -> None:
    configured_md = _configure_continuous()
    model_md = baseline_model(configured_md, "shared")

    comparison = baseline_comparison(model_md)
    transformed = bayes_compare(model_md)

    assert transformed.shape == comparison.shape
    pd.testing.assert_frame_equal(transformed, comparison.round(2))

    configured_or = _configure_binary()
    model_or = baseline_model(configured_or, "shared")
    transformed_or = bayes_compare(model_or)
    baseline_or = baseline_comparison(model_or)
    pd.testing.assert_frame_equal(transformed_or, np.exp(baseline_or).round(2))



def test_baseline_model_validation_errors() -> None:
    configured = _configure_continuous()

    with pytest.raises(ValueError, match="configured_data must be of class configured_data"):
        baseline_model("bad", "shared")

    with pytest.raises(ValueError, match=r"model must be an object created by bayes_model\(\) or covariate_model\(\)"):
        bayes_compare({"outcome_measure": "MD"})
    with pytest.raises(ValueError, match="regressor_type must be of class character"):
        baseline_model(configured, 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="regressor_type must be"):
        baseline_model(configured, "invalid")

    invalid_measure = dataclasses.replace(_configure_continuous(), outcome_measure="SMD")
    with pytest.raises(ValueError, match="configured data must have an outcome_measure of 'OR', 'RR' or 'MD'"):
        baseline_model(invalid_measure, "shared")

    with pytest.raises(ValueError, match="model must be of class baseline_model"):
        baseline_regression("bad", configured)

    with pytest.raises(ValueError, match="configured_data must be of class configured_data"):
        baseline_regression(model=baseline_model(configured, "shared"), configured_data="bad")

    with pytest.raises(ValueError, match="model must be an object created by baseline_model"):
        baseline_deviance("bad")

    # _replace doesn't exist if tests run on frozen dataclass in older pythons; guard:
    with pytest.raises(ValueError, match="configured_data must be of class configured_data"):
        baseline_summary("not-configured")


def test_baseline_metaregression_plot_validations() -> None:
    configured = _configure_continuous()
    model = baseline_model(configured, "shared")
    regression = baseline_regression(model, configured)

    comparators = [configured.treatments["Label"].iloc[2], configured.treatments["Label"].iloc[4]]
    plot = metaregression_plot(model, configured, regression, comparators, include_covariate=True)
    assert plot.startswith("<svg")

    with pytest.raises(ValueError, match=r"model must be an object created by baseline_model\(\) or covariate_model\(\)"):
        metaregression_plot("bad", configured, regression, comparators)

    with pytest.raises(ValueError, match=r"regression_data must be an object created by baseline_regression\(\) or covariate_regression\(\)"):
        metaregression_plot(model, configured, "bad", comparators)

    with pytest.raises(ValueError, match="comparators must be present in the configured data"):
        metaregression_plot(model, configured, regression, ["not-a-trt"])

    with pytest.raises(ValueError, match="comparators cannot contain the reference treatment"):
        metaregression_plot(model, configured, regression, [configured.reference_treatment])

    with pytest.raises(ValueError, match="credible_opacity must be between 0 and 1"):
        metaregression_plot(model, configured, regression, comparators, include_credible=True, credible_opacity=2)


def test_baseline_ranking_and_mcmc_helpers() -> None:
    configured = _configure_continuous()
    model = baseline_model(configured, "shared")

    rank = baseline_ranking(model, configured)
    assert set(rank.keys()) == {"SUCRA", "Colour", "Cumulative", "Probabilities", "BUGSnetData"}


def test_baseline_ranking_respects_ranking_option() -> None:
    configured = _configure_continuous()
    configured_bad = dataclasses.replace(configured, ranking_option="bad")
    model = baseline_model(configured, "shared")

    rank_good = baseline_ranking(model, configured)
    rank_bad = baseline_ranking(model, configured_bad)

    assert rank_good["SUCRA"].loc[0, "Treatment"] == "the_Butcher"
    assert rank_bad["SUCRA"].loc[0, "Treatment"] == "the_Great"
    assert rank_good["SUCRA"].loc[0, "Treatment"] != rank_bad["SUCRA"].loc[0, "Treatment"]

    table = ranking_table(rank_good)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns)[0] == "Treatment"
    assert table.shape[1] == len(configured.treatments) + 2
    assert table.loc[0, "Treatment"] == "the_Butcher"

    # each treatment should have a full probability distribution that sums to 100%
    prob_cols = [c for c in rank_good["Probabilities"].columns if c.startswith("Rank ")]
    assert len(prob_cols) == len(configured.treatments)
    row_sums = rank_good["Probabilities"][prob_cols].sum(axis=1)
    assert all(abs(float(v) - 100.0) < 1e-6 for v in row_sums)

    cumulative_last = rank_good["Cumulative"].groupby("Treatment").tail(1)
    assert (abs(cumulative_last["Cumulative_Probability"] - 100.0) < 1e-9).all()
    assert rank_good["SUCRA"]["SUCRA"].between(0, 100).all()

    litmus = LitmusRankOGram(rank_good)
    radial = RadialSUCRA(rank_good)
    assert litmus.startswith("<svg")
    assert radial.startswith("<svg")

    results = baseline_results(model)
    assert results.count("<br") == 5

    details = baseline_details(model)
    assert set(details.keys()) == {"mcmc", "priors"}

    mcmc = baseline_mcmc(model)
    assert set(mcmc.keys()) == {"parameters", "gelman_data", "n_cols", "n_rows", "n_rows_rmd"}
    assert len(mcmc["gelman_data"]) == len(mcmc["parameters"])

    gels = gelman_plots(mcmc["gelman_data"], mcmc["parameters"])
    assert len(gels) == len(mcmc["parameters"])
    assert gels[0].startswith("<svg")


def test_baseline_async_flags_run_synchronously_with_warning() -> None:
    configured = _configure_continuous()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        model = baseline_model(configured, "shared", async_=True)
        regression = baseline_regression(model, configured, async_=True)
        ranking = baseline_forest(model, async_=True)
        deviance = baseline_deviance(model, async_=True)
        plot = metaregression_plot(model, configured, regression, [configured.treatments.iloc[1]["Label"]], async_=True)

    assert model["regressor"] == "shared"
    assert set(regression.keys()) == {"directness", "credible_regions", "_model_type"}
    assert ranking.startswith("<svg")
    assert "deviance_mtc" in deviance
    assert plot.startswith("<svg")
    assert any("async" in str(msg.message).lower() for msg in captured)
