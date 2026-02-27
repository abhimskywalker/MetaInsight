from pathlib import Path

import pandas as pd
import pytest
import warnings

from metainsight import (
    freq_compare,
    freq_forest,
    freq_forest_annotation,
    freq_forest_limits,
    freq_inconsistent,
    freq_summary,
)
from metainsight.setup import setup_configure, setup_exclude, setup_load
from metainsight.frequentist import create_list_of_wide_columns, frequentist


_TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "testthat" / "data"


def _configured_continuous() -> object:
    loaded = setup_load(data_path=_TEST_DATA_DIR / "Cont_long.csv", outcome="continuous")
    return setup_configure(
        loaded,
        reference_treatment=loaded.treatments["Label"].iloc[0],
        effects="random",
        outcome_measure="MD",
        ranking_option="good",
        seed=111,
    )


def _configured_binary() -> object:
    loaded = setup_load(data_path=_TEST_DATA_DIR / "Binary_long.csv", outcome="binary")
    return setup_configure(
        loaded,
        reference_treatment=loaded.treatments["Label"].iloc[0],
        effects="random",
        outcome_measure="OR",
        ranking_option="good",
        seed=111,
    )


def _excluded_continuous() -> object:
    configured = _configured_continuous()
    return setup_exclude(configured, exclusions=["Leo", "Constantine"])


@pytest.mark.parametrize(
    "path,outcome,measure",
    [
        (_TEST_DATA_DIR / "Binary_long.csv", "binary", "OR"),
        (_TEST_DATA_DIR / "Cont_long.csv", "continuous", "MD"),
    ],
)
def test_frequentist_payload_structure(path: Path, outcome: str, measure: str) -> None:
    loaded = setup_load(data_path=path, outcome=outcome)
    reference_treatment = str(loaded.treatments["Label"].iloc[0])

    result = frequentist(
        non_covariate_data=loaded.data,
        outcome=outcome,
        treatments=loaded.treatments,
        outcome_measure=measure,
        effects="random",
        reference_treatment=reference_treatment,
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"net1", "lstx", "ntx", "d0", "d1"}

    assert result["lstx"] == [str(label) for label in loaded.treatments["Label"].astype(str).tolist()]
    assert reference_treatment in result["lstx"]
    assert len(result["lstx"]) == len(loaded.treatments)
    assert isinstance(result["ntx"], int)
    assert result["ntx"] == len(loaded.treatments)
    assert isinstance(result["d0"], pd.DataFrame)
    assert isinstance(result["d1"], pd.DataFrame)

    assert set(["TE", "seTE", "treat1", "treat2", "studlab"]).issubset(set(result["d0"].columns))
    assert set(["TE", "seTE", "treat1", "treat2", "studlab"]).issubset(set(result["d1"].columns))
    assert len(result["d0"]) > 0
    assert len(result["d1"]) > 0

    net1 = result["net1"]
    assert net1["effects"] == "random"
    assert net1["reference_treatment"] == reference_treatment
    assert net1["outcome_measure"] == measure


def test_frequentist_follows_setup_configure() -> None:
    binary = setup_load(data_path=_TEST_DATA_DIR / "Binary_long.csv", outcome="binary")
    continuous = setup_load(data_path=_TEST_DATA_DIR / "Cont_long.csv", outcome="continuous")

    for loaded in [binary, continuous]:
        reference_treatment = str(loaded.treatments["Label"].iloc[0])
        configured = setup_configure(
            loaded,
            reference_treatment=reference_treatment,
            effects="fixed",
            outcome_measure="OR" if loaded.outcome == "binary" else "MD",
            ranking_option="good",
            seed=111,
        )

        freq = frequentist(
            non_covariate_data=configured.non_covariate_data,
            outcome=configured.outcome,
            treatments=configured.treatments,
            outcome_measure=configured.outcome_measure,
            effects=configured.effects,
            reference_treatment=configured.reference_treatment,
        )

        assert configured.freq["ntx"] == freq["ntx"]
        assert configured.freq["lstx"] == freq["lstx"]
        assert len(configured.freq["d0"]) == len(freq["d0"])
        assert len(configured.freq["d1"]) == len(freq["d1"])
        assert configured.freq["net1"]["effects"] == freq["net1"]["effects"]


def test_frequentist_rejects_invalid_arguments() -> None:
    loaded = setup_load(data_path=_TEST_DATA_DIR / "Cont_long.csv", outcome="continuous")

    with pytest.raises(ValueError, match="outcome must be either 'binary' or 'continuous'"):
        frequentist(
            non_covariate_data=loaded.data,
            outcome="invalid",
            treatments=loaded.treatments,
            outcome_measure="MD",
            effects="random",
            reference_treatment="the Great",
        )

    with pytest.raises(ValueError, match="effects must be either random or fixed"):
        frequentist(
            non_covariate_data=loaded.data,
            outcome="continuous",
            treatments=loaded.treatments,
            outcome_measure="MD",
            effects="invalid",
            reference_treatment="the Great",
        )


def test_create_list_of_wide_columns_preserves_order() -> None:
    frame = pd.DataFrame(
        {
            "Study": ["A"],
            "T.2": [2],
            "T": [1],
            "N.2": [10],
            "N": [12],
            "T.1": [3],
            "N.1": [15],
            "Mean": [4.2],
            "Mean.3": [4.4],
            "Mean.1": [4.0],
            "SD.1": [1.0],
            "SD": [0.9],
            "SD.2": [1.1],
            "SD.3": [1.2],
        }
    )

    assert create_list_of_wide_columns(frame, "T") == ["T", "T.1", "T.2"]
    assert create_list_of_wide_columns(frame, "N") == ["N", "N.1", "N.2"]


def test_freq_compare_generates_full_matrix() -> None:
    configured = _configured_continuous()

    comp = freq_compare(configured)
    assert isinstance(comp, pd.DataFrame)
    assert comp.shape == (len(configured.treatments), len(configured.treatments))
    assert comp.index.tolist() == [str(label) for label in configured.treatments["Label"]]
    assert (comp.iloc[0, 0] == 0.0)

    first = str(configured.treatments["Label"].iloc[0])
    second = str(configured.treatments["Label"].iloc[1])
    assert comp.loc[first, second] == -comp.loc[second, first]


def test_freq_compare_supports_excluded_data() -> None:
    excluded = _excluded_continuous()

    comp = freq_compare(excluded)
    assert isinstance(comp, pd.DataFrame)
    assert comp.shape == (len(excluded.treatments), len(excluded.treatments))
    assert not comp.isna().any().any()
    assert all(comp.loc[label, label] == 0.0 for label in excluded.treatments["Label"])


def test_freq_forest_limits_and_annotation() -> None:
    configured = _configured_continuous()

    xmin, xmax = freq_forest_limits(configured.freq, configured.outcome)
    assert isinstance(xmin, float)
    assert isinstance(xmax, float)
    assert xmin <= xmax

    annotation = freq_forest_annotation(configured.freq, configured.effects, configured.outcome_measure)
    assert "Number of studies" in annotation
    assert "Number of comparisons" in annotation
    assert "Between-study standard deviation" in annotation

    loaded = setup_load(data_path=_TEST_DATA_DIR / "Cont_long.csv", outcome="continuous")
    configured_fixed = setup_configure(
        loaded,
        reference_treatment=str(loaded.treatments["Label"].iloc[0]),
        effects="fixed",
        outcome_measure="MD",
        ranking_option="good",
        seed=111,
    )
    annotation_fixed = freq_forest_annotation(configured_fixed.freq, configured_fixed.effects, configured_fixed.outcome_measure)
    assert "set at 0." in annotation_fixed


def test_freq_forest_generates_svg() -> None:
    configured = _configured_binary()

    svg_default = freq_forest(configured)
    svg_custom = freq_forest(configured, title="freq")
    assert svg_default.startswith("<svg")
    assert svg_custom.startswith("<svg")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        svg_async = freq_forest(configured, async_=True)

    assert svg_async.startswith("<svg")
    assert any("async" in str(msg.message).lower() for msg in captured)

    with pytest.raises(ValueError, match="configured_data must be of class configured_data"):
        freq_forest("not-configured")
    with pytest.raises(ValueError, match="title must be of class character"):
        freq_forest(configured, title=123)
    with pytest.raises(ValueError, match="xmin must be of class numeric"):
        freq_forest(configured, xmin="bad", xmax=1)
    with pytest.raises(ValueError, match="xmax must be of class numeric"):
        freq_forest(configured, xmin=0, xmax="bad")


def test_freq_inconsistent_and_summary() -> None:
    configured = _configured_continuous()
    comp = freq_compare(configured)

    incons = freq_inconsistent(configured)
    assert isinstance(incons, pd.DataFrame)
    expected_rows = sum(range(1, len(configured.treatments)))
    assert len(incons) == expected_rows
    assert list(incons.columns) == [
        "Comparison",
        "No.Studies",
        "NMA",
        "Direct",
        "Indirect",
        "Difference",
        "Diff_95CI_lower",
        "Diff_95CI_upper",
        "pValue",
    ]

    # each row should have a deterministic numerical inconsistency summary
    assert (incons["pValue"] >= 0).all() and (incons["pValue"] <= 1).all()
    assert (incons["No.Studies"] >= 0).all()

    first = str(configured.treatments["Label"].iloc[0])
    second = str(configured.treatments["Label"].iloc[1])
    first_row = incons[incons["Comparison"] == f"{first} vs {second}"].iloc[0]
    assert first_row["NMA"] == comp.loc[first, second]
    assert pd.notna(first_row["Diff_95CI_lower"])
    assert pd.notna(first_row["Diff_95CI_upper"])

    excluded = _excluded_continuous()
    incons_sub = freq_inconsistent(excluded)
    assert incons_sub.shape[0] == sum(range(1, len(excluded.treatments)))

    summary_svg = freq_summary(configured, "title")
    assert summary_svg.startswith("<svg")
    assert "Top treatments" in summary_svg
    summary_svg_sub = freq_summary(excluded, "title")
    assert summary_svg_sub.startswith("<svg")
    assert "Too few treatments" in summary_svg_sub
    with pytest.raises(ValueError, match="plot_title must be of class character"):
        freq_summary(configured, plot_title=123)
