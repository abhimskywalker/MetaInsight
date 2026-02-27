from pathlib import Path

import pandas as pd
import pytest

from metainsight import setup_load
from metainsight.network import identify_subnetworks, is_nodesplittable


_TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "testthat" / "data"


def test_identify_subnetworks_continuous_disconnected_long() -> None:
    loaded = setup_load(
        data_path=_TEST_DATA_DIR / "continuous_long_disconnected.csv",
        outcome="continuous",
    )

    result = identify_subnetworks(loaded.data, loaded.treatments)

    assert list(result.keys()) == ["subnet_1", "subnet_2", "subnet_3"]
    assert result["subnet_1"]["treatments"] == [1, 2, 3, 4, 8]
    assert set(result["subnet_1"]["studies"]) == {"Uno", "Deux", "Three", "Cinque", "Six"}
    assert result["subnet_2"]["treatments"] == [5, 6, 7, 9]
    assert set(result["subnet_2"]["studies"]) == {"Quatro", "Sept"}
    assert result["subnet_3"]["treatments"] == [10, 11]
    assert set(result["subnet_3"]["studies"]) == {"Ocho"}


def test_identify_subnetworks_continuous_disconnected_wide_ref_E() -> None:
    loaded = setup_load(
        data_path=_TEST_DATA_DIR / "continuous_wide_disconnected.csv",
        outcome="continuous",
    )

    result = identify_subnetworks(loaded.data, loaded.treatments, reference_treatment_name="E")

    assert list(result.keys()) == ["subnet_1", "subnet_2", "subnet_3"]
    assert result["subnet_1"]["treatments"] == [3, 7, 9, 11]
    assert set(result["subnet_1"]["studies"]) == {"Quatro", "Sept"}
    assert result["subnet_2"]["treatments"] == [1, 2, 5, 6, 8]
    assert set(result["subnet_2"]["studies"]) == {"Uno", "Deux", "Three", "Cinque", "Six"}
    assert result["subnet_3"]["treatments"] == [4, 10]
    assert set(result["subnet_3"]["studies"]) == {"Ocho"}


def test_identify_subnetworks_invalid_reference_defaults_to_first() -> None:
    loaded = setup_load(
        data_path=_TEST_DATA_DIR / "continuous_long_disconnected.csv",
        outcome="continuous",
    )

    with pytest.warns(UserWarning):
        result = identify_subnetworks(loaded.data, loaded.treatments, reference_treatment_name="Omlette")

    assert list(result.keys()) == ["subnet_1", "subnet_2", "subnet_3"]
    assert result["subnet_1"]["treatments"] == [1, 2, 3, 4, 8]


def test_nodesplittable_true_with_loop_splittable() -> None:
    data = pd.DataFrame(
        {
            "Study": ["A", "A", "B", "B", "C", "C"],
            "T": ["Placebo", "Hydrogen", "Placebo", "Oxygen", "Hydrogen", "Oxygen"],
        }
    )

    result = is_nodesplittable(data=data, treatments=["Placebo", "Hydrogen", "Oxygen"])

    assert result["is_nodesplittable"] is True
    assert result["reason"] is None


def test_nodesplittable_false_when_no_loops() -> None:
    data = pd.DataFrame(
        {
            "Study": ["A", "A", "B", "B"],
            "T": ["Placebo", "Hydrogen", "Placebo", "Oxygen"],
        }
    )

    result = is_nodesplittable(data=data, treatments=["Placebo", "Hydrogen", "Oxygen"])

    assert result["is_nodesplittable"] is False
    assert result["reason"].startswith("There are no loops in the network")


def test_nodesplittable_false_when_loops_not_distinguishable() -> None:
    data = pd.DataFrame(
        {
            "Study": ["A", "A", "A", "B", "B"],
            "T": ["Placebo", "Hydrogen", "Oxygen", "Placebo", "Hydrogen"],
        }
    )

    result = is_nodesplittable(data=data, treatments=["Placebo", "Hydrogen", "Oxygen"])

    assert result["is_nodesplittable"] is False
    assert result["reason"].startswith("In all loops, heterogeneity and inconsistency cannot be distinguished")
