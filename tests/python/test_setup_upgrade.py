from pathlib import Path

import pandas as pd
import pytest

from metainsight import setup_upgrade


_TEST_EXTDATA_DIR = Path(__file__).resolve().parents[2] / "inst" / "extdata"


@pytest.mark.parametrize(
    "data_path, treatments, expected_type",
    [
        (123, "A,B,C,D,E", TypeError),
        ("word_doc.docx", "A,B,C,D,E", ValueError),
        ("non_existent.csv", "A,B,C,D,E", FileNotFoundError),
    ],
)
def test_setup_upgrade_validation_errors(data_path, treatments, expected_type):
    with pytest.raises(expected_type):
        setup_upgrade(data_path=data_path, treatments=treatments)


def test_setup_upgrade_checks_treatment_names_and_count() -> None:
    source = _TEST_EXTDATA_DIR / "old_data.csv"

    with pytest.raises(TypeError, match="treatments must be of class character"):
        setup_upgrade(data_path=source, treatments=123)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="The treatment names must only contain words separated by commas",
    ):
        setup_upgrade(data_path=source, treatments="123")

    with pytest.raises(ValueError, match="Your input data contains 2 treatments"):
        setup_upgrade(data_path=source, treatments="A,B")


def test_setup_upgrade_functions_correctly_with_long_data() -> None:
    source = _TEST_EXTDATA_DIR / "old_data.csv"
    upgraded = setup_upgrade(source, "A,B,C,D,E")

    assert isinstance(upgraded, pd.DataFrame)
    assert list(upgraded.columns) == ["Study", "T", "OtherText"]
    assert len(upgraded) == 7

    assert list(upgraded["T"]) == ["A", "B", "A", "C", "A", "D", "E"]
    assert upgraded["Study"].tolist() == ["A", "A", "B", "B", "C", "C", "C"]


def test_setup_upgrade_functions_with_wide_data(tmp_path: Path) -> None:
    wide_data = """Study,T,N,OtherText,T.2,N.2,OtherText.2\nS1,1,2,alpha,2,3,beta\nS2,2,4,gamma,1,5,delta\n"""
    source = tmp_path / "legacy_wide.csv"
    source.write_text(wide_data)

    upgraded = setup_upgrade(source, "Trt_A,Trt_B")

    assert "T" in upgraded.columns
    assert "T.2" in upgraded.columns
    assert upgraded.loc[0, "T"] == "Trt_A"
    assert upgraded.loc[0, "T.2"] == "Trt_B"
    assert upgraded.loc[1, "T"] == "Trt_B"
    assert upgraded.loc[1, "T.2"] == "Trt_A"
