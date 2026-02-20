from pathlib import Path

import pandas as pd
import pytest

from metainsight.setup import LoadedData, setup_load, validate_uploaded_data


_TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "testthat" / "data"


@pytest.mark.parametrize(
    "bad_file,expected_msg",
    [
        ("invalid_data/binary-missing-long.csv", "Missing columns for binary data: N"),
        ("invalid_data/continuous-missing-long.csv", "Missing columns for continuous data: N"),
        ("invalid_data/continuous-mistyped-columns-wide.csv", "Some columns have incorrect data types: Study should be of type character (text), Mean.1 should be of type numeric"),
        ("invalid_data/binary-misnumbered-wide.csv", "For wide format data, numbered columns (T, N, R) must all have matching sequential indices, starting from 1"),
    ],
)
def test_validate_uploaded_data(bad_file, expected_msg):
    df = pd.read_csv(_TEST_DATA_DIR / bad_file)
    result = validate_uploaded_data(df, outcome="binary" if "binary" in bad_file else "continuous")
    assert result.valid is False
    assert result.message == expected_msg


def test_validate_uploaded_data_empty():
    result = validate_uploaded_data(pd.DataFrame(), outcome="binary")
    assert not result.valid
    assert result.message == "File is empty"


def test_setup_load_defaults_are_loaded():
    continuous = setup_load(outcome="continuous")
    binary = setup_load(outcome="binary")

    assert isinstance(continuous, LoadedData)
    assert continuous.is_data_valid
    assert continuous.is_data_uploaded is False
    assert continuous.outcome == "continuous"
    assert continuous.treatments is not None
    assert list(continuous.treatments.columns) == ["Number", "Label"]

    assert isinstance(binary, LoadedData)
    assert binary.is_data_valid
    assert binary.is_data_uploaded is False
    assert binary.outcome == "binary"
    assert binary.treatments is not None


def test_setup_load_raises_on_bad_path_type():
    with pytest.raises(TypeError, match="data_path must be of class character"):
        setup_load(data_path=123, outcome="binary")


def test_setup_load_raises_on_bad_extension():
    with pytest.raises(ValueError, match="data_path must link to either a \.csv or \.xlsx file"):
        setup_load(data_path=_TEST_DATA_DIR / "invalid_data", outcome="binary")


def test_setup_load_raises_when_file_missing():
    with pytest.raises(FileNotFoundError, match="The specified file does not exist"):
        setup_load(data_path=_TEST_DATA_DIR / "does_not_exist.csv", outcome="binary")


def test_setup_load_invalid_data_raises_if_no_logger():
    with pytest.raises(ValueError, match="Uploaded data was invalid because"):
        setup_load(data_path=_TEST_DATA_DIR / "invalid_data" / "binary-missing-long.csv", outcome="binary")


def test_setup_load_invalid_data_returns_payload_with_logger():
    messages = []

    def logger(msg: str) -> None:
        messages.append(msg)

    loaded = setup_load(
        data_path=_TEST_DATA_DIR / "invalid_data" / "binary-missing-long.csv",
        outcome="binary",
        logger=logger,
    )

    assert loaded.is_data_uploaded
    assert not loaded.is_data_valid
    assert loaded.treatments is None
    assert messages
