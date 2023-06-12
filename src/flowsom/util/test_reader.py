import pytest

from flowsom.util import read_input


def test_wrong_input():
    text = "Input must be a file, directory, numpy object, pandas dataframe or an AnnData"
    with pytest.raises(ValueError) as error:
        read_input([], [])
    assert error.type is ValueError
    assert error.match(text)

    with pytest.raises(ValueError) as error:
        read_input(3, [])
    assert error.type is ValueError
    assert error.match(text)

def test_wrong_directory():
    with pytest.raises(AssertionError) as error:
        read_input("../../../empty", [])
    assert error.type is AssertionError
    assert error.match("Directory cannot be empty")

    with pytest.raises(AssertionError) as error:
        read_input("../../../Gelabelde_datasets", [])
    assert error.type is AssertionError
    assert error.match("All files must have the same markers")


