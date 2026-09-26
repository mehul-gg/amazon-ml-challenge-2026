import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metric import entity_fbeta, macro_fbeta, parse_id_list  # noqa: E402


def test_spec_example():
    # Worked example from the problem statement: P=2/3, R=1 -> 0.714
    true = {"S2-00047", "S3-00812"}
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    assert round(entity_fbeta(true, pred), 3) == 0.714


def test_singletons():
    assert entity_fbeta(set(), set()) == 1.0
    assert entity_fbeta(set(), {"S2-1"}) == 0.0


def test_missed_entity():
    assert entity_fbeta({"S2-1"}, set()) == 0.0


def test_macro_average_and_parsing():
    truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    pred = {"S1-1": parse_id_list("S2-1"), "S1-2": parse_id_list(float("nan"))}
    assert macro_fbeta(truth, pred) == 1.0
    assert macro_fbeta(truth, {}) == 0.5  # missing prediction = empty list


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
