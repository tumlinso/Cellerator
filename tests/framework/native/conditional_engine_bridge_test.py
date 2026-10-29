"""Optional built-native end-to-end engine bridge check."""
import importlib
import importlib.util
from pathlib import Path
from cellerator.adapters.conditional import NativeIdentityMap
from cellerator.execution import ParameterKey


def main():
    try:
        native = importlib.import_module("cellerator._conditional_native")
    except ModuleNotFoundError:
        native = importlib.import_module("_conditional_native")
    path = Path(__file__).resolve().parents[3] / "examples/framework/scheduled_native_experts.py"
    spec = importlib.util.spec_from_file_location("scheduled_native_experts", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    assert example.run(native) == {"cell-a": [4.0], "cell-b": [15.0]}
    first, second = ParameterKey("a", 0), ParameterKey("b", 0)
    for experts, origins in (
        ({first: (1, 0), second: (1, 0)}, {}),
        ({first: (1, 0), ParameterKey("a", 1): (2, 1)}, {}),
        ({first: (1, 0), ParameterKey("a", 1): (1, 0)}, {}),
        ({first: (1, 0)}, {("cpu", "a"): (0, 1, 1), ("cpu", "b"): (0, 1, 1)}),
        ({first: (1, -1)}, {}),
    ):
        try:
            NativeIdentityMap(experts, origins)
        except ValueError:
            pass
        else:
            raise AssertionError("native logical identity collision/overflow accepted")


if __name__ == "__main__":
    main()
