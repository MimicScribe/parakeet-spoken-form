"""en_digit_mask: written-form label columns (+1 after) are freed from English KL; nothing else changes.
Loads only digit_mask_cols from ft_train.py (the training stack is not needed)."""
import ast
import os

SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ft_train.py")).read()
_fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "digit_mask_cols")
_ns = {}
exec(compile(ast.Module([_fn], []), "ft_train.py", "exec"), _ns)
digit_mask_cols = _ns["digit_mask_cols"]


def test_digit_cols():
    wf = {7, 8}  # pretend ids 7, 8 are digit pieces
    assert digit_mask_cols([1, 2, 7, 8, 3], wf) == [2, 3, 4]  # columns before each digit piece + the one after the last
    assert digit_mask_cols([1, 2, 3], wf) == []                # no digits: untouched
    assert digit_mask_cols([7], wf) == [0, 1]                  # a lone code piece at the start


if __name__ == "__main__":
    test_digit_cols(); print("ok")
