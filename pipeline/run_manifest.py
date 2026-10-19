"""Read-only facts about a pipeline export, for the run launcher (no modal, no NeMo needed): which modules the
training image mounts, their sha256, the run-2 language list, and whether an arm exists.

    python3 run_manifest.py <pipeline dir> modules|hashes|langs|arm <ARM>

modal_app.py is parsed, never imported, so the export can be inspected on a machine without the training stack.
"""
import ast
import hashlib
import os
import sys


def _tree(pipeline_dir: str) -> ast.Module:
    with open(os.path.join(pipeline_dir, "modal_app.py")) as f:
        return ast.parse(f.read())


def mounted_modules(pipeline_dir: str, image: str = "nemo_image") -> list:
    """The string arguments of every `.add_local_python_source(...)` in `<image> = (...)`: the modules the training
    function runs with. Order kept."""
    for node in _tree(pipeline_dir).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == image for t in node.targets):
            return [a.value for c in ast.walk(node.value)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "add_local_python_source"
                    for a in c.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    raise ValueError(f"no {image} = ... in modal_app.py")


def module_hashes(pipeline_dir: str, image: str = "nemo_image") -> list:
    """[(name, sha256)] of modal_app.py and every mounted module's file; a missing file is an error."""
    out = []
    for name in ["modal_app"] + mounted_modules(pipeline_dir, image):
        path = os.path.join(pipeline_dir, name + ".py")
        with open(path, "rb") as f:
            out.append((name, hashlib.sha256(f.read()).hexdigest()))
    return out


def run2_langs(pipeline_dir: str) -> str:
    """RUN2_LANGS as the comma string the arm recipes compare against. Only the three constant assignments it is built
    from (WAVE1, MODERATE_LANGS, RUN2_LANGS) are executed, in file order, with nothing else in scope."""
    ns = {}
    for node in _tree(pipeline_dir).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("WAVE1", "MODERATE_LANGS", "RUN2_LANGS")
                                                for t in node.targets):
            exec(compile(ast.Module([node], []), "modal_app.py", "exec"), {"__builtins__": {}}, ns)
    return ",".join(ns["RUN2_LANGS"])


def has_arm(pipeline_dir: str, arm: str) -> bool:
    """The arm is defined as a literal key in an R3_ARMS dict (R3_ARMS = {...} or R3_ARMS.update({...}))."""
    for node in ast.walk(_tree(pipeline_dir)):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and k.value == arm:
                    return True
    return False


def main(argv) -> int:
    d, what = argv[0], argv[1]
    if what == "modules":
        print("\n".join(mounted_modules(d)))
    elif what == "hashes":
        print("\n".join(f"{h}  {n}.py" for n, h in module_hashes(d)))
    elif what == "langs":
        print(run2_langs(d))
    elif what == "arm":
        return 0 if has_arm(d, argv[2]) else 1
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
