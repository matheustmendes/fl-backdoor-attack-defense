import ast
import sys
from pathlib import Path


def check(path: Path) -> list[str]:
    violations = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "robust_aggregation":
            continue
        lineno = getattr(node, "lineno", "?")
        kind = type(node).__name__
        name = getattr(node, "name", "")
        violations.append(f"line {lineno}: disallowed module-scope {kind} {name!r} "
                          f"(only imports and robust_aggregation allowed)")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "robust_aggregation" for n in tree.body):
        violations.append("missing module-scope def robust_aggregation")
    return violations


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or [Path("DEFENSE.py")]
    ok = True
    for path in paths:
        if not path.is_file():
            print(f"{path}: NOT FOUND")
            ok = False
            continue
        violations = check(path)
        if violations:
            ok = False
            print(f"{path}: FAIL")
            for v in violations:
                print(f"  - {v}")
        else:
            print(f"{path}: PASS (portal-guard clean)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
