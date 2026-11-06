"""The dispatched-operations table in docs/api/auto.md matches the code.

``fastfields.auto`` builds its functions at import time, so the API reference
generator cannot see them and the page lists them by hand. This keeps that list
honest: every dispatched operation is in the table, with exactly the backends
its dispatch table names, and nothing else is.
"""

from __future__ import annotations

import re
from pathlib import Path

import fastfields.auto as fa

PAGE = Path(__file__).resolve().parent.parent / "docs" / "api" / "auto.md"
NOT_OPERATIONS = {"Spline", "Bound", "__version__", "dlpack_backend"}
ROW = re.compile(r"^\| `(?P<op>\w+)` \| (?P<links>.+) \|$", re.MULTILINE)
LINK = re.compile(r"\[(?P<ns>\w+)\]\[fastfields\.(?P=ns)\.(?P<name>\w+)\]")


def _dispatch_table(op: str) -> dict[str, str]:
    func = getattr(fa, op)
    cells = dict(zip(func.__code__.co_freevars, func.__closure__))
    return cells["table"].cell_contents


def _documented() -> dict[str, set[str]]:
    rows = {}
    for match in ROW.finditer(PAGE.read_text()):
        links = LINK.findall(match["links"])
        assert all(name == match["op"] for _, name in links), match["op"]
        rows[match["op"]] = {ns for ns, _ in links}
    return rows


def test_every_dispatched_operation_is_listed_with_its_backends():
    documented = _documented()
    dispatched = [n for n in fa.__all__ if n not in NOT_OPERATIONS]
    assert sorted(documented) == sorted(dispatched)
    for op in dispatched:
        assert documented[op] == set(_dispatch_table(op)), op
