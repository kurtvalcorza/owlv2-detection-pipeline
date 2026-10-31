"""DAT20: a refused BYOD input in Section 4 shows exactly one line, ``Refused: <file>: <reason>``, with no traceback,
and the cell raises so Run all stops there; the router relays it as the same line. The in-cell forgetting of the
previous data is kept, and a genuine failure keeps its traceback. Section 4 is executed from the committed notebook."""
# ruff: noqa: E501

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from test_owlv2_detection_colab_review_fixes import (  # noqa: F401  (fixture)
    _byod_namespace,
    _byod_prefix,
    _cell_with,
    nb,
)

REFUSAL_CELL = "owlv2_detection-16"


def _refused(nb: dict, path: str, ns: dict | None = None) -> tuple[str, dict]:  # noqa: F811
    ns = _byod_namespace() if ns is None else ns
    with pytest.raises(ValueError) as caught:
        exec(_byod_prefix(nb, path), ns)
    exc = caught.value
    assert type(exc).__name__ == "BYODRefusal" and exc.dimer_refusal is True and exc.__suppress_context__
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    return rendered[0], ns


def test_the_scripted_refusal_input_is_one_line_naming_the_file(nb: dict, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    monkeypatch.chdir(tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    line, ns = _refused(nb, str(hostname))
    assert line.startswith("Refused: hostname: expected a zip of images plus boxes.csv") and "hostname: hostname" not in line
    assert line.endswith("run this cell again.") and "BYOD input refused" not in line
    assert "train_records" not in ns and "splits" not in ns  # the previous data is forgotten


def test_a_bad_folder_and_no_upload_dialog_are_refused_by_name(nb: dict, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    monkeypatch.chdir(tmp_path)
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "hostname").write_text("runtime-host\n", encoding="utf-8")
    line, _ = _refused(nb, str(tmp_path / "etc"))
    assert line.startswith("Refused: etc: ")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    line, _ = _refused(nb, "")
    assert line.startswith("Refused: (no file): BYOD is on but its path field is empty")


def test_a_genuine_failure_keeps_its_type(nb: dict, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mine").mkdir()

    def broken(path):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug"):
        exec(_byod_prefix(nb, str(tmp_path / "mine")), _byod_namespace(load_byod_dataset=broken))


def test_the_refusal_cell_id_and_the_router_line(nb: dict) -> None:  # noqa: F811
    cell = next(c for c in nb["cells"] if c.get("id") == REFUSAL_CELL)
    source = "".join(cell["source"])
    assert source == _cell_with(nb, "class BYODRefusal(ValueError):") and "BYOD_PATH = ''" in source
    assert "raise refuse_byod(exc, byod_label) from None" in source and 'return [f"Refused: {self}"]' in source
    router = _cell_with(nb, "class IsolatedCellRefusal(Exception):")
    assert 'return [f"Refused: {self}"]' in router and 'getattr(exc, "dimer_refusal", False)' in router
