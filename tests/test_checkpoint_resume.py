"""Resume tells apart files that share a stem but not an extension."""

from __future__ import annotations

from pathlib import Path

from womblex.store.checkpoint import CheckpointManager, relpath_key


def _mgr(tmp_path: Path) -> CheckpointManager:
    mgr = CheckpointManager(tmp_path / "ckpt", "ds")
    mgr.load()
    return mgr


def test_same_stem_sibling_is_not_skipped(tmp_path: Path) -> None:
    root = tmp_path / "in"
    pdf, docx = root / "report.pdf", root / "report.docx"
    mgr = _mgr(tmp_path)
    mgr.update(["report"], 1, 0, 1, paths=[relpath_key(pdf, root)])

    assert mgr.filter_unprocessed([pdf, docx], root) == [docx]


def test_same_stem_in_two_folders_is_kept_apart(tmp_path: Path) -> None:
    root = tmp_path / "in"
    a, b = root / "a" / "x.pdf", root / "b" / "x.pdf"
    mgr = _mgr(tmp_path)
    mgr.update(["x"], 1, 0, 1, paths=["a/x.pdf"])

    assert mgr.filter_unprocessed([a, b], root) == [b]


def test_checkpoint_without_paths_falls_back_to_the_stem(tmp_path: Path) -> None:
    root = tmp_path / "in"
    mgr = _mgr(tmp_path)
    mgr.update(["old"], 1, 0, 1)

    assert mgr.filter_unprocessed([root / "old.pdf", root / "new.pdf"], root) == [
        root / "new.pdf"
    ]


def test_paths_survive_a_reload_and_drop_clears_them(tmp_path: Path) -> None:
    root = tmp_path / "in"
    pdf = root / "report.pdf"
    mgr = _mgr(tmp_path)
    mgr.update(["report"], 1, 0, 1, paths=["report.pdf"])

    again = _mgr(tmp_path)
    assert again.filter_unprocessed([pdf], root) == []
    again.drop(["report"])
    assert again.filter_unprocessed([pdf], root) == [pdf]
