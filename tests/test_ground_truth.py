"""The ground-truth producer: segment -> render -> baseline + sidecar.

Builds a programmatic extraction shard (real ``write_results`` output, so the
manifest, footer stamp and four siblings are exactly what a run writes) and
exercises :func:`build_ground_truth` end to end, offline: a word-count
``count_fn`` stands in for the kanon-2 tokeniser. Each assertion pins one
field's derivation against ``docs/ground-truth-units.md``.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from womblex.config import SegmentationConfig
from womblex.ingest.elements import Cell, Element, FieldEntry
from womblex.ingest.views import ExtractionResult
from womblex.process.ground_truth import BASELINE_SUFFIX, build_ground_truth
from womblex.process.renderer import RENDERER_VERSION, baseline_digest
from womblex.store.ground_truth_output import (
    SIDECAR_SUFFIX,
    UNFILLED,
    read_sidecar,
    unit_id,
)
from womblex.store.normalise_output import write_normalised_text
from womblex.store.output import read_manifest, write_results
from womblex.store.run_stamp import RunStamp
from womblex.store.source_provenance import IngestProvenance

_RUN_ID = "run-2026-09"
_VERSION = "9.9.9"
_DIGEST = "sha256:cafef00dcafef00d"


def words(texts: list[str]) -> list[int]:
    """Deterministic, offline stand-in for a real tokeniser."""
    return [len(t.split()) for t in texts]


def para(order: int, text: str, page: int | None = 0) -> Element:
    return Element(order=order, kind="paragraph", extractor="test", page=page, text=text)


def _write_shard(
    tmp_path: Path,
    elements: list[Element],
    *,
    doc_id: str = "doc-a",
    stamp: bool = True,
    provenance: bool = True,
) -> Path:
    """Write one real batch shard and return its directory."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    src = corpus / f"{doc_id}.pdf"
    src.write_bytes(b"%PDF-1.4 fake source bytes for " + doc_id.encode())

    shard_dir = tmp_path / "documents"
    shard_dir.mkdir(exist_ok=True)
    prov = (
        IngestProvenance.declare(str(corpus), "testcol") if provenance else None
    )
    run_stamp = (
        RunStamp(
            run_id=_RUN_ID, version=_VERSION, commit="deadbeef",
            config_digest=_DIGEST, stage="extract",
        )
        if stamp else None
    )
    write_results(
        [(doc_id, str(src), ExtractionResult(elements=elements, method="native"))],
        shard_dir / "batch-0001.parquet",
        provenance=prov, stamp=run_stamp,
    )
    return shard_dir


def _sidecars(output_dir: Path) -> list[dict]:
    return [read_sidecar(p) for p in sorted(output_dir.glob(f"*{SIDECAR_SUFFIX}"))]


def test_writes_one_baseline_and_one_sidecar_per_segment(tmp_path: Path) -> None:
    shard_dir = _write_shard(tmp_path, [para(i, " ".join(["w"] * 4)) for i in range(10)])
    out = tmp_path / "gt"

    result = build_ground_truth(shard_dir, out, SegmentationConfig(token_budget=10), count_fn=words)

    baselines = sorted(out.glob(f"*{BASELINE_SUFFIX}"))
    sidecars = sorted(out.glob(f"*{SIDECAR_SUFFIX}"))
    assert result.units_written > 1
    assert len(baselines) == result.units_written == len(sidecars)
    assert result.documents == 1
    assert all(_sidecars(out))


def test_identity_is_read_from_the_manifest(tmp_path: Path) -> None:
    shard_dir = _write_shard(tmp_path, [para(0, "only one segment")])
    out = tmp_path / "gt"
    build_ground_truth(shard_dir, out, SegmentationConfig(), count_fn=words)

    row = read_manifest(shard_dir / "batch-0001.parquet").to_pylist()[0]
    identity = _sidecars(out)[0]["identity"]
    assert identity["source_hash"] == row["source_hash"]
    assert identity["ingest_root"] == row["ingest_root"] == "file://" + str(tmp_path / "corpus")
    assert identity["source_relpath"] == row["source_relpath"] == "doc-a.pdf"
    assert identity["collection"] == row["collection_id"] == "testcol"
    assert unit_id(_sidecars(out)[0]).startswith(row["source_hash"] + ":")


def test_element_ranges_tile_the_document(tmp_path: Path) -> None:
    shard_dir = _write_shard(tmp_path, [para(i, " ".join(["w"] * 4)) for i in range(6)])
    out = tmp_path / "gt"
    build_ground_truth(shard_dir, out, SegmentationConfig(token_budget=10), count_fn=words)

    ranges = sorted(tuple(s["identity"]["element_range"]) for s in _sidecars(out))
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 6
    for (_, prev_end), (next_start, _) in pairwise(ranges):
        assert prev_end == next_start


def test_derivation_comes_from_the_footer_and_the_renderer(tmp_path: Path) -> None:
    shard_dir = _write_shard(tmp_path, [para(0, "alpha beta gamma")])
    out = tmp_path / "gt"
    build_ground_truth(shard_dir, out, SegmentationConfig(), count_fn=words)

    sidecar = _sidecars(out)[0]
    derivation = sidecar["derivation"]
    assert derivation["run_id"] == _RUN_ID
    assert derivation["preset_digest"] == _DIGEST
    assert derivation["parser_version"] == _VERSION
    assert derivation["text_source"] == "elements"
    assert derivation["renderer_version"] == RENDERER_VERSION
    assert derivation["preset"] == UNFILLED

    md = min(out.glob(f"*{BASELINE_SUFFIX}")).read_text(encoding="utf-8")
    assert derivation["baseline_digest"] == baseline_digest(md)

    review = sidecar["review"]
    assert review["review_class"] == "unreviewed"
    assert review["reviewer"] == review["reviewed_date"] == review["edit_distance"] == UNFILLED


def test_unstamped_shard_leaves_derivation_facts_unfilled(tmp_path: Path) -> None:
    shard_dir = _write_shard(tmp_path, [para(0, "no run stamp here")], stamp=False)
    out = tmp_path / "gt"
    build_ground_truth(shard_dir, out, SegmentationConfig(), count_fn=words)

    derivation = _sidecars(out)[0]["derivation"]
    assert derivation["run_id"] == UNFILLED
    assert derivation["preset_digest"] == UNFILLED
    assert derivation["parser_version"] == UNFILLED
    assert derivation["renderer_version"] == RENDERER_VERSION
    assert derivation["baseline_digest"].startswith("sha256:")


def test_page_range_is_the_segment_range_when_paged(tmp_path: Path) -> None:
    shard_dir = _write_shard(tmp_path, [para(0, "a", page=3), para(1, "b", page=4)])
    out = tmp_path / "gt"
    build_ground_truth(shard_dir, out, SegmentationConfig(token_budget=1000), count_fn=words)
    assert _sidecars(out)[0]["identity"]["page_range"] == [3, 5]


def test_page_range_is_null_for_a_source_with_no_pages(tmp_path: Path) -> None:
    shard_dir = _write_shard(tmp_path, [para(i, "a", page=None) for i in range(2)])
    out = tmp_path / "gt"
    build_ground_truth(shard_dir, out, SegmentationConfig(), count_fn=words)
    assert _sidecars(out)[0]["identity"]["page_range"] is None


def test_page_less_segment_of_a_paged_source_takes_the_sentinel(tmp_path: Path) -> None:
    tbl = Element(
        order=2, kind="table", extractor="test", page=None,
        cells=[Cell(row=r, col=c, value="x") for r in range(4) for c in range(3)],
        header_rows=[0],
    )
    elements = [para(0, "a", page=0), para(1, "b", page=0), tbl]
    shard_dir = _write_shard(tmp_path, elements)
    out = tmp_path / "gt"
    build_ground_truth(shard_dir, out, SegmentationConfig(token_budget=4), count_fn=words)

    ranges = {tuple(s["identity"]["element_range"]): s["identity"]["page_range"]
              for s in _sidecars(out)}
    # The paged prose keeps its range; the page-less table takes the sentinel,
    # never null — the source plainly has pages.
    assert ranges[(0, 2)] == [0, 1]
    assert ranges[(2, 3)] == UNFILLED


def test_form_fields_are_restored_into_the_baseline(tmp_path: Path) -> None:
    form = Element(
        order=1, kind="form", extractor="test", page=0,
        fields=[FieldEntry(name="Applicant", value="Ada Lovelace")],
    )
    shard_dir = _write_shard(tmp_path, [para(0, "cover"), form])
    out = tmp_path / "gt"
    build_ground_truth(shard_dir, out, SegmentationConfig(token_budget=1000), count_fn=words)

    md = "\n".join(p.read_text(encoding="utf-8") for p in out.glob(f"*{BASELINE_SUFFIX}"))
    # `_load_elements` drops form fields; the producer must stitch them back.
    assert "Applicant: Ada Lovelace" in md


def test_normalised_overlay_is_rendered_and_recorded(tmp_path: Path) -> None:
    shard_dir = _write_shard(tmp_path, [para(0, "raw   verbatim")])
    base = shard_dir / "batch-0001.parquet"
    src_hash = read_manifest(base).to_pylist()[0]["source_hash"]
    write_normalised_text(
        [{"source_hash": src_hash, "elem_order": 0, "kind": "paragraph",
          "page": 0, "text": "clean normalised", "n_changes": 1}],
        base,
    )
    out = tmp_path / "gt"
    build_ground_truth(shard_dir, out, SegmentationConfig(), text_source="normalised", count_fn=words)

    md = min(out.glob(f"*{BASELINE_SUFFIX}")).read_text(encoding="utf-8")
    assert "clean normalised" in md
    assert "raw   verbatim" not in md
    assert _sidecars(out)[0]["derivation"]["text_source"] == "normalised"


def test_document_without_provenance_is_skipped_not_fatal(tmp_path: Path) -> None:
    shard_dir = _write_shard(tmp_path, [para(0, "text")], provenance=False)
    out = tmp_path / "gt"

    result = build_ground_truth(shard_dir, out, SegmentationConfig(), count_fn=words)

    # Counted as a document, but its identity is unfillable so no unit is written.
    assert result.documents == 1
    assert result.units_written == 0
    assert not list(out.glob(f"*{SIDECAR_SUFFIX}"))
