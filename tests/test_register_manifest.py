"""Tests for the register-ingest manifest (store/register_manifest.py).

Exercised against the real ABN and G-NAF ingest writers so the metadata
keys the manifest reads stay in sync with what the ingests attach.
"""

from pathlib import Path

import pyarrow.parquet as pq

from womblex.ingest.abn_bulk import ingest_abn_xml
from womblex.store.register_manifest import (
    REGISTER_MANIFEST_SCHEMA,
    RUN_MANIFEST_FILENAME,
    write_register_manifest,
)

_IND_RECORD = (
    '<ABR recordLastUpdatedDate="20260301" replaced="N">'
    '<ABN status="ACT" ABNStatusFromDate="20100401">11111111111</ABN>'
    "<EntityType><EntityTypeInd>IND</EntityTypeInd>"
    "<EntityTypeText>Individual/Sole Trader</EntityTypeText></EntityType>"
    '<LegalEntity><IndividualName type="LGL">'
    "<GivenName>JANE</GivenName><FamilyName>CITIZEN</FamilyName>"
    "</IndividualName></LegalEntity>"
    '<OtherEntity><NonIndividualName type="TRD">'
    "<NonIndividualNameText>JANES PLUMBING</NonIndividualNameText>"
    "</NonIndividualName></OtherEntity>"
    "</ABR>"
)


def _write_extract(path: Path, records: str) -> None:
    path.write_text(
        '<Transfer error="false"><TransferInfo></TransferInfo>'
        f"{records}</Transfer>",
        encoding="utf-8",
    )


def test_manifest_indexes_abn_outputs(tmp_path: Path):
    out = tmp_path / "out"
    xml = tmp_path / "20260601Public01.xml"
    _write_extract(xml, _IND_RECORD)
    ingest_abn_xml(xml, out, compute_md5=True)

    manifest_path = write_register_manifest(out)
    assert manifest_path == out / RUN_MANIFEST_FILENAME
    assert manifest_path.exists()

    table = pq.read_table(manifest_path)
    assert table.schema.equals(REGISTER_MANIFEST_SCHEMA)
    rows = {r["output_file"]: r for r in table.to_pylist()}

    # Both the records file and the names sidecar are indexed.
    assert "20260601Public01.parquet" in rows
    assert "20260601Public01_names.parquet" in rows
    # The manifest itself is never indexed.
    assert RUN_MANIFEST_FILENAME not in rows

    records = rows["20260601Public01.parquet"]
    names = rows["20260601Public01_names.parquet"]
    assert records["role"] == "records"
    assert names["role"] == "names"
    assert records["source_file"] == "20260601Public01.xml"
    assert records["row_count"] == 1
    # LGL display name + TRD trading name -> two name rows.
    assert names["row_count"] == 2
    assert records["schema_version"]
    assert records["source_md5"]  # compute_md5=True populated it


def test_manifest_empty_dir_is_schema_correct(tmp_path: Path):
    out = tmp_path / "empty"
    out.mkdir()
    manifest_path = write_register_manifest(out)
    table = pq.read_table(manifest_path)
    assert table.num_rows == 0
    assert table.schema.equals(REGISTER_MANIFEST_SCHEMA)


def test_manifest_excludes_itself_on_rerun(tmp_path: Path):
    out = tmp_path / "out"
    xml = tmp_path / "20260601Public01.xml"
    _write_extract(xml, _IND_RECORD)
    ingest_abn_xml(xml, out, compute_md5=False)

    write_register_manifest(out)
    # Re-running must not index the manifest written by the first run.
    table = pq.read_table(write_register_manifest(out))
    assert RUN_MANIFEST_FILENAME not in {r["output_file"] for r in table.to_pylist()}
    assert table.num_rows == 2
