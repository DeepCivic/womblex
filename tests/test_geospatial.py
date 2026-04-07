"""Tests for geospatial SHP → GeoParquet ingest."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from womblex.ingest.geospatial import (
    GeospatialIngestResult,
    discover_shapefiles,
    ingest_geospatial_directory,
    ingest_shapefile,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures" / "womblex-collection"
_SHP_DIR = FIXTURE_DIR / "_SHP" / "ntd_register_nat_shp"
_SHP_FILE = _SHP_DIR / "NTD_Register_Nat.shp"


# ── Real fixture tests ──────────────────────────────────────────────────────


class TestRealShapefileIngest:
    """Tests against the NTD Register fixture (20 features, EPSG:7844)."""

    @pytest.fixture(autouse=True)
    def _require_fixture(self):
        if not _SHP_FILE.exists():
            pytest.skip("SHP fixture not available")

    def test_ingest_produces_geoparquet(self, tmp_path: Path) -> None:
        result = ingest_shapefile(_SHP_FILE, tmp_path)
        assert result.error is None
        assert result.output is not None
        assert result.output.exists()
        assert result.output.suffix == ".parquet"

    def test_feature_count_preserved(self, tmp_path: Path) -> None:
        result = ingest_shapefile(_SHP_FILE, tmp_path)
        assert result.features == 20

    def test_crs_preserved(self, tmp_path: Path) -> None:
        result = ingest_shapefile(_SHP_FILE, tmp_path)
        assert result.crs == "EPSG:7844"

    def test_geometry_type(self, tmp_path: Path) -> None:
        result = ingest_shapefile(_SHP_FILE, tmp_path)
        assert result.geometry_type == "Polygon"

    def test_provenance_metadata(self, tmp_path: Path) -> None:
        result = ingest_shapefile(_SHP_FILE, tmp_path)
        table = pq.read_table(str(result.output))
        meta = table.schema.metadata
        assert meta[b"geospatial.source_file"] == b"NTD_Register_Nat.shp"
        assert meta[b"geospatial.feature_count"] == b"20"
        assert meta[b"geospatial.crs"] == b"EPSG:7844"
        assert b"geospatial.source_md5" in meta

    def test_no_md5(self, tmp_path: Path) -> None:
        result = ingest_shapefile(_SHP_FILE, tmp_path, compute_md5=False)
        table = pq.read_table(str(result.output))
        assert b"geospatial.source_md5" not in table.schema.metadata

    def test_attributes_preserved(self, tmp_path: Path) -> None:
        """All source attribute columns appear in the output."""
        import geopandas as gpd

        result = ingest_shapefile(_SHP_FILE, tmp_path)
        source = gpd.read_file(str(_SHP_FILE), engine="pyogrio")
        output = gpd.read_parquet(str(result.output))

        # All non-geometry columns from source should be in output.
        src_cols = set(source.columns) - {"geometry"}
        out_cols = set(output.columns) - {"geometry"}
        assert src_cols == out_cols

    def test_row_count_matches(self, tmp_path: Path) -> None:
        import geopandas as gpd

        result = ingest_shapefile(_SHP_FILE, tmp_path)
        output = gpd.read_parquet(str(result.output))
        assert len(output) == 20

    def test_geometry_validity(self, tmp_path: Path) -> None:
        import geopandas as gpd

        result = ingest_shapefile(_SHP_FILE, tmp_path)
        output = gpd.read_parquet(str(result.output))
        assert output.geometry.is_valid.all()

    def test_output_is_readable_as_geodataframe(self, tmp_path: Path) -> None:
        """Output GeoParquet can be read back as a GeoDataFrame with CRS."""
        import geopandas as gpd

        result = ingest_shapefile(_SHP_FILE, tmp_path)
        gdf = gpd.read_parquet(str(result.output))
        assert gdf.crs is not None
        assert "7844" in str(gdf.crs)


# ── Directory ingest ────────────────────────────────────────────────────────


class TestDirectoryIngest:
    @pytest.fixture(autouse=True)
    def _require_fixture(self):
        if not _SHP_FILE.exists():
            pytest.skip("SHP fixture not available")

    def test_discover_shapefiles(self) -> None:
        found = discover_shapefiles(_SHP_DIR)
        assert len(found) == 1
        assert found[0].name == "NTD_Register_Nat.shp"

    def test_ingest_directory(self, tmp_path: Path) -> None:
        results = ingest_geospatial_directory(_SHP_DIR, tmp_path)
        assert len(results) == 1
        assert results[0].error is None
        assert results[0].output is not None

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        results = ingest_geospatial_directory(empty, tmp_path / "out")
        assert results == []


# ── Error handling ──────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = ingest_shapefile(tmp_path / "missing.shp", tmp_path / "out")
        assert result.error is not None
        assert result.output is None
