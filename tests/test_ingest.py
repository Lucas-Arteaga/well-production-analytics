"""Tests for source resolution, chunked reading and the Parquet cache.

None of these touch the network: resource selection takes the CKAN payload as an argument, and
ingestion is exercised against a local CSV written into ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wellprod.ingest import (
    IngestError,
    IngestManifest,
    cache_paths,
    ingest_year,
    load_year,
    pick_resource,
    read_source,
)

RESOURCES: list[object] = [
    {"format": "SHP", "name": "Capítulo IV - Pozos", "url": "http://example.test/pozos.shp"},
    {
        "format": "CSV",
        "name": "Producción de Pozos de Gas y Petróleo - 2026",
        "url": "http://example.test/anual-2026.csv",
    },
    {
        "format": "CSV",
        "name": "Producción de Pozos de Gas y Petróleo - 2026 (DDJJ abiertas y cerradas)",
        "url": "http://example.test/ddjj-2026.csv",
    },
    {
        "format": "CSV",
        "name": "Producción de Pozos de Gas y Petróleo - 2025",
        "url": "http://example.test/anual-2025.csv",
    },
    "not a dict",
]


def write_csv(path: Path, rows: int = 5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["anio,mes,idpozo,prod_pet,prod_gas,prod_agua,cuenca,empresa"]
    for index in range(rows):
        lines.append(
            f"2026,{(index % 12) + 1},{1000 + index},"
            f"{(index + 1) * 10},{(index + 1) * 0.5},{(index + 1) * 3},"
            f"NEUQUINA,OPERADORA SA"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------
# Resource selection
# ---------------------------------------------------------------------------------------


def test_pick_resource_prefers_the_plain_annual_file() -> None:
    resource = pick_resource(RESOURCES, 2026)

    assert resource.url.endswith("anual-2026.csv")
    assert "DDJJ" not in resource.name


def test_pick_resource_can_choose_the_sworn_statement_variant() -> None:
    resource = pick_resource(RESOURCES, 2026, prefer="ddjj")

    assert resource.url.endswith("ddjj-2026.csv")


def test_pick_resource_selects_the_requested_year() -> None:
    assert pick_resource(RESOURCES, 2025).url.endswith("anual-2025.csv")


def test_pick_resource_ignores_non_csv_and_non_dict_entries() -> None:
    """The shapefile and the stray string must not be selected, or the year selection breaks."""
    resource = pick_resource(RESOURCES, 2026)

    assert resource.url.endswith(".csv")


def test_pick_resource_raises_for_a_year_the_dataset_does_not_carry() -> None:
    with pytest.raises(IngestError, match="no CSV resource found for 1999"):
        pick_resource(RESOURCES, 1999)


def test_pick_resource_raises_when_the_variant_is_absent() -> None:
    only_annual = [RESOURCES[1]]

    with pytest.raises(IngestError, match="no 'ddjj' variant"):
        pick_resource(only_annual, 2026, prefer="ddjj")


# ---------------------------------------------------------------------------------------
# Chunked reading
# ---------------------------------------------------------------------------------------


def test_read_source_coerces_volumes_to_float_and_ids_to_integer(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "sample.csv", rows=3)

    frame = next(iter(read_source(source)))

    assert frame["prod_pet"].dtype == "float64"
    assert str(frame["idpozo"].dtype) == "Int64"
    # pandas 3.0 gives text columns its own ``str`` dtype rather than ``object``, so the check is
    # expressed as "is text" instead of naming a dtype that only one major version uses.
    assert pd.api.types.is_string_dtype(frame["cuenca"])
    assert frame["cuenca"].iloc[0] == "NEUQUINA"


def test_read_source_splits_into_chunks(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "sample.csv", rows=10)

    chunks = list(read_source(source, chunk_size=4))

    assert [len(chunk) for chunk in chunks] == [4, 4, 2]


def test_read_source_limit_truncates_exactly(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "sample.csv", rows=10)

    chunks = list(read_source(source, chunk_size=4, limit=6))

    assert sum(len(chunk) for chunk in chunks) == 6


def test_read_source_rejects_a_non_positive_chunk_size(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "sample.csv")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        next(iter(read_source(source, chunk_size=0)))


# ---------------------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------------------


def test_cache_paths_are_year_scoped(tmp_path: Path) -> None:
    directory, csv, manifest = cache_paths(tmp_path, 2026)

    assert directory == tmp_path / "year=2026"
    assert csv.parent == directory
    assert manifest.name == "manifest.json"


def test_ingest_year_caches_parts_and_a_manifest(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "source.csv", rows=7)
    cache = tmp_path / "cache"

    manifest = ingest_year(2026, cache, source_url=str(source), chunk_size=3, progress=False)

    assert manifest.rows == 7
    assert manifest.chunks == 3
    assert manifest.is_complete
    assert manifest.truncated_at_row is None
    assert manifest.source_bytes > 0
    assert len(manifest.source_sha256) == 64
    assert manifest.columns_missing, "the fixture has only a few contract columns"

    directory, _, manifest_file = cache_paths(cache, 2026)
    assert manifest_file.exists()
    for part in manifest.parts:
        assert (directory / part).exists()


def test_ingest_year_records_truncation_instead_of_hiding_it(tmp_path: Path) -> None:
    """A head-of-file sample must be recorded, or a report can present it as the whole year."""
    source = write_csv(tmp_path / "source.csv", rows=100)

    manifest = ingest_year(
        2026, tmp_path / "cache", source_url=str(source), chunk_size=10, limit=25, progress=False
    )

    assert manifest.rows == 25
    assert manifest.truncated_at_row == 25
    assert not manifest.is_complete


def test_ingest_year_returns_the_cache_on_a_second_call(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "source.csv", rows=4)
    cache = tmp_path / "cache"

    first = ingest_year(2026, cache, source_url=str(source), progress=False)
    second = ingest_year(2026, cache, source_url="http://does-not-exist.test/x.csv", progress=False)

    assert second.rows == first.rows
    assert second.source_url == first.source_url


def test_ingest_year_rejects_a_missing_local_source(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="local source not found"):
        ingest_year(2026, tmp_path / "cache", source_url=str(tmp_path / "nope.csv"), progress=False)


# ---------------------------------------------------------------------------------------
# Loading back
# ---------------------------------------------------------------------------------------


def test_load_year_round_trips_the_cache(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "source.csv", rows=5)
    cache = tmp_path / "cache"
    ingest_year(2026, cache, source_url=str(source), chunk_size=2, progress=False)

    frame = load_year(2026, cache)

    assert len(frame) == 5
    assert set(frame.columns) >= {"prod_pet", "prod_gas", "prod_agua"}
    assert frame["prod_pet"].sum() == pytest.approx(10 + 20 + 30 + 40 + 50)


def test_load_year_can_project_columns(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "source.csv", rows=3)
    cache = tmp_path / "cache"
    ingest_year(2026, cache, source_url=str(source), progress=False)

    frame = load_year(2026, cache, columns=["prod_pet"])

    assert list(frame.columns) == ["prod_pet"]


def test_load_year_without_a_cache_explains_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no cache for 2026"):
        load_year(2026, tmp_path / "empty")


# ---------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------


def test_manifest_json_round_trips(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "source.csv", rows=3)
    manifest = ingest_year(2026, tmp_path / "cache", source_url=str(source), progress=False)

    restored = IngestManifest.from_json(manifest.to_json())

    assert restored == manifest
    assert restored.is_complete


def test_manifest_rejects_invalid_json() -> None:
    with pytest.raises(IngestError, match="not valid JSON"):
        IngestManifest.from_json("{not json")


def test_manifest_rejects_a_non_object_payload() -> None:
    with pytest.raises(IngestError, match="must be a JSON object"):
        IngestManifest.from_json("[1, 2, 3]")


def test_manifest_drops_unknown_keys_when_reading_older_versions() -> None:
    payload = IngestManifest(
        year=2026,
        source_name="x",
        source_url="y",
        source_sha256="0" * 64,
        source_bytes=1,
        rows=1,
        chunks=1,
    ).to_json()

    forward_compatible = payload.replace('"rows": 1', '"rows": 1,\n  "invented_later": true')

    assert IngestManifest.from_json(forward_compatible).rows == 1


def test_is_complete_reflects_truncation() -> None:
    base = IngestManifest(
        year=2026,
        source_name="x",
        source_url="y",
        source_sha256="0" * 64,
        source_bytes=1,
        rows=1,
        chunks=1,
    )

    assert base.is_complete
    base.truncated_at_row = 10
    assert not base.is_complete


def test_read_source_produces_a_dataframe_per_chunk_type(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "source.csv", rows=2)

    for chunk in read_source(source):
        assert isinstance(chunk, pd.DataFrame)
