"""End-to-end tests for the command line interface.

The cache is built from a local CSV, so the whole CLI is exercised without a network call or a
200 MB download.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wellprod.cli import DEFAULT_PROFILE_DIMENSIONS, build_parser, main
from wellprod.ingest import ingest_year

HEADER = (
    "anio,mes,idpozo,prod_pet,prod_gas,prod_agua,iny_agua,iny_gas,iny_co2,iny_otro,"
    "areayacimiento,provincia,empresa,cuenca,formacion,tipo_de_recurso"
)
BASINS = ("NEUQUINA", "GOLFO SAN JORGE", "CUYANA")


def build_cache(tmp_path: Path, rows: int = 12) -> Path:
    """Write a small CSV shaped like the published one and ingest it into a temp cache."""
    source = tmp_path / "source.csv"
    lines = [HEADER]
    for index in range(rows):
        lines.append(
            f"2026,{(index % 12) + 1},{1000 + index},"
            f"{100 + index},{5 + index},{50 + index},0,0,0,0,"
            f"CAMPO NORTE,Neuquén,OPERADORA {index % 2},{BASINS[index % 3]},"
            f"MULICHINCO,CONVENCIONAL"
        )
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cache = tmp_path / "cache"
    ingest_year(2026, cache, source_url=str(source), chunk_size=5, progress=False)
    return cache


# ---------------------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------------------


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_every_declared_subcommand_parses() -> None:
    parser = build_parser()

    for argv in (
        ["ingest", "--year", "2026"],
        ["quality", "--year", "2026"],
        ["profile", "--year", "2026", "--by", "cuenca"],
        ["report", "--year", "2026"],
    ):
        assert parser.parse_args(argv).year == 2026


# ---------------------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------------------


def test_quality_writes_a_markdown_report(tmp_path: Path) -> None:
    cache = build_cache(tmp_path)
    out = tmp_path / "reports"

    code = main(["quality", "--year", "2026", "--cache", str(cache), "--out", str(out)])

    assert code == 0
    report = (out / "quality-2026.md").read_text(encoding="utf-8")
    assert "# Data quality — 2026" in report
    assert "Modeling-eligible share" in report
    assert "columns read:" in report


def test_quality_can_also_emit_json(tmp_path: Path) -> None:
    cache = build_cache(tmp_path)
    destination = tmp_path / "quality.json"

    code = main(
        [
            "quality",
            "--year",
            "2026",
            "--cache",
            str(cache),
            "--json",
            str(destination),
        ]
    )

    assert code == 0
    assert '"n_rows": 12' in destination.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------------------


def test_profile_prints_a_table_for_one_dimension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = build_cache(tmp_path)

    code = main(
        ["profile", "--year", "2026", "--cache", str(cache), "--by", "cuenca", "--top", "2"]
    )

    assert code == 0
    printed = capsys.readouterr().out
    # The fixture's oil rises with the row index, so the two largest basins are the last two
    # written — CUYANA then GOLFO SAN JORGE. NEUQUINA is genuinely outside the top two, which is
    # what makes this a test of the sort rather than of the renderer.
    assert printed.startswith("# Production profile by cuenca")
    assert "CUYANA" in printed
    assert "NEUQUINA" not in printed


def test_written_reports_keep_the_unit_symbols(tmp_path: Path) -> None:
    """The artifact on disk is the deliverable, so its UTF-8 units must survive."""
    cache = build_cache(tmp_path)
    out = tmp_path / "reports"

    main(["report", "--year", "2026", "--cache", str(cache), "--out", str(out)])

    report = (out / "profile-cuenca-2026.md").read_text(encoding="utf-8")
    assert "Oil (m³)" in report
    assert "GOR median (m³/m³)" in report


# ---------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------


def test_report_writes_quality_plus_every_default_profile(tmp_path: Path) -> None:
    cache = build_cache(tmp_path)
    out = tmp_path / "reports"

    code = main(["report", "--year", "2026", "--cache", str(cache), "--out", str(out)])

    assert code == 0
    assert (out / "quality-2026.md").exists()
    for dimension in DEFAULT_PROFILE_DIMENSIONS:
        assert (out / f"profile-{dimension}-2026.md").exists()


def test_report_states_where_the_numbers_came_from(tmp_path: Path) -> None:
    """A report without provenance is a claim, so the sha and the row count must be in it."""
    cache = build_cache(tmp_path)
    out = tmp_path / "reports"

    main(["report", "--year", "2026", "--cache", str(cache), "--out", str(out)])

    report = (out / "quality-2026.md").read_text(encoding="utf-8")
    assert "source: **" in report
    assert "sha256: `" in report
    assert "complete published file" in report


# ---------------------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------------------


def test_a_missing_cache_returns_two_instead_of_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["quality", "--year", "2026", "--cache", str(tmp_path / "empty")])

    assert code == 2
    assert "error: no cache for 2026" in capsys.readouterr().err


def test_a_missing_local_source_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "ingest",
            "--year",
            "2026",
            "--cache",
            str(tmp_path / "c"),
            "--source",
            str(tmp_path / "no.csv"),
        ]
    )

    assert code == 2
    assert "error:" in capsys.readouterr().err
