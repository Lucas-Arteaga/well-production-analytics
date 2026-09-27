"""Command line interface for wellprod.

The commands are deliberately thin: each one resolves a cache, reads a **projection** of the
columns it needs, and hands the work to a module that is tested on its own. A full year of the
published file does not fit comfortably in memory, so nothing here loads the whole contract.

::

    wellprod ingest  --year 2026 --limit 200000
    wellprod quality --year 2026 --out reports
    wellprod profile --year 2026 --by cuenca --top 10
    wellprod report  --year 2026 --out reports
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

from wellprod.ingest import IngestError, cache_paths, ingest_year, load_year
from wellprod.profile import PROFILE_DIMENSIONS, profile_by, to_markdown
from wellprod.quality import build_report
from wellprod.schema import COLUMNS, VOLUME_COLUMNS
from wellprod.series import (
    SERIES_METRICS,
    annual_totals,
    common_months,
    coverage_table,
    growth_markdown,
    month_coverage,
    year_over_year,
)
from wellprod.series import to_markdown as series_to_markdown

#: Columns the quality check reads. A projection of the contract, on purpose.
QUALITY_COLUMNS: tuple[str, ...] = (
    "anio",
    "mes",
    "idpozo",
    *VOLUME_COLUMNS,
    "areayacimiento",
    "provincia",
    "empresa",
    "cuenca",
    "formacion",
    "tipo_de_recurso",
)

#: Columns the profiling reads.
PROFILE_COLUMNS: tuple[str, ...] = ("anio", "idpozo", "prod_pet", "prod_gas", "prod_agua")

#: Dimensions profiled by ``wellprod report`` when ``--by`` is not given.
DEFAULT_PROFILE_DIMENSIONS: tuple[str, ...] = (
    "cuenca",
    "empresa",
    "formacion",
    "tipo_de_recurso",
)


#: Columns the multi-year series reads.
SERIES_COLUMNS: tuple[str, ...] = (
    "anio",
    "mes",
    "idpozo",
    "prod_pet",
    "prod_gas",
    "prod_agua",
)


def _projection(base: Sequence[str], extra: Sequence[str]) -> list[str]:
    columns = list(base)
    for name in extra:
        if name not in columns:
            columns.append(name)
    return columns


def _provenance(year: int, cache: Path) -> str:
    """A one-line statement of where the numbers came from, and whether the file was complete.

    A report without provenance is a claim. This makes it checkable, including the honest case
    where only the head of the file was read.
    """
    _, _, manifest_file = cache_paths(cache, year)
    if not manifest_file.exists():
        return f"source: cache for {year} (no manifest)"
    from wellprod.ingest import IngestManifest

    manifest = IngestManifest.from_json(manifest_file.read_text(encoding="utf-8"))
    completeness = (
        "complete published file"
        if manifest.is_complete
        else f"**truncated to the first {manifest.truncated_at_row:,} rows — a biased head of "
        "the file, ordered by operator code, not a random sample**"
    )
    return (
        f"source: **{manifest.source_name}**  \n"
        f"rows: {manifest.rows:,} · sha256: `{manifest.source_sha256[:16]}…` · {completeness}"
    )


def _projection_note() -> str:
    """State the projection explicitly, so a reader knows the report is not over every column."""
    return f"columns read: {len(QUALITY_COLUMNS)} of {len(COLUMNS)} in the published contract"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    manifest = ingest_year(
        args.year,
        args.cache,
        source_url=args.source,
        chunk_size=args.chunk_size,
        limit=args.limit,
        force=args.force,
    )
    print(manifest.to_json(indent=2))
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    frame = load_year(args.year, args.cache, columns=list(QUALITY_COLUMNS))
    report = build_report(frame, check_gor=not args.no_gor, expected_columns=QUALITY_COLUMNS)

    lines = [
        f"# Data quality — {args.year}",
        "",
        _provenance(args.year, Path(args.cache)),
        _projection_note(),
        "",
        report.to_markdown(),
        "",
    ]
    body = "\n".join(lines)

    if args.out:
        destination = _write(Path(args.out) / f"quality-{args.year}.md", body)
        print(f"wrote {destination}")
    if args.json:
        _write(Path(args.json), __import__("json").dumps(report.to_dict(), indent=2) + "\n")
        print(f"wrote {args.json}")
    if not args.out and not args.json:
        print(body)
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    columns = _projection(PROFILE_COLUMNS, [args.by])
    frame = load_year(args.year, args.cache, columns=columns)
    profile = profile_by(frame, args.by, top=args.top)

    lines = [
        f"# Production profile by {args.by} — {args.year}",
        "",
        _provenance(args.year, Path(args.cache)),
        "",
        to_markdown(profile, dimension=args.by),
        "",
    ]
    body = "\n".join(lines)

    if args.out:
        destination = _write(Path(args.out) / f"profile-{args.by}-{args.year}.md", body)
        print(f"wrote {destination}")
    else:
        print(body)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Run the quality check and every default dimension, writing one bundle of reports."""
    out = Path(args.out)
    written: list[Path] = []

    quality_frame = load_year(args.year, args.cache, columns=list(QUALITY_COLUMNS))
    report = build_report(quality_frame, expected_columns=QUALITY_COLUMNS)
    written.append(
        _write(
            out / f"quality-{args.year}.md",
            "\n".join(
                [
                    f"# Data quality — {args.year}",
                    "",
                    _provenance(args.year, Path(args.cache)),
                    _projection_note(),
                    "",
                    report.to_markdown(),
                    "",
                ]
            ),
        )
    )
    del quality_frame

    for dimension in DEFAULT_PROFILE_DIMENSIONS:
        frame = load_year(args.year, args.cache, columns=_projection(PROFILE_COLUMNS, [dimension]))
        profile = profile_by(frame, dimension, top=args.top)
        written.append(
            _write(
                out / f"profile-{dimension}-{args.year}.md",
                "\n".join(
                    [
                        f"# Production profile by {dimension} — {args.year}",
                        "",
                        _provenance(args.year, Path(args.cache)),
                        "",
                        to_markdown(profile, dimension=dimension),
                        "",
                    ]
                ),
            )
        )
        del frame

    print(f"wrote {len(written)} report(s) to {out}")
    for path in written:
        print(f"  {path}")
    return 0


def _coverage(cache: Path, years: Sequence[int]) -> dict[int, list[int]]:
    """Which months each cached year actually covers.

    A cheap projection: this decides whether two years can be compared at all, so it runs before
    anything expensive is loaded.
    """
    coverage: dict[int, list[int]] = {}
    for year in years:
        frame = load_year(year, cache, columns=["mes"])
        coverage[year] = month_coverage(frame)
        del frame
    return coverage


def cmd_series(args: argparse.Namespace) -> int:
    """Compare years over a window of months present in every one of them."""
    cache = Path(args.cache)
    years = list(args.years)
    coverage = _coverage(cache, years)
    months = common_months(coverage.values())
    if not months:
        print("error: no month is present in every requested year", file=sys.stderr)
        return 2

    columns = _projection(SERIES_COLUMNS, [args.by] if args.by else [])

    def streamed() -> Iterator[tuple[int, pd.DataFrame]]:
        """Yield one year at a time so three years are never resident together."""
        for year in years:
            frame = load_year(year, cache, columns=columns)
            yield year, frame
            del frame

    totals = annual_totals(streamed(), metric=args.metric, dimension=args.by, months=months)
    growth = year_over_year(totals)

    window_note = (
        f"**Comparable window: months {', '.join(str(month) for month in months)}.** Every "
        "total below is restricted to it. A file that stops in August is a *shorter* year, not a "
        "smaller one: comparing raw annual totals would show a collapse that is purely an "
        "artefact of the publication date."
    )
    body = "\n".join(
        [
            f"# Multi-year series — {args.metric}",
            "",
            window_note,
            "",
            "## Coverage per year",
            "",
            coverage_table(coverage),
            "",
            "## Annual totals",
            "",
            series_to_markdown(totals, metric=args.metric, months=months),
            "",
            "## Year over year",
            "",
            growth_markdown(growth),
            "",
            "## Provenance",
            "",
        ]
    )
    for year in years:
        body += f"- {year}: {_provenance(year, cache)}\n"

    suffix = f"-{args.by}" if args.by else ""
    destination = _write(
        Path(args.out) / f"series-{args.metric}{suffix}-{years[0]}-{years[-1]}.md", body
    )
    print(f"wrote {destination}")
    return 0


# ---------------------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------------------


def _ensure_printable_output() -> None:
    """Make printing UTF-8 where possible, and never crash where it is not.

    Reports carry ``m³`` and em-dashes. A Windows console with a legacy code page raises
    UnicodeEncodeError on those, which would turn a successful run into a traceback, and at best
    prints the unit as a replacement character. The files are always written as UTF-8; this is
    only about the terminal.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is None:
        return
    with contextlib.suppress(ValueError, OSError):
        reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wellprod",
        description=(
            "Production analytics over Argentina's official open per-well oil & gas dataset "
            "(Secretaría de Energía, CC-BY-4.0)."
        ),
    )
    parser.add_argument("--version", action="version", version="wellprod 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--year", type=int, required=True, help="production year to work on")
    common.add_argument(
        "--cache",
        type=Path,
        default=Path("data/cache"),
        help="cache directory (default: data/cache)",
    )

    ingest = subparsers.add_parser(
        "ingest", parents=[common], help="download and cache a year as Parquet"
    )
    ingest.add_argument(
        "--source",
        default=None,
        help="an explicit CSV url or local path, instead of resolving the official one",
    )
    ingest.add_argument(
        "--limit", type=int, default=None, help="stop after N rows (records that it truncated)"
    )
    ingest.add_argument("--chunk-size", type=int, default=250_000, help="rows per parsed chunk")
    ingest.add_argument("--force", action="store_true", help="ignore any existing cache")
    ingest.set_defaults(func=cmd_ingest)

    quality = subparsers.add_parser(
        "quality", parents=[common], help="write the data-quality report"
    )
    quality.add_argument("--out", type=Path, default=None, help="directory for the Markdown report")
    quality.add_argument("--json", default=None, help="also write the raw report as JSON here")
    quality.add_argument("--no-gor", action="store_true", help="skip the gas-oil ratio check")
    quality.set_defaults(func=cmd_quality)

    profile = subparsers.add_parser(
        "profile", parents=[common], help="aggregate production by one dimension"
    )
    profile.add_argument(
        "--by",
        choices=PROFILE_DIMENSIONS,
        default="cuenca",
        help="dimension to aggregate by (default: cuenca)",
    )
    profile.add_argument("--top", type=int, default=15, help="keep the top N rows (default: 15)")
    profile.add_argument("--out", type=Path, default=None, help="directory for the Markdown report")
    profile.set_defaults(func=cmd_profile)

    report = subparsers.add_parser(
        "report", parents=[common], help="quality plus every default profile, in one directory"
    )
    report.add_argument("--out", type=Path, default=Path("reports"), help="output directory")
    report.add_argument("--top", type=int, default=15, help="rows per profile (default: 15)")
    report.set_defaults(func=cmd_report)

    series = subparsers.add_parser(
        "series",
        help="compare years over a window of months present in every one of them",
    )
    series.add_argument(
        "--years", type=int, nargs="+", required=True, help="years to compare, in order"
    )
    series.add_argument(
        "--metric",
        choices=SERIES_METRICS,
        default="oil_m3",
        help="what to compare over time (default: oil_m3)",
    )
    series.add_argument(
        "--by",
        choices=PROFILE_DIMENSIONS,
        default=None,
        help="split the series by a dimension, for example tipo_de_recurso",
    )
    series.add_argument("--cache", type=Path, default=Path("data/cache"), help="cache directory")
    series.add_argument("--out", type=Path, default=Path("reports"), help="output directory")
    series.set_defaults(func=cmd_series)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_printable_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (IngestError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
