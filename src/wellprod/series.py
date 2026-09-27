"""Time series over several years of the published dataset.

The trap this module exists to avoid
------------------------------------
The yearly files are published as they fill up. The 2026 file covers January to August; the 2024
and 2025 files cover the full twelve months. Comparing their annual totals directly says that
national oil production fell by a quarter in 2026 — which is the exact opposite of the truth, and
it is the conclusion a careful-looking analysis reaches by not checking month coverage first.

So nothing here compares raw annual totals. A window of :func:`common_months` is computed across
the years being compared, every total is restricted to it, and the window travels with the report.

Memory
------
A year is about a million rows. :func:`annual_totals` therefore takes an **iterable** of
``(year, frame)`` pairs rather than a mapping, so a caller can load one year, aggregate it, free
it and move on, instead of holding every year at once.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Final, Literal

import pandas as pd

from wellprod.convert import as_float, as_int
from wellprod.profile import metrics_frame

#: Metrics the series can report. Every volume is in cubic metres.
SERIES_METRICS: Final[tuple[str, ...]] = (
    "oil_m3",
    "gas_m3",
    "water_m3",
    "water_cut",
    "gor_m3_m3",
    "wells",
)

#: How each metric collapses within a period. A volume sums; a ratio does not.
AggName = Literal["sum", "mean", "median", "nunique"]

AGGREGATIONS: Final[dict[str, AggName]] = {
    "oil_m3": "sum",
    "gas_m3": "sum",
    "water_m3": "sum",
    "water_cut": "mean",
    "gor_m3_m3": "median",
    "wells": "nunique",
}

#: Metric label and unit, for the reports.
METRIC_LABELS: Final[dict[str, tuple[str, str]]] = {
    "oil_m3": ("Oil", "m³"),
    "gas_m3": ("Gas", "m³"),
    "water_m3": ("Water", "m³"),
    "water_cut": ("Water cut", "%"),
    "gor_m3_m3": ("GOR median", "m³/m³"),
    "wells": ("Wells", "count"),
}

_SOURCE_COLUMNS: Final[dict[str, str]] = {
    "oil_m3": "prod_pet_m3",
    "gas_m3": "prod_gas_m3",
    "water_m3": "prod_agua_m3",
    "water_cut": "water_cut",
    "gor_m3_m3": "gor_m3_m3",
    "wells": "well",
}


def _check_metric(metric: str) -> None:
    if metric not in SERIES_METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {', '.join(SERIES_METRICS)}")


def _prepare(
    frame: pd.DataFrame,
    *,
    months: Iterable[int] | None = None,
    dimension: str | None = None,
) -> pd.DataFrame:
    """A frame with the metric columns, the month, and optionally the dimension, as columns."""
    working = metrics_frame(frame)
    working["well"] = frame["idpozo"] if "idpozo" in frame.columns else pd.NA
    working["mes"] = pd.to_numeric(frame["mes"], errors="coerce").astype("Int64")
    working["anio"] = pd.to_numeric(frame["anio"], errors="coerce").astype("Int64")

    if dimension is not None:
        if dimension not in frame.columns:
            raise KeyError(f"missing column {dimension!r}")
        working[dimension] = frame[dimension].astype("string")

    if months is not None:
        window = list(months)
        working = working[working["mes"].isin(window)]

    return working


def month_coverage(frame: pd.DataFrame) -> list[int]:
    """The months present in a frame, sorted.

    A file that stops in August is not a smaller year, it is a shorter one, and the difference
    decides whether two years can be compared at all.
    """
    if "mes" not in frame.columns:
        raise KeyError("missing column 'mes'")
    months = pd.to_numeric(frame["mes"], errors="coerce").dropna().unique()
    return sorted(as_int(month, context="month") for month in months)


def common_months(coverages: Iterable[Iterable[int]]) -> list[int]:
    """Months present in **every** coverage — the window in which the years are comparable."""
    sets = [set(coverage) for coverage in coverages]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def period_index(frame: pd.DataFrame) -> pd.PeriodIndex:
    """A monthly period per row, from the ``anio`` and ``mes`` columns."""
    for required in ("anio", "mes"):
        if required not in frame.columns:
            raise KeyError(f"missing column {required!r}")
    stamp = pd.to_datetime(
        {
            "year": pd.to_numeric(frame["anio"], errors="coerce"),
            "month": pd.to_numeric(frame["mes"], errors="coerce"),
            "day": 1,
        },
        errors="coerce",
    )
    return pd.PeriodIndex(stamp, freq="M")


def monthly_series(
    frame: pd.DataFrame,
    *,
    metric: str = "oil_m3",
    dimension: str | None = None,
) -> pd.DataFrame:
    """One row per month, one column per dimension value (or a single column for the metric).

    With a dimension, the result is a pivot: months down, dimension values across. That shape is
    what makes a share shift visible — unconventional gaining on conventional reads as one column
    growing past another.
    """
    _check_metric(metric)
    working = _prepare(frame, dimension=dimension)
    working["period"] = period_index(frame)
    source = _SOURCE_COLUMNS[metric]
    how = AGGREGATIONS[metric]

    if dimension is None:
        grouped = working.groupby("period")[source].agg(how)
        return grouped.to_frame(metric).sort_index()

    table = working.pivot_table(
        index="period",
        columns=dimension,
        values=source,
        aggfunc=how,
        observed=True,
    )
    return table.sort_index()


def annual_totals(
    years_frames: Iterable[tuple[int, pd.DataFrame]],
    *,
    metric: str = "oil_m3",
    dimension: str | None = None,
    months: Iterable[int] | None = None,
) -> pd.DataFrame:
    """One row per year, restricted to ``months`` when given.

    Takes an iterable so the caller can stream years: load one, aggregate it, let it go.
    """
    _check_metric(metric)
    source = _SOURCE_COLUMNS[metric]
    how = AGGREGATIONS[metric]

    rows: list[dict[str, object]] = []
    for year, frame in years_frames:
        working = _prepare(frame, months=months, dimension=dimension)
        if dimension is None:
            rows.append({"year": year, metric: working[source].agg(how)})
            continue
        grouped = working.groupby(dimension, dropna=False)[source].agg(how)
        entry: dict[str, object] = {"year": year}
        entry.update({str(key): value for key, value in grouped.items()})
        rows.append(entry)

    return pd.DataFrame(rows).set_index("year").sort_index()


def year_over_year(totals: pd.DataFrame) -> pd.DataFrame:
    """Percentage change against the previous year, per column.

    The first year has no predecessor, so it stays ``NaN`` rather than being reported as zero
    growth — which would read as a flat year instead of an unknown one.
    """
    return totals.pct_change() * 100.0


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------


def _number_or_none(value: object, *, context: str) -> float | None:
    """The value as a plain float, or ``None`` when it is missing or not numeric.

    Funnels the two ways a cell can fail to be a number — an unconvertible value and a ``NaN`` —
    through one place, so the renderers below never call ``pd.isna`` on an ``object``
    and never meet a ``nan`` in an f-string.
    """
    try:
        number = as_float(value, context=context)
    except TypeError:
        return None
    return None if math.isnan(number) else number


def _format(value: object, metric: str) -> str:
    number = _number_or_none(value, context=metric)
    if number is None:
        return "—"
    if metric == "water_cut":
        return f"{number:.1%}"
    if metric == "wells":
        return f"{number:,.0f}"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.2f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:,.1f}k"
    return f"{number:,.1f}"


def to_markdown(totals: pd.DataFrame, *, metric: str, months: Iterable[int] | None = None) -> str:
    """Render annual totals as a Markdown table, stating the window it covers."""
    _check_metric(metric)
    label, unit = METRIC_LABELS[metric]

    header = [str(column) for column in totals.columns]
    lines = [
        "| Year | " + " | ".join(header) + " |",
        "| ---: | " + " | ".join("---:" for _ in header) + " |",
    ]
    for year, row in totals.iterrows():
        cells = [_format(row[column], metric) for column in totals.columns]
        lines.append(f"| {as_int(year, context='year')} | " + " | ".join(cells) + " |")

    window = ""
    if months is not None:
        window = " · window: months " + ", ".join(str(month) for month in months)
    lines.append("")
    lines.append(f"_{label} in {unit} per year{window}._")
    return "\n".join(lines)


def growth_markdown(growth: pd.DataFrame) -> str:
    """Render year-over-year percentages, with missing years left blank."""
    header = [str(column) for column in growth.columns]
    lines = [
        "| Year | " + " | ".join(header) + " |",
        "| ---: | " + " | ".join("---:" for _ in header) + " |",
    ]
    for year, row in growth.iterrows():
        cells = []
        for column in growth.columns:
            number = _number_or_none(row[column], context=str(column))
            cells.append("—" if number is None else f"{number:+.1f}%")
        lines.append(f"| {as_int(year, context='year')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def coverage_table(coverages: Mapping[int, Iterable[int]]) -> str:
    """State, per year, which months the file actually covers."""
    lines = [
        "| Year | Months covered | Comparable |",
        "| ---: | :--- | :--- |",
    ]
    for year, months in sorted(coverages.items()):
        present = sorted(as_int(month, context="month") for month in months)
        complete = "yes" if len(present) == 12 else f"**no — {len(present)} of 12**"
        lines.append(f"| {year} | {', '.join(str(m) for m in present)} | {complete} |")
    return "\n".join(lines)
