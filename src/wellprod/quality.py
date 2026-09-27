"""Data-quality reporting for the official per-well production dataset.

The dataset is published as-is: it contains wells with no movement, placeholder field
names, administrative provinces, amended records and duplicated rows. Modelling on top
of that produces a confident answer to a question nobody asked.

This module answers one question, cheaply and repeatably: **what is actually in the file,
and how much of it is usable?** The report is meant to be committed next to the analysis
so that a reader can see what was dropped and why.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from wellprod.convert import as_float as _as_float
from wellprod.convert import as_int as _as_int
from wellprod.schema import (
    COLUMNS,
    EXCLUDED_PROVINCIA,
    PLACEHOLDER_YACIMIENTOS,
    PRODUCTION_COLUMNS,
    VOLUME_COLUMNS,
    gas_oil_ratio,
    is_plausible_gor,
)

# ---------------------------------------------------------------------------------------
# The pandas boundary.
#
# `pd.to_numeric` is declared with a very broad return type, and `DataFrame.isna().sum()`
# resolves to an untyped union, so static analysis loses track of the types and the errors
# propagate to every call site. The narrowing lives in these three helpers instead, once.
# ---------------------------------------------------------------------------------------


def _numeric(series: pd.Series) -> pd.Series:
    """A float64 view of a column. Unparseable and missing values become ``NaN``."""
    return pd.Series(pd.to_numeric(series, errors="coerce"), dtype="float64")


def _count_true(mask: pd.Series) -> int:
    """How many values in a boolean mask are true."""
    return _as_int(mask.sum(), context="boolean mask")


def _null_counts(df: pd.DataFrame) -> dict[str, int]:
    """Columns that contain at least one missing value, and how many."""
    counts = df.isna().sum()
    result: dict[str, int] = {}
    for name, value in counts.items():
        total = _as_int(value, context=f"null count for {name}")
        if total > 0:
            result[str(name)] = total
    return result


def _column_stats(values: pd.Series, *, column: str, unit: str | None) -> dict[str, Any]:
    """Descriptive statistics for one fluid column, with its unit declared.

    ``column`` is a parameter rather than a closure over the caller's loop variable: a
    closure binds late, so a deferred call would have labelled every row with the last
    column's name.
    """
    return {
        "unit": unit,
        "sum": _as_float(values.sum(), context=f"{column} sum"),
        "mean": _as_float(values.mean(), context=f"{column} mean"),
        "p50": _as_float(values.median(), context=f"{column} median"),
        "p95": _as_float(values.quantile(0.95), context=f"{column} p95"),
        "max": _as_float(values.max(), context=f"{column} max"),
        "nulls": _count_true(values.isna()),
    }


@dataclass(slots=True)
class DataQualityReport:
    """A plain, serialisable summary of what a slice of the dataset contains."""

    n_rows: int = 0
    n_columns: int = 0
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    negative_volume_counts: dict[str, int] = field(default_factory=dict)
    zero_oil_and_gas_rows: int = 0
    placeholder_field_rows: int = 0
    administrative_province_rows: int = 0
    implausible_gor_rows: int = 0
    columns_requested: int = 0

    @property
    def usable_share(self) -> float:
        """Share of rows with a non-zero oil or gas volume — the modelling-eligible set."""
        if not self.n_rows:
            return 0.0
        return 1.0 - (self.zero_oil_and_gas_rows / self.n_rows)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["usable_share"] = round(self.usable_share, 6)
        return data

    def to_markdown(self) -> str:
        lines = [
            "| Check | Value |",
            "| :--- | ---: |",
            f"| Rows | {self.n_rows:,} |",
            f"| Columns | {self.n_columns} |",
            f"| Missing contract columns | {len(self.missing_columns)} |",
            f"| Unexpected columns | {len(self.unexpected_columns)} |",
            f"| Duplicate rows | {self.duplicate_rows:,} |",
            f"| Rows with zero oil **and** zero gas | {self.zero_oil_and_gas_rows:,} |",
            f"| Rows in placeholder fields | {self.placeholder_field_rows:,} |",
            f"| Rows with administrative province | {self.administrative_province_rows:,} |",
            f"| Rows with implausible gas-oil ratio | {self.implausible_gor_rows:,} |",
            f"| Modeling-eligible share | {self.usable_share:.2%} |",
        ]
        if self.missing_columns:
            lines += ["", f"Missing: {', '.join(f'`{c}`' for c in self.missing_columns)}"]
        if self.unexpected_columns:
            lines += ["", f"Unexpected: {', '.join(f'`{c}`' for c in self.unexpected_columns)}"]
        return "\n".join(lines)


def build_report(
    df: pd.DataFrame,
    *,
    check_gor: bool = True,
    expected_columns: Collection[str] | None = None,
) -> DataQualityReport:
    """Inspect a dataframe against the column contract in :mod:`wellprod.schema`.

    ``expected_columns`` defaults to the full published contract. A caller that deliberately
    reads a **projection** of the file — which is what the CLI does, because a full year does not
    fit comfortably in memory — passes the projection instead, so that ``missing_columns`` keeps
    meaning "absent from what I asked for" rather than listing twenty-six columns the run never
    requested. ``columns_requested`` records which case applied.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")

    expected = set(COLUMNS) if expected_columns is None else set(expected_columns)

    report = DataQualityReport(
        n_rows=_as_int(df.shape[0], context="row count"),
        n_columns=_as_int(df.shape[1], context="column count"),
        columns_requested=len(expected),
    )

    present = set(df.columns)
    report.missing_columns = sorted(expected - present)
    report.unexpected_columns = sorted(present - expected)

    report.null_counts = _null_counts(df)
    report.duplicate_rows = _as_int(df.duplicated().sum(), context="duplicate rows")

    for column in VOLUME_COLUMNS:
        if column not in present:
            continue
        negatives = _count_true(_numeric(df[column]) < 0)
        if negatives:
            report.negative_volume_counts[column] = negatives

    if {"prod_pet", "prod_gas"}.issubset(present):
        oil = _numeric(df["prod_pet"]).fillna(0.0)
        gas = _numeric(df["prod_gas"]).fillna(0.0)
        report.zero_oil_and_gas_rows = _count_true((oil <= 0) & (gas <= 0))

    if "areayacimiento" in present:
        report.placeholder_field_rows = _count_true(
            df["areayacimiento"].isin(PLACEHOLDER_YACIMIENTOS)
        )

    if "provincia" in present:
        report.administrative_province_rows = _count_true(df["provincia"] == EXCLUDED_PROVINCIA)

    if check_gor and {"prod_gas", "prod_pet"}.issubset(present):
        ratio = gas_oil_ratio(df["prod_gas"], df["prod_pet"]).dropna()
        if not ratio.empty:
            report.implausible_gor_rows = _count_true(~is_plausible_gor(ratio))

    return report


def production_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column descriptive statistics for the produced fluids, units made explicit.

    The returned frame keeps ``*_published`` units next to ``*_m3`` equivalents, so a reader
    can never mistake a published gas figure for cubic metres. Columns already published in
    m³ are not duplicated — the duplication is only where a conversion actually happens.
    """
    missing = [column for column in PRODUCTION_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"missing production columns: {missing}")

    rows: dict[str, dict[str, Any]] = {}
    for column in PRODUCTION_COLUMNS:
        spec = COLUMNS[column]
        values = _numeric(df[column])
        factor = spec.m3_per_unit if spec.m3_per_unit is not None else 1.0

        rows[f"{column}_published"] = _column_stats(values, column=column, unit=spec.unit)
        if factor != 1.0:
            rows[f"{column}_m3"] = _column_stats(values * factor, column=column, unit="m3")

    return pd.DataFrame(rows).T
