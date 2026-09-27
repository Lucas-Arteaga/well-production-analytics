"""Aggregation of production by any geological or administrative dimension.

Every volume reported here is in **cubic metres**, converted through the factors declared in
:mod:`wellprod.schema`. Gas is published in thousands of m³, so a profile that summed the
published column would understate gas by a factor of 1000 — the exact failure mode the schema
module exists to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from wellprod.convert import as_float, as_int
from wellprod.schema import COLUMNS, PRODUCTION_COLUMNS, gas_oil_ratio

#: Dimensions the dataset exposes that are meaningful to aggregate over.
PROFILE_DIMENSIONS: Final[tuple[str, ...]] = (
    "cuenca",
    "empresa",
    "provincia",
    "formacion",
    "tipo_de_recurso",
    "clasificacion",
    "subclasificacion",
    "tipopozo",
    "tipoextraccion",
)


def _m3(frame: pd.DataFrame, column: str) -> pd.Series:
    """A production column expressed in cubic metres, using the schema's factor."""
    spec = COLUMNS[column]
    factor = spec.m3_per_unit if spec.m3_per_unit is not None else 1.0
    return pd.Series(pd.to_numeric(frame[column], errors="coerce"), dtype="float64") * factor


def metrics_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The standard production metrics for a frame, all volumes in cubic metres.

    Defined once and shared by the profiler and the time series, so that a metric can never
    mean two different things in two reports. ``water_cut`` and ``gor_m3_m3`` are ``NaN`` where
    they are undefined rather than zero, because a well that produced nothing has neither.
    """
    for required in PRODUCTION_COLUMNS:
        if required not in frame.columns:
            raise KeyError(f"missing column {required!r}")

    return pd.DataFrame(
        {
            "prod_pet_m3": _m3(frame, "prod_pet"),
            "prod_gas_m3": _m3(frame, "prod_gas"),
            "prod_agua_m3": _m3(frame, "prod_agua"),
            "gor_m3_m3": gas_oil_ratio(frame["prod_gas"], frame["prod_pet"]),
            "water_cut": water_cut(frame),
        },
        index=frame.index,
    )


def water_cut(frame: pd.DataFrame) -> pd.Series:
    """Water cut as a fraction of total liquid, ``NaN`` where there is no liquid at all.

    A well that produced nothing has no water cut. Returning zero would silently pull every
    average down.
    """
    water = _m3(frame, "prod_agua")
    oil = _m3(frame, "prod_pet")
    liquid = water + oil
    return (water / liquid).where(liquid > 0, other=np.nan)


def profile_by(
    frame: pd.DataFrame,
    dimension: str,
    *,
    top: int | None = None,
    min_wells: int = 1,
) -> pd.DataFrame:
    """Aggregate production by ``dimension``, sorted by oil produced.

    Returns one row per dimension value with:

    ``n_rows``
        Well-month records.
    ``n_wells``
        Distinct wells.
    ``prod_pet_m3``, ``prod_gas_m3``, ``prod_agua_m3``
        Totals in cubic metres. Gas is converted from the published thousands of m³.
    ``gor_median_m3_m3``
        Median gas-oil ratio. The median rather than the mean: a handful of gas wells produce
        ratios four orders of magnitude above the rest, and a mean would describe those wells
        rather than the basin.
    ``water_cut``
        Mean water cut over records with liquid production.
    ``share_of_oil``
        This row's share of total oil in the frame.
    """
    if dimension not in PROFILE_DIMENSIONS:
        raise ValueError(
            f"unknown dimension {dimension!r}; expected one of {', '.join(PROFILE_DIMENSIONS)}"
        )
    if dimension not in frame.columns:
        raise KeyError(f"missing column {dimension!r}")

    working = metrics_frame(frame)
    working["_dim"] = frame[dimension].astype("string")
    working["_well"] = frame["idpozo"] if "idpozo" in frame.columns else pd.NA

    grouped = working.groupby("_dim", dropna=False)
    result = pd.DataFrame(
        {
            "n_rows": grouped.size(),
            "n_wells": grouped["_well"].nunique(dropna=True),
            "prod_pet_m3": grouped["prod_pet_m3"].sum(),
            "prod_gas_m3": grouped["prod_gas_m3"].sum(),
            "prod_agua_m3": grouped["prod_agua_m3"].sum(),
            "gor_median_m3_m3": grouped["gor_m3_m3"].median(),
            "water_cut": grouped["water_cut"].mean(),
        }
    )

    total_oil = as_float(result["prod_pet_m3"].sum(), context="total oil")
    total_gas = as_float(result["prod_gas_m3"].sum(), context="total gas")
    result["share_of_oil"] = result["prod_pet_m3"] / total_oil if total_oil else np.nan
    result["share_of_gas"] = result["prod_gas_m3"] / total_gas if total_gas else np.nan

    result = result[result["n_wells"] >= min_wells]
    result = result.sort_values("prod_pet_m3", ascending=False)
    if top is not None:
        result = result.head(top)
    return result


def profile_many(
    frame: pd.DataFrame,
    dimensions: Sequence[str],
    *,
    top: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Profile several dimensions at once, skipping any the frame does not carry."""
    return {
        dimension: profile_by(frame, dimension, top=top)
        for dimension in dimensions
        if dimension in frame.columns
    }


def to_markdown(profile: pd.DataFrame, *, dimension: str, floats: int = 2) -> str:
    """Render a profile as a Markdown table, with units in the headers.

    Volumes are formatted with thousands separators because the interesting numbers here are in
    the millions, where a missing separator turns into a misread.
    """
    if profile.empty:
        return "_no rows_"

    header = {
        "n_rows": "Records",
        "n_wells": "Wells",
        "prod_pet_m3": "Oil (m³)",
        "prod_gas_m3": "Gas (m³)",
        "prod_agua_m3": "Water (m³)",
        "gor_median_m3_m3": "GOR median (m³/m³)",
        "water_cut": "Water cut",
        "share_of_oil": "Share of oil",
        "share_of_gas": "Share of gas",
    }
    columns = [name for name in header if name in profile.columns]
    lines = [
        f"| {dimension} | " + " | ".join(header[name] for name in columns) + " |",
        "| :--- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for value, row in profile.iterrows():
        cells = []
        for name in columns:
            cell = row[name]
            available = pd.notna(cell)
            if name in {"share_of_oil", "share_of_gas", "water_cut"}:
                cells.append(f"{as_float(cell, context=name):.1%}" if available else "—")
            elif name in {"n_rows", "n_wells"}:
                cells.append(f"{as_int(cell, context=name):,}")
            elif name == "gor_median_m3_m3":
                cells.append(f"{as_float(cell, context=name):,.1f}" if available else "—")
            else:
                cells.append(f"{as_float(cell, context=name):,.{floats}f}")
        lines.append(f"| {value} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
