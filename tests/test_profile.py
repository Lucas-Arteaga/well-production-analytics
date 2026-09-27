"""Tests for the production profile aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wellprod.convert import as_float
from wellprod.profile import PROFILE_DIMENSIONS, profile_by, to_markdown, water_cut


def frame() -> pd.DataFrame:
    """Two basins, one dry well, and one gas well with a ratio far above the rest.

    ``idpozo`` 4 doubles as the case that makes the median the right statistic:
    its gas-oil ratio is four orders of magnitude above everything else.
    """
    return pd.DataFrame(
        {
            "cuenca": ["NEUQUINA", "NEUQUINA", "GOLFO SAN JORGE", "NEUQUINA"],
            "empresa": ["YPF", "YPF", "PAE", "YPF"],
            "idpozo": [1, 2, 3, 4],
            "prod_pet": [100.0, 200.0, 0.0, 1.0],
            "prod_gas": [5.0, 10.0, 0.0, 500.0],
            "prod_agua": [50.0, 0.0, 0.0, 0.0],
        }
    )


# ---------------------------------------------------------------------------------------
# The unit that everything else depends on.
# ---------------------------------------------------------------------------------------


def test_gas_is_aggregated_in_cubic_metres_not_published_units() -> None:
    """15 published gas units are 15 000 m3, not 15.

    Summing the published column would understate gas by a factor of 1000 — the same class of
    mistake the schema module exists to prevent, one layer up.
    """
    result = profile_by(frame(), "cuenca")

    assert result.at["NEUQUINA", "prod_gas_m3"] == pytest.approx((5.0 + 10.0 + 500.0) * 1_000)
    assert result.at["NEUQUINA", "prod_pet_m3"] == pytest.approx(301.0)


# ---------------------------------------------------------------------------------------
# Statistics that a mean would get wrong.
# ---------------------------------------------------------------------------------------


def test_gor_is_the_median_so_one_gas_well_does_not_describe_the_basin() -> None:
    result = profile_by(frame(), "cuenca")

    ratios = [5_000 / 100, 10_000 / 200, 500_000 / 1]
    # `DataFrame.at` is typed as a broad Scalar union, so the package's own boundary helper is
    # what narrows it — the same helper the reports use, which is rather the point.
    median = as_float(result.at["NEUQUINA", "gor_median_m3_m3"], context="median gas-oil ratio")

    assert median == pytest.approx(float(np.median(ratios)))
    assert median < float(np.mean(ratios)) / 10


def test_water_cut_is_nan_when_there_is_no_liquid() -> None:
    """A well that produced nothing has no water cut; zero would drag every average down."""
    cut = water_cut(frame())

    assert cut.iloc[0] == pytest.approx(50 / 150)
    assert cut.iloc[1] == 0.0
    assert np.isnan(cut.iloc[2])


def test_shares_sum_to_one_within_the_frame() -> None:
    result = profile_by(frame(), "cuenca")

    assert result["share_of_oil"].sum() == pytest.approx(1.0)
    assert result["share_of_gas"].sum() == pytest.approx(1.0)


def test_shares_are_zero_safe_when_nothing_was_produced() -> None:
    silent = frame()
    silent["prod_pet"] = 0.0
    silent["prod_gas"] = 0.0

    result = profile_by(silent, "cuenca")

    assert result["share_of_oil"].isna().all()


# ---------------------------------------------------------------------------------------
# Shape, ordering and filters.
# ---------------------------------------------------------------------------------------


def test_sorted_by_oil_produced_descending() -> None:
    result = profile_by(frame(), "cuenca")

    assert list(result.index) == ["NEUQUINA", "GOLFO SAN JORGE"]
    assert result["prod_pet_m3"].is_monotonic_decreasing


def test_top_truncates_after_sorting() -> None:
    result = profile_by(frame(), "empresa", top=1)

    assert list(result.index) == ["YPF"]


def test_min_wells_filters_thin_dimension_values() -> None:
    result = profile_by(frame(), "empresa", min_wells=2)

    assert list(result.index) == ["YPF"]


def test_counts_records_and_distinct_wells() -> None:
    result = profile_by(frame(), "cuenca")

    assert result.loc["NEUQUINA", "n_rows"] == 3
    assert result.loc["NEUQUINA", "n_wells"] == 3


def test_every_declared_dimension_can_be_profiled() -> None:
    wide = frame()
    for dimension in PROFILE_DIMENSIONS:
        wide[dimension] = "value"

    for dimension in PROFILE_DIMENSIONS:
        assert not profile_by(wide, dimension).empty


# ---------------------------------------------------------------------------------------
# Refusing bad input.
# ---------------------------------------------------------------------------------------


def test_unknown_dimension_is_rejected_with_the_valid_list() -> None:
    with pytest.raises(ValueError, match="unknown dimension 'yacimiento'"):
        profile_by(frame(), "yacimiento")


def test_missing_column_is_named() -> None:
    with pytest.raises(KeyError, match="prod_gas"):
        profile_by(frame().drop(columns=["prod_gas"]), "cuenca")


def test_missing_dimension_column_is_named() -> None:
    with pytest.raises(KeyError, match="formacion"):
        profile_by(frame(), "formacion")


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------


def test_markdown_has_headers_with_units_and_a_row_per_value() -> None:
    markdown = to_markdown(profile_by(frame(), "cuenca"), dimension="cuenca")

    assert "cuenca" in markdown.splitlines()[0]
    assert "Oil (m³)" in markdown
    assert "GOR median (m³/m³)" in markdown
    assert markdown.count("NEUQUINA") == 1
    assert markdown.count("GOLFO SAN JORGE") == 1


def test_markdown_renders_missing_values_as_a_dash() -> None:
    markdown = to_markdown(profile_by(frame(), "cuenca"), dimension="cuenca")

    assert "—" in markdown


def test_markdown_of_an_empty_profile_says_so() -> None:
    assert to_markdown(pd.DataFrame(), dimension="cuenca") == "_no rows_"
