"""Tests for the multi-year series.

No network and no data files: the frames are built to be small enough to check by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wellprod.series import (
    SERIES_METRICS,
    annual_totals,
    common_months,
    coverage_table,
    growth_markdown,
    month_coverage,
    monthly_series,
    period_index,
    to_markdown,
    year_over_year,
)


def year_frame(year: int, oil: float, *, months: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Two rows per year, alternating resource type, so shares are easy to verify by hand.

    Gas is written in the published unit (thousands of m³) at one per cent of oil, which makes the
    gas-oil ratio exactly 10 m³/m³ for every row.
    """
    rows = []
    for index, month in enumerate(months):
        rows.append(
            {
                "anio": year,
                "mes": month,
                "idpozo": 1000 + index,
                "prod_pet": oil,
                "prod_gas": oil / 100,
                "prod_agua": oil / 10,
                "tipo_de_recurso": "CONVENCIONAL" if index % 2 == 0 else "NO CONVENCIONAL",
            }
        )
    return pd.DataFrame(rows)


def three_years() -> list[tuple[int, pd.DataFrame]]:
    return [
        (2024, year_frame(2024, 100.0)),
        (2025, year_frame(2025, 120.0)),
        (2026, year_frame(2026, 150.0)),
    ]


# ---------------------------------------------------------------------------------------
# Coverage. This is the whole point of the module.
# ---------------------------------------------------------------------------------------


def test_month_coverage_is_sorted_and_deduplicated() -> None:
    frame = pd.concat([year_frame(2026, 10.0), year_frame(2026, 10.0)])

    assert month_coverage(frame) == [1, 2]


def test_common_months_intersects_rather_than_unions() -> None:
    """A year that stops in February must shrink the window for everyone, not be dropped."""
    assert common_months([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3]]) == [1, 2, 3]


def test_common_months_of_nothing_is_empty() -> None:
    assert common_months([]) == []


def test_common_months_of_disjoint_years_is_empty() -> None:
    assert common_months([[1, 2], [3, 4]]) == []


def test_coverage_table_flags_an_incomplete_year() -> None:
    table = coverage_table({2024: range(1, 13), 2026: range(1, 9)})

    assert "yes" in table
    assert "**no — 8 of 12**" in table


# ---------------------------------------------------------------------------------------
# Periods and monthly series.
# ---------------------------------------------------------------------------------------


def test_period_index_is_monthly() -> None:
    periods = period_index(year_frame(2026, 10.0, months=(3, 4)))

    assert str(periods.dtype) == "period[M]"
    assert [str(period) for period in periods] == ["2026-03", "2026-04"]


def test_period_index_requires_both_columns() -> None:
    with pytest.raises(KeyError, match="anio"):
        period_index(pd.DataFrame({"mes": [1]}))


def test_monthly_series_returns_one_column_without_a_dimension() -> None:
    series = monthly_series(year_frame(2026, 100.0))

    assert "oil_m3" in series.columns
    assert len(series) == 2
    assert series["oil_m3"].sum() == pytest.approx(200.0)


def test_monthly_series_splits_by_dimension() -> None:
    series = monthly_series(year_frame(2026, 100.0), dimension="tipo_de_recurso")

    assert set(series.columns) == {"CONVENCIONAL", "NO CONVENCIONAL"}
    assert series["CONVENCIONAL"].sum() == pytest.approx(100.0)


def test_monthly_series_reports_gas_in_cubic_metres() -> None:
    """The unit trap, one layer up: summing the published column would understate gas by 1000."""
    series = monthly_series(year_frame(2026, 100.0), metric="gas_m3")

    assert series["gas_m3"].sum() == pytest.approx(2_000.0)


def test_monthly_series_rejects_an_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric 'barrels'"):
        monthly_series(year_frame(2026, 10.0), metric="barrels")


def test_monthly_series_rejects_a_missing_dimension_column() -> None:
    with pytest.raises(KeyError, match="cuenca"):
        monthly_series(year_frame(2026, 10.0), dimension="cuenca")


# ---------------------------------------------------------------------------------------
# Annual totals and the window.
# ---------------------------------------------------------------------------------------


def test_annual_totals_has_one_row_per_year() -> None:
    totals = annual_totals(three_years())

    assert list(totals.index) == [2024, 2025, 2026]
    assert totals["oil_m3"].iloc[0] == pytest.approx(200.0)


def test_annual_totals_restricts_to_the_given_months() -> None:
    """Month 2 excluded, so each year contributes only its January row."""
    totals = annual_totals(three_years(), months=[1])

    assert totals["oil_m3"].tolist() == pytest.approx([100.0, 120.0, 150.0])


def test_annual_totals_accepts_a_generator_so_years_can_be_streamed() -> None:
    def streamed():
        yield from three_years()

    totals = annual_totals(streamed())

    assert len(totals) == 3


def test_annual_totals_splits_by_dimension() -> None:
    totals = annual_totals(three_years(), dimension="tipo_de_recurso")

    assert "CONVENCIONAL" in totals.columns
    assert "NO CONVENCIONAL" in totals.columns
    assert totals["CONVENCIONAL"].iloc[0] == pytest.approx(100.0)


def test_annual_totals_of_wells_counts_distinct_wells_not_rows() -> None:
    totals = annual_totals(three_years(), metric="wells")

    assert totals["wells"].iloc[0] == 2


def test_annual_totals_of_a_ratio_uses_the_median_not_the_sum() -> None:
    totals = annual_totals(three_years(), metric="gor_m3_m3")

    assert totals["gor_m3_m3"].iloc[0] == pytest.approx(10.0)


def test_every_declared_metric_is_aggregatable() -> None:
    for metric in SERIES_METRICS:
        totals = annual_totals(three_years(), metric=metric)

        assert len(totals) == 3, metric


# ---------------------------------------------------------------------------------------
# Growth.
# ---------------------------------------------------------------------------------------


def test_year_over_year_leaves_the_first_year_unknown() -> None:
    """Not zero: a first year with no predecessor is unknown, not flat."""
    growth = year_over_year(annual_totals(three_years()))

    assert pd.isna(growth["oil_m3"].iloc[0])
    assert growth["oil_m3"].iloc[1] == pytest.approx(20.0)
    assert growth["oil_m3"].iloc[2] == pytest.approx(25.0)


def test_growth_markdown_renders_signs_and_blanks() -> None:
    text = growth_markdown(year_over_year(annual_totals(three_years())))

    assert "+20.0%" in text
    assert "+25.0%" in text
    assert "—" in text


# ---------------------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------------------


def test_to_markdown_states_the_window() -> None:
    text = to_markdown(annual_totals(three_years(), months=[1, 2]), metric="oil_m3", months=[1, 2])

    assert "| Year |" in text
    assert "window: months 1, 2" in text
    assert "2024" in text


def test_to_markdown_scales_large_numbers_so_they_stay_readable() -> None:
    big = [(2024, year_frame(2024, 5_000_000.0, months=(1,)))]

    text = to_markdown(annual_totals(big), metric="oil_m3")

    assert "M" in text


def test_to_markdown_renders_water_cut_as_a_percentage() -> None:
    text = to_markdown(annual_totals(three_years(), metric="water_cut"), metric="water_cut")

    assert "%" in text


def test_to_markdown_renders_well_counts_without_decimals() -> None:
    text = to_markdown(annual_totals(three_years(), metric="wells"), metric="wells")

    assert "| 2024 | 2 |" in text


def test_to_markdown_rejects_an_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        to_markdown(annual_totals(three_years()), metric="degrees")
