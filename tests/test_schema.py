"""Tests for the column contract, the unit conversions and the gas-oil ratio."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
import pytest

from wellprod.schema import (
    COLUMNS,
    GAS_M3_PER_UNIT,
    INJECTION_COLUMNS,
    PLAUSIBLE_GOR_M3_M3,
    PRODUCTION_COLUMNS,
    VOLUME_COLUMNS,
    gas_oil_ratio,
    is_plausible_gor,
)


def series_ratio(gas: Sequence[float], oil: Sequence[float]) -> pd.Series:
    """``gas_oil_ratio`` through the Series overload, with the return type spelled out.

    Declaring the return type here keeps the tests readable and keeps them independent of
    overload inference, which needs the third-party stubs to be resolvable by whatever
    checker happens to be looking at the file.
    """
    return gas_oil_ratio(pd.Series(gas, dtype="float64"), pd.Series(oil, dtype="float64"))


def series_plausible(ratios: Sequence[float]) -> pd.Series:
    """``is_plausible_gor`` through the element-wise overload."""
    return is_plausible_gor(pd.Series(ratios, dtype="float64"))


# ---------------------------------------------------------------------------------------
# The regression test that justifies this module.
# ---------------------------------------------------------------------------------------


def test_gas_oil_ratio_uses_thousands_not_millions() -> None:
    """100 m3 of oil and '5 Mm3' of gas is a GOR of 50 m3/m3 — not 50 000.

    In this dataset ``Mm3`` means *miles de m3* (thousands). Treating it as millions inflates
    every gas-oil ratio by exactly 1000 and produces a number that still looks plausible to a
    reader who is not checking units.
    """
    ratio = series_ratio([5.0], [100.0]).iloc[0]

    assert GAS_M3_PER_UNIT == 1_000.0
    assert ratio == pytest.approx(50.0)
    # The exact shape of the mistake this module exists to prevent.
    assert ratio != pytest.approx(50_000.0)


def test_gas_oil_ratio_is_scale_invariant_in_the_expected_way() -> None:
    """Doubling both fluids leaves the ratio unchanged; doubling only gas doubles it."""
    base = series_ratio([10.0], [200.0]).iloc[0]
    both = series_ratio([20.0], [400.0]).iloc[0]
    gas_only = series_ratio([20.0], [200.0]).iloc[0]

    assert both == pytest.approx(base)
    assert gas_only == pytest.approx(base * 2)


# ---------------------------------------------------------------------------------------
# Refusing to invent a number.
# ---------------------------------------------------------------------------------------


def test_gas_oil_ratio_is_nan_when_oil_is_zero() -> None:
    ratio = series_ratio([5.0, 5.0], [0.0, 100.0])

    assert math.isnan(ratio.iloc[0])
    assert ratio.iloc[1] == pytest.approx(50.0)


def test_gas_oil_ratio_is_nan_for_negative_volumes() -> None:
    ratio = series_ratio([-1.0, 5.0], [100.0, 100.0])

    assert math.isnan(ratio.iloc[0])


def test_gas_oil_ratio_is_nan_for_missing_values() -> None:
    gas = pd.Series([None, 5.0], dtype="float64")
    oil = pd.Series([100.0, None], dtype="float64")

    result: pd.Series = gas_oil_ratio(gas, oil)

    assert result.isna().all()


def test_gas_oil_ratio_keeps_the_series_index() -> None:
    index = pd.Index([101, 202, 303], name="idpozo")
    gas = pd.Series([5.0, 6.0, 7.0], index=index, dtype="float64")
    oil = pd.Series([100.0] * 3, index=index, dtype="float64")

    result: pd.Series = gas_oil_ratio(gas, oil)

    assert result.index.equals(index)


def test_gas_oil_ratio_accepts_scalars_and_returns_a_float() -> None:
    ratio = gas_oil_ratio(5.0, 100.0)

    assert isinstance(ratio, float)
    assert ratio == pytest.approx(50.0)


def test_gas_oil_ratio_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        series_ratio([1.0, 2.0], [1.0])


def test_gas_oil_ratio_rejects_a_non_numeric_scalar() -> None:
    with pytest.raises(TypeError, match="expected a number"):
        gas_oil_ratio("not a volume", 100.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# Plausibility band.
# ---------------------------------------------------------------------------------------


def test_is_plausible_gor_respects_the_declared_band() -> None:
    low, high = PLAUSIBLE_GOR_M3_M3

    assert is_plausible_gor(low)
    assert is_plausible_gor(high)
    assert not is_plausible_gor(0.0)
    assert not is_plausible_gor(high * 10)
    assert not is_plausible_gor(float("nan"))


def test_is_plausible_gor_works_element_wise() -> None:
    result = series_plausible([50.0, 0.0, 1e9])

    assert list(result) == [True, False, False]


# ---------------------------------------------------------------------------------------
# The contract itself.
# ---------------------------------------------------------------------------------------


def test_contract_declares_every_published_column() -> None:
    """The official CSV publishes 38 columns; the contract must not drift from it."""
    assert len(COLUMNS) == 38


def test_every_volume_column_declares_a_conversion_factor() -> None:
    """A volume column without a unit is a bug waiting to happen."""
    for name in VOLUME_COLUMNS:
        spec = COLUMNS[name]
        assert spec.unit, f"{name} has no unit"
        assert spec.m3_per_unit is not None, f"{name} has no conversion factor"
        assert spec.m3_per_unit > 0


def test_gas_columns_convert_by_a_thousand_and_liquids_do_not() -> None:
    assert COLUMNS["prod_gas"].m3_per_unit == 1_000.0
    assert COLUMNS["iny_gas"].m3_per_unit == 1_000.0
    assert COLUMNS["iny_co2"].m3_per_unit == 1_000.0

    for liquid in ("prod_pet", "prod_agua", "iny_agua", "iny_otro"):
        assert COLUMNS[liquid].m3_per_unit == 1.0


def test_production_and_injection_groups_match_the_contract() -> None:
    assert PRODUCTION_COLUMNS == ("prod_pet", "prod_gas", "prod_agua")
    assert INJECTION_COLUMNS == ("iny_agua", "iny_gas", "iny_co2", "iny_otro")
    for name in PRODUCTION_COLUMNS + INJECTION_COLUMNS:
        assert COLUMNS[name].kind == "volume"


def test_only_sub_tipo_recurso_is_optional() -> None:
    """If a new column becomes optional, that is a decision worth failing a test over."""
    optional = {name for name, spec in COLUMNS.items() if not spec.required}

    assert optional == {"sub_tipo_recurso"}
