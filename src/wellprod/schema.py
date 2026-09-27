"""The column contract for the official Argentine per-well production dataset.

Why this module exists
----------------------
The single most expensive mistake in production data is a **unit error**, because the
wrong answer still looks like a plausible number.

In this dataset the gas column is labelled ``prod_gas`` and documented as ``Mm3``. The
published values are **miles de metros cúbicos** — *thousands* of cubic metres, not
millions. A conversion of ``1e6`` instead of ``1e3`` inflates every gas-oil ratio by a
factor of 1000, and 98 700 m³/m³ looks like a number someone might believe.

So: every production column declares its unit and its conversion factor here, once.
Nothing downstream is allowed to hardcode it, and :func:`gas_oil_ratio` is the only
sanctioned way to compute a ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast, overload

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Conversion factors. m3 per published unit. These are the only numbers that matter.
# --------------------------------------------------------------------------------------

#: ``Mm3`` in this dataset means *miles de m3* (thousands), so one published unit is 1 000 m3.
#: Using 1e6 here is the classic error and it is off by three orders of magnitude.
GAS_M3_PER_UNIT: Final[float] = 1_000.0

#: Oil and water are already published in m3.
OIL_M3_PER_UNIT: Final[float] = 1.0
WATER_M3_PER_UNIT: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class Column:
    """One column of the published dataset, with its unit and its meaning."""

    name: str
    kind: str  # volume | descriptor | identifier | date | flag
    unit: str | None = None
    m3_per_unit: float | None = None
    description: str = ""
    required: bool = True


def _c(
    name: str,
    kind: str,
    description: str,
    unit: str | None = None,
    m3_per_unit: float | None = None,
    required: bool = True,
) -> tuple[str, Column]:
    return name, Column(name, kind, unit, m3_per_unit, description, required)


#: The full published schema, in the order the official CSV presents it.
COLUMNS: Final[dict[str, Column]] = dict(
    [
        _c("idempresa", "identifier", "Operator company code"),
        _c("anio", "descriptor", "Production year"),
        _c("mes", "descriptor", "Production month, 1-12"),
        _c("idpozo", "identifier", "Well identifier (stable across the dataset)"),
        _c("prod_pet", "volume", "Monthly oil production", "m3", OIL_M3_PER_UNIT),
        _c(
            "prod_gas",
            "volume",
            "Monthly gas production. Published as Mm3 meaning MILES de m3, not millions.",
            "Mm3 (thousands of m3)",
            GAS_M3_PER_UNIT,
        ),
        _c("prod_agua", "volume", "Monthly water production", "m3", WATER_M3_PER_UNIT),
        _c("iny_agua", "volume", "Monthly water injection", "m3", WATER_M3_PER_UNIT),
        _c("iny_gas", "volume", "Monthly gas injection", "Mm3 (thousands of m3)", GAS_M3_PER_UNIT),
        _c("iny_co2", "volume", "Monthly CO2 injection", "Mm3 (thousands of m3)", GAS_M3_PER_UNIT),
        _c("iny_otro", "volume", "Monthly other-fluid injection", "m3", WATER_M3_PER_UNIT),
        _c("tef", "descriptor", "Effective operating time (días efectivos de funcionamiento)"),
        _c("vida_util", "descriptor", "Useful life"),
        _c("tipoextraccion", "descriptor", "Lift / extraction method"),
        _c("tipoestado", "descriptor", "Well status (Extracción Efectiva, Abandonado, ...)"),
        _c("tipopozo", "descriptor", "Well type (Petrolífero, Gasífero, ...)"),
        _c("observaciones", "descriptor", "Free-text remarks"),
        _c("fechaingreso", "date", "Record ingestion timestamp"),
        _c("rectificado", "flag", "Record was amended"),
        _c("habilitado", "flag", "Record is enabled"),
        _c("idusuario", "identifier", "Reporting user id"),
        _c("empresa", "descriptor", "Operator company name"),
        _c("sigla", "identifier", "Well short name"),
        _c("formprod", "descriptor", "Producing formation code"),
        _c("profundidad", "descriptor", "Well depth", "m"),
        _c("formacion", "descriptor", "Geological formation"),
        _c("idareapermisoconcesion", "identifier", "Concession area id"),
        _c("areapermisoconcesion", "descriptor", "Concession area name"),
        _c("idareayacimiento", "identifier", "Field (yacimiento) id"),
        _c("areayacimiento", "descriptor", "Field (yacimiento) name"),
        _c("cuenca", "descriptor", "Basin"),
        _c("provincia", "descriptor", "Province"),
        _c("tipo_de_recurso", "descriptor", "Resource type (CONVENCIONAL / NO CONVENCIONAL)"),
        _c("proyecto", "descriptor", "Project"),
        _c("clasificacion", "descriptor", "Classification (EXPLOTACION, EXPLORACION, ...)"),
        _c("subclasificacion", "descriptor", "Sub-classification (DESARROLLO, ...)"),
        _c("sub_tipo_recurso", "descriptor", "Resource sub-type", required=False),
        _c("fecha_data", "date", "Data date for the row"),
    ]
)

#: Columns that carry a volume and therefore must never be negative.
VOLUME_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name, col in COLUMNS.items() if col.kind == "volume"
)

#: The three produced fluids.
PRODUCTION_COLUMNS: Final[tuple[str, ...]] = ("prod_pet", "prod_gas", "prod_agua")

#: The four injected fluids.
INJECTION_COLUMNS: Final[tuple[str, ...]] = ("iny_agua", "iny_gas", "iny_co2", "iny_otro")

#: Statuses that represent real production. Anything else is not a producing record.
VALID_TIPOESTADO: Final[frozenset[str]] = frozenset({"Extracción Efectiva"})

#: Resource types the dataset distinguishes.
VALID_TIPO_RECURSO: Final[frozenset[str]] = frozenset({"CONVENCIONAL", "NO CONVENCIONAL"})

#: Well types that report hydrocarbon volumes.
VALID_TIPOPOZO: Final[frozenset[str]] = frozenset({"Petrolífero", "Gasífero"})

#: Placeholder field names that carry no geological information.
PLACEHOLDER_YACIMIENTOS: Final[frozenset[str]] = frozenset(
    {"POZOS SIN YACIMIENTO", "SIN NOMBRE", "PUESTO SIN NOMBRE"}
)

#: Rows with this province are administrative, not geological.
EXCLUDED_PROVINCIA: Final[str] = "Estado Nacional"

#: A gas-oil ratio in m3/m3. Below the low bound the record is an artefact; above the high
#: bound it is almost always a unit error or a gas well with negligible oil.
#: Reference points: conventional black oil sits in the tens to low hundreds; a dry gas
#: well can legitimately exceed 100 000 m3/m3.
PLAUSIBLE_GOR_M3_M3: Final[tuple[float, float]] = (1.0, 500_000.0)


def _as_series(value: float | pd.Series) -> tuple[pd.Series, bool]:
    """Accept a scalar or a Series and always work on a Series internally."""
    if isinstance(value, pd.Series):
        return value.astype("float64"), False
    try:
        return pd.Series([float(value)], dtype="float64"), True
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"expected a number or a pandas Series, got {type(value).__name__}: {value!r}"
        ) from exc


@overload
def gas_oil_ratio(prod_gas: float, prod_pet: float) -> float:
    """Scalar in, scalar out."""
    ...


@overload
def gas_oil_ratio(prod_gas: pd.Series, prod_pet: pd.Series) -> pd.Series:
    """Series in, Series out."""
    ...


def gas_oil_ratio(prod_gas: float | pd.Series, prod_pet: float | pd.Series) -> float | pd.Series:
    """Gas-oil ratio in m3/m3, computed with the units of *this* dataset.

    ``prod_gas`` is taken in the published unit (thousands of m3) and converted with
    :data:`GAS_M3_PER_UNIT`. ``prod_pet`` is already in m3.

    Returns ``NaN`` wherever the ratio is not defined or not physically meaningful:

    * oil production <= 0 — a division by zero is not a ratio,
    * gas production < 0 — a negative volume is not physical,
    * either input missing.

    A ratio computed from oil produced by a well that is not actually producing is noise,
    not data. Filtering belongs upstream; this function refuses to invent a number.
    """
    gas, gas_scalar = _as_series(prod_gas)
    oil, oil_scalar = _as_series(prod_pet)

    if len(gas) != len(oil):
        raise ValueError(f"length mismatch: prod_gas has {len(gas)}, prod_pet has {len(oil)}")

    gas_m3 = gas * GAS_M3_PER_UNIT

    invalid = (oil <= 0) | (gas < 0) | oil.isna() | gas.isna()
    ratio = (gas_m3 / oil).astype("float64")
    ratio = ratio.where(~invalid, other=np.nan)

    if gas_scalar and oil_scalar:
        return cast(float, ratio.iloc[0])
    return ratio


@overload
def is_plausible_gor(ratio: float) -> bool:
    """Scalar in, scalar out."""
    ...


@overload
def is_plausible_gor(ratio: pd.Series) -> pd.Series:
    """Series in, element-wise Series out."""
    ...


def is_plausible_gor(ratio: float | pd.Series) -> bool | pd.Series:
    """Whether a gas-oil ratio falls inside :data:`PLAUSIBLE_GOR_M3_M3`."""
    low, high = PLAUSIBLE_GOR_M3_M3
    series, scalar = _as_series(ratio)
    inside = series.between(low, high, inclusive="both")
    if scalar:
        return cast(bool, inside.iloc[0])
    return inside
