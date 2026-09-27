"""wellprod — production analytics over Argentina's official open well dataset.

Source of the data
------------------
Secretaría de Energía de la Nación, dataset *"Producción de petróleo y gas por pozo
(Capítulo IV)"*, published on <https://datos.gob.ar> under **CC-BY-4.0**.

The dataset is per well and per month: one row is one producing well in one month,
with oil, gas and water volumes, injection volumes, and geological descriptors
(basin, formation, province, resource type).

Design rule of this package
---------------------------
Units are declared **once**, in :mod:`wellprod.schema`, attached to the column that
carries them. No other module is allowed to hardcode a conversion factor.
"""

from wellprod.schema import (
    COLUMNS,
    INJECTION_COLUMNS,
    PRODUCTION_COLUMNS,
    gas_oil_ratio,
)

__all__ = [
    "COLUMNS",
    "INJECTION_COLUMNS",
    "PRODUCTION_COLUMNS",
    "gas_oil_ratio",
]
__version__ = "0.1.0"
