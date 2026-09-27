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

Typical use
-----------
::

    wellprod ingest  --year 2026 --limit 200000
    wellprod quality --year 2026 --out reports
    wellprod profile --year 2026 --by cuenca --top 10
    wellprod report  --year 2026 --out reports
"""

from wellprod.ingest import IngestManifest, ingest_year, load_years, resolve_source
from wellprod.profile import profile_by
from wellprod.quality import DataQualityReport, build_report, production_summary
from wellprod.schema import (
    COLUMNS,
    GAS_M3_PER_UNIT,
    INJECTION_COLUMNS,
    PRODUCTION_COLUMNS,
    gas_oil_ratio,
)
from wellprod.series import (
    SERIES_METRICS,
    annual_totals,
    common_months,
    month_coverage,
    monthly_series,
    year_over_year,
)

__all__ = [
    "COLUMNS",
    "GAS_M3_PER_UNIT",
    "INJECTION_COLUMNS",
    "PRODUCTION_COLUMNS",
    "SERIES_METRICS",
    "DataQualityReport",
    "IngestManifest",
    "annual_totals",
    "build_report",
    "common_months",
    "gas_oil_ratio",
    "ingest_year",
    "load_years",
    "month_coverage",
    "monthly_series",
    "profile_by",
    "production_summary",
    "resolve_source",
    "year_over_year",
]
__version__ = "0.1.0"
