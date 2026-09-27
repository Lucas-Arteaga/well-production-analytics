"""Tests for the data-quality report."""

from __future__ import annotations

import pandas as pd
import pytest

from wellprod.quality import build_report, production_summary
from wellprod.schema import COLUMNS


def make_frame(n: int = 4, **overrides: list[object]) -> pd.DataFrame:
    """A frame that actually satisfies the contract, so each test breaks exactly one thing.

    Defaults are derived from each column's ``kind``, so the frame has no nulls and no
    accidental defects. A helper that fills everything with ``None`` is not a clean frame —
    it is a frame with twenty defects waiting to satisfy a wrong assertion.
    """
    data: dict[str, list[object]] = {}
    for name, spec in COLUMNS.items():
        if spec.kind == "volume":
            data[name] = [5.0] * n
        elif spec.kind == "identifier":
            data[name] = [1] * n
        elif spec.kind == "date":
            data[name] = ["2026-01-31"] * n
        elif spec.kind == "flag":
            data[name] = ["t"] * n
        else:
            data[name] = ["X"] * n

    data.update(
        {
            "anio": [2026] * n,
            "mes": list(range(1, n + 1)),
            "idpozo": list(range(1, n + 1)),
            "prod_pet": [100.0] * n,
            "prod_gas": [5.0] * n,
            "prod_agua": [10.0] * n,
            "iny_agua": [0.0] * n,
            "iny_gas": [0.0] * n,
            "iny_co2": [0.0] * n,
            "iny_otro": [0.0] * n,
            "tipoestado": ["Extracción Efectiva"] * n,
            "tipopozo": ["Petrolífero"] * n,
            "empresa": ["OPERADORA SA"] * n,
            "cuenca": ["NEUQUINA"] * n,
            "provincia": ["Neuquén"] * n,
            "areayacimiento": ["CAMPO NORTE"] * n,
            "tipo_de_recurso": ["CONVENCIONAL"] * n,
        }
    )
    data.update(overrides)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------------------
# Contract conformance.
# ---------------------------------------------------------------------------------------


def test_a_clean_frame_reports_no_defects() -> None:
    report = build_report(make_frame())

    assert report.n_rows == 4
    assert report.missing_columns == []
    assert report.unexpected_columns == []
    assert report.duplicate_rows == 0
    assert report.negative_volume_counts == {}
    assert report.zero_oil_and_gas_rows == 0
    assert report.implausible_gor_rows == 0
    assert report.usable_share == pytest.approx(1.0)


def test_missing_contract_columns_are_named() -> None:
    frame = make_frame().drop(columns=["cuenca", "formacion"])

    report = build_report(frame)

    assert report.missing_columns == ["cuenca", "formacion"]


def test_unexpected_columns_are_named() -> None:
    frame = make_frame()
    frame["columna_nueva_2027"] = 1

    report = build_report(frame)

    assert report.unexpected_columns == ["columna_nueva_2027"]


def test_duplicate_rows_are_counted() -> None:
    frame = make_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    report = build_report(frame)

    assert report.duplicate_rows == 1


def test_negative_volumes_are_counted_per_column() -> None:
    frame = make_frame(prod_agua=[-5.0, 10.0, 10.0, 10.0], prod_pet=[-1.0, 100.0, 100.0, 100.0])

    report = build_report(frame)

    assert report.negative_volume_counts == {"prod_agua": 1, "prod_pet": 1}


def test_rows_with_no_oil_and_no_gas_are_counted_and_lower_usable_share() -> None:
    frame = make_frame(prod_pet=[0.0, 100.0, 100.0, 100.0], prod_gas=[0.0, 5.0, 5.0, 5.0])

    report = build_report(frame)

    assert report.zero_oil_and_gas_rows == 1
    assert report.usable_share == pytest.approx(0.75)


def test_placeholder_fields_and_administrative_province_are_counted() -> None:
    frame = make_frame(
        areayacimiento=["POZOS SIN YACIMIENTO", "CAMPO NORTE", "SIN NOMBRE", "CAMPO NORTE"],
        provincia=["Neuquén", "Estado Nacional", "Neuquén", "Neuquén"],
    )

    report = build_report(frame)

    assert report.placeholder_field_rows == 2
    assert report.administrative_province_rows == 1


def test_implausible_gas_oil_ratio_is_detected() -> None:
    # 5 "Mm3" of gas over 0.0001 m3 of oil is a GOR of 50 million m3/m3.
    frame = make_frame(prod_pet=[0.0001, 100.0, 100.0, 100.0])

    report = build_report(frame)

    assert report.implausible_gor_rows == 1


def test_null_counts_only_list_columns_that_have_nulls() -> None:
    frame = make_frame(formacion=["MULICHINCO", None, "MULICHINCO", "MULICHINCO"])

    report = build_report(frame)

    assert report.null_counts == {"formacion": 1}


def test_report_rejects_a_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        build_report([{"prod_pet": 1}])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# Serialisation.
# ---------------------------------------------------------------------------------------


def test_to_dict_is_json_serialisable_and_includes_usable_share() -> None:
    import json

    payload = build_report(make_frame()).to_dict()

    assert payload["usable_share"] == pytest.approx(1.0)
    json.dumps(payload)  # must not raise


def test_to_markdown_renders_every_check() -> None:
    markdown = build_report(make_frame()).to_markdown()

    for expected in ("Rows", "Duplicate rows", "Modeling-eligible share"):
        assert expected in markdown


def test_to_markdown_lists_missing_and_unexpected_when_present() -> None:
    frame = make_frame().drop(columns=["cuenca"])
    frame["otra"] = 1

    markdown = build_report(frame).to_markdown()

    assert "Missing: `cuenca`" in markdown
    assert "Unexpected: `otra`" in markdown


# ---------------------------------------------------------------------------------------
# production_summary keeps units visible.
# ---------------------------------------------------------------------------------------


def test_production_summary_exposes_both_published_and_m3_for_gas() -> None:
    summary = production_summary(make_frame(prod_gas=[5.0, 5.0, 5.0, 5.0]))

    assert "prod_gas_published" in summary.index
    assert "prod_gas_m3" in summary.index
    assert summary.loc["prod_gas_published", "sum"] == pytest.approx(20.0)
    assert summary.loc["prod_gas_m3", "sum"] == pytest.approx(20_000.0)


def test_production_summary_does_not_duplicate_liquid_columns() -> None:
    summary = production_summary(make_frame())

    assert "prod_pet_published" in summary.index
    assert "prod_pet_m3" not in summary.index
    assert summary.loc["prod_pet_published", "unit"] == "m3"


def test_production_summary_requires_the_production_columns() -> None:
    frame = make_frame().drop(columns=["prod_gas"])

    with pytest.raises(KeyError, match="prod_gas"):
        production_summary(frame)
