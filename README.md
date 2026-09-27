# wellprod — production analytics over Argentina's open well data

Data contract, quality reporting and (soon) production analytics over the **official per-well
oil & gas production dataset** published by the Secretaría de Energía de la Nación.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-3fb950?style=flat-square)
![Data: CC-BY-4.0](https://img.shields.io/badge/data-CC--BY--4.0-f5b544?style=flat-square)

**[English](#english) · [Español](#español)**

---

<a id="english"></a>

## English

> **Status: early, and honest about it.** What works today is the data contract and the
> quality layer, both tested. The analytics, the model service and the dashboard are next —
> the roadmap is at the bottom and it says what is not built yet.

### The problem this starts with

In this dataset the gas column is labelled `prod_gas` and documented as **`Mm3`**. The
published values are **miles de metros cúbicos** — *thousands* of cubic metres, not millions.

Treat that as millions and every gas-oil ratio you compute is **1000× too large**. The
resulting number is 98 700 instead of 98.7 m³/m³ — and 98 700 still looks like something a
person might believe. Nothing crashes. No warning is raised. The analysis is simply wrong.

So this project declares the unit of every column **once**, in `wellprod.schema`, attached to
the column that carries it. `gas_oil_ratio()` is the only sanctioned way to compute a ratio,
and a regression test pins the conversion factor at 1 000:

```python
def test_gas_oil_ratio_uses_thousands_not_millions():
    ratio = gas_oil_ratio(pd.Series([5.0]), pd.Series([100.0])).iloc[0]

    assert GAS_M3_PER_UNIT == 1_000.0
    assert ratio == pytest.approx(50.0)          # not 50 000
```

That test exists because I made this exact mistake in a previous project. The data there was
under an operator's confidentiality, so the correction could not be published. This is the
public version, on public data, where anyone can check the arithmetic.

### What is in the box today

| Module | What it does |
| :--- | :--- |
| `wellprod.schema` | The column contract: all 38 published columns with their unit and conversion factor, the business-valid value sets, and `gas_oil_ratio()` |
| `wellprod.quality` | `build_report(df)` — nulls, duplicates, negative volumes, rows with no oil and no gas, placeholder field names, administrative provinces, implausible ratios, and the share of rows eligible for modelling. `production_summary(df)` — per-fluid statistics with published units and m³ kept side by side |

### The data

* **Source:** Secretaría de Energía de la Nación — *"Producción de petróleo y gas por pozo (Capítulo IV)"*
* **Where:** <https://datos.gob.ar/dataset/produccion-de-petroleo-y-gas-por-pozo>
* **Licence:** **CC-BY-4.0**
* **Granularity:** one row = one well in one month. Oil, gas and water volumes, injection
  volumes, and geological descriptors (basin, formation, field, province, resource type).
* **Size:** roughly 200 MB per year of file, tens of millions of rows across the series.

The data is **not** redistributed here — you download it. Only the code is in this repository.

### Install and run

```bash
git clone https://github.com/Lucas-Arteaga/well-production-analytics.git
cd well-production-analytics
uv sync --extra dev

uv run pytest                 # the full suite: no network, no data files needed
uv run ruff check .
```

### Design decisions worth stating

1. **Units live with the column, never inline.** A conversion factor hardcoded anywhere but
   `schema.py` is a bug, not an optimisation.
2. **The ratio function refuses to invent a number.** If oil production is zero, the ratio is
   `NaN` — not infinity, not zero. A ratio from a well that is not producing is noise, and
   filtering belongs upstream where the analyst can see it happen.
3. **The quality report is meant to be committed.** It is the record of what was dropped and
   why. An analysis without it is an unverifiable claim.
4. **No data files in CI.** Tests build their own frames from the contract, so the suite runs
   in seconds and proves the logic without depending on a 200 MB download.

### Roadmap

Built toward what the market actually asks for, not toward what is fun to build:

- [x] Data contract with units and conversion factors
- [x] Data-quality report, serialisable to JSON and Markdown
- [ ] Chunked multi-year ingestion with a Parquet cache
- [ ] Basin / operator / formation profiling
- [ ] Gas-oil ratio and water cut, with documented validity filters
- [ ] Well clustering and Arps decline-curve fitting
- [ ] **Physics-informed** GOR model: decline physics for the trend, machine learning for the
      residual — and no target leakage (oil and gas production must not be predictors of a
      ratio defined by oil and gas production)
- [ ] **Model served as an API** (FastAPI + Docker), with CI and basic drift monitoring
- [ ] **Power BI** report published on top of the same outputs
- [ ] Static dashboard

### Licence

MIT for the code — see [LICENSE](LICENSE). The data is CC-BY-4.0 and belongs to the
Secretaría de Energía.

---

<a id="español"></a>
<details>
<summary><b>🇪🇸 Leer en castellano</b></summary>

<br>

> **Estado: temprano, y lo digo.** Lo que hoy funciona es el contrato de datos y la capa de
> calidad, ambos con tests. La analítica, el servicio del modelo y el tablero vienen después:
> el roadmap está al final y dice explícitamente qué **no** está construido.

### El problema con el que arranca

En este dataset la columna de gas se llama `prod_gas` y está documentada como **`Mm3`**. Los
valores publicados son **miles de metros cúbicos**, no millones.

Si los tratás como millones, **toda relación gas-petróleo que calcules queda 1000× más
grande**. El resultado da 98.700 en lugar de 98,7 m³/m³ — y 98.700 sigue pareciendo un número
que alguien podría creer. No falla nada, no salta ninguna advertencia. El análisis simplemente
está mal.

Por eso este proyecto declara la unidad de cada columna **una sola vez**, en `wellprod.schema`,
pegada a la columna que la lleva. `gas_oil_ratio()` es la única forma habilitada de calcular
una relación, y un test de regresión fija el factor de conversión en 1.000.

Ese test existe porque **cometí exactamente este error** en un trabajo anterior. Los datos de
ese trabajo estaban bajo confidencialidad de una operadora, así que la corrección no se pudo
publicar. Esta es la versión pública, sobre datos públicos, donde cualquiera puede verificar
la aritmética.

### Qué hay hoy

| Módulo | Qué hace |
| :--- | :--- |
| `wellprod.schema` | El contrato de columnas: las 38 columnas publicadas con su unidad y factor de conversión, los conjuntos de valores válidos de negocio, y `gas_oil_ratio()` |
| `wellprod.quality` | `build_report(df)` — nulos, duplicados, volúmenes negativos, filas sin petróleo ni gas, nombres de yacimiento placeholder, provincias administrativas, relaciones implausibles y la proporción de filas aptas para modelar. `production_summary(df)` — estadísticas por fluido con la unidad publicada y su equivalente en m³, uno al lado del otro |

### Los datos

* **Fuente:** Secretaría de Energía de la Nación — *"Producción de petróleo y gas por pozo (Capítulo IV)"*
* **Dónde:** <https://datos.gob.ar/dataset/produccion-de-petroleo-y-gas-por-pozo>
* **Licencia:** **CC-BY-4.0**
* **Granularidad:** una fila = un pozo en un mes. Volúmenes de petróleo, gas y agua,
  volúmenes de inyección, y descriptores geológicos (cuenca, formación, yacimiento, provincia,
  tipo de recurso).
* **Tamaño:** alrededor de 200 MB por año de archivo, decenas de millones de filas en la serie.

Los datos **no** se redistribuyen acá: los descargás vos. En este repositorio está sólo el código.

### Instalar y correr

```bash
git clone https://github.com/Lucas-Arteaga/well-production-analytics.git
cd well-production-analytics
uv sync --extra dev

uv run pytest                 # la suite completa: sin red y sin archivos de datos
uv run ruff check .
```

### Decisiones de diseño que vale la pena declarar

1. **Las unidades viven con la columna, nunca sueltas en el código.** Un factor de conversión
   escrito en cualquier lugar que no sea `schema.py` es un bug, no una optimización.
2. **La función de la relación se niega a inventar un número.** Si la producción de petróleo es
   cero, la relación es `NaN` — no infinito, no cero. Una relación de un pozo que no produce es
   ruido, y filtrar es responsabilidad de la etapa anterior, donde el analista lo ve pasar.
3. **El reporte de calidad está pensado para commitearse.** Es el registro de qué se descartó y
   por qué. Un análisis sin ese registro es una afirmación no verificable.
4. **Sin archivos de datos en CI.** Los tests construyen sus propios dataframes a partir del
   contrato, así que la suite corre en segundos y prueba la lógica sin depender de una descarga
   de 200 MB.

### Roadmap

Construido hacia lo que el mercado pide, no hacia lo que es divertido de construir:

- [x] Contrato de datos con unidades y factores de conversión
- [x] Reporte de calidad, serializable a JSON y Markdown
- [ ] Ingesta multi-año por bloques con caché en Parquet
- [ ] Perfilado por cuenca / operadora / formación
- [ ] Relación gas-petróleo y corte de agua, con filtros de validez documentados
- [ ] Clustering de pozos y ajuste de curvas de declinación (Arps)
- [ ] Modelo de RGP **physics-informed**: la física de declinación para la tendencia, machine
      learning para el residuo — y **sin fuga de target** (la producción de petróleo y gas no
      puede ser predictora de una relación definida por la producción de petróleo y gas)
- [ ] **Modelo servido como API** (FastAPI + Docker), con CI y monitoreo básico de drift
- [ ] Reporte de **Power BI** publicado sobre las mismas salidas
- [ ] Tablero estático

### Licencia

MIT para el código — ver [LICENSE](LICENSE). Los datos son CC-BY-4.0 y pertenecen a la
Secretaría de Energía.

</details>
