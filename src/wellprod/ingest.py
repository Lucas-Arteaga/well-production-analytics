"""Ingestion of the official per-well production files.

The dataset is published as one CSV per year, roughly 200 MB each. The download URL is a
per-resource UUID that changes whenever the Secretaría replaces a file, so hardcoding it would
mean the project silently breaks the next time a file is re-uploaded. The URL is therefore
resolved from the official CKAN API at run time, and the resolution is recorded in the manifest
so that every report says exactly which resource it came from.

The file is read in chunks and cached as Parquet. Nothing is loaded into memory unless it is
asked for: ``limit`` makes a fast smoke run possible without a 200 MB download.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

from wellprod.schema import COLUMNS, VOLUME_COLUMNS

#: CKAN package for the dataset. Resolving through this endpoint keeps the project working when
#: the Secretaría re-uploads a yearly file and its resource UUID changes.
CKAN_PACKAGE_URL: Final[str] = (
    "https://datos.gob.ar/api/3/action/package_show?id=produccion-de-petroleo-y-gas-por-pozo"
)

DEFAULT_CHUNK_SIZE: Final[int] = 250_000
DEFAULT_BLOCK_SIZE: Final[int] = 1 << 20  # 1 MiB
USER_AGENT: Final[str] = (
    "wellprod/0.1 (+https://github.com/Lucas-Arteaga/well-production-analytics)"
)

#: Columns that are numeric but are not volumes.
_INT_COLUMNS: Final[tuple[str, ...]] = ("anio", "mes", "idpozo")
_FLOAT_COLUMNS: Final[tuple[str, ...]] = (*VOLUME_COLUMNS, "profundidad")


class IngestError(RuntimeError):
    """Raised when the dataset cannot be resolved, downloaded or parsed."""


# ---------------------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceResource:
    """One CSV resource of the official dataset."""

    name: str
    url: str
    year: int


def _fetch_json(url: str, *, timeout: float = 60.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise IngestError(f"could not read {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IngestError(f"unexpected response shape from {url}")
    return payload


def resolve_source(year: int, *, prefer: str = "annual", timeout: float = 60.0) -> SourceResource:
    """Find the official CSV for ``year`` through the CKAN API.

    Each year has two variants: the plain annual file and one labelled *"DDJJ abiertas y
    cerradas"* (sworn statements, open and closed). ``prefer`` selects between them:

    * ``"annual"`` — the plain annual file. The default, and the one used for analysis here.
    * ``"ddjj"`` — the sworn-statement variant, updated as operators close declarations.

    Raises :class:`IngestError` when no CSV matches, rather than failing later with a confusing
    parse error.
    """
    if prefer not in {"annual", "ddjj"}:
        raise ValueError(f"prefer must be 'annual' or 'ddjj', got {prefer!r}")

    payload = _fetch_json(CKAN_PACKAGE_URL, timeout=timeout)
    resources = payload.get("result", {}).get("resources", [])
    if not isinstance(resources, list):
        raise IngestError("the CKAN response carried no resource list")
    return pick_resource(resources, year, prefer=prefer)


def pick_resource(
    resources: Sequence[object], year: int, *, prefer: str = "annual"
) -> SourceResource:
    """Choose the right CSV out of a CKAN resource list.

    Split out of :func:`resolve_source` so the selection rules — which are the part that can be
    wrong — are testable without a network call.
    """
    year_token = str(year)
    candidates: list[tuple[bool, SourceResource]] = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        if str(resource.get("format", "")).upper() != "CSV":
            continue
        name = str(resource.get("name") or "")
        url = str(resource.get("url") or "")
        if year_token not in name or not url.lower().endswith(".csv"):
            continue
        candidates.append(("ddjj" in name.lower(), SourceResource(name=name, url=url, year=year)))

    if not candidates:
        raise IngestError(
            f"no CSV resource found for {year}. The dataset may not carry that year yet; browse "
            f"https://datos.gob.ar/dataset/produccion-de-petroleo-y-gas-por-pozo to confirm."
        )

    want_ddjj = prefer == "ddjj"
    exact = [resource for is_ddjj, resource in candidates if is_ddjj == want_ddjj]
    if not exact:
        raise IngestError(f"no '{prefer}' variant found for {year}")
    return exact[0]


# ---------------------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------------------


def _sha256(path: Path, *, block: int = DEFAULT_BLOCK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    destination: Path,
    *,
    force: bool = False,
    block: int = DEFAULT_BLOCK_SIZE,
    progress: bool = True,
) -> Path:
    """Download ``url`` to ``destination``, skipping the transfer when it already exists.

    The file is streamed to a temporary name and moved into place only when complete, so an
    interrupted download can never be mistaken for a usable one.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force and destination.stat().st_size > 0:
        if progress:
            print(f"  source already cached: {destination} ({destination.stat().st_size:,} bytes)")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            with partial.open("wb") as handle:
                while chunk := response.read(block):
                    handle.write(chunk)
                    written += len(chunk)
                    if progress and total:
                        pct = written * 100 // total
                        if pct % 10 == 0:
                            print(f"  downloading: {pct}%", end="\r", flush=True)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise IngestError(f"download failed for {url}: {exc}") from exc

    if progress:
        print(f"  downloaded {written:,} bytes" + " " * 8)
    partial.replace(destination)
    return destination


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every column the type the rest of the package expects.

    Everything is read as text first: inferring dtypes across chunks makes the same column
    change type between chunks, and a Parquet cache with inconsistent types is worse than no
    cache. Volumes come from the schema, so the coercion cannot drift from the contract.
    """
    for column in _FLOAT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in _INT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame


def read_source(
    source: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield consecutive chunks of the CSV as typed frames.

    ``limit`` stops after that many rows. The dataset is ordered by operator code, so a
    truncated read is a **biased head** of the file, not a random sample — the manifest
    records the truncation so no report can silently present it as the full year.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    remaining = limit
    reader = pd.read_csv(
        source,
        dtype=str,
        chunksize=chunk_size,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    for chunk in reader:
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]
            remaining -= len(chunk)
        yield _coerce(chunk)


# ---------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------


@dataclass(slots=True)
class IngestManifest:
    """What was ingested, from where, and whether it is the whole file."""

    year: int
    source_name: str
    source_url: str
    source_sha256: str
    source_bytes: int
    rows: int
    chunks: int
    parts: list[str] = field(default_factory=list)
    columns_present: list[str] = field(default_factory=list)
    columns_missing: list[str] = field(default_factory=list)
    truncated_at_row: int | None = None
    created_at: str = ""

    @property
    def is_complete(self) -> bool:
        """Whether the cache holds the whole published file."""
        return self.truncated_at_row is None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_complete"] = self.is_complete
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> IngestManifest:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise IngestError(f"the manifest is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise IngestError("the manifest must be a JSON object")
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ---------------------------------------------------------------------------------------
# Ingestion and loading
# ---------------------------------------------------------------------------------------


def cache_paths(cache_dir: Path, year: int) -> tuple[Path, Path, Path]:
    """The (directory, source csv, manifest) paths for a year."""
    directory = cache_dir / f"year={year}"
    return directory, directory / "source.csv", directory / "manifest.json"


def ingest_year(
    year: int,
    cache_dir: Path | str = Path("data/cache"),
    *,
    source_url: str | None = None,
    source_name: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
    force: bool = False,
    progress: bool = True,
) -> IngestManifest:
    """Resolve, download and cache one year of the dataset as Parquet parts.

    Returns the :class:`IngestManifest` describing exactly what was written.
    """
    cache_dir = Path(cache_dir)
    directory, local_csv, manifest_file = cache_paths(cache_dir, year)

    if manifest_file.exists() and not force:
        existing = IngestManifest.from_json(manifest_file.read_text(encoding="utf-8"))
        if progress:
            print(f"  cache hit: {existing.rows:,} rows from {existing.source_name}")
        return existing

    resolved = (
        SourceResource(name=source_name or local_csv.name, url=source_url, year=year)
        if source_url is not None
        else resolve_source(year)
    )
    if progress:
        print(f"  source: {resolved.name}")

    if resolved.url.startswith(("http://", "https://")):
        download(resolved.url, local_csv, force=force, progress=progress)
    else:
        local_csv = Path(resolved.url)
        if not local_csv.exists():
            raise IngestError(f"local source not found: {local_csv}")

    directory.mkdir(parents=True, exist_ok=True)
    rows = 0
    chunks = 0
    parts: list[str] = []
    columns_present: list[str] = []
    truncated: int | None = None

    for index, chunk in enumerate(read_source(local_csv, chunk_size=chunk_size, limit=limit)):
        part = directory / f"part-{index:05d}.parquet"
        chunk.to_parquet(part, index=False)
        rows += len(chunk)
        chunks += 1
        parts.append(part.name)
        if not columns_present:
            columns_present = [str(c) for c in chunk.columns]
        if limit is not None and rows >= limit:
            truncated = rows
            break

    manifest = IngestManifest(
        year=year,
        source_name=resolved.name,
        source_url=resolved.url,
        source_sha256=_sha256(local_csv),
        source_bytes=local_csv.stat().st_size,
        rows=rows,
        chunks=chunks,
        parts=parts,
        columns_present=columns_present,
        columns_missing=sorted(set(COLUMNS) - set(columns_present)),
        truncated_at_row=truncated,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    manifest_file.write_text(manifest.to_json(), encoding="utf-8")

    if progress:
        flag = "" if manifest.is_complete else f" (TRUNCATED at {truncated:,} rows)"
        print(f"  cached {rows:,} rows in {chunks} part(s){flag}")

    return manifest


def load_years(
    years: Iterable[int],
    cache_dir: Path | str = Path("data/cache"),
    *,
    columns: list[str] | None = None,
) -> dict[int, pd.DataFrame]:
    """Load several cached years, one frame per year.

    Returns a mapping, so the caller decides whether to keep every year alive. When memory
    matters, prefer the streaming shape that :func:`wellprod.series.annual_totals` accepts: load a
    year, aggregate it, let it go.
    """
    return {year: load_year(year, cache_dir, columns=columns) for year in years}


def load_year(
    year: int,
    cache_dir: Path | str = Path("data/cache"),
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a cached year back into memory.

    Each Parquet part is read on its own so that a multi-year or multi-part cache never needs
    every file open at once. Use ``columns`` to keep memory bounded on wide frames.
    """
    directory, _, manifest_file = cache_paths(Path(cache_dir), year)
    if not manifest_file.exists():
        raise IngestError(f"no cache for {year}. Run `wellprod ingest --year {year}` first.")

    manifest = IngestManifest.from_json(manifest_file.read_text(encoding="utf-8"))
    frames = [
        pd.read_parquet(directory / part, columns=columns)  # type: ignore[call-overload]
        for part in manifest.parts
    ]
    if not frames:
        raise IngestError(f"the cache for {year} holds no parts")
    return pd.concat(frames, ignore_index=True)
