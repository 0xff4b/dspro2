"""Municipal statistics for the Quality-of-Life Index (QoLI): BFS SDMX, ESTV and BFS communes.

Vacancy rates (BFS DF_LWZ_1) and City Statistics (BFS DF_CITYSTAT_1) from the stats.swiss SDMX
API, the ESTV tax burden per municipality and the BFS correspondence table that carries values
of an older municipality list to the current one (mergers get the mean of their parts).
"""

import io
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from rentml.qoli_sources import COMMUNES_API, HTTP_HEADERS

SDMX_BASE = "https://disseminate.stats.swiss/rest/data"
_SDMX_CSV = "application/vnd.sdmx.data+csv;version=1.0.0;labels=both"


# --- BFS SDMX: vacancy and City Statistics ---------------------------------------------------


def fetch_sdmx_csv(
    flow: str,
    cache_path: Path,
    *,
    key: str = "all",
    params: Mapping[str, str] | None = None,
    overwrite: bool = False,
    timeout: float = 300.0,
) -> pd.DataFrame:
    """Download an SDMX dataflow of the BFS (stats.swiss) as CSV with codes and labels, cached.

    Args:
        flow: ``"<agency>,<dataflow>,<version>"``, e.g. ``"CH1.LWZ,DF_LWZ_1,1.1.0"``.
        cache_path: CSV cache file.
        key: SDMX series key (``"all"`` = every series).
        params: Query parameters such as ``{"startPeriod": "2026"}``.
        overwrite: Query again even if cached.
        timeout: HTTP timeout in seconds.

    Returns:
        The raw CSV table (columns ``"<ID>: <label>"``, ``OBS_VALUE`` ...).

    Raises:
        RuntimeError: If the request fails.
    """
    if cache_path.is_file() and not overwrite:
        return pd.read_csv(cache_path, dtype=str)
    url = f"{SDMX_BASE}/{flow}/{key}"
    try:
        resp = requests.get(
            url,
            params=dict(params or {}),
            headers=HTTP_HEADERS | {"Accept": _SDMX_CSV},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"SDMX request {url} failed: {exc}") from exc
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)
    return pd.read_csv(io.BytesIO(resp.content), dtype=str)


def sdmx_code(series: pd.Series) -> pd.Series:
    """Code part of an SDMX ``"<code>: <label>"`` column."""
    return series.astype(str).str.split(":", n=1).str[0].str.strip()


def vacancy_rates(table: pd.DataFrame) -> pd.Series:
    """Municipal vacancy rate (all dwelling types and sizes) from the DF_LWZ_1 table.

    Args:
        table: Output of ``fetch_sdmx_csv`` for ``CH1.LWZ,DF_LWZ_1``.

    Returns:
        Series ``vacancy_rate`` (% of dwellings) indexed by ``municipality_id`` for the latest
        period in the table; districts, cantons and Switzerland are dropped.
    """
    cols = {c.split(":")[0]: c for c in table.columns}
    unit = sdmx_code(table[cols["GR_KT_GDE"]])
    keep = (
        sdmx_code(table[cols["WOHN_ANZAHL"]]).eq("_T")
        & sdmx_code(table[cols["LEERWOHN_TYP"]]).eq("_T")
        & sdmx_code(table[cols["MEASURE_DIMENSION"]]).eq("RATE")
        & unit.str.fullmatch(r"\d+")
    )
    sub = table.loc[keep]
    sub = sub.loc[sub[cols["TIME_PERIOD"]].eq(sub[cols["TIME_PERIOD"]].max())]
    values = pd.to_numeric(sub["OBS_VALUE"], errors="coerce").to_numpy()
    index = pd.Index(unit[sub.index].astype("int64").to_numpy(), name="municipality_id")
    return pd.Series(values, index=index, name="vacancy_rate").dropna().sort_index()


def city_statistics(table: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    """Latest observed value per core city and variable from the City Statistics table.

    Core cities have spatial codes ``<BFS municipality number> 901`` (e.g. 261901 = Zürich).

    Args:
        table: Output of ``fetch_sdmx_csv`` for ``CH1.CITYSTAT,DF_CITYSTAT_1``.
        variables: Variable codes, e.g. ``["EC3063I", "SA1025I"]``.

    Returns:
        DataFrame indexed by ``municipality_id`` with one column per variable (latest year with
        an observed value) and ``<variable>_year``.
    """
    cols = {c.split(":")[0]: c for c in table.columns}
    unit = sdmx_code(table[cols["ID_RAUM"]])
    var = sdmx_code(table[cols["ID_INDIC_DISPR"]])
    value = pd.to_numeric(table["OBS_VALUE"], errors="coerce")
    keep = (
        unit.str.endswith("901")
        & var.isin(list(variables))
        & sdmx_code(table[cols["STATISTICAL_OPERATION"]]).eq("OBS")
        & value.notna()
    )
    frame = pd.DataFrame(
        {
            "municipality_id": unit[keep].str[:-3].astype("int64"),
            "variable": var[keep],
            "year": table.loc[keep, cols["TIME_PERIOD"]].astype(int),
            "value": value[keep],
        }
    )
    latest = frame.sort_values("year").groupby(["municipality_id", "variable"]).last()
    wide = latest["value"].unstack("variable").reindex(columns=list(variables))
    years = latest["year"].unstack("variable").reindex(columns=list(variables))
    return wide.join(years.add_suffix("_year"))


# --- ESTV tax burden and municipality correspondence -----------------------------------------


def load_tax_burden(xlsx: Path, *, profile: str = "Ledig", gross_income: int = 80_000) -> pd.Series:
    """Cantonal, municipal and church tax in % of gross labour income per municipality (ESTV).

    Args:
        xlsx: ESTV ``statistik-belastung-gden-<year>-de.xlsx``.
        profile: Sheet of the household profile (``"Ledig"``, ``"VOK"``, ``"VMK"``, ...).
        gross_income: Gross labour income in CHF (a column of the sheet).

    Returns:
        Series ``tax_burden_pct`` indexed by the municipality number of the file's year.

    Raises:
        KeyError: If the income column is not in the sheet.
    """
    raw = pd.read_excel(xlsx, sheet_name=profile, header=None)
    header_row = raw.index[raw.iloc[:, 1].astype(str).str.contains("Gemeindenummer", na=False)][0]
    incomes = pd.to_numeric(raw.iloc[header_row - 1], errors="coerce")
    match = np.flatnonzero(incomes.to_numpy() == gross_income)
    if match.size == 0:
        raise KeyError(f"No column for a gross income of {gross_income} in sheet {profile!r}")
    body = raw.iloc[header_row + 1 :]
    ids = pd.to_numeric(body.iloc[:, 1], errors="coerce")
    values = pd.to_numeric(body.iloc[:, match[0]], errors="coerce")
    ok = ids.notna() & values.notna()
    index = pd.Index(ids[ok].astype("int64").to_numpy(), name="municipality_id")
    return pd.Series(values[ok].to_numpy(float), index=index, name="tax_burden_pct")


def fetch_commune_correspondence(
    start: str, end: str, cache_path: Path, *, overwrite: bool = False, timeout: float = 120.0
) -> pd.DataFrame:
    """BFS correspondence of municipality numbers between two dates (mergers, splits).

    Args:
        start: Start date ``DD-MM-YYYY``.
        end: End date ``DD-MM-YYYY``.
        cache_path: CSV cache file.
        overwrite: Query again even if cached.
        timeout: HTTP timeout in seconds.

    Returns:
        DataFrame with ``initial_id``, ``initial_name``, ``terminal_id``, ``terminal_name``.

    Raises:
        RuntimeError: If the request fails.
    """
    if overwrite or not cache_path.is_file():
        params = {
            "includeUnmodified": "true",
            "includeTerritoryExchange": "false",
            "startPeriod": start,
            "endPeriod": end,
        }
        try:
            resp = requests.get(COMMUNES_API, params=params, headers=HTTP_HEADERS, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Commune correspondence request failed: {exc}") from exc
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
    raw = pd.read_csv(cache_path)
    return pd.DataFrame(
        {
            "initial_id": raw["InitialCode"].astype("int64"),
            "initial_name": raw["InitialName"].astype(str),
            "terminal_id": raw["TerminalCode"].astype("int64"),
            "terminal_name": raw["TerminalName"].astype(str),
        }
    ).drop_duplicates(["initial_id", "terminal_id"])


def remap_municipalities(values: pd.Series, correspondence: pd.DataFrame) -> pd.Series:
    """Carry municipal values to later municipality numbers (merged ones get the mean).

    Args:
        values: Values indexed by the initial municipality numbers.
        correspondence: Output of ``fetch_commune_correspondence``.

    Returns:
        Series indexed by ``municipality_id`` (terminal numbers); unmatched values are dropped.
    """
    mapped = correspondence.merge(
        values.rename("value"), left_on="initial_id", right_index=True, how="inner"
    )
    out = mapped.groupby("terminal_id")["value"].mean()
    out.index.name = "municipality_id"
    return out.rename(values.name)
