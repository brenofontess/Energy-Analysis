"""
Data Ingestion Module for Energy Price Agents project.
Handles extraction from OMIE, Open-Meteo, and Financial Commodities,
applying thread-safe local caching, DST handling, 15-minute aggregation, and Parquet persistence.
"""

import concurrent.futures
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests
import yfinance as yf
from src.utils import get_project_root, load_config

_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
  """Returns a per-thread requests.Session (Session is not thread-safe)."""
  if not hasattr(_thread_local, "session"):
    _thread_local.session = requests.Session()
  return _thread_local.session


def fetch_omie_day(
    d: date, raw_omie_dir: Path, session: Optional[requests.Session] = None
) -> Optional[Path]:
  """Downloads and caches raw marginalpdbc file for a given date if not already cached."""
  f_str = d.strftime("%Y%m%d")
  target_file = raw_omie_dir / f"marginalpdbc_{f_str}.1"
  if target_file.exists() and target_file.stat().st_size > 100:
    return target_file

  if session is None:
    session = _get_thread_session()

  url = (
      "https://www.omie.es/en/file-download?"
      f"parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{f_str}.1"
  )
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }

  for attempt in range(3):
    try:
      resp = session.get(url, headers=headers, timeout=12)
      if resp.status_code == 200 and len(resp.content) > 100:
        with open(target_file, "wb") as f:
          f.write(resp.content)
        return target_file
      if resp.status_code == 404:
        return None
    except Exception:
      time.sleep(1.0 * (attempt + 1))
  return None


def parse_omie_file(filepath: Path) -> List[Dict[str, Any]]:
  """Parses an OMIE marginalpdbc file using explicit period indices.

  Handles pre-2025-10-01 hourly and post-2025-10-01 15-min MTU15 data without
  positional drift if an interval is missing.
  """
  with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    raw_lines = [line.strip() for line in f if line.strip()]

  data_lines = [
      line
      for line in raw_lines
      if line and not line.startswith("MARGINALPDBC") and not line.startswith("*")
  ]
  if not data_lines:
    return []

  parsed_records = []
  for line in data_lines:
    parts = line.split(";")
    if len(parts) >= 6:
      try:
        y = int(parts[0])
        m = int(parts[1])
        d = int(parts[2])
        period = int(parts[3])
        p_sp = float(parts[4])
        p_pt = float(parts[5])
        if period >= 1:
          parsed_records.append((y, m, d, period, p_sp, p_pt))
      except (ValueError, IndexError):
        continue

  if not parsed_records:
    return []

  y, m, d = parsed_records[0][0], parsed_records[0][1], parsed_records[0][2]
  max_period = max(r[3] for r in parsed_records)
  is_15min = (
      date(y, m, d) >= date(2025, 10, 1)
      or len(parsed_records) > 40
      or max_period > 25
  )

  day_start_madrid = pd.Timestamp(y, m, d, 0, 0, tz="Europe/Madrid")
  day_start_utc = day_start_madrid.tz_convert("UTC")

  price_min = -500.0
  price_max = 4000.0

  hourly_buckets: Dict[int, List[Tuple[float, float]]] = {}
  for _, _, _, period, p_sp, p_pt in parsed_records:
    hour_idx = (period - 1) // 4 if is_15min else (period - 1)
    hourly_buckets.setdefault(hour_idx, []).append((p_sp, p_pt))

  hourly_outputs = []
  for hour_idx in sorted(hourly_buckets):
    block = hourly_buckets[hour_idx]
    avg_p_sp = sum(r[0] for r in block) / len(block)
    avg_p_pt = sum(r[1] for r in block) / len(block)
    avg_p_pt = avg_p_pt if price_min <= avg_p_pt <= price_max else float("nan")
    avg_p_sp = avg_p_sp if price_min <= avg_p_sp <= price_max else float("nan")
    ts_utc = day_start_utc + pd.Timedelta(hours=hour_idx)
    ts_local = ts_utc.tz_convert("Europe/Lisbon")
    hourly_outputs.append({
        "timestamp_utc": ts_utc,
        "timestamp_local": ts_local,
        "price_pt_eur_mwh": round(avg_p_pt, 2),
        "price_sp_eur_mwh": round(avg_p_sp, 2),
    })

  return hourly_outputs


def ingest_omie_dataset(
    start_date: str,
    end_date: str,
    raw_dir: Optional[Path] = None,
    max_workers: int = 12,
) -> pd.DataFrame:
  """Downloads and parses OMIE prices across the specified date range."""
  if raw_dir is None:
    raw_dir = get_project_root() / "data" / "raw" / "omie"
  raw_dir.mkdir(parents=True, exist_ok=True)

  s_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
  e_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
  date_list = [s_dt + timedelta(days=i) for i in range((e_dt - s_dt).days + 1)]

  print(
      f"Fetching OMIE files for {len(date_list)} days ({start_date} to"
      f" {end_date})..."
  )

  files_to_parse = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = [ex.submit(fetch_omie_day, d, raw_dir) for d in date_list]
    for fut in concurrent.futures.as_completed(futures):
      res = fut.result()
      if res:
        files_to_parse.append(res)

  print(
      f"Downloaded/Cached {len(files_to_parse)} OMIE daily files. Parsing"
      " records..."
  )

  all_records = []
  for fpath in files_to_parse:
    all_records.extend(parse_omie_file(fpath))

  df_omie = pd.DataFrame(all_records)
  if df_omie.empty:
    raise ValueError(
        "No OMIE records parsed. Check network connectivity or date range."
    )

  df_omie = (
      df_omie.drop_duplicates(subset=["timestamp_utc"])
      .sort_values("timestamp_utc")
      .reset_index(drop=True)
  )
  return df_omie


def ingest_weather_dataset(
    start_date: str,
    end_date: str,
    raw_dir: Optional[Path] = None,
    locations: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
  """Downloads historical weather from Open-Meteo (ERA5 proxy) for configured locations.

  Caches responses per location and date range, then computes weighted averages.
  """
  if raw_dir is None:
    raw_dir = get_project_root() / "data" / "raw" / "weather"
  raw_dir.mkdir(parents=True, exist_ok=True)

  if locations is None:
    cfg = load_config()
    locations = cfg["data_sources"]["open_meteo"]["locations"]

  weather_specs = [
      ("temperature_2m", "temp_pt_c", 2, None),
      ("wind_speed_100m", "wind_speed_100m_pt_kmh", 2, None),
      ("direct_radiation", "solar_radiation_pt_wm2", 2, None),
      ("cloud_cover", "cloud_cover_pt_pct", 1, None),
      ("precipitation", "precipitation_pt_mm", 3, 0.0),
      ("relative_humidity_2m", "humidity_pt_pct", 1, 70.0),
  ]

  loc_dfs: Dict[str, pd.DataFrame] = {}
  for loc_name, info in locations.items():
    cache_file = raw_dir / f"open_meteo_{loc_name}_{start_date}_{end_date}.json"
    if not cache_file.exists():
      url = (
          "https://archive-api.open-meteo.com/v1/archive?"
          f"latitude={info['lat']}&longitude={info['lon']}&"
          f"start_date={start_date}&end_date={end_date}&"
          "hourly=temperature_2m,wind_speed_100m,direct_radiation,cloud_cover,"
          "precipitation,relative_humidity_2m"
      )
      print(
          f"Requesting Open-Meteo for {loc_name} ({info['lat']},"
          f" {info['lon']})..."
      )
      resp = requests.get(url, timeout=45)
      resp.raise_for_status()
      with open(cache_file, "w", encoding="utf-8") as f:
        f.write(resp.text)
      data = resp.json()
    else:
      with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    hourly = data["hourly"]
    n_pts = len(hourly["time"])
    df_loc = pd.DataFrame({"timestamp_utc": pd.to_datetime(hourly["time"], utc=True)})
    for raw_key, _, _, default_val in weather_specs:
      if raw_key in hourly:
        df_loc[raw_key] = hourly[raw_key]
      elif default_val is not None:
        df_loc[raw_key] = [default_val] * n_pts
      else:
        raise KeyError(f"Missing required weather variable '{raw_key}' in {cache_file}")
    loc_dfs[loc_name] = df_loc

  first_loc = next(iter(locations))
  df_w = loc_dfs[first_loc][["timestamp_utc"]].copy()
  for raw_key, out_col, decimals, _ in weather_specs:
    weighted = sum(
        loc_dfs[loc_name][raw_key] * info["weight"]
        for loc_name, info in locations.items()
    )
    df_w[out_col] = weighted.round(decimals)

  return df_w


def ingest_commodities_dataset(
    start_date: str, end_date: str, raw_dir: Optional[Path] = None
) -> pd.DataFrame:
  """Downloads TTF Natural Gas futures and Carbon Allowances ETF proxy (KRBN)."""
  if raw_dir is None:
    raw_dir = get_project_root() / "data" / "raw" / "commodities"
  raw_dir.mkdir(parents=True, exist_ok=True)

  ttf_file = raw_dir / "ttf_gas_raw.parquet"
  krbn_file = raw_dir / "krbn_carbon_raw.parquet"

  if not ttf_file.exists():
    print("Downloading TTF Natural Gas (TTF=F) via yfinance...")
    df_ttf = yf.download("TTF=F", start=start_date, end=end_date, progress=False)
    if isinstance(df_ttf.columns, pd.MultiIndex):
      df_ttf.columns = [c[0] for c in df_ttf.columns]
    df_ttf.to_parquet(ttf_file)
  else:
    df_ttf = pd.read_parquet(ttf_file)

  if not krbn_file.exists():
    print("Downloading Carbon EUA Proxy (KRBN) via yfinance...")
    df_krbn = yf.download("KRBN", start=start_date, end=end_date, progress=False)
    if isinstance(df_krbn.columns, pd.MultiIndex):
      df_krbn.columns = [c[0] for c in df_krbn.columns]
    df_krbn.to_parquet(krbn_file)
  else:
    df_krbn = pd.read_parquet(krbn_file)

  close_ttf = df_ttf["Close"].copy().rename("ttf_gas_eur_mwh")
  close_krbn = df_krbn["Close"].copy().rename("carbon_proxy_usd")

  df_comm = pd.concat([close_ttf, close_krbn], axis=1)
  df_comm.index = pd.to_datetime(df_comm.index).tz_localize(None)

  all_days = pd.date_range(start_date, end_date, freq="D")
  df_comm = df_comm.reindex(all_days).ffill()
  if df_comm.isnull().any().any():
    first_month_median = df_comm.iloc[:31].median()
    df_comm = df_comm.fillna(first_month_median)
  df_comm["date_str"] = df_comm.index.strftime("%Y-%m-%d")
  return df_comm


def summarize_missing_gaps(missing_hours: pd.DatetimeIndex) -> List[str]:
  """Formats contiguous missing hour intervals as human-readable strings."""
  if len(missing_hours) == 0:
    return []
  sorted_ts = missing_hours.sort_values()
  gaps = []
  gap_start = sorted_ts[0]
  prev_ts = sorted_ts[0]
  for ts in sorted_ts[1:]:
    if ts - prev_ts > pd.Timedelta(hours=1):
      gaps.append(f"{gap_start} -> {prev_ts}")
      gap_start = ts
    prev_ts = ts
  gaps.append(f"{gap_start} -> {prev_ts}")
  return gaps


def build_base_dataset(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interim_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Path, Path]:
  """Orchestrates end-to-end ingestion into data/interim/dataset_base.parquet."""
  config = load_config()
  if start_date is None:
    start_date = config["temporal_settings"]["dataset_start"]
  if end_date is None:
    end_date = config["temporal_settings"]["dataset_end"]

  if interim_dir is None:
    interim_dir = get_project_root() / "data" / "interim"
  interim_dir.mkdir(parents=True, exist_ok=True)

  df_omie = ingest_omie_dataset(start_date, end_date)
  df_weather = ingest_weather_dataset(start_date, end_date)
  df_comm = ingest_commodities_dataset(start_date, end_date)

  df_base = pd.merge(df_omie, df_weather, on="timestamp_utc", how="inner")

  df_base["date_str"] = df_base["timestamp_local"].dt.strftime("%Y-%m-%d")
  df_base = pd.merge(
      df_base,
      df_comm[["date_str", "ttf_gas_eur_mwh", "carbon_proxy_usd"]],
      on="date_str",
      how="left",
  )
  df_base.drop(columns=["date_str"], inplace=True)

  df_base["price_spread_pt_sp"] = (
      df_base["price_pt_eur_mwh"] - df_base["price_sp_eur_mwh"]
  ).round(2)

  expected_hours = pd.date_range(
      df_base["timestamp_utc"].min(),
      df_base["timestamp_utc"].max(),
      freq="h",
  )
  missing_hours = expected_hours.difference(pd.DatetimeIndex(df_base["timestamp_utc"]))
  if len(missing_hours) > 0:
    gaps = summarize_missing_gaps(missing_hours)
    print(f"[AVISO] {len(missing_hours)} horas em falta no dataset base. Lacunas: {gaps}")
  else:
    print(
        f"[OK] Continuidade temporal verificada: {len(df_base)} horas"
        " cronologicas sem lacunas."
    )

  columns_order = [
      "timestamp_utc",
      "timestamp_local",
      "price_pt_eur_mwh",
      "price_sp_eur_mwh",
      "price_spread_pt_sp",
      "temp_pt_c",
      "wind_speed_100m_pt_kmh",
      "solar_radiation_pt_wm2",
      "cloud_cover_pt_pct",
      "precipitation_pt_mm",
      "humidity_pt_pct",
      "ttf_gas_eur_mwh",
      "carbon_proxy_usd",
  ]
  df_base = df_base[columns_order]

  output_parquet = interim_dir / "dataset_base.parquet"
  df_base.to_parquet(output_parquet, compression="snappy", index=False)

  dict_records = [
      {
          "name": "timestamp_utc",
          "type": "datetime64[ns, UTC]",
          "unit": "ISO 8601 UTC",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Timestamp canónico universal em UTC (início da hora de entrega)",
          "source": "Calculado / OMIE & Open-Meteo",
      },
      {
          "name": "timestamp_local",
          "type": "datetime64[ns, Europe/Lisbon]",
          "unit": "Hora Local",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Timestamp local em Portugal Continental (trata DST WET/WEST)",
          "source": "Conversão Europe/Lisbon",
      },
      {
          "name": "price_pt_eur_mwh",
          "type": "float64",
          "unit": "EUR/MWh",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Preço marginal horário no mercado diário MIBEL para Portugal (Variável-Alvo)",
          "source": "OMIE (marginalpdbc)",
      },
      {
          "name": "price_sp_eur_mwh",
          "type": "float64",
          "unit": "EUR/MWh",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Preço marginal horário no mercado diário MIBEL para Espanha",
          "source": "OMIE (marginalpdbc)",
      },
      {
          "name": "price_spread_pt_sp",
          "type": "float64",
          "unit": "EUR/MWh",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Diferencial de preço entre Portugal e Espanha (indicador de desacoplamento/congestão)",
          "source": "Cálculo derivado (Preço PT - Preço ES)",
      },
      {
          "name": "temp_pt_c",
          "type": "float64",
          "unit": "°C",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Temperatura ambiente ponderada para Portugal a 2m (proxy ERA5 de previsão NWP)",
          "source": "Open-Meteo ERA5 (Proxy)",
      },
      {
          "name": "wind_speed_100m_pt_kmh",
          "type": "float64",
          "unit": "km/h",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Velocidade do vento ponderada à cota de 100m (proxy ERA5 de previsão NWP)",
          "source": "Open-Meteo ERA5 (Proxy)",
      },
      {
          "name": "solar_radiation_pt_wm2",
          "type": "float64",
          "unit": "W/m²",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Radiação solar direta incidente ponderada à superfície (proxy ERA5 de previsão NWP)",
          "source": "Open-Meteo ERA5 (Proxy)",
      },
      {
          "name": "cloud_cover_pt_pct",
          "type": "float64",
          "unit": "%",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Percentagem de cobertura de nuvens ponderada (proxy ERA5 de previsão NWP)",
          "source": "Open-Meteo ERA5 (Proxy)",
      },
      {
          "name": "precipitation_pt_mm",
          "type": "float64",
          "unit": "mm/h",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Precipitação horária ponderada em Portugal (usada apenas com lag >= 24h)",
          "source": "Open-Meteo ERA5",
      },
      {
          "name": "humidity_pt_pct",
          "type": "float64",
          "unit": "%",
          "relevance_tier": "Tier 2 (Relevância Alta)",
          "description": "Humidade relativa ponderada a 2m em Portugal",
          "source": "Open-Meteo ERA5",
      },
      {
          "name": "ttf_gas_eur_mwh",
          "type": "float64",
          "unit": "EUR/MWh",
          "relevance_tier": "Tier 1 (Relevância Crítica)",
          "description": "Preço de fecho diário dos contratos de futuros de gás natural Dutch TTF",
          "source": "ICE / Yahoo Finance (TTF=F)",
      },
      {
          "name": "carbon_proxy_usd",
          "type": "float64",
          "unit": "USD",
          "relevance_tier": "Tier 2 (Relevância Alta)",
          "description": "Preço de fecho em USD do ETF de licenças de emissão de carbono",
          "source": "KraneShares / Yahoo Finance (KRBN)",
      },
  ]
  dict_path = interim_dir / "data_dictionary.csv"
  pd.DataFrame(dict_records).to_csv(dict_path, index=False, encoding="utf-8")

  return df_base, output_parquet, dict_path
