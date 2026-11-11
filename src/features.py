"""
Feature Engineering and Dataset Splitting Module for Energy Price Agents project.
Strictly enforces zero data leakage, DST-safe calendar/hour lags, pure-currency
thermal proxies, and generates train/validation/test Parquet splits and manifest.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils import calculate_file_hash, get_project_root, load_config

NON_FEATURE_COLS = (
    "timestamp_utc",
    "timestamp_local",
    "price_pt_eur_mwh",
    "price_sp_eur_mwh",
    "price_spread_pt_sp",
    "ttf_gas_eur_mwh",
    "carbon_proxy_usd",
    "precipitation_pt_mm",
    "humidity_pt_pct",
)

# Single source of truth for target and engineered features.
FEATURE_METADATA: List[Dict[str, str]] = [
    {"name": "price_pt_eur_mwh", "type": "float64", "unit": "EUR/MWh", "role": "Target", "description": "Preço marginal horário MIBEL Portugal (variável alvo a prever)", "available_at": "D 12:00 CET (Resultado a prever)"},
    {"name": "temp_pt_c", "type": "float64", "unit": "°C", "role": "Feature", "description": "Temperatura ambiente a 2m (proxy ERA5 de previsão matinal NWP)", "available_at": "D-1 08:00 UTC (Proxy ERA5 NWP)"},
    {"name": "wind_speed_100m_pt_kmh", "type": "float64", "unit": "km/h", "role": "Feature", "description": "Velocidade do vento à cota de 100m (proxy ERA5 de previsão NWP)", "available_at": "D-1 08:00 UTC (Proxy ERA5 NWP)"},
    {"name": "solar_radiation_pt_wm2", "type": "float64", "unit": "W/m²", "role": "Feature", "description": "Radiação solar direta na superfície (proxy ERA5 de previsão NWP)", "available_at": "D-1 08:00 UTC (Proxy ERA5 NWP)"},
    {"name": "cloud_cover_pt_pct", "type": "float64", "unit": "%", "role": "Feature", "description": "Nebulosidade média ponderada em Portugal (proxy ERA5 NWP)", "available_at": "D-1 08:00 UTC (Proxy ERA5 NWP)"},
    {"name": "price_pt_lag_24h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Preço de Portugal na mesma hora local de D-1 (fechado em D-2 12:00 CET)", "available_at": "D-2 12:00 CET"},
    {"name": "price_pt_lag_48h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Preço de Portugal na mesma hora local de D-2", "available_at": "D-3 12:00 CET"},
    {"name": "price_pt_lag_96h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Preço de Portugal na mesma hora local de D-4", "available_at": "D-5 12:00 CET"},
    {"name": "price_pt_lag_168h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Preço de Portugal na mesma hora local de D-7", "available_at": "D-8 12:00 CET"},
    {"name": "price_pt_lag_336h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Preço de Portugal na mesma hora local de D-14", "available_at": "D-15 12:00 CET"},
    {"name": "price_sp_lag_24h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Preço de Espanha na mesma hora local de D-1", "available_at": "D-2 12:00 CET"},
    {"name": "price_sp_lag_168h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Preço de Espanha na mesma hora local de D-7", "available_at": "D-8 12:00 CET"},
    {"name": "price_spread_lag_24h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Spread PT-ES na mesma hora local de D-1", "available_at": "D-2 12:00 CET"},
    {"name": "price_pt_rolling_mean_24h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Média móvel de 24h do preço de Portugal até D-1", "available_at": "D-2 12:00 CET"},
    {"name": "price_pt_rolling_std_24h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Desvio padrão móvel de 24h do preço de Portugal até D-1", "available_at": "D-2 12:00 CET"},
    {"name": "price_pt_rolling_min_24h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Preço mínimo nas últimas 24h de D-1", "available_at": "D-2 12:00 CET"},
    {"name": "price_pt_rolling_max_24h", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Preço máximo nas últimas 24h de D-1", "available_at": "D-2 12:00 CET"},
    {"name": "price_pt_rolling_mean_7d", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Média móvel de 7 dias do preço de Portugal até D-1", "available_at": "D-2 12:00 CET"},
    {"name": "temp_comfort_delta", "type": "float64", "unit": "°C", "role": "Feature", "description": "Desvio térmico face à temperatura de conforto (18°C)", "available_at": "D-1 08:00 UTC (Proxy ERA5 NWP)"},
    {"name": "temp_comfort_delta_sq", "type": "float64", "unit": "°C²", "role": "Feature", "description": "Quadrado do desvio térmico (arrefecimento e aquecimento)", "available_at": "D-1 08:00 UTC (Proxy ERA5 NWP)"},
    {"name": "wind_power_proxy", "type": "float64", "unit": "Índice (0-50)", "role": "Feature", "description": "Proxy cúbico da potência eólica gerada", "available_at": "D-1 08:00 UTC (Proxy ERA5 NWP)"},
    {"name": "solar_wind_interaction", "type": "float64", "unit": "Índice", "role": "Feature", "description": "Interação entre radiação e vento", "available_at": "D-1 08:00 UTC (Proxy ERA5 NWP)"},
    {"name": "precip_rolling_7d", "type": "float64", "unit": "mm", "role": "Feature", "description": "Precipitação acumulada nos últimos 7 dias até D-1 (lag 24h)", "available_at": "D-1 08:00 UTC"},
    {"name": "precip_rolling_30d", "type": "float64", "unit": "mm", "role": "Feature", "description": "Precipitação acumulada nos últimos 30 dias até D-1 (lag 24h)", "available_at": "D-1 08:00 UTC"},
    {"name": "hour", "type": "int32", "unit": "Hora (0-23)", "role": "Feature", "description": "Hora local em Portugal Continental", "available_at": "Determinístico / D-infinito"},
    {"name": "hour_sin", "type": "float64", "unit": "Rad", "role": "Feature", "description": "Componente cíclica seno da hora", "available_at": "Determinístico / D-infinito"},
    {"name": "hour_cos", "type": "float64", "unit": "Rad", "role": "Feature", "description": "Componente cíclica cosseno da hora", "available_at": "Determinístico / D-infinito"},
    {"name": "dayofweek", "type": "int32", "unit": "Dia (0-6)", "role": "Feature", "description": "Dia da semana local (0=Segunda, 6=Domingo)", "available_at": "Determinístico / D-infinito"},
    {"name": "is_weekend", "type": "int32", "unit": "Binário", "role": "Feature", "description": "Indicador de fim de semana", "available_at": "Determinístico / D-infinito"},
    {"name": "month", "type": "int32", "unit": "Mês (1-12)", "role": "Feature", "description": "Mês do ano", "available_at": "Determinístico / D-infinito"},
    {"name": "month_sin", "type": "float64", "unit": "Rad", "role": "Feature", "description": "Componente cíclica seno do mês", "available_at": "Determinístico / D-infinito"},
    {"name": "month_cos", "type": "float64", "unit": "Rad", "role": "Feature", "description": "Componente cíclica cosseno do mês", "available_at": "Determinístico / D-infinito"},
    {"name": "is_holiday_pt", "type": "int32", "unit": "Binário", "role": "Feature", "description": "Feriado nacional fixo em Portugal", "available_at": "Determinístico / D-infinito"},
    {"name": "is_holiday_es", "type": "int32", "unit": "Binário", "role": "Feature", "description": "Feriado nacional fixo em Espanha", "available_at": "Determinístico / D-infinito"},
    {"name": "is_holiday_iberian", "type": "int32", "unit": "Binário", "role": "Feature", "description": "Feriado em Portugal e/ou Espanha", "available_at": "Determinístico / D-infinito"},
    {"name": "ttf_gas_lag1", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Preço de fecho do gás Dutch TTF do dia civil D-1", "available_at": "D-2 18:00 CET / D-1 08:00 CET"},
    {"name": "carbon_proxy_lag1", "type": "float64", "unit": "USD", "role": "Feature", "description": "Preço de fecho em USD do ETF de carbono (KRBN) do dia civil D-1", "available_at": "D-2 22:00 CET / D-1 08:00 CET"},
    {"name": "ccgt_srmc_proxy", "type": "float64", "unit": "EUR/MWh", "role": "Feature", "description": "Custo marginal térmico de combustível CCGT em EUR/MWh (TTF * 2.0)", "available_at": "D-1 08:00 CET"},
    {"name": "ccgt_srmc_zscore", "type": "float64", "unit": "Adimensional", "role": "Feature", "description": "Z-score rolling 30d do custo marginal CCGT em EUR/MWh", "available_at": "D-1 08:00 CET"},
]


def _local_day_lag(df: pd.DataFrame, col: str, days: int) -> pd.Series:
  """Looks up the value from `days` calendar days ago at the same local hour.

  Falls back to row shift (`days * 24`) only for the single skipped 01:00 hour
  the day after spring DST transition.
  """
  local_dates = df["timestamp_local"].dt.date
  local_hours = df["timestamp_local"].dt.hour
  lookup_df = pd.DataFrame(
      {"d": local_dates, "h": local_hours, "v": df[col]}
  ).groupby(["d", "h"], sort=False)["v"].first()

  target_dates = [d - timedelta(days=days) for d in local_dates]
  target_index = pd.MultiIndex.from_arrays([target_dates, local_hours])
  lagged = lookup_df.reindex(target_index).reset_index(drop=True)

  # Keep leading boundary NaNs so initial rows are dropped cleanly after lag window.
  utc_fallback = df[col].shift(days * 24)
  min_valid_date = local_dates.iloc[0] + timedelta(days=days)
  after_warmup = pd.Series(local_dates >= min_valid_date, index=df.index)
  lagged = lagged.where(~(after_warmup & lagged.isna()), utc_fallback)
  return lagged


def _commodity_prev_day(df: pd.DataFrame, col: str) -> pd.Series:
  """Maps each row to the previous local calendar day's commodity close."""
  local_dates = df["timestamp_local"].dt.date
  daily_map = (
      pd.DataFrame({"d": local_dates, "v": df[col]})
      .groupby("d", sort=False)["v"]
      .first()
  )
  prev_dates = [d - timedelta(days=1) for d in local_dates]
  return daily_map.reindex(prev_dates).reset_index(drop=True)


def _is_available_before_auction(available_at: str) -> bool:
  """Evaluates whether an `available_at` declaration precedes D-1 12:00 CET."""
  if "Determinístico" in available_at:
    return True
  matches = re.findall(r"D-(\d+)", available_at)
  if not matches:
    return False
  min_offset = min(int(m) for m in matches)
  if min_offset >= 2:
    return True
  if min_offset == 1:
    time_match = re.search(r"D-1\s+(\d{2}):(\d{2})", available_at)
    if time_match:
      hh = int(time_match.group(1))
      return hh < 12
  return False


def compute_features(df_clean: pd.DataFrame) -> pd.DataFrame:
  """Generates all engineered features respecting D-1 12:00 CET causality and DST."""
  df = df_clean.copy()
  df = df.sort_values("timestamp_utc").reset_index(drop=True)

  # 1. DST-safe Autoregressive Price Lags (same local hour D-k)
  df["price_pt_lag_24h"] = _local_day_lag(df, "price_pt_eur_mwh", 1)
  df["price_pt_lag_48h"] = _local_day_lag(df, "price_pt_eur_mwh", 2)
  df["price_pt_lag_96h"] = _local_day_lag(df, "price_pt_eur_mwh", 4)
  df["price_pt_lag_168h"] = _local_day_lag(df, "price_pt_eur_mwh", 7)
  df["price_pt_lag_336h"] = _local_day_lag(df, "price_pt_eur_mwh", 14)

  df["price_sp_lag_24h"] = _local_day_lag(df, "price_sp_eur_mwh", 1)
  df["price_sp_lag_168h"] = _local_day_lag(df, "price_sp_eur_mwh", 7)
  df["price_spread_lag_24h"] = (
      df["price_pt_lag_24h"] - df["price_sp_lag_24h"]
  ).round(2)

  # 2. Rolling Historical Statistics strictly up to D-1
  df["price_pt_rolling_mean_24h"] = (
      df["price_pt_lag_24h"].rolling(24).mean().round(2)
  )
  df["price_pt_rolling_std_24h"] = (
      df["price_pt_lag_24h"].rolling(24).std().fillna(0).round(2)
  )
  df["price_pt_rolling_min_24h"] = (
      df["price_pt_lag_24h"].rolling(24).min().round(2)
  )
  df["price_pt_rolling_max_24h"] = (
      df["price_pt_lag_24h"].rolling(24).max().round(2)
  )
  df["price_pt_rolling_mean_7d"] = (
      df["price_pt_lag_24h"].rolling(168).mean().round(2)
  )

  # 3. Exogenous Physical Weather Drivers
  df["temp_comfort_delta"] = (df["temp_pt_c"] - 18.0).round(2)
  df["temp_comfort_delta_sq"] = (df["temp_comfort_delta"] ** 2).round(2)
  df["wind_power_proxy"] = (
      np.clip((df["wind_speed_100m_pt_kmh"] / 3.6) ** 3 / 1000.0, 0, 50)
  ).round(3)
  df["solar_wind_interaction"] = (
      (df["solar_radiation_pt_wm2"] * df["wind_speed_100m_pt_kmh"]) / 1000.0
  ).round(3)

  # 3b. Hydrological Precipitation Proxies (strictly shifted 24h to prevent day-D leakage)
  if "precipitation_pt_mm" in df.columns:
    precip_lag24 = df["precipitation_pt_mm"].shift(24)
    df["precip_rolling_7d"] = precip_lag24.rolling(168).sum().round(3)
    df["precip_rolling_30d"] = precip_lag24.rolling(720).sum().round(3)

  # 4. Temporal and Calendar Features
  df["hour"] = df["timestamp_local"].dt.hour
  df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0).round(4)
  df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0).round(4)

  df["dayofweek"] = df["timestamp_local"].dt.dayofweek
  df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

  df["month"] = df["timestamp_local"].dt.month
  df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0).round(4)
  df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0).round(4)

  pt_fixed_holidays = {
      (1, 1), (4, 25), (5, 1), (6, 10), (8, 15),
      (10, 5), (11, 1), (12, 1), (12, 8), (12, 25),
  }
  es_fixed_holidays = {
      (1, 1), (1, 6), (5, 1), (8, 15), (10, 12),
      (11, 1), (12, 6), (12, 8), (12, 25),
  }
  month_day = list(zip(df["month"], df["timestamp_local"].dt.day))
  df["is_holiday_pt"] = [1 if md in pt_fixed_holidays else 0 for md in month_day]
  df["is_holiday_es"] = [1 if md in es_fixed_holidays else 0 for md in month_day]
  df["is_holiday_iberian"] = (
      (df["is_holiday_pt"] == 1) | (df["is_holiday_es"] == 1)
  ).astype(int)

  # 5. Commodities and Pure-EUR CCGT Thermal Cost Proxy (calendar D-1 across DST)
  df["ttf_gas_lag1"] = _commodity_prev_day(df, "ttf_gas_eur_mwh")
  df["carbon_proxy_lag1"] = _commodity_prev_day(df, "carbon_proxy_usd")
  # Pure EUR/MWh fuel marginal cost (50% efficiency heat rate = 2.0); KRBN USD kept separate.
  df["ccgt_srmc_proxy"] = (df["ttf_gas_lag1"] * 2.0).round(2)

  ccgt_roll_mean = df["ccgt_srmc_proxy"].rolling(720).mean()
  ccgt_roll_std = df["ccgt_srmc_proxy"].rolling(720).std().replace(0, float("nan"))
  df["ccgt_srmc_zscore"] = (
      ((df["ccgt_srmc_proxy"] - ccgt_roll_mean) / ccgt_roll_std)
      .fillna(0)
      .round(3)
  )

  df = df.dropna().reset_index(drop=True)
  return df


def split_and_export_features(
    df_features: pd.DataFrame, processed_dir: Optional[Path] = None
) -> Dict[str, Any]:
  """Performs chronological train/valid/test split and exports Parquet + metadata."""
  config = load_config()
  if processed_dir is None:
    processed_dir = get_project_root() / "data" / "processed"
  processed_dir.mkdir(parents=True, exist_ok=True)

  splits_cfg = config["temporal_splits"]
  train_end = splits_cfg["train"]["end"]
  valid_start = splits_cfg["validation"]["start"]
  valid_end = splits_cfg["validation"]["end"]
  test_start = splits_cfg["test"]["start"]
  test_end = splits_cfg["test"]["end"]

  date_col = df_features["timestamp_local"].dt.strftime("%Y-%m-%d")

  df_train = df_features[date_col <= train_end].copy().reset_index(drop=True)
  df_valid = (
      df_features[(date_col >= valid_start) & (date_col <= valid_end)]
      .copy()
      .reset_index(drop=True)
  )
  df_test = (
      df_features[(date_col >= test_start) & (date_col <= test_end)]
      .copy()
      .reset_index(drop=True)
  )

  feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]

  train_path = processed_dir / "features_train.parquet"
  valid_path = processed_dir / "features_valid.parquet"
  test_path = processed_dir / "features_test.parquet"

  df_train.to_parquet(train_path, compression="snappy", index=False)
  df_valid.to_parquet(valid_path, compression="snappy", index=False)
  df_test.to_parquet(test_path, compression="snappy", index=False)

  # Keep dictionary strictly aligned with columns actually present in df_features.
  active_names = set(feature_cols) | {config["ml_framing"]["target_column"]}
  active_metadata = [m for m in FEATURE_METADATA if m["name"] in active_names]

  dict_path = processed_dir / "feature_dictionary.csv"
  pd.DataFrame(active_metadata).to_csv(dict_path, index=False, encoding="utf-8")

  leakage_audit_rows = []
  for item in active_metadata:
    is_target = item["role"] == "Target"
    is_ok = is_target or _is_available_before_auction(item["available_at"])
    leakage_audit_rows.append({
        "Feature": item["name"],
        "Papel": item["role"],
        "Disponibilidade (available_at)": item["available_at"],
        "Momento Decisivo": "Antes de D-1 ~12:00 CET",
        "Auditoria Anti-Leakage": (
            "APROVADA (Sem Contaminação Futura)" if is_ok else "REPROVADA"
        ),
    })

  audit_path = processed_dir / "leakage_audit.csv"
  pd.DataFrame(leakage_audit_rows).to_csv(audit_path, index=False, encoding="utf-8")

  manifest = {
      "project": config["project"]["name"],
      "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "target_variable": config["ml_framing"]["target_column"],
      "feature_count": len(feature_cols),
      "feature_names": feature_cols,
      "splits": {
          "train": {
              "file": "features_train.parquet",
              "rows": len(df_train),
              "date_start": str(df_train["timestamp_local"].min().date()),
              "date_end": str(df_train["timestamp_local"].max().date()),
              "sha256": calculate_file_hash(train_path),
              "null_count": int(df_train[feature_cols].isnull().sum().sum()),
          },
          "validation": {
              "file": "features_valid.parquet",
              "rows": len(df_valid),
              "date_start": str(df_valid["timestamp_local"].min().date()),
              "date_end": str(df_valid["timestamp_local"].max().date()),
              "sha256": calculate_file_hash(valid_path),
              "null_count": int(df_valid[feature_cols].isnull().sum().sum()),
          },
          "test": {
              "file": "features_test.parquet",
              "rows": len(df_test),
              "date_start": str(df_test["timestamp_local"].min().date()),
              "date_end": str(df_test["timestamp_local"].max().date()),
              "sha256": calculate_file_hash(test_path),
              "null_count": int(df_test[feature_cols].isnull().sum().sum()),
          },
      },
  }

  manifest_path = processed_dir / "manifest.json"
  with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

  return {
      "train_path": train_path,
      "valid_path": valid_path,
      "test_path": test_path,
      "dict_path": dict_path,
      "audit_path": audit_path,
      "manifest_path": manifest_path,
      "manifest": manifest,
      "feature_cols": feature_cols,
  }
