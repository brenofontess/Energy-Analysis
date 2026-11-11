"""Automated regression tests for Energy Analysis pipeline (uses synthetic data & tmp_path)."""

import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from scripts.canvas_server import CanvasHTTPRequestHandler, is_allowed_static_path
from src.features import compute_features, split_and_export_features
from src.ingestion import parse_omie_file, summarize_missing_gaps
from src.modeling import compute_dynamic_blend, diebold_mariano_test, get_feature_columns
from src.utils import log_correction


def test_parse_omie_file_mtu15_missing_period_does_not_shift_hours(tmp_path: Path):
  # Skip period 2 (hour 0) and keep periods 5..8 (hour 1) intact.
  lines = [
      "MARGINALPDBC;",
      "2025;10;01;1;40.0;40.0;",
      "2025;10;01;3;44.0;44.0;",
      "2025;10;01;4;48.0;48.0;",
      "2025;10;01;5;100.0;100.0;",
      "2025;10;01;6;100.0;100.0;",
      "2025;10;01;7;100.0;100.0;",
      "2025;10;01;8;100.0;100.0;",
      "*",
  ]
  fpath = tmp_path / "marginalpdbc_20251001.1"
  fpath.write_text("\n".join(lines), encoding="utf-8")

  rows = parse_omie_file(fpath)
  assert len(rows) == 2
  assert rows[0]["price_pt_eur_mwh"] == 44.0
  assert rows[1]["price_pt_eur_mwh"] == 100.0
  assert rows[1]["timestamp_utc"] - rows[0]["timestamp_utc"] == pd.Timedelta(hours=1)


def test_parse_omie_file_dst_23h_and_25h(tmp_path: Path):
  # Spring forward (2024-03-31): 23 hourly periods; Autumn fallback (2024-10-27): 25 hourly periods.
  for fname, y, m, d, n_hours in [
      ("marginalpdbc_20240331.1", 2024, 3, 31, 23),
      ("marginalpdbc_20241027.1", 2024, 10, 27, 25),
  ]:
    lines = ["MARGINALPDBC;"] + [
        f"{y};{m:02d};{d:02d};{p};50.0;50.0;" for p in range(1, n_hours + 1)
    ] + ["*"]
    fpath = tmp_path / fname
    fpath.write_text("\n".join(lines), encoding="utf-8")
    rows = parse_omie_file(fpath)
    assert len(rows) == n_hours
    ts = [r["timestamp_utc"] for r in rows]
    assert all(ts[i + 1] - ts[i] == pd.Timedelta(hours=1) for i in range(len(ts) - 1))


def _make_synthetic_clean_df(start: str = "2024-09-15", end: str = "2024-11-05") -> pd.DataFrame:
  ts_utc = pd.date_range(start, end, freq="h", tz="UTC")
  ts_local = ts_utc.tz_convert("Europe/Lisbon")
  n = len(ts_utc)
  # Encode calendar date in TTF price so any same-day leakage on DST day is detectable.
  local_day_num = pd.Series(ts_local).dt.strftime("%Y%m%d").astype(float).values
  rain = np.zeros(n)
  rain[-24:] = 50.0  # Rain strictly on the final day D should NOT enter D's lag-24 rolling sum.
  return pd.DataFrame({
      "timestamp_utc": ts_utc,
      "timestamp_local": ts_local,
      "price_pt_eur_mwh": np.arange(n, dtype=float),
      "price_sp_eur_mwh": np.arange(n, dtype=float) + 1.0,
      "price_spread_pt_sp": np.full(n, -1.0),
      "temp_pt_c": np.full(n, 18.0),
      "wind_speed_100m_pt_kmh": np.full(n, 15.0),
      "solar_radiation_pt_wm2": np.full(n, 200.0),
      "cloud_cover_pt_pct": np.full(n, 30.0),
      "precipitation_pt_mm": rain,
      "humidity_pt_pct": np.full(n, 60.0),
      "ttf_gas_eur_mwh": local_day_num,
      "carbon_proxy_usd": np.full(n, 30.0),
  })


def test_compute_features_dst_and_no_rain_leakage():
  df_clean = _make_synthetic_clean_df()
  df_feats = compute_features(df_clean)

  assert df_feats.isnull().sum().sum() == 0

  # 1. On Portugal's 25h DST day (2024-10-27), ALL 25 hours must get 2024-10-26's TTF close.
  oct27 = df_feats[df_feats["timestamp_local"].dt.strftime("%Y-%m-%d") == "2024-10-27"]
  assert len(oct27) == 25
  assert (oct27["ttf_gas_lag1"] == 20241026.0).all()

  # 2. Same-local-hour lag on 2024-10-28 10:00 must match 2024-10-27 10:00 (25 UTC hours earlier).
  row_oct28_10 = df_feats[
      (df_feats["timestamp_local"].dt.strftime("%Y-%m-%d %H") == "2024-10-28 10")
  ].iloc[0]
  val_oct27_10 = df_feats[
      (df_feats["timestamp_local"].dt.strftime("%Y-%m-%d %H") == "2024-10-27 10")
  ].iloc[0]["price_pt_eur_mwh"]
  assert row_oct28_10["price_pt_lag_24h"] == val_oct27_10

  # 3. Pure-EUR CCGT proxy equals ttf_gas_lag1 * 2.0 (no cross-currency USD addition).
  assert np.allclose(df_feats["ccgt_srmc_proxy"], df_feats["ttf_gas_lag1"] * 2.0)

  # 4. Precipitation rolling 7d at final hour must NOT include final day's rain (shifted 24h).
  assert df_feats["precip_rolling_7d"].iloc[-1] == 0.0


def test_split_and_export_leakage_audit_zero_failures(tmp_path: Path):
  df_clean = _make_synthetic_clean_df(start="2024-01-01", end="2026-09-30")
  df_feats = compute_features(df_clean)
  res = split_and_export_features(df_feats, processed_dir=tmp_path)

  df_audit = pd.read_csv(res["audit_path"])
  assert (df_audit["Auditoria Anti-Leakage"] == "REPROVADA").sum() == 0

  df_dict = pd.read_csv(res["dict_path"])
  feature_names_in_dict = set(df_dict[df_dict["role"] == "Feature"]["name"])
  assert feature_names_in_dict == set(res["manifest"]["feature_names"])

  # Verify manifest feature count matches Parquet columns excluding non-features.
  df_test = pd.read_parquet(res["test_path"])
  assert get_feature_columns(df_test, res["manifest"]) == res["manifest"]["feature_names"]


def test_log_correction_escapes_commas_and_quotes(tmp_path: Path):
  csv_path = tmp_path / "correction_log.csv"
  log_correction(
      correction_id="CORR_001",
      column="price_pt_eur_mwh, price_sp_eur_mwh",
      affected_rows=48,
      method='Reindex + "D-7"',
      justification="Gap on 2025-10-30, 2025-11-27",
      before_summary="48 nulls",
      after_summary="0 nulls",
      approved_by="Agente 4",
      log_csv_path=csv_path,
  )
  df = pd.read_csv(csv_path)
  assert len(df) == 1
  assert list(df.columns) == [
      "id",
      "timestamp_utc",
      "column",
      "affected_rows",
      "method",
      "justification",
      "before_summary",
      "after_summary",
      "approved_by",
  ]
  assert df.loc[0, "id"] == "CORR_001"
  assert df.loc[0, "column"] == "price_pt_eur_mwh, price_sp_eur_mwh"
  assert df.loc[0, "affected_rows"] == 48


def test_summarize_missing_gaps_identifies_separate_days():
  idx = pd.DatetimeIndex([
      "2025-10-30 00:00:00+00:00",
      "2025-10-30 01:00:00+00:00",
      "2025-11-27 00:00:00+00:00",
  ])
  gaps = summarize_missing_gaps(idx)
  assert len(gaps) == 2


def test_dynamic_blend_and_diebold_mariano():
  blend = compute_dynamic_blend(
      np.array([10.0, 20.0, 30.0]),
      np.array([20.0, 10.0, 10.0]),
      np.array([3, 13, 20]),
  )
  assert np.allclose(blend, [16.5, 18.0, 22.0])
  dm, p = diebold_mariano_test(
      np.array([10.0, 20.0, 30.0, 40.0]),
      np.array([11.0, 21.0, 28.5, 39.0]),
      np.array([15.0, 26.0, 38.0, 49.0]),
  )
  assert dm < 0 and p < 0.05


def test_canvas_server_blocks_dotfiles_and_preserves_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  import importlib
  import scripts.canvas_server as cs

  monkeypatch.chdir(tmp_path)
  importlib.reload(cs)
  assert Path.cwd().resolve() == tmp_path.resolve()
  assert is_allowed_static_path("/.env") is False
  assert is_allowed_static_path("/.git/config") is False
  assert is_allowed_static_path("/reports/workflow_canvas.html") is True
  assert hasattr(CanvasHTTPRequestHandler, "do_GET")
