"""
Shared Modeling, Cross-Validation, Tournament, and Evaluation Module.
Used by both Portuguese and English modeling/analysis notebooks so results stay identical.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from src.features import NON_FEATURE_COLS


def get_feature_columns(
    df: pd.DataFrame, manifest: Optional[Dict[str, Any]] = None
) -> List[str]:
  """Returns ordered active feature columns matching manifest when provided."""
  if manifest and "feature_names" in manifest:
    return [c for c in manifest["feature_names"] if c in df.columns]
  return [c for c in df.columns if c not in NON_FEATURE_COLS]


def calculate_mase(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_baseline_d1: np.ndarray | pd.Series,
) -> float:
  """Computes Mean Absolute Scaled Error relative to D-1 persistence."""
  mae_model = mean_absolute_error(y_true, y_pred)
  mae_base = mean_absolute_error(y_true, y_baseline_d1)
  return float(mae_model / mae_base) if mae_base > 0 else float("nan")


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_d1: Optional[np.ndarray | pd.Series] = None,
) -> Dict[str, float]:
  """Computes MAE, MASE, RMSE, R2, and mean bias (pred - actual)."""
  mae = float(mean_absolute_error(y_true, y_pred))
  rmse = float(root_mean_squared_error(y_true, y_pred))
  r2 = float(r2_score(y_true, y_pred))
  bias = float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))
  mase = calculate_mase(y_true, y_pred, y_d1) if y_d1 is not None else 1.0
  return {
      "mae": round(mae, 2),
      "mase": round(mase, 3),
      "rmse": round(rmse, 2),
      "r2": round(r2, 3),
      "bias": round(bias, 2),
  }


def compute_dynamic_blend(
    pred_xgb: np.ndarray, lag_d1: np.ndarray, hours: np.ndarray
) -> np.ndarray:
  """Combines XGBoost and D-1 persistence using regime-specific hourly weights."""
  h = np.asarray(hours)
  w_xgb = np.full(len(h), 0.50, dtype=float)
  w_xgb[(h >= 1) & (h <= 6)] = 0.35
  w_xgb[(h >= 11) & (h <= 16)] = 0.80
  w_xgb[(h >= 19) & (h <= 22)] = 0.60
  return w_xgb * np.asarray(pred_xgb) + (1.0 - w_xgb) * np.asarray(lag_d1)


def diebold_mariano_test(
    y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray
) -> Tuple[float, float]:
  """Two-sided Diebold-Mariano test on absolute forecast errors."""
  d = np.abs(y_true - y_pred1) - np.abs(y_true - y_pred2)
  n = len(d)
  if n < 2:
    return 0.0, 1.0
  gamma0 = float(np.var(d, ddof=1))
  if gamma0 <= 0:
    return 0.0, 1.0
  dm_stat = float(np.mean(d) / np.sqrt(gamma0 / n))
  p_val = float(2.0 * scipy_stats.norm.cdf(-abs(dm_stat)))
  return dm_stat, p_val


def run_walkforward_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    gap: int = 24,
) -> pd.DataFrame:
  """Runs forward-chaining TimeSeriesSplit CV with Pseudo-Huber XGBoost."""
  tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
  rows = []
  for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), 1):
    X_f_tr, y_f_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_f_va, y_f_va = X_train.iloc[va_idx], y_train.iloc[va_idx]
    cv_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        objective="reg:pseudohubererror",
        huber_slope=10.0,
        random_state=42,
        seed=42,
        eval_metric="mae",
        early_stopping_rounds=30,
        n_jobs=-1,
    )
    cv_model.fit(X_f_tr, y_f_tr, eval_set=[(X_f_va, y_f_va)], verbose=False)
    preds = cv_model.predict(X_f_va)
    rows.append({
        "Fold": fold,
        "Train Rows": len(tr_idx),
        "Val Rows": len(va_idx),
        "MAE": round(float(mean_absolute_error(y_f_va, preds)), 2),
        "RMSE": round(float(root_mean_squared_error(y_f_va, preds)), 2),
        "R2": round(float(r2_score(y_f_va, preds)), 3),
    })
  return pd.DataFrame(rows)


def build_experiment_configs(
    n_experiments: int = 100, seed: int = 42
) -> List[Dict[str, Any]]:
  """Builds the deterministic hyperparameter search configurations."""
  configs = [
      {"name": "Exp_001_Baseline_Default", "max_depth": 5, "learning_rate": 0.030, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.5, "reg_lambda": 2.0, "min_child_weight": 1},
      {"name": "Exp_002_Shallow_Conservative", "max_depth": 4, "learning_rate": 0.040, "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.2, "reg_lambda": 1.5, "min_child_weight": 2},
      {"name": "Exp_003_Deep_Interactions", "max_depth": 6, "learning_rate": 0.025, "subsample": 0.85, "colsample_bytree": 0.80, "reg_alpha": 1.0, "reg_lambda": 3.0, "min_child_weight": 3},
      {"name": "Exp_004_High_L2_Regularization", "max_depth": 5, "learning_rate": 0.030, "subsample": 0.80, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 5.0, "min_child_weight": 2},
      {"name": "Exp_005_Low_Learning_Rate_Robust", "max_depth": 6, "learning_rate": 0.015, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.8, "reg_lambda": 2.5, "min_child_weight": 1},
      {"name": "Exp_006_Feature_Subsampling", "max_depth": 5, "learning_rate": 0.035, "subsample": 0.85, "colsample_bytree": 0.70, "reg_alpha": 1.2, "reg_lambda": 2.0, "min_child_weight": 2},
      {"name": "Exp_007_L1_Sparsity_Focus", "max_depth": 6, "learning_rate": 0.030, "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 2.5, "reg_lambda": 1.0, "min_child_weight": 3},
      {"name": "Exp_008_High_Capacity_Deep", "max_depth": 7, "learning_rate": 0.020, "subsample": 0.90, "colsample_bytree": 0.75, "reg_alpha": 1.0, "reg_lambda": 3.5, "min_child_weight": 4},
      {"name": "Exp_009_Compact_Fast", "max_depth": 4, "learning_rate": 0.050, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 1},
      {"name": "Exp_010_Heavy_Regularized_Blend", "max_depth": 5, "learning_rate": 0.025, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 2.0, "reg_lambda": 4.0, "min_child_weight": 3},
      {"name": "Exp_011_Moderate_Depth_Subsample", "max_depth": 6, "learning_rate": 0.030, "subsample": 0.90, "colsample_bytree": 0.85, "reg_alpha": 0.5, "reg_lambda": 2.0, "min_child_weight": 2},
      {"name": "Exp_012_Ultra_Deep_Peaks", "max_depth": 8, "learning_rate": 0.015, "subsample": 0.80, "colsample_bytree": 0.70, "reg_alpha": 1.5, "reg_lambda": 4.0, "min_child_weight": 5},
  ]
  if n_experiments > len(configs):
    rng = np.random.RandomState(seed)
    for extra_i in range(len(configs) + 1, n_experiments + 1):
      configs.append({
          "name": f"Exp_{extra_i:03d}_Systematic_Search",
          "max_depth": int(rng.choice([3, 4, 5, 6, 7, 8, 9, 10])),
          "learning_rate": float(np.round(rng.uniform(0.015, 0.06), 4)),
          "subsample": float(np.round(rng.uniform(0.65, 0.95), 2)),
          "colsample_bytree": float(np.round(rng.uniform(0.60, 0.90), 2)),
          "reg_alpha": float(np.round(rng.uniform(0.0, 4.0), 2)),
          "reg_lambda": float(np.round(rng.uniform(0.5, 8.0), 2)),
          "min_child_weight": int(rng.choice([1, 2, 3, 4, 5, 6, 8])),
      })
  return configs[:n_experiments]


def run_tournament(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    d1_valid: pd.Series,
    n_experiments: int = 100,
) -> Tuple[xgb.XGBRegressor, pd.DataFrame, Dict[str, Any]]:
  """Runs N XGBoost experiments with early stopping on validation and returns winner."""
  configs = build_experiment_configs(n_experiments=n_experiments)
  records: List[Dict[str, Any]] = []
  best_model: Optional[xgb.XGBRegressor] = None
  best_score = (float("inf"), float("-inf"))

  for idx, p in enumerate(configs, 1):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_alpha=p["reg_alpha"],
        reg_lambda=p["reg_lambda"],
        min_child_weight=p["min_child_weight"],
        objective="reg:pseudohubererror",
        huber_slope=10.0,
        random_state=42,
        seed=42,
        eval_metric="mae",
        early_stopping_rounds=40,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=False,
    )
    pred_v = model.predict(X_valid)
    pred_tr = model.predict(X_train)

    mae_v = float(mean_absolute_error(y_valid, pred_v))
    rmse_v = float(root_mean_squared_error(y_valid, pred_v))
    r2_v = float(r2_score(y_valid, pred_v))
    mase_v = calculate_mase(y_valid, pred_v, d1_valid)

    mae_tr = float(mean_absolute_error(y_train, pred_tr))
    rmse_tr = float(root_mean_squared_error(y_train, pred_tr))
    r2_tr = float(r2_score(y_train, pred_tr))
    best_iter = int(model.best_iteration)
    overfit = mae_v / mae_tr if mae_tr > 0 else 1.0

    rec = {
        "exp_id": idx,
        "name": p["name"],
        "max_depth": int(p["max_depth"]),
        "learning_rate": float(p["learning_rate"]),
        "subsample": float(p["subsample"]),
        "colsample_bytree": float(p["colsample_bytree"]),
        "reg_alpha": float(p["reg_alpha"]),
        "reg_lambda": float(p["reg_lambda"]),
        "min_child_weight": int(p["min_child_weight"]),
        "best_iteration": best_iter,
        "train_mae": round(mae_tr, 2),
        "train_rmse": round(rmse_tr, 2),
        "train_r2": round(r2_tr, 3),
        "val_mae": round(mae_v, 2),
        "val_mase": round(mase_v, 3),
        "val_rmse": round(rmse_v, 2),
        "val_r2": round(r2_v, 3),
        "overfitting_ratio": round(overfit, 2),
    }
    records.append(rec)

    score = (rec["val_mae"], -rec["val_r2"])
    if score < best_score:
      best_score = score
      best_model = model

  df_lb = (
      pd.DataFrame(records)
      .sort_values(by=["val_mae", "val_r2"], ascending=[True, False])
      .reset_index(drop=True)
  )
  df_lb.insert(0, "Ranking", range(1, len(df_lb) + 1))
  winning_record = df_lb.iloc[0].to_dict()
  assert best_model is not None
  return best_model, df_lb, winning_record


def run_modeling_pipeline(
    project_root: Path, n_experiments: int = 100
) -> Dict[str, Any]:
  """Executes full modeling pipeline and persists model, predictions, and JSON logs."""
  data_dir = project_root / "data" / "processed"
  logs_dir = project_root / "logs"
  models_dir = project_root / "models"
  logs_dir.mkdir(parents=True, exist_ok=True)
  models_dir.mkdir(parents=True, exist_ok=True)

  with open(data_dir / "manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

  target_col = manifest["target_variable"]
  df_train = pd.read_parquet(data_dir / "features_train.parquet")
  df_valid = pd.read_parquet(data_dir / "features_valid.parquet")
  df_test = pd.read_parquet(data_dir / "features_test.parquet")

  assert len(df_train) == manifest["splits"]["train"]["rows"]
  assert len(df_valid) == manifest["splits"]["validation"]["rows"]
  assert len(df_test) == manifest["splits"]["test"]["rows"]

  feature_cols = get_feature_columns(df_train, manifest)
  X_train, y_train = df_train[feature_cols], df_train[target_col]
  X_valid, y_valid = df_valid[feature_cols], df_valid[target_col]
  X_test, y_test = df_test[feature_cols], df_test[target_col]

  # Baselines
  ridge = Ridge(alpha=100.0, random_state=42)
  ridge.fit(X_train, y_train)
  pred_ridge_v = ridge.predict(X_valid)
  pred_ridge_t = ridge.predict(X_test)

  # Walk-forward CV
  df_cv = run_walkforward_cv(X_train, y_train, n_splits=5, gap=24)

  # 100-run tournament
  best_xgb, df_lb, winning_row = run_tournament(
      X_train,
      y_train,
      X_valid,
      y_valid,
      df_valid["price_pt_lag_24h"],
      n_experiments=n_experiments,
  )

  pred_xgb_tr = best_xgb.predict(X_train)
  pred_xgb_v = best_xgb.predict(X_valid)
  pred_xgb_t = best_xgb.predict(X_test)

  hours_val = df_valid["timestamp_local"].dt.hour.values
  hours_tst = df_test["timestamp_local"].dt.hour.values
  pred_blend_v = compute_dynamic_blend(
      pred_xgb_v, df_valid["price_pt_lag_24h"].values, hours_val
  )
  pred_blend_t = compute_dynamic_blend(
      pred_xgb_t, df_test["price_pt_lag_24h"].values, hours_tst
  )

  # Save model & predictions
  model_path = models_dir / "xgboost_mibel_best.json"
  best_xgb.save_model(str(model_path))

  df_predictions = pd.DataFrame({
      "timestamp_utc": df_test["timestamp_utc"],
      "timestamp_local": df_test["timestamp_local"],
      "price_pt_actual": y_test,
      "price_pt_baseline_d1": df_test["price_pt_lag_24h"],
      "price_pt_baseline_d7": df_test["price_pt_lag_168h"],
      "price_pt_pred_ridge": np.round(pred_ridge_t, 2),
      "price_pt_pred_xgboost": np.round(pred_xgb_t, 2),
      "price_pt_pred_blend": np.round(pred_blend_t, 2),
      "residual_xgboost": np.round(y_test - pred_xgb_t, 2),
      "residual_blend": np.round(y_test - pred_blend_t, 2),
  })
  pred_path = data_dir / "predictions.parquet"
  df_predictions.to_parquet(pred_path, compression="snappy", index=False)

  # Save leaderboard
  lb_path = logs_dir / "experiments_leaderboard.json"
  with open(lb_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "total_experiments": int(len(df_lb)),
            "selection_criteria": ["min(val_mae)", "max(val_r2)"],
            "winning_experiment_id": int(winning_row["exp_id"]),
            "leaderboard": df_lb.to_dict(orient="records"),
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

  # Unified metrics summary
  m_tr_xgb = compute_metrics(y_train, pred_xgb_tr, df_train["price_pt_lag_24h"])
  m_v_d1 = compute_metrics(y_valid, df_valid["price_pt_lag_24h"], df_valid["price_pt_lag_24h"])
  m_v_d7 = compute_metrics(y_valid, df_valid["price_pt_lag_168h"], df_valid["price_pt_lag_24h"])
  m_v_ridge = compute_metrics(y_valid, pred_ridge_v, df_valid["price_pt_lag_24h"])
  m_v_xgb = compute_metrics(y_valid, pred_xgb_v, df_valid["price_pt_lag_24h"])
  m_v_blend = compute_metrics(y_valid, pred_blend_v, df_valid["price_pt_lag_24h"])

  m_t_d1 = compute_metrics(y_test, df_test["price_pt_lag_24h"], df_test["price_pt_lag_24h"])
  m_t_d7 = compute_metrics(y_test, df_test["price_pt_lag_168h"], df_test["price_pt_lag_24h"])
  m_t_ridge = compute_metrics(y_test, pred_ridge_t, df_test["price_pt_lag_24h"])
  m_t_xgb = compute_metrics(y_test, pred_xgb_t, df_test["price_pt_lag_24h"])
  m_t_blend = compute_metrics(y_test, pred_blend_t, df_test["price_pt_lag_24h"])

  win_cfg = {
      "max_depth": int(winning_row["max_depth"]),
      "learning_rate": float(winning_row["learning_rate"]),
      "subsample": float(winning_row["subsample"]),
      "colsample_bytree": float(winning_row["colsample_bytree"]),
      "reg_alpha": float(winning_row["reg_alpha"]),
      "reg_lambda": float(winning_row["reg_lambda"]),
      "min_child_weight": int(winning_row["min_child_weight"]),
      "objective": "reg:pseudohubererror",
      "huber_slope": 10.0,
      "best_iteration": int(winning_row["best_iteration"]),
  }

  tournament_meta = {
      "total_experiments": int(len(df_lb)),
      "winning_experiment_id": int(winning_row["exp_id"]),
      "winning_exp_id": int(winning_row["exp_id"]),
      "winning_name": str(winning_row["name"]),
      "winning_config": win_cfg,
  }

  metrics_summary = {
      "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "feature_count": len(feature_cols),
      "tournament": tournament_meta,
      "experiments": tournament_meta,
      "train": {
          "xgboost": m_tr_xgb,
          "xgboost_best": m_tr_xgb,
          "xgboost_champion": m_tr_xgb,
      },
      "validation": {
          "baseline_d1": m_v_d1,
          "baseline_d7": m_v_d7,
          "ridge": m_v_ridge,
          "ridge_linear": m_v_ridge,
          "xgboost": m_v_xgb,
          "xgboost_best": m_v_xgb,
          "xgboost_champion": m_v_xgb,
          "blend": m_v_blend,
          "hybrid_blend": m_v_blend,
      },
      "test": {
          "baseline_d1": m_t_d1,
          "baseline_d7": m_t_d7,
          "ridge": m_t_ridge,
          "ridge_linear": m_t_ridge,
          "xgboost": m_t_xgb,
          "xgboost_best": m_t_xgb,
          "xgboost_champion": m_t_xgb,
          "blend": m_t_blend,
          "hybrid_blend": m_t_blend,
      },
  }
  metrics_path = logs_dir / "metrics_summary.json"
  with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

  dm_xgb_d1 = diebold_mariano_test(
      y_test.values, pred_xgb_t, df_test["price_pt_lag_24h"].values
  )
  dm_bld_d1 = diebold_mariano_test(
      y_test.values, pred_blend_t, df_test["price_pt_lag_24h"].values
  )
  dm_bld_xgb = diebold_mariano_test(y_test.values, pred_blend_t, pred_xgb_t)

  return {
      "manifest": manifest,
      "feature_cols": feature_cols,
      "df_train": df_train,
      "df_valid": df_valid,
      "df_test": df_test,
      "df_cv": df_cv,
      "df_leaderboard": df_lb,
      "winning_row": winning_row,
      "best_xgb": best_xgb,
      "df_predictions": df_predictions,
      "metrics_summary": metrics_summary,
      "dm_tests": {
          "xgb_vs_d1": dm_xgb_d1,
          "blend_vs_d1": dm_bld_d1,
          "blend_vs_xgb": dm_bld_xgb,
      },
      "paths": {
          "model": model_path,
          "predictions": pred_path,
          "leaderboard": lb_path,
          "metrics": metrics_path,
      },
  }
