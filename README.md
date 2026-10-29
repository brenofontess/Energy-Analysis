# Electricity Price Forecasting & SHAP Explainability in the Day-Ahead Market (MIBEL Portugal)
# Previsão e Explicabilidade do Preço da Eletricidade no Mercado Diário (MIBEL Portugal)

[![Metodologia CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-blue.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-green.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP%20TreeExplainer-red.svg)](https://shap.readthedocs.io/)
[![100% Local Execution](https://img.shields.io/badge/Execution-100%25%20Local-blue.svg)](README.md)
[![Zero-Secrets Certified](https://img.shields.io/badge/Security-Zero--PII%20%7C%20Zero--Tokens-success.svg)](SECURITY.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An enterprise-grade, portfolio Data Science project developed under the **CRISP-DM** methodology via a coordinated **7-agent specialized architecture**. The system forecasts and explains hourly electricity clearing prices in the Iberian wholesale day-ahead market (**MIBEL, Portugal bidding zone**) using **XGBoost** and **SHAP (Shapley Additive Explanations)**, driven entirely by real data from public APIs (OMIE, Open-Meteo, Yahoo Finance).

> **Dual Language Support:** Complete reports and documentation are provided in both **European Portuguese (`pt-PT`)** and **Technical English (`en`)**.

---

## 1. Executive Highlights & Market Insights

- **Continuous 2.75-Year Verified Horizon:** **24,094 continuous hourly timestamps** (January 1, 2024 to September 30, 2026), 100% validated against `manifest.json` with **0 missing hours, 0 null values**, and strict UTC / `Europe/Lisbon` dual indexing.
- **Harmonized 15-Minute Market Transition:** Full operational handling of the European SDAC "Big Bang" transition (October 1, 2025) from 60-minute to 15-minute market time units (MTU15), aggregated into canonical hourly blocks via verified arithmetic means.
- **Strict Anti-Leakage Feature Engineering:** 31 ex-ante features strictly limited to information available prior to the Day-Ahead auction gate closure ($D-1$ ~12:00 CET). Zero target leakage; forward-chaining temporal cross-validation (`TimeSeriesSplit`).
- **100-Experiment Optimization Tournament:** Systematic exploration of tree depths, learning rates, subsampling, and regularization penalties ($L_1$/$L_2$) per local run, with early stopping monitored exclusively on validation data.
- **Out-of-Sample Blind Test Superiority (2026):**
  - **$R^2 = 0.866$** and **MAE = 18.01 EUR/MWh** (RMSE = 24.08 EUR/MWh) on the Hybrid Blend (85% Champion XGBoost + 15% Ridge Benchmark).
  - Outperformed market baselines: $D-1$ Naive Persistence (19.34 EUR/MWh), $D-7$ Seasonal Persistence (26.73 EUR/MWh), and Linear Ridge (20.18 EUR/MWh).
  - Delivered an estimated **EUR 174,000+ annual imbalance cost reduction** for a reference 100 MW operational asset portfolio.
- **Empirical Market Laws Verified with SHAP (`TreeExplainer`):**
  1. *Solar Cannibalization:* Midday solar irradiance compresses wholesale prices by up to **$-25$ to $-30$ EUR/MWh** (*Duck Curve*).
  2. *Wind Power Dampening:* High wind generation dampens marginal prices by up to **$-20$ EUR/MWh**, creating single-digit night prices.
  3. *Thermal Evening Ramp:* Natural gas (Dutch TTF) and EU ETS CO₂ allowances set the marginal clearing floor during sunset peaks (**120-180+ EUR/MWh**).
  4. *Structural Persistence:* $D-1$ inertia acts as an anchor while allowing non-linear tree splits during abrupt weather shifts.
- **Actionable Energy Contract Strategies:** Detailed contracting and consumption guidance tailored to **Industry**, **Commerce & Services**, and **Residential** segments.

---

## 2. Multi-Agent CRISP-DM Architecture & Deliverables

The project is structured around 7 specialized autonomous agents, with artifacts published in both Portuguese and English:

| CRISP-DM Phase | Agent | Portuguese Notebook | English Notebook | Mission & Quality Gate |
|---|---|---|---|---|
| **Global Governance** | Coordinator Agent | [`00_coordenador_plano.ipynb`](reports/00_coordenador_plano.ipynb) | — | Master roadmap, risk matrix, technical contracts & Gate 0. |
| **Business Understanding** | Agent 1 (Research) | [`01_pesquisa.ipynb`](reports/01_pesquisa.ipynb) | — | Sector research, weighted source evaluation, MTU15 analysis, ML framing. Gate 1 (PASS). |
| **Data Understanding** | Agent 2 (Dataset Builder) | [`02_dataset.ipynb`](reports/02_dataset.ipynb) | — | Robust public API ingestion with raw caching, retry backoff & DST handling. Gate 2 (PASS). |
| **Data Understanding / EDA** | Agent 3 (Validation & EDA) | [`03_validacao_eda.ipynb`](reports/03_validacao_eda.ipynb) | — | Statistical profiling, extreme regimes (negative prices & spikes), Duck Curve EDA. Gate 3 (PASS). |
| **Data Preparation** | Agent 4 (Data Hygiene) | [`04_correcao.ipynb`](reports/04_correcao.ipynb) | — | Canonical grid reindexing and documented $D-7$ seasonal imputations in `logs/correction_log.csv`. Gate 4 (PASS). |
| **Data Preparation** | Agent 5 (Feature Engineering) | [`05_features.ipynb`](reports/05_features.ipynb) | — | 31 ex-ante features, anti-leakage audit, temporal partitions, and `manifest.json`. Gate 5 (PASS). |
| **Modeling** | Agent 6 (Machine Learning) | [`06_ml.ipynb`](reports/06_ml.ipynb) | [`06_ml_en.ipynb`](reports/06_ml_en.ipynb) | Mandatory baselines, TimeSeriesSplit CV, **100-experiment tournament**, early stopping, champion persistence. Gate 6 (PASS). |
| **Evaluation** | Agent 7 (Results & Explainability) | [`07_analise_resultados.ipynb`](reports/07_analise_resultados.ipynb) | [`07_results_analysis_en.ipynb`](reports/07_results_analysis_en.ipynb) | Meta-analysis of 100 runs, Duck Curve error decomposition, SHAP beeswarm/waterfalls, **Continuous Recommendation Engine**. Gate 7 (PASS). |
| **Deployment & Synthesis** | Coordinator Agent | [`99_relatorio_final.ipynb`](reports/99_relatorio_final.ipynb) | [`99_final_report_en.ipynb`](reports/99_final_report_en.ipynb) | Consolidated executive report, quality gate audit trail, and energy contract advisory. |

---

## 3. Energy Contract Advisory & Consumption Strategies

Section 7 of the Executive Report synthesizes empirical Duck Curve dynamics into actionable operational recommendations across three consumer segments:

```
          TYPICAL MIBEL DAILY PRICE REGIMES & OPERATIONAL ACTION WINDOWS
    [01:00 - 06:00]        [11:00 - 16:00]                   [19:00 - 22:00]
  Deep Off-Peak Window   Solar Opportunity Window          Thermal Evening Peak
  ~40-60 EUR/MWh         ~15-30 EUR/MWh (Zero/Negative)    ~120-180+ EUR/MWh
  Wind Surplus           Solar Cannibalization             CCGT Gas & CO2 Marginal Units
  ACTION: Baseload & EV  ACTION: Load Shifting & BESS      ACTION: Peak Shaving & Storage
```

### 3.1 Industry (Large Consumers - MAT / AT / MT)
- **Contract Structures:**
  - *Hybrid Hedged Contracts:* 40% to 60% fixed baseload via forward contracts (OMIP) + 40% to 60% spot-indexed to MIBEL to capture zero-price solar hours while capping tail risk.
  - *Corporate Solar PPAs:* 10 to 15-year on-site or off-site PPAs at levelized costs of **30-45 EUR/MWh**.
  - *Tetra-Hourly Network Access Tariffs (TAR):* Rigorous scheduling around ERSE peak penalty hours.
- **Operational Strategies:**
  - *Load Shifting:* Move high-power batch operations (crushing, inductive melting, pumping, electrolysis) to the **11:00 to 16:00** window.
  - *Battery Storage (BESS) & Peak Shaving:* Charge batteries during solar hours (11h-16h) and discharge during the evening ramp (**19:00 to 22:00**).
  - *Demand Response:* Participate in tertiary frequency reserve balancing markets with the grid operator (REN).

### 3.2 Commerce & Services (Medium Consumers - BTE / Commercial BTN)
- **Contract Structures:**
  - *Tri-Hourly Tariffs:* Optimize weekly cycles (Off-Peak nights/weekends, Shoulder daytime, Peak restricted to 3-4 hours).
  - *Capped Dynamic Spot Tariffs:* Real-time MIBEL indexation protected by a hard cap (e.g. 120 EUR/MWh).
  - *Green Guarantee of Origin (GoO) Tariffs:* 100% certified renewable energy for ESG reporting.
- **Operational Strategies:**
  - *HVAC Thermal Pre-Cooling:* Deep-chill commercial offices and malls between **11:00 and 15:00**, idling chillers during the **18:00 to 21:00** peak.
  - *Supermarket Cold-Chain Management:* Overcool freezer cabinets during midday and reduce compressor load during evening peak hours without breaching safety thresholds.
  - *Rooftop Solar (UPAC):* Self-consume on-site solar power to cover 40% to 70% of daytime refrigeration and lighting.
  - *Logistics Fleets:* Schedule commercial EV van charging overnight or during the 13:00 to 15:00 post-lunch window.

### 3.3 Residential (Domestic Consumers - Normal Low Voltage BTN)
- **Contract Structures:**
  - *Bi-Hourly Tariff:* Recommended for homes shifting $\ge 40\%$ of consumption to off-peak hours (22:00 to 08:00), providing 25% to 35% cost savings.
  - *Dynamic Spot Tariff:* For tech-savvy households with smart tele-metering and automated home scheduling.
  - *Simple Fixed Tariff:* Budget stability for households with inflexible schedules.
- **Operational Strategies:**
  - *Smart Appliance Scheduling:* Run dishwashers and laundry machines between **12:00 and 16:00** (solar index) or overnight (**01:00 to 07:00** in bi-hourly mode).
  - *Water Heater Timers:* Heat domestic hot water tanks between **13:00 and 16:00** or overnight, never during the 19:00 to 22:00 peak.
  - *Electric Vehicle (EV) Smart Wallbox:* Automate EV charging to start strictly after **01:00**, capturing wind surpluses.
  - *Peak Evening Appliance Separation:* Avoid running high-draw appliances concurrently between 19:00 and 22:00.

---

## 4. Continuous Improvement & Recommendation Engine

Agent 7 features a built-in automated recommendation engine (`logs/next_run_recommendations.json`) that analyzes the **100-run tournament leaderboard**, SHAP importances, and Duck Curve residual profiles to prescribe refined parameters for subsequent training cycles:

```json
{
  "tournament_analyzed": { "total_runs": 100, "champion_exp_id": 84 },
  "recommended_hyperparameters": {
    "max_depth": { "recommended_min": 4, "recommended_max": 6, "optimal_mode": 5 },
    "learning_rate": { "recommended_min": 0.025, "recommended_max": 0.055, "optimal_mean": 0.038 },
    "reg_lambda": { "recommended_min": 1.0, "recommended_max": 3.0, "optimal_mean": 1.8 }
  },
  "feature_pruning_candidates": ["cloud_cover_pt_pct", "wind_power_proxy"],
  "architectural_guidance": [
    "Constrain hyperparameter sampling to optimal neighborhood for faster convergence.",
    "Adopt dynamic hourly weights for the Hybrid Blend (80% XGBoost midday / 65% D-1 nighttime).",
    "Test Huber or quantile loss (reg:pseudohubererror) to capture extreme scarcity spikes without gradient explosion."
  ]
}
```

---

## 5. Security Hardening & Zero-Vulnerability Guarantee

This repository is certified for public GitHub deployment with rigorous security protections:

- **Zero PII & Credentials:** Every notebook, script, log, and configuration file is audited to guarantee zero usernames, zero personal emails, zero machine identifiers, and zero absolute local drive paths.
- **Zero Tokens or API Keys:** Public endpoints (OMIE, Open-Meteo, Yahoo Finance) operate without secret keys. The `.env` file is strictly ignored by Git, with only `.env.example` committed.
- **Zero Remote Access or Backdoors:** 100% local Python execution. No remote daemon, no open listening network ports, and zero background listeners.
- **Automated Security Scanner:** A dedicated security audit script ([`scripts/security_check.py`](scripts/security_check.py)) scans the codebase across 45+ files and enforces:
  - Regex detection for AWS keys, GitHub tokens, private keys (`BEGIN PRIVATE KEY`), password assignments, and Databricks endpoints.
  - Verification that sensitive files (`.env`, `*.log`, `*.tmp`, `*.key`) are matched in [`.gitignore`](.gitignore).
  - Clean audit verification: **100% PASS / 0 Findings**.
- **Formal Security Policy:** Documented threat model and vulnerability disclosure in [`SECURITY.md`](SECURITY.md).

---

## 6. Repository Layout

```text
energy-price-agents/
├── config.yaml                       # Centralized project configuration (dates, parameters)
├── .env.example                      # Template for environment variables (zero secrets required)
├── .gitignore                        # Enterprise-grade ignore rules (blocking .env, logs, caches)
├── SECURITY.md                       # Security policy, threat model & vulnerability disclosure
├── README.md                         # Main portfolio documentation (English & Portuguese)
├── data/
│   ├── raw/                          # Immutable raw API responses (OMIE, weather, commodities)
│   ├── interim/                      # dataset_base.parquet, dataset_clean.parquet, data_dictionary.csv
│   └── processed/                    # features_train/valid/test.parquet, manifest.json
├── models/
│   └── xgboost_mibel_best.json       # Champion XGBoost model artifact (JSON format)
├── reports/                          # Complete runnable Jupyter Notebooks
│   ├── 00_coordenador_plano.ipynb    # Master orchestration roadmap (pt-PT)
│   ├── 01_pesquisa.ipynb             # Research & ML framing (pt-PT)
│   ├── 02_dataset.ipynb              # Base dataset ingestion (pt-PT)
│   ├── 03_validacao_eda.ipynb        # Data validation & Duck Curve EDA (pt-PT)
│   ├── 04_correcao.ipynb             # Data hygiene & DST handling (pt-PT)
│   ├── 05_features.ipynb             # Feature engineering & anti-leakage audit (pt-PT)
│   ├── 06_ml.ipynb                   # XGBoost Modeling (100-run tournament) (pt-PT)
│   ├── 06_ml_en.ipynb                # XGBoost Modeling (100-run tournament) (English)
│   ├── 07_analise_resultados.ipynb   # SHAP analysis & recommendation engine (pt-PT)
│   ├── 07_results_analysis_en.ipynb  # SHAP analysis & recommendation engine (English)
│   ├── 99_relatorio_final.ipynb      # Final executive report & contract advisory (pt-PT)
│   └── 99_final_report_en.ipynb      # Final executive report & contract advisory (English)
├── logs/
│   ├── correction_log.csv            # Detailed record of all hygiene imputations
│   ├── gate_results.json             # Quality Gate transition states & test results
│   ├── metrics_summary.json          # Consolidated performance metrics across all models
│   ├── experiments_leaderboard.json  # Full 100-run tournament leaderboard
│   └── next_run_recommendations.json # Prescriptive recommendations for next training cycle
├── scripts/
│   ├── execute_notebook.py           # Headless notebook runner with inline plot capture
│   └── security_check.py             # Automated PII, secret, and security scanner
└── src/                              # Modular production code
    ├── __init__.py
    ├── utils.py                      # Configurations, SHA-256 hashing, logging
    ├── ingestion.py                  # Concurrent API retrieval & DST normalization
    └── features.py                   # Ex-ante feature generation & validation
```

---

## 7. Getting Started & Local Reproduction

### Prerequisites
- Python 3.10+ (tested and validated on Python 3.12 and 3.14).
- Standard package manager (`pip`).

### Step 1: Clone and Install Dependencies
```bash
git clone https://github.com/your-username/energy-price-agents.git
cd energy-price-agents

# Install required dependencies
pip install pandas numpy polars pyarrow pyyaml requests httpx yfinance xgboost shap scikit-learn matplotlib seaborn tabulate
```

### Step 2: Environment Setup
```bash
cp .env.example .env
```
*(All data sources are public and open; no proprietary paid API keys are required.)*

### Step 3: Run the Pipeline (Headless or in PyCharm / VS Code)
All figures are captured inline (`image/png`), ensuring clean rendering inside PyCharm, VS Code, or JupyterLab without external popups.

#### Option A: Run via Headless Runner CLI
```bash
# Execute English Modeling, Analysis & Executive Report:
python scripts/execute_notebook.py reports/06_ml_en.ipynb
python scripts/execute_notebook.py reports/07_results_analysis_en.ipynb
python scripts/execute_notebook.py reports/99_final_report_en.ipynb
```

#### Option B: Run in PyCharm / JupyterLab
Open any notebook in `reports/` and select **Kernel -> Restart & Run All**. Every notebook runs completely end-to-end without errors.

### Step 4: Run the Security & PII Verification
```bash
python scripts/security_check.py
```

---

## 8. Quality Gates Audit Trail

All 10 project acceptance criteria and quality gates were verified and archived in [`logs/gate_results.json`](logs/gate_results.json):

1. ✅ **All 12 Notebooks Execute End-to-End:** Verified with zero errors (*Restart & Run All*).
2. ✅ **All Quality Gates Passed:** Gate 1, Gate 2, Gate 3, Gate 4, Gate 5, Gate 6, and Gate 7 formally approved.
3. ✅ **Strict Temporal Walk-Forward Validation:** No shuffling or future leakage; causality preserved.
4. ✅ **Mandatory Baselines Computed:** $D-1$ persistence, $D-7$ seasonal persistence, and Ridge regression benchmarks evaluated across validation and blind test sets.
5. ✅ **100-Experiment Optimization Tournament:** Systematic parameter search per local run with early stopping.
6. ✅ **SHAP Explainability Complete:** Global beeswarm, partial dependence response curves, and local operational waterfall decompositions.
7. ✅ **Continuous Recommendation Engine:** Actionable parameter updates exported to `logs/next_run_recommendations.json`.
8. ✅ **Energy Contract Advisory:** Comprehensive contract recommendations and consumption strategies for Industry, Commerce, and Residential sectors.
9. ✅ **Bilingual Availability:** Dedicated English notebooks for Modeling, Evaluation, and Consolidated Synthesis (`_en.ipynb`).
10. ✅ **Zero Security Findings:** Full repository audit certifies zero PII, zero tokens, and zero local absolute paths.

---

## 9. License

This project is licensed under the [MIT License](LICENSE).
Public data from OMIE, Open-Meteo (CC-BY 4.0), and Yahoo Finance are used strictly in compliance with their respective open data terms.
