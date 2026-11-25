# SyncGuard AI - Pricing / Inventory Sync Anomaly Detector

**Domain:** E-Commerce and Retail  |  **Stack:** Python, Pandas, NumPy, Scikit-learn (Isolation Forest), Streamlit, Plotly

Retailers that sell on several channels (own store, Amazon, eBay, ...) rely on integrations that keep prices and
stock levels in sync with a master catalogue. When an integration breaks or lags, customers see wrong prices and
products get oversold. SyncGuard AI learns what *normal* synchronisation looks like using an **unsupervised
Isolation Forest** and flags deviations - with a risk score and a plain-English explanation - before they reach customers.

## Features

- Synthetic dataset generator: 20,000 records, 5 channels, 7 categories, 6 anomaly types (labels used only for evaluation)
- Preprocessing + feature engineering with **per-channel learned price baselines**
- Isolation Forest (RobustScaler + IsolationForest pipeline), trained with no labels, saved with `joblib`
- Evaluation: precision, recall, F1, confusion matrix, ROC / PR curves, recall by anomaly type
- Streamlit dashboard: CSV upload, KPI cards, anomaly table (score, risk, severity, explanation), filters,
  record inspector, sensitivity slider, CSV download of detected anomalies
- Plotly charts for **pricing differences**, **inventory mismatches** and **synchronisation delays**

## Project structure

```
SyncGuard-AI/
|-- app.py                     Streamlit dashboard (entry point)
|-- run_pipeline.py            Generate data -> train -> evaluate (one command)
|-- requirements.txt
|-- setup_windows.bat          One-time setup (creates venv, installs packages)
|-- run_app.bat                Starts the dashboard
|-- rebuild_data_and_model.bat Re-runs the whole pipeline
|-- src/
|   |-- config.py              Paths, channel profiles, hyper-parameters
|   |-- data_generator.py      Synthetic data + anomaly injection
|   |-- preprocessing.py       Validation, cleaning, FeatureEngineer
|   |-- model.py               SyncGuardModel, save / load / auto-retrain
|   |-- train.py               Train on data/train.csv and save the model
|   |-- explain.py             Reason codes for flagged records
|   |-- evaluate.py            Metrics and matplotlib charts
|   `-- charts.py              Plotly charts used by the dashboard
|-- data/
|   |-- train.csv  test.csv    Generated training / testing datasets (labelled)
|   |-- sample_upload.csv      Unlabelled sample for testing the upload feature
|   `-- sample_upload_with_labels.csv
|-- models/syncguard_model.joblib   Trained model used by the app
|-- reports/                   evaluation_metrics.json, classification_report.txt,
|   |                          test_predictions.csv, test_detected_anomalies.csv
|   `-- charts/                confusion matrix, ROC/PR, score distribution, ...
|-- docs/
|   |-- SyncGuard_AI_Project_Report.docx
|   `-- Presentation_Outline.md
|-- tests/test_pipeline.py     Automated tests
`-- tools/build_report.py      Regenerates the Word report from the results
```

## Run on Windows with VS Code

**Prerequisite:** Python 3.10 or newer from <https://www.python.org> (tick *"Add Python to PATH"* during install).

1. Unzip the project, then in VS Code choose **File > Open Folder...** and select the `SyncGuard-AI` folder.
2. Open the terminal: **Terminal > New Terminal** (PowerShell is fine).
3. Create a virtual environment and install packages:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *(If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first, or use Command Prompt.)*
4. Start the dashboard:
   ```powershell
   streamlit run app.py
   ```
   The browser opens at <http://localhost:8501>.

**Shortcut:** double-click `setup_windows.bat` once, then `run_app.bat` whenever you want to start the app.

In VS Code, select the interpreter via **Ctrl+Shift+P > Python: Select Interpreter > .\venv\Scripts\python.exe**.
`.vscode/launch.json` also provides a "SyncGuard: Streamlit dashboard" debug configuration.

### Re-creating the data, model and charts (optional)

The ZIP already contains generated datasets, a trained model and evaluation charts. To rebuild everything:

```powershell
python run_pipeline.py
```

Run the automated tests:

```powershell
python -m unittest discover -s tests -v
```

Rebuild the Word report (needs `pip install python-docx`): `python tools/build_report.py`

## Using the dashboard

1. Pick a data source in the sidebar: the built-in labelled test set, the demo sample, or **Upload my own CSV**.
2. Adjust **Sensitivity** (50 = model default). Higher flags more records (more recall, more false alarms).
3. Explore the tabs: *Overview*, *Detected anomalies*, *Pricing*, *Inventory*, *Sync delay*, *Model performance*, *Data & help*.
4. In *Detected anomalies*, filter and press **Download** to export the flagged records as CSV.

### CSV format for uploads

| Column | Status | Description |
|---|---|---|
| `sku` | required | Product identifier |
| `channel` | required | Sales channel (e.g. Shopify, Amazon) |
| `master_price` | required | Price in the master catalogue |
| `channel_price` | required | Price live on the channel |
| `master_stock` | required | Units in master inventory |
| `channel_stock` | required | Units shown on the channel |
| `sync_delay_minutes` | required | Minutes since the last successful sync |
| `record_id`, `timestamp`, `category` | optional | Identifier, time (enables timeline), category |
| `sales_velocity_per_day` | optional | Units sold per day (default 5) |
| `sync_frequency_minutes` | optional | Expected sync interval (learned per channel if missing) |
| `is_anomaly` | optional | 0/1 ground truth; enables live precision / recall |

Column names are case-insensitive and spaces are treated as underscores. Unknown channels are handled using global baselines.

## How it works

1. **Features** - for each record the engineer computes the channel-adjusted price deviation, the *unexplained* stock
   gap (channel - master - expected sales since last sync), and the sync delay relative to the channel's normal cadence.
2. **Model** - a RobustScaler followed by an Isolation Forest (300 trees). Records that are isolated quickly get high
   anomaly scores. The score is also mapped to a 0-100 risk score (Critical / High / Medium).
3. **Explanations** - tolerances are learned from records the model considers normal; each flagged record is checked
   against them to produce reason codes (price mismatch, overselling risk, false stock-out, sync delay, invalid data).
4. **Persistence** - everything (baselines, scaler, forest, explainer) is saved to `models/syncguard_model.joblib`.
   If the file is missing or was saved by an incompatible scikit-learn version, the app retrains automatically from
   `data/train.csv` (takes a few seconds) and shows a notice.

## Results (held-out synthetic test set, default threshold)

| Metric | Value |
|---|---|
| Precision | 0.811 |
| Recall | 0.830 |
| F1-score | 0.820 |
| ROC-AUC | 0.992 |

Confusion matrix: TN 4,642 | FP 58 | FN 51 | TP 249. Charts are in `reports/charts/`; full details in the project report.

> These numbers describe performance on **synthetic** data. Real retailer feeds will need threshold tuning.
> The contamination setting (0.06) mirrors the synthetic anomaly rate; use the sensitivity slider on real data.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognised | Activate the venv first: `venv\Scripts\activate`, then `pip install -r requirements.txt` |
| PowerShell: "running scripts is disabled" | `Set-ExecutionPolicy -Scope Process Bypass`, or use Command Prompt |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Model-version warning in the app | Harmless; the app retrained a compatible model automatically |
| Upload rejected | Check the required column names listed above |
| Charts / data missing | Run `python run_pipeline.py` |
