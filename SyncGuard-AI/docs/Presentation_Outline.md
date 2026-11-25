# SyncGuard AI - Presentation Outline

Suggested length: 12 slides, about 10-12 minutes plus a 3-minute live demo.

---

### Slide 1 - Title
**SyncGuard AI: Pricing / Inventory Sync Anomaly Detector** - Domain: E-Commerce and Retail
Your name, roll number, guide, institution.

### Slide 2 - The problem
- Retailers sell on many channels (own store, Amazon, eBay, Walmart, Flipkart ...)
- Integrations keep price and stock in sync with a master catalogue
- When they fail: wrong prices, overselling, false "out of stock", lost revenue and trust
- *Speaker note:* give one concrete example, e.g. a price feed shifting a decimal so an 8.44 item lists at 84.40.

### Slide 3 - Why simple rules are not enough
- Every channel differs legitimately (e.g. Amazon about +4%, Walmart about -1.5%)
- Stock on a channel lags master by units sold since the last sync
- A fixed "alert if > 5%" rule is noisy and misses subtle issues
- Need a model that **learns normal behaviour per channel**

### Slide 4 - Objectives
- Synthetic data with normal + anomalous sync records
- Feature engineering, unsupervised model, evaluation
- Dashboard with upload, scores, explanations, charts, CSV export
- Reusable saved model

### Slide 5 - Architecture
Data generator -> preprocessing / features -> RobustScaler + Isolation Forest -> threshold -> explanation engine -> Streamlit dashboard.
Show the architecture diagram (draw boxes from the report, section 3).

### Slide 6 - Dataset
- 20,000 records, 600 SKUs, 7 categories, 5 channels, 60 days; 75/25 split
- 6 anomaly types: price drift, decimal error, stock oversell, zeroed stock, sync delay, combined
- About 8% "benign but unusual" normal records (promos, slow syncs, cart holds) so the task is realistic
- Labels used **only** for evaluation

### Slide 7 - Feature engineering
- Channel-adjusted price deviation (learned median + MAD per channel)
- Unexplained stock gap = channel - master - expected sales lag
- Sync delay relative to channel cadence
- Flags: false stock-out, invalid values
- Signed-log transform + RobustScaler to tame extreme outliers

### Slide 8 - Model: Isolation Forest
- Random splits isolate anomalies in fewer steps -> shorter path -> higher anomaly score
- No labels needed (realistic for retail)
- 300 trees, contamination 0.06, saved with joblib
- Score mapped to 0-100 risk and Critical / High / Medium severity

### Slide 9 - Explainability
- Reason codes: price mismatch, overselling risk, false stock-out, sync delay, invalid data
- Tolerances learned from records the model considers normal
- Example: *"Price mismatch: Shopify lists 84.40 vs master 8.44 (+900.0%); normal for Shopify is +0.0%. Looks like a decimal-place / unit error."* (real output from the test set)

### Slide 10 - Results
Precision 0.81 | Recall 0.83 | F1 0.82 | ROC-AUC 0.99 (held-out test set)
Show: confusion matrix, recall-by-type chart, sensitivity trade-off chart.
- Obvious faults caught about 100%; subtle ones (small drifts / small stock gaps) are harder
- Sensitivity slider trades precision against recall

### Slide 11 - Live demo (3 minutes)
1. Open the dashboard with the built-in test set; point at KPI cards
2. *Detected anomalies* tab: filter by channel, open a record, read the explanation
3. *Pricing / Inventory / Sync delay* tabs: show the charts
4. Upload `data/sample_upload.csv`; download the anomalies CSV
5. Move the sensitivity slider and show the *Model performance* tab change

### Slide 12 - Limitations, future work, conclusion
- Limitations: synthetic data; contamination mirrors the synthetic rate; independent records (no time context)
- Future: live Shopify / Amazon connectors, Slack / email alerts, rolling per-SKU baselines, feedback loop and retraining, SHAP-style attributions
- Conclusion: unsupervised, channel-aware, explainable detection of sync errors before customers see them

---

## Likely questions (with short answers)

**Why Isolation Forest?** It is unsupervised, fast, handles mixed-scale numeric features and works without labelled failures, which retailers rarely have.

**Why not just use thresholds?** Each channel has its own normal markup and stock lag; learned baselines remove that systematic variation and leave real errors.

**How do you know it works if the data is synthetic?** We measure it on held-out records with known injected anomalies; we are explicit that real data needs tuning. The architecture and feature logic carry over; thresholds would be re-calibrated.

**Why is recall not 100%?** Subtle anomalies overlap with legitimate promotions and order-timing noise. The sensitivity slider lets users pick their trade-off.

**What does contamination mean?** The expected share of anomalies; it sets the default threshold. In production it is unknown, so the slider and review of flagged records replace it.

**How would you deploy it?** Replace the CSV input with scheduled API pulls from channel and ERP systems, score each batch with the saved model, send alerts for High / Critical records and feed reviewer verdicts back for retraining.
