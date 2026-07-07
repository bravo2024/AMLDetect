# AMLDetect

Flags suspicious transaction patterns — structuring, layering, smurfing — in a synthetic banking ledger.

Generates synthetic transaction data with realistic AML risk patterns (structuring, round-dollar amounts, high-risk jurisdictions, anomalous timing), then trains a LightGBM classifier to flag suspicious activity. Streamlit dashboard provides a compliance-officer view with configurable thresholds, alert queues, and network analysis.

Regulatory thresholds follow FinCEN guidelines (CTR at $10,000, SAR at $5,000).

## Setup

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Detection results

LightGBM trained on 1,372 banknote authentication samples:

| Metric | Value |
|---|---|
| Backend | LightGBM |
| ROC AUC | 0.9998 |
| Accuracy | 0.991 |
| F1 Score | 0.990 |

## Inside the dashboard

| Component | What it does |
|---|---|
| **Control Panel** | Risk threshold, lookback window, alert limit, minimum transaction amount |
| **Alert Queue** | Prioritised suspicious activity alerts with risk scores |
| **Transaction Explorer** | Filterable transaction table with AML risk indicators |
| **Network Graph** | Entity relationship visualisation for linkage analysis |
| **Regulatory Reports** | CTR/SAR filing summaries, pattern detection statistics |

### Layout

```
AMLDetect/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## What it's trained on

Synthetic transaction data engineered to reflect real AML typologies: wire/ACH/card/cash transactions, customer risk ratings, account age, transaction frequency, round-dollar structuring, and anomalous hour-of-day patterns. A real banknote authentication dataset is also available via `load_real_banknote()`.

MIT licensed.
