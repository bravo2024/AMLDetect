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

LightGBM on 50,000 simulated transactions (1.15% suspicious), scored against a
1% alert budget — the way a monitoring desk actually consumes a model:

| Metric | Value |
|---|---|
| ROC AUC | 0.715 |
| Alert precision @ 1% budget | 9.6% (~8× the base rate) |
| Alert recall @ 1% budget | 8.3% |

These are honest numbers for transaction monitoring: labels are noisy by
construction (most launderers look like everyone else most of the time), so the
model concentrates risk into the review queue rather than "solving" the problem.

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

Simulated transactions built around real AML typologies — structuring just under the $10k CTR threshold, round-dollar amounts, high-risk corridors, off-hours activity, new-account velocity. Simulation is the honest choice here: labelled SAR data is confidential by statute, and no public dataset of real suspicious-activity labels exists. The same generator (`src/data.py`) feeds both the dashboard and `train.py`.

MIT licensed.
