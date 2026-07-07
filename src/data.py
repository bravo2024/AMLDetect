"""Transaction simulator shared by the dashboard and train.py.

There is no public labelled AML dataset — SARs are confidential by statute —
so the data is simulated around real typologies: structuring just under the
$10k CTR threshold, round-dollar amounts, high-risk corridors, off-hours
activity, new accounts, and velocity spikes.
"""
import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "amount", "customer_age", "account_age_days", "num_transactions_30d",
    "avg_amount_30d", "amount_to_avg_ratio", "is_round_number", "hour_of_day",
    "day_of_week", "structuring_indicator", "country_risk_num", "beneficiary_risk_num",
]


def make_transactions(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    amount      = np.clip(rng.lognormal(mean=5.5, sigma=1.2, size=n), 1, 500_000)
    tx_types    = rng.choice(['wire', 'ACH', 'card', 'cash'], size=n, p=[0.25, 0.35, 0.30, 0.10])
    risk_levels = ['low', 'medium', 'high', 'very_high']
    c_risk      = rng.choice(risk_levels, size=n, p=[0.55, 0.25, 0.15, 0.05])
    b_risk      = rng.choice(risk_levels, size=n, p=[0.50, 0.25, 0.17, 0.08])
    cust_age    = rng.integers(18, 85, size=n)
    acc_age     = rng.integers(1, 7300, size=n)
    n_tx_30d    = rng.integers(1, 150, size=n)
    avg_amt_30d = rng.lognormal(5.0, 1.0, size=n)
    amt_ratio   = amount / (avg_amt_30d + 1e-6)
    is_round    = ((amount % 1000 < 1) | (amount % 500 < 1)).astype(int)
    hour        = rng.integers(0, 24, size=n)
    dow         = rng.integers(0, 7, size=n)
    structuring = ((amount >= 8000) & (amount < 10000)).astype(int)

    def risk_num(arr):
        return np.where(arr == 'very_high', 4, np.where(arr == 'high', 3, np.where(arr == 'medium', 2, 1)))

    c_num = risk_num(c_risk)
    b_num = risk_num(b_risk)

    logit = (
        -6.5
        + 2.2 * structuring
        + 1.4 * is_round
        + 1.2 * (c_num >= 3).astype(float)
        + 1.2 * (b_num >= 3).astype(float)
        + 1.0 * (amt_ratio > 5).astype(float)
        + 0.8 * (n_tx_30d > 100).astype(float)
        + 1.0 * (tx_types == 'cash').astype(float)
        + 0.8 * ((hour < 6) | (hour > 22)).astype(float)
        + 0.6 * (acc_age < 90).astype(float)
    )
    p       = 1 / (1 + np.exp(-logit))
    is_susp = (rng.random(n) < p).astype(int)

    base_date  = pd.Timestamp('2024-01-01')
    timestamps = (base_date
                  + pd.to_timedelta(rng.integers(0, 365, size=n), unit='D')
                  + pd.to_timedelta(hour * 3600 + rng.integers(0, 3600, size=n), unit='s'))

    df = pd.DataFrame({
        'transaction_id'          : [f"TXN{i:07d}" for i in range(n)],
        'account_id'              : [f"ACC{rng.integers(1, 5000):05d}" for _ in range(n)],
        'timestamp'               : timestamps,
        'amount'                  : amount,
        'transaction_type'        : tx_types,
        'country_risk'            : c_risk,
        'beneficiary_country_risk': b_risk,
        'customer_age'            : cust_age,
        'account_age_days'        : acc_age,
        'num_transactions_30d'    : n_tx_30d,
        'avg_amount_30d'          : avg_amt_30d,
        'amount_to_avg_ratio'     : amt_ratio,
        'is_round_number'         : is_round,
        'hour_of_day'             : hour,
        'day_of_week'             : dow,
        'structuring_indicator'   : structuring,
        'country_risk_num'        : c_num,
        'beneficiary_risk_num'    : b_num,
        'is_suspicious'           : is_susp,
    })
    return df.reset_index(drop=True)


def make_model_data(n: int = 50_000, seed: int = 42) -> dict:
    """Numeric matrix view of the simulator for the supervised model."""
    df = make_transactions(n=n, seed=seed)
    return {"X": df[MODEL_FEATURES].to_numpy(float),
            "y": df["is_suspicious"].to_numpy(int),
            "features": MODEL_FEATURES}
