import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMLDetect – Anti-Money Laundering Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Control Panel")
    st.markdown("---")
    risk_threshold = st.slider("Risk Score Threshold", 0, 100, 60, 5)
    lookback_days  = st.slider("Lookback Window (days)", 7, 90, 30, 7)
    alert_limit    = st.slider("Max Alerts to Display", 10, 200, 50, 10)
    min_amount     = st.number_input("Min Transaction Amount ($)", 0, 50000, 100, 100)
    st.markdown("---")
    st.markdown("**Regulatory Thresholds**")
    st.info("CTR: Cash > $10,000\nSAR: Suspicious activity ≥ $5,000")
    st.markdown("---")
    st.caption("AMLDetect v2.0 | FinCEN Compliant")

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION  (cached by seed + n)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_aml_dataset(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    amount      = np.clip(rng.lognormal(mean=5.5, sigma=1.2, size=n), 1, 500_000)
    tx_types    = rng.choice(['wire','ACH','card','cash'], size=n, p=[0.25,0.35,0.30,0.10])
    risk_levels = ['low','medium','high','very_high']
    c_risk      = rng.choice(risk_levels, size=n, p=[0.55,0.25,0.15,0.05])
    b_risk      = rng.choice(risk_levels, size=n, p=[0.50,0.25,0.17,0.08])
    cust_age    = rng.integers(18, 85, size=n)
    acc_age     = rng.integers(1, 7300, size=n)
    n_tx_30d    = rng.integers(1, 150, size=n)
    avg_amt_30d = rng.lognormal(5.0, 1.0, size=n)
    amt_ratio   = amount / (avg_amt_30d + 1e-6)
    is_round    = ((amount % 1000 < 1) | (amount % 500 < 1)).astype(int)
    hour        = rng.integers(0, 24, size=n)
    dow         = rng.integers(0, 7,  size=n)
    structuring = ((amount >= 8000) & (amount < 10000)).astype(int)

    def risk_num(arr):
        return np.where(arr=='very_high',4,np.where(arr=='high',3,np.where(arr=='medium',2,1)))

    c_num = risk_num(c_risk)
    b_num = risk_num(b_risk)

    logit = (
        -5.5
        + 0.8  * structuring
        + 0.6  * is_round
        + 0.5  * (c_num >= 3).astype(float)
        + 0.5  * (b_num >= 3).astype(float)
        + 0.4  * (amt_ratio > 5).astype(float)
        + 0.3  * (n_tx_30d > 100).astype(float)
        + 0.4  * (tx_types == 'cash').astype(float)
        + 0.3  * ((hour < 6) | (hour > 22)).astype(float)
        + 0.2  * (acc_age < 90).astype(float)
    )
    p           = 1 / (1 + np.exp(-logit))
    is_susp     = (rng.random(n) < p).astype(int)

    base_date   = pd.Timestamp('2024-01-01')
    timestamps  = (base_date
                   + pd.to_timedelta(rng.integers(0, 365, size=n), unit='D')
                   + pd.to_timedelta(hour * 3600 + rng.integers(0, 3600, size=n), unit='s'))

    df = pd.DataFrame({
        'transaction_id'         : [f"TXN{i:07d}" for i in range(n)],
        'account_id'             : [f"ACC{rng.integers(1,5000):05d}" for _ in range(n)],
        'timestamp'              : timestamps,
        'amount'                 : amount,
        'transaction_type'       : tx_types,
        'country_risk'           : c_risk,
        'beneficiary_country_risk': b_risk,
        'customer_age'           : cust_age,
        'account_age_days'       : acc_age,
        'num_transactions_30d'   : n_tx_30d,
        'avg_amount_30d'         : avg_amt_30d,
        'amount_to_avg_ratio'    : amt_ratio,
        'is_round_number'        : is_round,
        'hour_of_day'            : hour,
        'day_of_week'            : dow,
        'structuring_indicator'  : structuring,
        'country_risk_num'       : c_num,
        'beneficiary_risk_num'   : b_num,
        'is_suspicious'          : is_susp,
    })
    return df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# AML RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def apply_rule_engine(df: pd.DataFrame) -> pd.DataFrame:
    rules = pd.DataFrame(index=df.index)
    rules['structuring']           = ((df['amount'] >= 8000) & (df['amount'] < 10000)).astype(int)
    rules['round_number']          = (
        (np.abs(df['amount'] - 1000) < 1) |
        (np.abs(df['amount'] - 5000) < 1) |
        (np.abs(df['amount'] - 10000) < 1) |
        (df['is_round_number'] == 1)
    ).astype(int)
    rules['rapid_movement']        = (
        (df['num_transactions_30d'] > 50) &
        ((df['hour_of_day'] < 5) | (df['hour_of_day'] > 22))
    ).astype(int)
    rules['high_risk_jurisdiction']= (
        df['country_risk'].isin(['high','very_high']) |
        df['beneficiary_country_risk'].isin(['high','very_high'])
    ).astype(int)
    rules['velocity_spike']        = (df['amount_to_avg_ratio'] > 3.0).astype(int)
    rules['rule_score']            = rules[['structuring','round_number','rapid_movement',
                                            'high_risk_jurisdiction','velocity_spike']].sum(axis=1)
    rules['rule_flag']             = (rules['rule_score'] >= 1).astype(int)
    return rules

# ─────────────────────────────────────────────────────────────────────────────
# ML ANOMALY DETECTION  (cached by data fingerprint)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_anomaly_models(data_key: str, X_raw: np.ndarray) -> dict:
    """
    data_key  – string used only as a cache key (ignored inside)
    X_raw     – shape (n, 8) float array of normalized-ready features
    """
    rng = np.random.default_rng(42)
    n   = len(X_raw)

    mu  = X_raw.mean(axis=0)
    sd  = X_raw.std(axis=0) + 1e-8
    X   = (X_raw - mu) / sd

    # ── Method 1: Z-score ──────────────────────────────────────────────────
    z   = np.abs(X)
    z_s = z.max(axis=1)
    z_s = (z_s - z_s.min()) / (z_s.max() - z_s.min() + 1e-8)

    # ── Method 2: LOF approximation (k-NN distance) ────────────────────────
    sample_idx = rng.choice(n, size=min(1500, n), replace=False)
    X_samp     = X[sample_idx]
    k          = 5
    lof        = np.zeros(n)
    bs         = 500
    for i in range(0, n, bs):
        Xb    = X[i:i+bs]
        dists = np.sqrt(((Xb[:, None, :] - X_samp[None, :, :]) ** 2).sum(axis=2))
        lof[i:i+bs] = np.sort(dists, axis=1)[:, :k].mean(axis=1)
    lof = (lof - lof.min()) / (lof.max() - lof.min() + 1e-8)

    # ── Method 3: NumPy Autoencoder (8→4→8) ───────────────────────────────
    rng2 = np.random.default_rng(123)
    n_feat, n_hid = X.shape[1], 4
    W1 = rng2.normal(0, 0.1, (n_feat, n_hid))
    b1 = np.zeros(n_hid)
    W2 = rng2.normal(0, 0.1, (n_hid, n_feat))
    b2 = np.zeros(n_feat)

    tr_idx = rng2.choice(n, size=min(10000, n), replace=False)
    X_tr   = X[tr_idx]
    lr     = 0.001

    def relu(x): return np.maximum(0, x)

    for _ in range(60):
        bi = rng2.choice(len(X_tr), size=256, replace=False)
        Xb = X_tr[bi]
        h  = relu(Xb @ W1 + b1)
        r  = h @ W2 + b2
        d  = 2 * (r - Xb) / len(Xb)
        dW2 = h.T @ d;  db2 = d.sum(0)
        dh  = (d @ W2.T) * (h > 0)
        dW1 = Xb.T @ dh; db1 = dh.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    recon = np.zeros(n)
    for i in range(0, n, bs):
        Xb = X[i:i+bs]
        h  = relu(Xb @ W1 + b1)
        recon[i:i+bs] = ((Xb - h @ W2 - b2) ** 2).mean(axis=1)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)

    ensemble = 0.35 * z_s + 0.35 * lof + 0.30 * recon
    ensemble = (ensemble - ensemble.min()) / (ensemble.max() - ensemble.min() + 1e-8)

    return {'z_score': z_s, 'lof': lof, 'recon_error': recon, 'ensemble': ensemble}

# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────
def compute_risk_scores(df, rules, ml):
    rule_comp  = (rules['rule_score'] / 5.0).values * 40
    ml_comp    = ml['ensemble'] * 60
    risk       = np.clip(rule_comp + ml_comp, 0, 100)
    tier       = pd.cut(risk, bins=[0,30,60,80,100],
                        labels=['Low','Medium','High','Critical'],
                        include_lowest=True)
    return risk, tier

# ─────────────────────────────────────────────────────────────────────────────
# ROC / PR  (manual, no sklearn)
# ─────────────────────────────────────────────────────────────────────────────
def compute_roc_pr(y_true, scores):
    idx    = np.argsort(scores)[::-1]
    ys     = y_true[idx]
    n_pos  = y_true.sum(); n_neg = len(y_true) - n_pos
    tp     = np.cumsum(ys); fp = np.cumsum(1 - ys)
    tpr    = tp / (n_pos + 1e-10); fpr = fp / (n_neg + 1e-10)
    prec   = tp / (tp + fp + 1e-10); rec = tpr
    fpr_c  = np.concatenate([[0], fpr, [1]]); tpr_c = np.concatenate([[0], tpr, [1]])
    auc    = float(np.trapz(tpr_c, fpr_c))
    ap     = float(np.sum((rec[1:] - rec[:-1]) * prec[1:])) if len(rec) > 1 else 0.0
    return fpr_c, tpr_c, prec, rec, auc, ap

def precision_at_k(y_true, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].sum()) / k

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RULE_NAMES  = ['structuring','round_number','rapid_movement','high_risk_jurisdiction','velocity_spike']
RULE_LABELS = ['Structuring (<$10k)','Round Numbers','Rapid Movement','High-Risk Jurisdiction','Velocity Spike']
FATF_RECS   = {
    'structuring'           : ('Rec. 3',  'Money Laundering Offence – structuring below reporting threshold'),
    'round_number'          : ('Rec. 10', 'Customer Due Diligence – unusual round-amount patterns'),
    'rapid_movement'        : ('Rec. 16', 'Wire Transfers – rapid layering through multiple accounts'),
    'high_risk_jurisdiction': ('Rec. 19', 'Higher-Risk Countries – FATF grey-list jurisdictions'),
    'velocity_spike'        : ('Rec. 20', 'Reporting Suspicious Transactions – abnormal velocity'),
}

# ─────────────────────────────────────────────────────────────────────────────
# SAR NARRATIVE
# ─────────────────────────────────────────────────────────────────────────────
def generate_sar_narrative(row, rule_row, risk_score):
    indicators = []
    if rule_row['structuring']:       indicators.append("Amount structured just below $10,000 CTR threshold")
    if rule_row['round_number']:      indicators.append("Transaction involves a suspiciously round dollar amount")
    if rule_row['rapid_movement']:    indicators.append("Abnormal velocity during off-hours window")
    if rule_row['high_risk_jurisdiction']: indicators.append(f"Involves high-risk jurisdiction (origin: {row['country_risk']})")
    if rule_row['velocity_spike']:    indicators.append(f"Amount is {row['amount_to_avg_ratio']:.1f}× the 30-day average")
    if not indicators:                indicators.append("General anomaly flagged by ML ensemble model")

    fatf_lines = [f"- {FATF_RECS[r][0]}: {FATF_RECS[r][1]}" for r in RULE_NAMES if rule_row.get(r, 0)]
    if not fatf_lines: fatf_lines = ["- Rec. 20: General suspicious transaction reporting"]

    ctr_note = ("CTR (FinCEN Form 112) **also required** – cash transaction > $10,000"
                if row['amount'] > 10000 and row['transaction_type'] == 'cash'
                else "CTR not required for this transaction")

    return f"""
**SUSPICIOUS ACTIVITY REPORT (SAR)**
**Filing Institution:** AMLDetect Financial Services | **Form:** FinCEN 111

---
**SUBJECT INFORMATION**
- Account: `{row['account_id']}` | Age: {row['account_age_days']} days | Customer Age: {row['customer_age']} yrs

**TRANSACTION DETAILS**
- ID: `{row['transaction_id']}` | Date: `{str(row['timestamp'])[:16]}`
- Amount: **${row['amount']:,.2f}** | Type: {row['transaction_type'].upper()}
- Origin Risk: {row['country_risk'].upper()} | Beneficiary Risk: {row['beneficiary_country_risk'].upper()}

**SUSPICIOUS INDICATORS**
{''.join(f"- {i}" + chr(10) for i in indicators)}
**COMPOSITE RISK SCORE:** {risk_score:.1f} / 100

**TYPOLOGY:** {"Layering / Structuring" if rule_row['structuring'] else "Placement / Integration"}

**FATF 40 RECOMMENDATIONS IMPLICATED**
{''.join(l + chr(10) for l in fatf_lines)}
**REGULATORY REQUIREMENT**
- {"**Mandatory** SAR filing (BSA 31 U.S.C. § 5318(g))" if risk_score >= 80 else "Voluntary SAR filing recommended"}
- {ctr_note}
- FinCEN SAR deadline: 30 calendar days from detection
"""

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & COMPUTE SCORES
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Generating 50,000 synthetic AML transactions..."):
    df_full = generate_aml_dataset(50_000, 42)

df = df_full[df_full['amount'] >= min_amount].reset_index(drop=True)

rules_df = apply_rule_engine(df)

ML_FEATURES = ['amount','num_transactions_30d','avg_amount_30d','amount_to_avg_ratio',
               'country_risk_num','beneficiary_risk_num','account_age_days','hour_of_day']
X_ml = df[ML_FEATURES].values.astype(float)
data_key = f"{len(df)}_{min_amount}"

with st.spinner("Training anomaly detection ensemble..."):
    ml = train_anomaly_models(data_key, X_ml)

risk_scores, risk_tiers = compute_risk_scores(df, rules_df, ml)

df['risk_score']  = risk_scores
df['risk_tier']   = risk_tiers.astype(str)
df['rule_flag']   = rules_df['rule_flag'].values
df['rule_score']  = rules_df['rule_score'].values
df['ml_score']    = ml['ensemble']
for r in RULE_NAMES:
    df[r] = rules_df[r].values

n_alerts  = int((df['risk_score'] >= risk_threshold).sum())
sar_rate  = 0.08
n_sar     = max(1, int(n_alerts * sar_rate))
alerts_df = df[df['risk_score'] >= risk_threshold]
fp_rate   = (1 - alerts_df['is_suspicious'].mean()) if len(alerts_df) > 0 else 0.85

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("AMLDetect – Anti-Money Laundering Detection Platform")
st.markdown("*Financial Crime Monitoring | FinCEN Compliant | Real-time Risk Scoring*")
st.markdown("---")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Transactions", f"{len(df):,}",
          f"{len(df_full)-len(df):,} filtered out")
k2.metric("Suspicious Rate",    f"{df['is_suspicious'].mean()*100:.2f}%",
          f"{df['is_suspicious'].sum():,} ground-truth labels")
k3.metric("Alerts Generated",   f"{n_alerts:,}",
          f"Threshold ≥ {risk_threshold}")
k4.metric("SARs Filed (Est.)",  f"{n_sar:,}",
          f"Conv. Rate {sar_rate*100:.0f}%")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏦 Transaction Explorer",
    "🔍 AML Rule Engine",
    "🤖 ML Anomaly Detection",
    "📊 Risk Scoring & Alerts",
    "📋 SAR Report Generator",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – TRANSACTION EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Transaction Explorer")
    st.markdown("Explore 50,000 synthetic AML transactions with realistic financial crime patterns.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Amount Distribution (log scale)")
        fig, ax = plt.subplots(figsize=(7, 4))
        lo = max(df['amount'].min(), 1)
        bins = np.logspace(np.log10(lo), np.log10(df['amount'].max()), 60)
        ax.hist(df.loc[df['is_suspicious']==0,'amount'], bins=bins, alpha=0.6,
                color='steelblue', density=True, label='Normal')
        ax.hist(df.loc[df['is_suspicious']==1,'amount'], bins=bins, alpha=0.75,
                color='crimson',   density=True, label='Suspicious')
        ax.set_xscale('log'); ax.set_xlabel('Amount ($) [log]'); ax.set_ylabel('Density')
        ax.set_title('Transaction Amount Distribution'); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Type Breakdown vs Suspicious Rate")
        ts = df.groupby('transaction_type').agg(
            count=('amount','count'), susp_rate=('is_suspicious','mean')).reset_index()
        fig, ax1 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(ts))
        ax1.bar(x, ts['count'], color=['#4472C4','#ED7D31','#A9D18E','#FF0000'], alpha=0.8)
        ax1.set_ylabel('Count'); ax1.set_xticks(x); ax1.set_xticklabels(ts['transaction_type'])
        ax1.set_title('Transaction Type: Volume vs Suspicious Rate')
        ax2 = ax1.twinx()
        ax2.plot(x, ts['susp_rate']*100, 'ro-', lw=2, ms=8, label='Suspicious Rate %')
        ax2.set_ylabel('Suspicious Rate (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Country Risk Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        rc = df['country_risk'].value_counts()
        pie_c = {'low':'#2ecc71','medium':'#f39c12','high':'#e74c3c','very_high':'#8e44ad'}
        ax.pie(rc.values, labels=rc.index,
               colors=[pie_c.get(r,'gray') for r in rc.index],
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Country Risk Distribution')
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Hourly Pattern – Suspicious vs Normal")
        hr = df.groupby('hour_of_day').agg(total=('amount','count'),
                                            susp=('is_suspicious','sum')).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(hr['hour_of_day'], hr['total'], alpha=0.4, color='steelblue', label='All Txns')
        ax.fill_between(hr['hour_of_day'], hr['susp']*20, alpha=0.75, color='crimson', label='Suspicious (×20)')
        ax.set_xlabel('Hour of Day'); ax.set_ylabel('Count')
        ax.set_title('Hourly Transaction Pattern')
        ax.legend(); ax.set_xticks(range(0,24,2)); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

    st.subheader("Segment Summary Statistics")
    seg = df.groupby('country_risk').agg(
        n=('amount','count'), avg_amount=('amount','mean'),
        median_amount=('amount','median'), suspicious_count=('is_suspicious','sum'),
        suspicious_rate=('is_suspicious','mean'), avg_risk_score=('risk_score','mean')
    ).round(3)
    seg['suspicious_rate'] = (seg['suspicious_rate']*100).round(2).astype(str) + '%'
    seg['avg_amount']      = seg['avg_amount'].round(0).astype(int)
    seg['avg_risk_score']  = seg['avg_risk_score'].round(1)
    st.dataframe(seg, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – AML RULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("AML Rule Engine")
    st.markdown("Classic typology rules based on BSA / FATF guidelines.")

    y_true = df['is_suspicious'].values
    n_pos  = y_true.sum(); n_neg = len(y_true) - n_pos

    metrics_rows = []
    for rule, label in zip(RULE_NAMES, RULE_LABELS):
        flag = rules_df[rule].values
        tp   = int((flag & y_true).sum()); fp = int((flag & (1-y_true)).sum())
        fn   = int(((1-flag) & y_true).sum())
        metrics_rows.append({
            'Rule'       : label,
            'Hit Rate'   : f"{flag.mean()*100:.1f}%",
            'Precision'  : f"{tp/(tp+fp+1e-10):.3f}",
            'Recall'     : f"{tp/(tp+fn+1e-10):.3f}",
            'FP Rate'    : f"{fp/(n_neg+1e-10)*100:.2f}%",
            'Alert Count': f"{int(flag.sum()):,}",
        })

    st.subheader("Rule Performance Metrics")
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hit Rate vs Precision")
        hit_rates  = [rules_df[r].mean()*100 for r in RULE_NAMES]
        precisions = []
        for r in RULE_NAMES:
            flag = rules_df[r].values
            tp = int((flag & y_true).sum()); fp = int((flag & (1-y_true)).sum())
            precisions.append(tp/(tp+fp+1e-10)*100)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(RULE_LABELS)); w = 0.35
        ax.bar(x-w/2, hit_rates,  w, label='Hit Rate %',  color='steelblue', alpha=0.8)
        ax.bar(x+w/2, precisions, w, label='Precision %', color='orange',    alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(RULE_LABELS, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Rule Hit Rate vs Precision')
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Rule Score vs ML Score")
        rng_sc  = np.random.default_rng(42)
        si      = rng_sc.choice(len(df), size=min(2000,len(df)), replace=False)
        df_s    = df.iloc[si]
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_s = ['crimson' if s else 'steelblue' for s in df_s['is_suspicious']]
        ax.scatter(df_s['rule_score'], df_s['ml_score'], c=colors_s, alpha=0.4, s=15)
        ax.set_xlabel('Rule Score (0–5 rules hit)'); ax.set_ylabel('ML Anomaly Score (0–1)')
        ax.set_title('Rule Score vs ML Score')
        ax.legend(handles=[mpatches.Patch(color='crimson',label='Suspicious'),
                           mpatches.Patch(color='steelblue',label='Normal')])
        ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

    st.subheader("Rules-as-Code")
    with st.expander("View AML Rule Implementations"):
        st.code("""
# Rule 1: Structuring (Smurfing) – BSA 31 U.S.C. § 5324
structuring = (amount >= 8_000) & (amount < 10_000)

# Rule 2: Round Number Clustering
round_number = (abs(amount - 1_000) < 1) | (abs(amount - 5_000) < 1) |
               (abs(amount - 10_000) < 1) | (amount % 1_000 == 0)

# Rule 3: Rapid Movement – off-hours high velocity
rapid_movement = (num_transactions_30d > 50) & ((hour < 5) | (hour > 22))

# Rule 4: High-Risk Jurisdiction – FATF grey-list
high_risk_jurisdiction = country_risk.isin(['high','very_high']) |
                         beneficiary_country_risk.isin(['high','very_high'])

# Rule 5: Velocity Spike – amount >3× 30-day average
velocity_spike = amount_to_avg_ratio > 3.0

# Combined Rule Score (0–5)
rule_score = structuring + round_number + rapid_movement +
             high_risk_jurisdiction + velocity_spike
        """, language='python')

    st.info("**Rule-based vs ML:** Rules catch *known* codified patterns and satisfy explainability requirements for regulators. ML catches *unknown/evolving* typologies that have not yet been codified. Together they maximize coverage while maintaining audit trails.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ML ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("ML Anomaly Detection")
    st.markdown("Ensemble of three unsupervised anomaly detectors trained with pure NumPy.")

    st.subheader("Isolation Forest – Theoretical Foundation")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**Expected path length:**")
        st.latex(r"E[h(x)] = 2H(\psi-1) - \frac{2(\psi-1)}{\psi}")
        st.markdown(r"$H(i) = \ln(i) + 0.5772$ (Euler–Mascheroni)")
    with ec2:
        st.markdown("**Anomaly score:**")
        st.latex(r"s(x,n) = 2^{-\frac{E[h(x)]}{c(n)}},\quad c(n)=2H(n-1)-\frac{2(n-1)}{n}")
        st.markdown("Score → 1 means highly anomalous; → 0 means normal.")

    st.subheader("NumPy Autoencoder – Reconstruction Error")
    st.latex(r"\mathcal{L}(x)=\|x-\hat{x}\|^2=\|x-W_2\cdot\mathrm{ReLU}(W_1 x+b_1)-b_2\|^2")
    st.markdown("Architecture: **8 → ReLU(4) → 8**. High reconstruction error flags anomalous transactions the network cannot compress.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distributions by Method")
        fig, axes = plt.subplots(3, 1, figsize=(7, 9))
        methods   = ['z_score','lof','recon_error']
        mlabels   = ['Z-Score Anomaly','LOF Approximation','Autoencoder Recon Error']
        susp_mask = y_true == 1
        for ax, meth, mlab in zip(axes, methods, mlabels):
            s = ml[meth]
            ax.hist(s[~susp_mask], bins=50, alpha=0.6, color='steelblue', density=True, label='Normal')
            ax.hist(s[susp_mask],  bins=50, alpha=0.75,color='crimson',   density=True, label='Suspicious')
            ax.set_title(mlab, fontsize=10); ax.set_xlabel('Score'); ax.set_ylabel('Density')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.suptitle('Score Distributions: Suspicious vs Normal', fontsize=12, y=1.01)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("ROC Curve & Precision-Recall")
        fpr_c, tpr_c, prec_c, rec_c, auc_v, ap_v = compute_roc_pr(y_true, ml['ensemble'])
        fig, axes = plt.subplots(2, 1, figsize=(7, 7))

        ax = axes[0]
        ax.plot(fpr_c, tpr_c, color='darkorange', lw=2, label=f'ROC (AUC={auc_v:.3f})')
        ax.plot([0,1],[0,1],'k--',lw=1,label='Random')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve (Ensemble)')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        step = max(1, len(rec_c)//300)
        ax.plot(rec_c[::step], prec_c[::step], color='steelblue', lw=2, label=f'PR (AP={ap_v:.4f})')
        ax.axhline(y=y_true.mean(), color='r', ls='--', label=f'Baseline ({y_true.mean():.3f})')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-Recall Curve')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Precision@K – Top-K Inspection Quality")
    pak_rows = []
    for k in [100, 500, 1000]:
        pak = precision_at_k(y_true, ml['ensemble'], k)
        pak_rows.append({'K': k, 'Precision@K': f"{pak:.4f}",
                         'True Positives': int(pak*k),
                         'Lift vs Baseline': f"{pak/(y_true.mean()+1e-10):.1f}×"})
    st.dataframe(pd.DataFrame(pak_rows), use_container_width=True)

    st.subheader("Method Comparison")
    comp = []
    for meth, mlab in zip(['z_score','lof','recon_error','ensemble'],
                           ['Z-Score','LOF Approx','Autoencoder','Ensemble']):
        _, _, _, _, a, ap = compute_roc_pr(y_true, ml[meth])
        pak = precision_at_k(y_true, ml[meth], 100)
        comp.append({'Method':mlab,'AUC-ROC':f"{a:.4f}",
                     'Avg Precision':f"{ap:.4f}",'P@100':f"{pak:.4f}"})
    st.dataframe(pd.DataFrame(comp), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – RISK SCORING & ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Risk Scoring & Alerts")
    st.markdown("Unified 0–100 risk score: **40 pts** from rule engine + **60 pts** from ML ensemble.")

    kc1,kc2,kc3,kc4 = st.columns(4)
    mean_alert_score = alerts_df['risk_score'].mean() if len(alerts_df) > 0 else 0.0
    kc1.metric("Total Alerts",      f"{n_alerts:,}")
    kc2.metric("False Positive Rate",f"{fp_rate*100:.1f}%")
    kc3.metric("Alert → SAR Rate",  f"{sar_rate*100:.0f}%")
    kc4.metric("Mean Alert Score",  f"{mean_alert_score:.1f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Tier Distribution")
        tier_c   = df['risk_tier'].value_counts()
        tc_colors= {'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c','Critical':'#8e44ad'}
        fig, ax  = plt.subplots(figsize=(6, 4))
        bars     = ax.bar(tier_c.index, tier_c.values,
                         color=[tc_colors.get(t,'gray') for t in tier_c.index], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+50,
                    f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Risk Tier'); ax.set_ylabel('Count')
        ax.set_title('Risk Tier Distribution'); ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Risk Score Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['risk_score'], bins=50, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(risk_threshold, color='red',    ls='--', lw=2, label=f'Alert Threshold ({risk_threshold})')
        ax.axvline(80,            color='purple',  ls=':',  lw=2, label='Critical (80)')
        ax.set_xlabel('Risk Score (0–100)'); ax.set_ylabel('Count')
        ax.set_title('Risk Score Distribution'); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

    st.subheader(f"Alert Queue – Top {alert_limit} Highest-Risk Transactions")
    aq = df.nlargest(alert_limit, 'risk_score')[[
        'transaction_id','account_id','timestamp','amount','transaction_type',
        'country_risk','risk_score','risk_tier',
        'structuring','round_number','rapid_movement','high_risk_jurisdiction','velocity_spike',
        'is_suspicious'
    ]].copy()
    aq['amount']     = aq['amount'].round(2)
    aq['risk_score'] = aq['risk_score'].round(1)
    aq['CTR']        = (aq['amount'] > 10000) & (aq['transaction_type']=='cash')
    aq['SAR_Rec']    = aq['risk_score'] >= 80
    st.dataframe(aq.reset_index(drop=True), use_container_width=True, height=380)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Risk Factor Contributions (Alert Avg)")
        if len(alerts_df) > 0:
            weights  = {'structuring':15,'round_number':8,'rapid_movement':10,
                        'high_risk_jurisdiction':12,'velocity_spike':10}
            contrib_v= [alerts_df[f].mean()*w for f,w in weights.items()]
            ml_cont  = alerts_df['ml_score'].mean() * 60
            all_labs = ['Structuring','Round Num','Rapid Mvmt','Hi-Risk Jur','Velocity','ML Score']
            all_vals = contrib_v + [ml_cont]
            fig, ax  = plt.subplots(figsize=(7, 4))
            c_wf = ['#e74c3c' if v>8 else '#f39c12' if v>3 else '#2ecc71' for v in all_vals]
            bars = ax.barh(all_labs, all_vals, color=c_wf, alpha=0.85)
            ax.set_xlabel('Avg Contribution'); ax.set_title('Risk Factor Contributions')
            ax.grid(True, alpha=0.3, axis='x')
            for bar, val in zip(bars, all_vals):
                ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                        f'{val:.1f}', va='center', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.subheader("SLA Compliance (Simulated)")
        rng_sla  = np.random.default_rng(77)
        n_sim    = max(1, min(len(alerts_df), 1000))
        inv_hrs  = rng_sla.exponential(scale=30, size=n_sim)
        sla_24   = (inv_hrs <= 24).mean()*100
        sla_48   = (inv_hrs <= 48).mean()*100
        sla_72   = (inv_hrs <= 72).mean()*100
        targets  = [60, 80, 95]
        fig, ax  = plt.subplots(figsize=(6, 4))
        x        = np.arange(3)
        ax.bar(x-0.2, [sla_24,sla_48,sla_72], 0.35, label='Actual', color='steelblue', alpha=0.8)
        ax.bar(x+0.2, targets,                 0.35, label='Target', color='orange',    alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(['24h','48h','72h'])
        ax.set_ylabel('% Investigated'); ax.set_title('SLA Compliance')
        ax.set_ylim(0,110); ax.legend(); ax.grid(True,alpha=0.3,axis='y')
        for i,(v,t) in enumerate(zip([sla_24,sla_48,sla_72],targets)):
            ax.text(i-0.2, v+1, f'{v:.0f}%', ha='center', fontsize=9)
            ax.text(i+0.2, t+1, f'{t}%',     ha='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Regulatory Thresholds")
    rc1, rc2 = st.columns(2)
    ctr_cnt  = int(((df['amount']>10000)&(df['transaction_type']=='cash')).sum())
    sar_cnt  = int((df['risk_score']>=80).sum())
    rc1.metric("CTR Required (Cash > $10k)", f"{ctr_cnt:,}", "FinCEN Form 112")
    rc2.metric("SAR Recommended (Score ≥ 80)", f"{sar_cnt:,}",  "FinCEN Form 111")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – SAR REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("SAR Report Generator")
    st.markdown("Automated Suspicious Activity Report narrative compliant with FinCEN requirements.")

    total_cases  = n_alerts
    closed_cases = int(total_cases * 0.60)
    esc_cases    = n_sar
    open_cases   = max(0, total_cases - closed_cases - esc_cases)

    cm1,cm2,cm3 = st.columns(3)
    cm1.metric("Open Cases",        f"{open_cases:,}")
    cm2.metric("Closed Cases",      f"{closed_cases:,}")
    cm3.metric("Escalated (SAR)",   f"{esc_cases:,}")
    st.markdown("---")

    top100 = df.nlargest(100, 'risk_score').reset_index(drop=True)
    options= [
        f"{row['transaction_id']} | ${row['amount']:,.0f} | Score: {row['risk_score']:.1f} | {row['transaction_type'].upper()}"
        for _, row in top100.iterrows()
    ]
    selected = st.selectbox("Select a flagged transaction to investigate:", options)

    tx_idx  = options.index(selected)
    tx_row  = top100.iloc[tx_idx]
    rule_row= rules_df.iloc[tx_row.name] if tx_row.name < len(rules_df) else rules_df.iloc[tx_idx]
    tx_risk = float(tx_row['risk_score'])

    col_sar1, col_sar2 = st.columns([1.6, 1])

    with col_sar1:
        st.subheader("SAR Narrative (Auto-Generated)")
        narrative = generate_sar_narrative(tx_row, rule_row, tx_risk)
        st.markdown(narrative)

    with col_sar2:
        st.subheader("FATF 40 Recommendations")
        fatf_hits = [{'Recommendation':FATF_RECS[r][0],'Description':FATF_RECS[r][1]}
                     for r in RULE_NAMES if rule_row.get(r, 0)]
        if fatf_hits:
            st.dataframe(pd.DataFrame(fatf_hits), use_container_width=True)
        else:
            st.info("No specific FATF rules triggered. General Rec. 20 applies.")

        st.subheader("Risk Factor Breakdown")
        w_map   = {'structuring':15,'round_number':8,'rapid_movement':10,
                   'high_risk_jurisdiction':12,'velocity_spike':10}
        ml_idx  = tx_idx  # same position in the filtered df
        ml_val  = float(ml['ensemble'][ml_idx]) if ml_idx < len(ml['ensemble']) else 0.0
        bd_labs = ['Structuring','Round Num','Rapid Mvmt','Hi-Risk Jur','Velocity','ML Ensemble']
        bd_vals = [float(rule_row.get(r,0))*w for r,w in w_map.items()] + [ml_val*60]
        fig, ax = plt.subplots(figsize=(5,3.5))
        c_bd    = ['#e74c3c' if v>8 else '#f39c12' if v>3 else '#2ecc71' for v in bd_vals]
        ax.barh(bd_labs, bd_vals, color=c_bd, alpha=0.85)
        ax.set_xlabel('Score Contribution')
        ax.set_title(f'Risk Breakdown – Total: {tx_risk:.1f}/100')
        ax.grid(True, alpha=0.3, axis='x'); plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Network Visualization
    st.markdown("---")
    st.subheader("Transaction Flow Graph (Synthetic Network)")

    rng_net = np.random.default_rng(hash(str(tx_row['account_id'])) % 2**31)
    n_nodes = 8
    center  = str(tx_row['account_id'])
    peers   = [f"ACC{rng_net.integers(1,99999):05d}" for _ in range(n_nodes-1)]
    nodes   = [center] + peers
    node_risks_net = np.concatenate([[tx_risk/100],
                                      rng_net.uniform(0.1,0.9,size=n_nodes-1)])
    edges   = [(center, peers[i], rng_net.lognormal(5,1))   for i in range(3)]
    edges  += [(peers[i], peers[i+1], rng_net.lognormal(5,1)) for i in range(3)]

    fig, ax = plt.subplots(figsize=(10, 5))
    angles  = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    px, py  = np.cos(angles), np.sin(angles)

    for src, dst, amt in edges:
        si, di = nodes.index(src), nodes.index(dst)
        ax.annotate('', xy=(px[di],py[di]), xytext=(px[si],py[si]),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        ax.text((px[si]+px[di])/2, (py[si]+py[di])/2,
                f'${amt:,.0f}', fontsize=7, ha='center', color='darkblue', alpha=0.8)

    cmap = plt.cm.RdYlGn_r
    for i, (nd, nr) in enumerate(zip(nodes, node_risks_net)):
        ax.scatter(px[i], py[i], s=900 if i==0 else 450,
                   c=[cmap(nr)], zorder=5, edgecolors='black', lw=1.5)
        ax.text(px[i], py[i]-0.18, nd[:10], ha='center', fontsize=7,
                fontweight='bold' if i==0 else 'normal')
        if i == 0:
            ax.text(px[i], py[i], '*', ha='center', va='center', fontsize=14, color='white')

    ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"Transaction Network: {center}  (* = Subject Account | Color = Risk)", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1)); sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, pad=0.02, label='Risk Level')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # BSA / FinCEN Reference
    st.markdown("---")
    st.subheader("Bank Secrecy Act – Reporting Requirements")
    bsa1, bsa2 = st.columns(2)
    with bsa1:
        st.markdown("""
**Currency Transaction Report (CTR)**
- Threshold: Cash transactions **> $10,000**
- Filing: FinCEN Form 112
- Deadline: 15 calendar days after transaction date
- Statute: 31 U.S.C. § 5313
        """)
    with bsa2:
        st.markdown("""
**Suspicious Activity Report (SAR)**
- Threshold: **≥ $5,000** with suspicious activity indicators
- Filing: FinCEN Form 111
- Deadline: 30 calendar days after initial detection
- Statute: 31 U.S.C. § 5318(g)
        """)

st.markdown("---")
st.caption("AMLDetect Platform v2.0  |  Built with Streamlit  |  For demonstration and educational purposes only  |  Not for production deployment without regulatory approval")
