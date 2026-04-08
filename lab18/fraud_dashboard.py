"""
=============================================================================
Credit Card Fraud Detection — Interactive Threshold & Model Comparison Dashboard
=============================================================================
Streamlit app that loads the Kaggle Credit Card Fraud dataset, trains Logistic
Regression and Random Forest classifiers, then lets you drag a threshold slider
to watch every metric update in real time.

Requirements (pip install):
    streamlit  scikit-learn  matplotlib  pandas  numpy

Run with:
    streamlit run fraud_dashboard.py

Dataset:
    Place `creditcard.csv` in the same directory, or update DATA_PATH below.
    Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------------------------
# 0. PAGE CONFIG & CUSTOM CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Lab",
    page_icon="🔍",
    layout="wide",
)

# Inject a minimal dark-accented style so the dashboard feels polished
# without depending on external assets or JS.
st.markdown(
    """
    <style>
    /* ---- global overrides ---- */
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        color: #f0f0f0;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .metric-card .label { font-size: 0.78rem; opacity: 0.7; margin-bottom: 2px; }
    .metric-card .value { font-size: 1.55rem; font-weight: 700; }
    .cost-card {
        background: linear-gradient(135deg, #6c1d1d 0%, #8b2e2e 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Credit Card Fraud Detection Dashboard")
st.caption(
    "Logistic Regression vs. Random Forest  ·  Threshold tuning  ·  Dollar-cost analysis"
)

# ---------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING (cached so it runs only once)
# ---------------------------------------------------------------------------
DATA_PATH = "creditcard.csv"


@st.cache_data(show_spinner="Loading and splitting dataset …")
def load_and_split(path: str):
    """
    Read the Kaggle credit-card CSV, scale the 'Amount' feature, drop 'Time',
    and return the standard train/test split objects your notebook already uses.
    """
    df = pd.read_csv(path)

    # 'Time' is seconds since first txn — not useful for a cross-sectional model
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    # Amount has a very wide range; the V1-V28 PCA features are already scaled
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. MODEL TRAINING (cached with st.cache_resource for sklearn estimators)
# ---------------------------------------------------------------------------
#   Why cache_resource and not cache_data?
#   Fitted sklearn estimators are not trivially serialisable; cache_resource
#   stores the Python object in memory by reference, which is both faster and
#   avoids hashing issues. The trade-off is that the object is shared across
#   sessions, which is fine for a read-only model.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Training Logistic Regression …")
def train_logistic_regression(_X_train, _y_train):
    """Fit a regularised logistic regression (matching most lab defaults)."""
    log_reg = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=42, n_jobs=-1
    )
    log_reg.fit(_X_train, _y_train)
    return log_reg


@st.cache_resource(show_spinner="Training Random Forest (this may take ~30 s) …")
def train_random_forest(_X_train, _y_train):
    """
    Fit a Random Forest. We keep n_estimators modest (100) so the dashboard
    stays responsive on a laptop; increase to 300+ for production work.
    """
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
    )
    rf.fit(_X_train, _y_train)
    return rf


# ---------------------------------------------------------------------------
# 3. LOAD DATA & TRAIN MODELS
# ---------------------------------------------------------------------------
try:
    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"**`{DATA_PATH}` not found.**  \n"
        "Download the Kaggle Credit Card Fraud dataset and place `creditcard.csv` "
        "in the same folder as this script, or update `DATA_PATH` at the top."
    )
    st.stop()

log_reg = train_logistic_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Predicted probabilities for the POSITIVE class (fraud = 1)
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Keep the notebook variable alias so students can cross-reference
y_prob = y_prob_lr  # default alias used in the lab notebook

# ---------------------------------------------------------------------------
# 4. SIDEBAR — THRESHOLD SLIDER & COST PARAMETERS
# ---------------------------------------------------------------------------
#   HOW THE SLIDER TRIGGERS RE-RUNS:
#   Streamlit's execution model is "top-to-bottom re-run on every widget
#   change". When you drag the slider, Streamlit stores the new value in its
#   widget state, then re-executes this entire script. Because the heavy work
#   (data loading, model training) is cached, only the lightweight metric
#   calculations below actually recompute — so updates feel instantaneous.
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Controls")
threshold = st.sidebar.slider(
    "Classification threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="P(fraud) ≥ threshold → predict fraud. Lower = more recalls, more false alarms.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("💵 Cost assumptions")

# Typical fraud-detection costs (adjustable):
#   • False Negative (missed fraud)  — the bank eats the full transaction amount;
#     a common proxy is the *average* fraudulent transaction (~$120-$150).
#   • False Positive (legitimate txn flagged) — manual review + customer friction;
#     industry estimates range $5-$50 per incident.
cost_fn = st.sidebar.number_input(
    "Cost per False Negative ($)", min_value=0, value=150, step=10,
    help="Average loss when a fraud slips through (missed detection).",
)
cost_fp = st.sidebar.number_input(
    "Cost per False Positive ($)", min_value=0, value=10, step=1,
    help="Cost of incorrectly flagging a legitimate transaction.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌲 Random Forest")
show_rf = st.sidebar.checkbox("Show RF comparison panel", value=True)


# ---------------------------------------------------------------------------
# 5. HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_proba, t):
    """Return a dict of classification metrics at threshold *t*."""
    y_pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # DOLLAR-COST METRIC
    # -------------------
    # Total cost = (missed frauds × avg fraud loss) + (false alarms × review cost)
    #
    #   cost = FN * cost_fn  +  FP * cost_fp
    #
    # This is an *expected* operational cost: it monetises the two error types
    # so you can compare thresholds on a single business-relevant scale.
    total_cost = fn * cost_fn + fp * cost_fp

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return dict(
        tn=tn, fp=fp, fn=fn, tp=tp,
        precision=prec, recall=rec, f1=f1,
        total_cost=total_cost, y_pred=y_pred,
    )


def metric_html(label: str, value: str, css_class: str = "metric-card") -> str:
    return (
        f'<div class="{css_class}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# 6. PANEL 1 — LOGISTIC REGRESSION: THRESHOLD EXPLORER
# ---------------------------------------------------------------------------

st.markdown("## Panel 1 · Logistic Regression — Threshold Explorer")

m = compute_metrics(y_test, y_prob_lr, threshold)

# ---- Metric cards row ----
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(metric_html("Precision", f"{m['precision']:.3f}"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_html("Recall", f"{m['recall']:.3f}"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_html("F1 Score", f"{m['f1']:.3f}"), unsafe_allow_html=True)
with c4:
    st.markdown(
        metric_html("Total Cost", f"${m['total_cost']:,.0f}", "metric-card cost-card"),
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(metric_html("Threshold", f"{threshold:.2f}"), unsafe_allow_html=True)

st.write("")  # spacer

# ---- Confusion matrix + cost curve side by side ----
col_cm, col_cost = st.columns([1, 1.4])

with col_cm:
    st.markdown("#### Confusion Matrix")
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3.2))
    cax = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                       fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Legit (pred)", "Fraud (pred)"])
    ax_cm.set_yticklabels(["Legit (true)", "Fraud (true)"])
    ax_cm.set_title(f"Threshold = {threshold:.2f}", fontsize=10)
    fig_cm.tight_layout()
    st.pyplot(fig_cm, width="stretch")
    plt.close(fig_cm)

with col_cost:
    st.markdown("#### Dollar Cost vs. Threshold")

    # Sweep thresholds from 0.01 to 0.99 to draw the full cost curve
    thresholds_sweep = np.linspace(0.01, 0.99, 200)
    costs_sweep = []
    f1_sweep = []
    for t in thresholds_sweep:
        _m = compute_metrics(y_test, y_prob_lr, t)
        costs_sweep.append(_m["total_cost"])
        f1_sweep.append(_m["f1"])

    # Find the cost-minimising and F1-maximising operating points
    idx_min_cost = int(np.argmin(costs_sweep))
    idx_max_f1 = int(np.argmax(f1_sweep))
    t_min_cost = thresholds_sweep[idx_min_cost]
    t_max_f1 = thresholds_sweep[idx_max_f1]

    fig_cost, ax_cost = plt.subplots(figsize=(5.5, 3.2))
    ax_cost.plot(thresholds_sweep, costs_sweep, color="#e74c3c", linewidth=2, label="Total cost")
    ax_cost.axvline(threshold, color="#3498db", linestyle="--", linewidth=1.3, label=f"Current t={threshold:.2f}")
    ax_cost.axvline(t_min_cost, color="#2ecc71", linestyle=":", linewidth=1.3,
                    label=f"Min cost t={t_min_cost:.2f}")
    ax_cost.axvline(t_max_f1, color="#f39c12", linestyle=":", linewidth=1.3,
                    label=f"Max F1 t={t_max_f1:.2f}")
    ax_cost.scatter([threshold], [m["total_cost"]], color="#3498db", s=60, zorder=5)
    ax_cost.set_xlabel("Threshold")
    ax_cost.set_ylabel("Total Cost ($)")
    ax_cost.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_cost.legend(fontsize=7, loc="upper right")
    ax_cost.set_title("FN×$cost_fn + FP×$cost_fp", fontsize=9)
    fig_cost.tight_layout()
    st.pyplot(fig_cost, width="stretch")
    plt.close(fig_cost)

# ---- ROC and PR curves for Logistic Regression ----
st.markdown("#### ROC & Precision-Recall Curves (Logistic Regression)")
col_roc, col_pr = st.columns(2)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
pr_auc_lr = auc(rec_lr, prec_lr)

with col_roc:
    fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.5))
    ax_roc.plot(fpr_lr, tpr_lr, color="#2980b9", linewidth=2,
                label=f"LogReg ROC-AUC = {roc_auc_lr:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random baseline")
    # Mark the current operating point on the ROC
    _y_pred_t = (y_prob_lr >= threshold).astype(int)
    _cm_t = confusion_matrix(y_test, _y_pred_t).ravel()
    fpr_t = _cm_t[1] / (_cm_t[1] + _cm_t[0])  # FP / (FP + TN)
    tpr_t = _cm_t[3] / (_cm_t[3] + _cm_t[2])  # TP / (TP + FN)
    ax_roc.scatter([fpr_t], [tpr_t], color="#e74c3c", s=70, zorder=5, label=f"t = {threshold:.2f}")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(fontsize=7)
    fig_roc.tight_layout()
    st.pyplot(fig_roc, width="stretch")
    plt.close(fig_roc)

with col_pr:
    fig_pr, ax_pr = plt.subplots(figsize=(4.5, 3.5))
    ax_pr.plot(rec_lr, prec_lr, color="#8e44ad", linewidth=2,
               label=f"LogReg PR-AUC = {pr_auc_lr:.4f}")
    # Baseline for PR curve is the prevalence (≈ 0.0017)
    prevalence = y_test.mean()
    ax_pr.axhline(prevalence, color="k", linestyle="--", linewidth=0.8,
                  label=f"Random baseline ({prevalence:.4f})")
    # Operating point
    prec_t = m["precision"]
    rec_t = m["recall"]
    ax_pr.scatter([rec_t], [prec_t], color="#e74c3c", s=70, zorder=5, label=f"t = {threshold:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(fontsize=7)
    fig_pr.tight_layout()
    st.pyplot(fig_pr, width="stretch")
    plt.close(fig_pr)

# ---------------------------------------------------------------------------
# 7. PANEL 2 — RANDOM FOREST COMPARISON (optional via sidebar checkbox)
# ---------------------------------------------------------------------------

if show_rf:
    st.markdown("---")
    st.markdown("## Panel 2 · Model Comparison — Logistic Regression vs. Random Forest")

    m_rf = compute_metrics(y_test, y_prob_rf, threshold)

    # Side-by-side metric comparison
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("**Logistic Regression**")
        st.markdown(
            f"Precision **{m['precision']:.3f}** · Recall **{m['recall']:.3f}** · "
            f"F1 **{m['f1']:.3f}** · Cost **${m['total_cost']:,.0f}**"
        )
    with comp_col2:
        st.markdown("**Random Forest**")
        st.markdown(
            f"Precision **{m_rf['precision']:.3f}** · Recall **{m_rf['recall']:.3f}** · "
            f"F1 **{m_rf['f1']:.3f}** · Cost **${m_rf['total_cost']:,.0f}**"
        )

    # ---- ROC comparison ----
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

    # ---- PR comparison ----
    prec_rf_curve, rec_rf_curve, _ = precision_recall_curve(y_test, y_prob_rf)
    pr_auc_rf = auc(rec_rf_curve, prec_rf_curve)

    col_roc2, col_pr2 = st.columns(2)

    with col_roc2:
        st.markdown("#### ROC Curve Comparison")
        fig_roc2, ax_roc2 = plt.subplots(figsize=(5, 4))
        ax_roc2.plot(fpr_lr, tpr_lr, color="#2980b9", linewidth=2,
                     label=f"LogReg  AUC = {roc_auc_lr:.4f}")
        ax_roc2.plot(fpr_rf, tpr_rf, color="#27ae60", linewidth=2,
                     label=f"RF      AUC = {roc_auc_rf:.4f}")
        ax_roc2.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax_roc2.set_xlabel("False Positive Rate")
        ax_roc2.set_ylabel("True Positive Rate")
        ax_roc2.set_title("ROC-AUC Comparison")
        ax_roc2.legend(fontsize=8)
        fig_roc2.tight_layout()
        st.pyplot(fig_roc2, width="stretch")
        plt.close(fig_roc2)

    with col_pr2:
        # WHY PR-AUC IS MORE INFORMATIVE THAN ROC-AUC FOR IMBALANCED DATA:
        # ------------------------------------------------------------------
        # ROC-AUC measures the trade-off between TPR and FPR. When negatives
        # vastly outnumber positives (here ~577:1), even a model with many
        # false positives shows a *tiny* FPR because the denominator (TN+FP)
        # is enormous. This makes ROC-AUC look deceptively high (often >0.95)
        # even for mediocre models.
        #
        # PR-AUC, by contrast, focuses on how well the model retrieves the
        # rare positives WITHOUT drowning them in false alarms. Precision
        # (TP / (TP+FP)) is directly hurt by false positives regardless of
        # how many true negatives exist. So PR-AUC gives you a much more
        # honest picture of performance on the minority class.
        st.markdown("#### Precision-Recall Curve Comparison")
        fig_pr2, ax_pr2 = plt.subplots(figsize=(5, 4))
        ax_pr2.plot(rec_lr, prec_lr, color="#8e44ad", linewidth=2,
                    label=f"LogReg  PR-AUC = {pr_auc_lr:.4f}")
        ax_pr2.plot(rec_rf_curve, prec_rf_curve, color="#d35400", linewidth=2,
                    label=f"RF      PR-AUC = {pr_auc_rf:.4f}")
        ax_pr2.axhline(prevalence, color="k", linestyle="--", linewidth=0.8,
                       label=f"Baseline = {prevalence:.4f}")
        ax_pr2.set_xlabel("Recall")
        ax_pr2.set_ylabel("Precision")
        ax_pr2.set_title("PR-AUC Comparison (more informative for fraud)")
        ax_pr2.legend(fontsize=8)
        fig_pr2.tight_layout()
        st.pyplot(fig_pr2, width="stretch")
        plt.close(fig_pr2)

    # ---- Dual cost curves ----
    st.markdown("#### Dollar-Cost Comparison Across Thresholds")
    costs_rf_sweep = []
    for t in thresholds_sweep:
        _m_rf = compute_metrics(y_test, y_prob_rf, t)
        costs_rf_sweep.append(_m_rf["total_cost"])

    fig_dual, ax_dual = plt.subplots(figsize=(8, 3.5))
    ax_dual.plot(thresholds_sweep, costs_sweep, color="#2980b9", linewidth=2, label="LogReg cost")
    ax_dual.plot(thresholds_sweep, costs_rf_sweep, color="#27ae60", linewidth=2, label="RF cost")
    ax_dual.axvline(threshold, color="#e74c3c", linestyle="--", linewidth=1.3,
                    label=f"Current t = {threshold:.2f}")
    ax_dual.set_xlabel("Threshold")
    ax_dual.set_ylabel("Total Cost ($)")
    ax_dual.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_dual.legend(fontsize=8)
    ax_dual.set_title("Which model costs less at each threshold?")
    fig_dual.tight_layout()
    st.pyplot(fig_dual, width="stretch")
    plt.close(fig_dual)


# ---------------------------------------------------------------------------
# 8. INTERPRETATION GUIDE (collapsible)
# ---------------------------------------------------------------------------

with st.expander("📖 How to interpret this dashboard", expanded=False):
    st.markdown(
        """
**What happens as you drag the threshold from 0.01 → 0.99?**

| Direction | Effect |
|-----------|--------|
| **Low threshold (→ 0.01)** | Almost every transaction is flagged as fraud. **Recall ≈ 1.0** (you catch all fraud), but **Precision plummets** (thousands of false alarms). FP cost dominates the dollar-cost curve. |
| **High threshold (→ 0.99)** | Almost nothing is flagged. **Precision may be high** (the few flags are correct), but **Recall → 0** (you miss most fraud). FN cost dominates because missed fraud is expensive. |
| **Sweet spot** | Somewhere in between, the total cost curve reaches a **minimum**. This is the *cost-minimising operating point* — the threshold where the combined penalty of false alarms and missed fraud is lowest. |

**Cost-minimising vs. F1-maximising threshold:**

- The **F1-maximising** threshold balances Precision and Recall *equally* in a
  harmonic mean. It has no notion of dollars.
- The **cost-minimising** threshold weights the two error types by their
  *business cost*. When `cost_fn >> cost_fp` (typical in fraud), the
  cost-optimal threshold is **lower** than the F1-optimal one, because missing
  a fraud is far more expensive than reviewing a false alarm.
- **Recommendation:** Use the cost-minimising threshold for deployment
  decisions, and F1 for model-selection comparisons on a level playing field.

**Why PR-AUC > ROC-AUC for this dataset:**

With only 0.17 % fraud, ROC-AUC is inflated because FPR stays tiny even with
many false positives. PR-AUC directly penalises false positives through
Precision, giving a more honest assessment of minority-class performance.
        """
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "ECON 5200 — Applied Data Analytics in Economics  \n"
    "Lab Extension: Fraud Detection Dashboard"
)
