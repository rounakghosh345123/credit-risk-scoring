import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Load everything saved from the notebook ----------
coefficients = joblib.load('coefficients.pkl')
bin_edges_dict = joblib.load('bin_edges.pkl')
woe_lookups = joblib.load('woe_lookups.pkl')
selected_features = joblib.load('selected_features.pkl')
scorecard_params = joblib.load('scorecard_params.pkl')

factor = scorecard_params['factor']
offset = scorecard_params['offset']
base_points = scorecard_params['base_points']

st.set_page_config(page_title="Credit Risk Scorecard", page_icon="💳")
st.title("💳 Credit Risk Scorecard")
st.write("Enter applicant details to generate a credit score and approval recommendation.")

# ---------- Input widgets ----------
st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    monthly_income = st.number_input("Monthly Income ($)", 0, 100000, 5000)
    debt_ratio = st.number_input("Debt Ratio (monthly debt payments / monthly income)", 0.0, 5.0, 0.3, step=0.01)
    revolving_util = st.number_input(
        "Revolving Credit Utilization (balance / credit limit, as a fraction)",
        0.0, 2.0, 0.3, step=0.01
    )
    num_dependents = st.number_input("Number of Dependents", 0, 20, 0)

with col2:
    open_credit_lines = st.number_input("Number of Open Credit Lines/Loans", 0, 50, 5)
    real_estate_loans = st.number_input("Number of Real Estate Loans/Lines", 0, 20, 1)
    late_30_59 = st.number_input("Times 30-59 Days Past Due (last 2 years)", 0, 20, 0)
    late_60_89 = st.number_input("Times 60-89 Days Past Due (last 2 years)", 0, 20, 0)
    late_90 = st.number_input("Times 90+ Days Late (last 2 years)", 0, 20, 0)

# ---------- Build a raw input row matching the ORIGINAL dataset's column names ----------
raw_input = {
    'RevolvingUtilizationOfUnsecuredLines': revolving_util,
    'age': age,
    'NumberOfTime30-59DaysPastDueNotWorse': min(late_30_59, 10),   # same cap applied as training
    'DebtRatio': debt_ratio,
    'MonthlyIncome': monthly_income,
    'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
    'NumberOfTimes90DaysLate': min(late_90, 10),                   # same cap applied as training
    'NumberRealEstateLoansOrLines': real_estate_loans,
    'NumberOfTime60-89DaysPastDueNotWorse': min(late_60_89, 10),   # same cap applied as training
    'NumberOfDependents': num_dependents,
}

raw_series = pd.Series(raw_input)

# ---------- Apply the SAME WOE binning/mapping used during training ----------
def apply_woe_single(value, feature, bin_edges_dict, woe_lookups):
    """
    Bins a single applicant's value using the bin edges saved from training,
    then looks up that bucket's WOE value via bin INDEX (0, 1, 2...) rather than
    the raw Interval object — pd.qcut's Interval boundaries during training don't
    exactly match pd.cut's Interval boundaries once the outer edges are widened to
    -inf/inf here, so an Interval-keyed lookup would silently fail. Falls back to
    0 (neutral) if something still falls outside every bucket.
    """
    edges = bin_edges_dict[feature].copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    bin_index = pd.cut(pd.Series([value]), bins=edges, labels=False, include_lowest=True)
    bucket = bin_index.iloc[0]

    woe_lookup = woe_lookups[feature]
    return woe_lookup.get(bucket, 0.0)

woe_input = {}
for feat in selected_features:
    woe_input[feat + '_WOE'] = apply_woe_single(raw_series[feat], feat, bin_edges_dict, woe_lookups)

# ---------- Calculate score using the saved scorecard formula ----------
points_breakdown = {}
for feat_woe_col, coef in coefficients.items():
    points_breakdown[feat_woe_col] = -factor * coef * woe_input[feat_woe_col]

total_score = base_points + sum(points_breakdown.values())
total_score = max(300, min(850, total_score))  # clip to the standard 300-850 display range

# ---------- Risk band ----------
def get_risk_band(score):
    if score >= 610:
        return "Excellent", "🟢"
    elif score >= 590:
        return "Good", "🟢"
    elif score >= 560:
        return "Fair", "🟡"
    else:
        return "Poor", "🔴"

band, emoji = get_risk_band(total_score)

# ---------- Approval cutoff ----------
approval_cutoff = 590  # adjust this based on your Phase 6 business-framing analysis

# ---------- Display ----------
if st.button("Calculate Credit Score"):
    st.subheader("Result")
    st.metric("Credit Score", f"{total_score:.0f}")
    st.write(f"**Risk Band:** {emoji} {band}")

    if total_score >= approval_cutoff:
        st.success(f"✅ Recommended: APPROVE (score ≥ cutoff of {approval_cutoff})")
    else:
        st.error(f"❌ Recommended: REJECT (score below cutoff of {approval_cutoff})")

    with st.expander("See full points breakdown"):
        breakdown_df = pd.DataFrame.from_dict(
            points_breakdown, orient='index', columns=['Points Contribution']
        ).sort_values('Points Contribution')
        st.write(f"Base points: {base_points:.1f}")
        st.dataframe(breakdown_df.style.format("{:.1f}"))
