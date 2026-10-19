import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

############################
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .block-container {
        max-width: 75%;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)
############################
# Same math as merton.py / public.calculate_merton_pd():
#   Trigger 2 (negative equity): P(T2) = N(-DD),
#       DD = (ln(V/L) + (mu - sigma^2/2) * t) / (sigma * sqrt(t))
#   Trigger 1 (income shock):    P(T1) = unemployment * DTI / (1 - DTI)
#   Joint PD:                    P1 + P2 - P1*P2 (triggers treated as independent)
# Inputs are in dollars: V = asset price, L = loan amount. LTV = L / V is derived, never entered.
# Asset price is also an axis option: the sweep holds the loan fixed and moves V.
############################

# key: (label, slider min, slider max, default, step, axis range default, is_percent)
# "asset" comes from the dollar input, not a slider; its axis range is set off the loan amount.
VARIABLES = {
    "asset": ("Asset price ($)",            None, None, None,  None,  None,          False),
    "dti":   ("Debt-to-income (DTI)",       0.05, 0.65, 0.43,  0.01,  (0.10, 0.60), True),
    "unemp": ("Unemployment rate",          0.01, 0.15, 0.041, 0.001, (0.02, 0.12), True),
    "mu":    ("Annual price drift (μ)",    -0.15, 0.15, 0.04,  0.005, (-0.10, 0.10), True),
    "sigma": ("Annual price volatility (σ)", 0.01, 0.15, 0.045, 0.001, (0.02, 0.12), True),
    "t":     ("Horizon (years)",            0.25, 5.00, 1.00,  0.25,  (0.25, 5.00), False),
}
METRICS = {
    "joint_pd": "Joint PD",
    "p_t2":     "P(T2) negative equity",
    "p_t1":     "P(T1) income shock",
}
GRID = 150
# color stops: green at 0, cream at COLOR_MID, full red at COLOR_RED and above (RdYlGn ends, like the yield heatmap)
COLOR_MID, COLOR_RED = 0.10, 0.25
# full RdYlGn ramp (not just its 3 anchors) so the blend reads as a smooth gradient, not bands
_GREEN_HALF = ["#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#ffffbf"]
_RED_HALF = ["#ffffbf", "#fee08b", "#fdae61", "#f46d43", "#d73027"]
COLOR_RANGE = _GREEN_HALF + _RED_HALF[1:]
COLOR_DOMAIN = ([COLOR_MID * i / 4 for i in range(5)]
                + [COLOR_MID + (COLOR_RED - COLOR_MID) * i / 4 for i in range(1, 5)])

_erf = np.frompyfunc(math.erf, 1, 1)


def norm_cdf(x):
    return 0.5 * (1.0 + np.asarray(_erf(np.asarray(x, dtype=float) / math.sqrt(2.0)), dtype=float))


def merton(ltv, dti, unemp, mu, sigma, t):
    """Vectorized dual-trigger model. All inputs as decimals; arrays broadcast."""
    dd = (np.log(1.0 / ltv) + (mu - sigma ** 2 / 2.0) * t) / (sigma * np.sqrt(t))
    p_t2 = 1.0 - norm_cdf(dd)
    p_t1 = unemp * dti / (1.0 - dti)
    joint = p_t1 + p_t2 - p_t1 * p_t2
    pct_drop = dd * sigma * np.sqrt(t)
    return dd, p_t1, p_t2, joint, pct_drop


def run(asset, loan, **rest):
    """Dollar inputs -> model. LTV = loan / asset price; returns (ltv, dd, p_t1, p_t2, joint, pct_drop)."""
    ltv = loan / asset
    return (ltv, *merton(ltv, **rest))


def risk_tier(pd_value):
    if pd_value >= 0.15:
        return "1 Critical"
    if pd_value >= 0.10:
        return "2 High"
    if pd_value >= 0.05:
        return "3 Elevated"
    if pd_value >= 0.02:
        return "4 Moderate"
    return "5 Low"


def fmt(key, value):
    if key == "asset":
        return f"${value:,.0f}"
    return f"{value:.1%}" if VARIABLES[key][6] else f"{value:.2f}"


############################
st.title("Mortgage Risk Scenario Heatmap")
st.sidebar.markdown("[How the model works ↗](https://danieljohnston638537.substack.com/p/i-built-what-the-rating-agencies)")
st.caption("Move the sliders to set a borrower and market, then pick any two inputs to sweep across the heatmap. "
           "Every square is the same Merton dual-trigger model used in the county map.")

with st.expander("How to read this"):
    st.markdown("""
- **Each square** is one scenario: the two axis inputs take the square's values, and every other input stays where the sidebar sliders put it.
- **Green is low risk, cream is about 10%, and red is 25% or more.** Hover over a square to see its joint PD and both triggers.
- **The black ring** marks your current slider scenario, when it falls inside the chart's range.
- **P(T1), income shock:** the chance a job loss pushes the borrower into default. It scales with unemployment and with DTI as an odds ratio, DTI / (1 − DTI). The model doesn't cap it, so extreme DTI and unemployment can push it past 100%; the chart clips at 100%.
- **P(T2), negative equity:** the chance the home's value falls below the loan balance within the horizon. It's driven by the equity cushion (LTV) and by how prices move (drift and volatility).
- **Joint PD** combines the two, treating them as independent: P1 + P2 − P1·P2.
- **Asset price and loan amount** are in dollars. LTV is worked out from them: loan ÷ asset price. To test a lower value, re-enter a lower asset price; the loan stays where you set it.
- **Asset price as an axis** sweeps the home's value with the loan held fixed, so you can see where the cushion runs out.
""")

############################
with st.sidebar:
    st.header("Scenario")
    asset = st.number_input("Asset price ($)", min_value=10_000, max_value=10_000_000,
                            value=400_000, step=10_000)
    loan = st.number_input("Loan amount ($)", min_value=1_000, max_value=10_000_000,
                           value=340_000, step=10_000)
    # rate and term only feed the payment card; the PD model doesn't use them
    rate = st.number_input("Interest rate (%)", min_value=0.0, max_value=25.0, value=6.5, step=0.125,
                           format="%.3f") / 100
    term_years = st.selectbox("Term (years)", [30, 20, 15, 10], index=0)
    scenario = {"asset": float(asset)}
    for key, (label, lo, hi, default, step, _, is_pct) in VARIABLES.items():
        if key == "asset":
            continue
        if is_pct:
            value = st.slider(f"{label} (%)", lo * 100, hi * 100, default * 100, step * 100,
                              format="%.1f", key=f"s_{key}")
            scenario[key] = value / 100
        else:
            scenario[key] = st.slider(label, lo, hi, default, step, key=f"s_{key}")

############################
ltv, dd, p_t1, p_t2, joint, pct_drop = (float(v) for v in run(loan=loan, **scenario))

with st.sidebar:
    st.caption(f"LTV {ltv:.1%} · equity \\${asset - loan:,.0f}")

n_payments, monthly_rate = term_years * 12, rate / 12
payment = loan / n_payments if monthly_rate == 0 else loan * monthly_rate / (1 - (1 + monthly_rate) ** -n_payments)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Payment", f"${payment:,.0f}",
          help="Implied monthly principal and interest from loan, rate and term. No taxes, insurance or escrow")
c2.metric("LTV", f"{ltv:.1%}", help="Loan amount ÷ asset price")
c3.metric("P(T1)", f"{p_t1:.2%}", help="Income shock: unemployment × DTI / (1 − DTI)")
c4.metric("P(T2)", f"{p_t2:.2%}", help="Negative equity: chance the home falls below the loan balance within the horizon")
c5.metric("Cushion", "Underwater" if dd < 0 else f"{pct_drop:.1%}",
          help="Price drop that would put the loan underwater")
c6.metric("Joint PD", f"{joint:.2%}", help="Chance of default from either trigger: P1 + P2 − P1·P2")
c7.metric("Tier", risk_tier(joint), help="Same cutoffs as the warehouse model: 15% / 10% / 5% / 2%")
cushion_dollars = "" if dd < 0 else f" · a {pct_drop:.1%} drop is \\${asset * pct_drop:,.0f}"
# floor, not a true income: DTI also covers other debts, taxes and insurance, the payment is P&I only
implied_income = payment * 12 / scenario["dti"]
st.caption(f"Asset \\${asset:,.0f} · loan \\${loan:,.0f} · implied income at least \\${implied_income:,.0f}/yr · "
           f"LTV {ltv:.1%}{cushion_dollars} · "
           f"distance to default {dd:.2f} standard deviations")

############################
keys = list(VARIABLES)
labels = {k: VARIABLES[k][0] for k in keys}

a1, a2, a3 = st.columns(3)
x_key = a1.selectbox("Across (x-axis)", keys, index=keys.index("sigma"), format_func=labels.get)
y_options = [k for k in keys if k != x_key]
y_key = a2.selectbox("Down (y-axis)", y_options,
                     index=y_options.index("asset") if "asset" in y_options else 0, format_func=labels.get)
metric_key = a3.selectbox("Color by", list(METRICS), format_func=METRICS.get)


def axis_range(key, column):
    if key == "asset":
        # bounds follow the loan; keyed on it so the range resets when the loan changes
        k = lambda m: int(round(loan * m, -3))
        r = column.slider("Asset price range ($)", k(0.5), k(2.5), (k(0.9), k(1.6)), 1_000,
                          format="$%d", key=f"r_asset_{loan}")
        return float(r[0]), float(r[1])
    label, lo, hi, _, step, (d_lo, d_hi), is_pct = VARIABLES[key]
    if is_pct:
        r = column.slider(f"{label} range (%)", lo * 100, hi * 100, (d_lo * 100, d_hi * 100), step * 100,
                          format="%.1f", key=f"r_{key}")
        return r[0] / 100, r[1] / 100
    return column.slider(f"{label} range", lo, hi, (d_lo, d_hi), step, key=f"r_{key}")


r1, r2 = st.columns(2)
x_lo, x_hi = axis_range(x_key, r1)
y_lo, y_hi = axis_range(y_key, r2)
if x_lo == x_hi or y_lo == y_hi:
    st.warning("Each axis range needs two different end points.")
    st.stop()

############################
xs = np.linspace(x_lo, x_hi, GRID)
ys = np.linspace(y_lo, y_hi, GRID)
xx, yy = np.meshgrid(xs, ys)
inputs = {k: np.full_like(xx, v) for k, v in scenario.items()}
inputs[x_key] = xx
inputs[y_key] = yy
g_ltv, _, g_p1, g_p2, g_joint, _ = run(loan=loan, **inputs)

grid = pd.DataFrame({
    "x": xx.ravel(),
    "y": yy.ravel(),
    "ltv": g_ltv.ravel(),
    "joint_pd": g_joint.ravel(),
    "p_t1": g_p1.ravel(),
    "p_t2": g_p2.ravel(),
})
grid["value"] = grid[metric_key].clip(upper=1.0)
grid["x_label"] = [fmt(x_key, v) for v in grid["x"]]
grid["y_label"] = [fmt(y_key, v) for v in grid["y"]]

x_step = (x_hi - x_lo) / (GRID - 1)
y_step = (y_hi - y_lo) / (GRID - 1)
# each square is centered on the value it was computed at
grid["x0"], grid["x2"] = grid["x"] - x_step / 2, grid["x"] + x_step * 0.8
grid["y0"], grid["y2"] = grid["y"] - y_step / 2, grid["y"] + y_step * 0.8
AXIS_FMT = {"asset": "$,.0f"}
x_fmt = AXIS_FMT.get(x_key, ".0%" if VARIABLES[x_key][6] else ".2f")
y_fmt = AXIS_FMT.get(y_key, ".0%" if VARIABLES[y_key][6] else ".2f")

heat = alt.Chart(grid).mark_rect(strokeWidth=0, clip=True).encode(
    x=alt.X("x0:Q", title=labels[x_key], axis=alt.Axis(format=x_fmt, grid=False, tickCount=8),
            scale=alt.Scale(domain=[x_lo - x_step / 2, x_hi + x_step / 2], nice=False)),
    x2="x2:Q",
    y=alt.Y("y0:Q", title=labels[y_key], axis=alt.Axis(format=y_fmt, grid=False, tickCount=8),
            scale=alt.Scale(domain=[y_lo - y_step / 2, y_hi + y_step / 2], nice=False)),
    y2="y2:Q",
    color=alt.Color("value:Q", title=METRICS[metric_key],
                    scale=alt.Scale(domain=COLOR_DOMAIN, range=COLOR_RANGE, interpolate="hcl", clamp=True),
                    legend=alt.Legend(format=".0%", gradientLength=320, values=[0, 0.05, 0.10, 0.15, 0.20, 0.25])),
    tooltip=[
        alt.Tooltip("x_label:N", title=labels[x_key]),
        alt.Tooltip("y_label:N", title=labels[y_key]),
        alt.Tooltip("ltv:Q", title="LTV", format=".1%"),
        alt.Tooltip("joint_pd:Q", title="Joint PD", format=".2%"),
        alt.Tooltip("p_t1:Q", title="P(T1) income shock", format=".2%"),
        alt.Tooltip("p_t2:Q", title="P(T2) negative equity", format=".2%"),
    ],
)

layers = [heat]
sx, sy = scenario[x_key], scenario[y_key]
if x_lo <= sx <= x_hi and y_lo <= sy <= y_hi:
    marker = pd.DataFrame({"x0": [sx], "y0": [sy]})
    # white halo under a black ring so the marker reads on both pale and dark-red squares
    layers.append(alt.Chart(marker).mark_point(shape="circle", size=240, filled=False,
                                               color="white", strokeWidth=5).encode(x="x0:Q", y="y0:Q"))
    layers.append(alt.Chart(marker).mark_point(shape="circle", size=240, filled=False,
                                               color="black", strokeWidth=2).encode(x="x0:Q", y="y0:Q"))

st.altair_chart(alt.layer(*layers).properties(height=560), width="stretch")

st.caption("Inputs not on an axis are held at the sidebar values. Color runs green (0%) to cream (10%) to red, "
           "and anything at 25% or above is full red. P(T1) is uncapped in the model. Model: Merton dual-trigger, same formulas as the county map (see Data Dictionary on the main page).")

with st.expander("Data for this heatmap"):
    table = grid[["x_label", "y_label", "ltv", "joint_pd", "p_t1", "p_t2"]].rename(columns={
        "x_label": labels[x_key], "y_label": labels[y_key], "ltv": "LTV", "joint_pd": "Joint PD",
        "p_t1": "P(T1)", "p_t2": "P(T2)",
    })
    st.dataframe(table.style.format({"LTV": "{:.1%}", "Joint PD": "{:.2%}", "P(T1)": "{:.2%}", "P(T2)": "{:.2%}"}),
                 width="stretch", hide_index=True)
