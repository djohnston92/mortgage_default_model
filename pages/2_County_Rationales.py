import html
import re

import pandas as pd
import pydeck as pdk
import streamlit as st
from sqlalchemy import create_engine

############################
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .block-container {
        max-width: 75%;
        margin-left: auto;
        margin-right: auto;
    }
    .story-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 4px 0 12px 0; font-size: 0.9rem; }
    .story-legend span { display: inline-flex; align-items: center; gap: 6px; }
    .story-legend i { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
    .rationale-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    .rationale-table th { text-align: left; font-weight: 600; padding: 6px 10px;
                          border-bottom: 2px solid rgba(128,128,128,0.4); }
    .rationale-table td { vertical-align: top; padding: 10px; border-bottom: 1px solid rgba(128,128,128,0.2); }
    .rationale-table td p { margin: 0 0 0.6em 0; }
    .rationale-table td p:last-child { margin-bottom: 0; }
    .rationale-table .county { white-space: nowrap; font-weight: 600; }
    .rationale-table .sub { font-weight: 400; font-size: 0.82rem; opacity: 0.75; margin: 2px 0 0 20px; }
    .rationale-table .num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .rationale-table .market { white-space: nowrap; font-size: 0.85rem; }
    .rationale-table .market div { display: flex; justify-content: space-between; gap: 12px; }
    .rationale-table .market span { opacity: 0.75; }
    .rationale-table .market b { font-weight: 600; font-variant-numeric: tabular-nums; }
    .rationale-table .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block;
                            margin-right: 8px; vertical-align: -1px; }
    </style>
""", unsafe_allow_html=True)
############################
# Story types come from the generator's classify() (python.llm_pd_county_explanations_enhanced).
# Colors are fixed per type, never by rank or count, so filtering never repaints a type.
# "stable" is the "nothing notable" bucket (about half of all counties), so it's a recessive
# gray; the other eight take the eight categorical slots of the validated default palette.
# Eight hues on one map are hard to tell apart for some pairs (and under color blindness),
# so the legend, the story type filter and the table's Story column always name the type.
STORY_TYPES = {
    # key: (label, hex, one-line meaning shown in "How to read this")
    "both_triggers":     ("Both risks rising", "#e34948", "Home price risk and income risk are rising together."),
    "persistently_high": ("Persistently high", "#e87ba4", "Risk has been well above national for a while, not newly rising."),
    "thin_equity":       ("Thin equity",       "#eb6834", "Buyers start with less equity than typical, so a small price drop matters."),
    "stalled_market":    ("Stalled market",    "#eda100", "Borrowers look normal; home price growth stalled under them."),
    "price_risk":        ("Price risk",        "#1baf7a", "Risk comes mainly from home prices, not incomes."),
    "income_shock":      ("Income shock",      "#2a78d6", "Risk comes mainly from borrowers' ability to keep paying."),
    "whiplash":          ("Whiplash",          "#6250d6", "Risk swung hard one year and reversed the next."),
    "improving":         ("Improving",         "#008300", "Risk has fallen over the last two years."),
    "stable":            ("Stable",            "#9a9a94", "Nothing dramatic; the county looks ordinary."),
}
UNKNOWN = ("Unclassified", "#9a9a94", "")


def story(key):
    return STORY_TYPES.get(key, UNKNOWN)


def hex_to_rgba(h, alpha=200):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


############################
# Reads Neon's public schema, same "neon" secret as the county map.
@st.cache_resource
def get_engine():
    # Pin the psycopg2 driver; SQLAlchemy 2.1+ defaults postgresql:// to psycopg (v3)
    return create_engine(st.secrets["neon"].replace("postgresql://", "postgresql+psycopg2://", 1))

@st.cache_data(ttl="1h")  # without a ttl a running app keeps its first pull until restart
def load_data():
    engine = get_engine()
    # One view carries the explanation, location and the county's numbers (PD, loans,
    # drift, volatility, cushion from python.hmda_merton_county_pd_3yr_history_vw, the
    # same matview the explanations were written from).
    df = pd.read_sql(f"""
        SELECT activity_year, geoid_clean, county_name, state_name, state_abbrev,
               latitude, longitude, story_type, pd_explanation,
               avg_joint_pd, loans, avg_mu_a, avg_sigma_a, pct_drop_to_default
        FROM public.hmda_merton_llm_county_rationale_w_location_detail_vw
    """, engine)
    for col in ("avg_joint_pd", "loans", "avg_mu_a", "avg_sigma_a", "pct_drop_to_default"):
        df[col] = df[col].astype(float)
    return df, df["loans"].min(), df["loans"].max()

df, loans_min, loans_max = load_data()
############################
st.title("County Risk Explanations")
st.sidebar.markdown("[How the model works ↗](https://danieljohnston638537.substack.com/p/i-built-what-the-rating-agencies)")
st.caption("A plain-language explanation of each county's mortgage default risk, written by a local language "
           "model from the same Merton model numbers used in the county map, then checked against those numbers.")

with st.expander("How to read this"):
    st.markdown(
        "- **Dot color** is the county's story type: the main reason its risk looks the way it does.\n"
        "- **Hover** a dot for its probability of default (PD). The table below has every county's full explanation, highest PD first.\n"
        "- **Dot size** is the number of home purchase loans.\n"
        "- **Market** in the table: *price growth (μ, drift)* is how fast home prices rose (or fell) over the last year; "
        "*price swings (σ, volatility)* is how much they typically move year to year; *drop to underwater* is how far prices "
        "would have to fall before the typical loan owes more than the home is worth.\n"
        "- Explanations are machine-written and checked for factual errors against the model's numbers. "
        "Counties with very few loans have noisier numbers.\n\n"
        "| Story type | Meaning |\n|---|---|\n"
        + "\n".join(f"| {label} | {meaning} |" for label, _, meaning in STORY_TYPES.values())
    )
############################
st.subheader("Filters")
col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Year", sorted(df["activity_year"].dropna().unique(), reverse=True))
year_df = df[df["activity_year"] == year]
with col2:
    state = st.selectbox("State", ["All"] + sorted(year_df["state_name"].dropna().unique()))
county_options = year_df if state == "All" else year_df[year_df["state_name"] == state]
with col3:
    county = st.selectbox("County Name", ["All"] + sorted(county_options["county_name"].dropna().unique()))

present = [k for k in STORY_TYPES if k in set(year_df["story_type"])]
picked = st.multiselect("Story type", present, default=present, format_func=lambda k: story(k)[0])

filtered = county_options if county == "All" else county_options[county_options["county_name"] == county]
filtered = filtered[filtered["story_type"].isin(picked)].copy()

if filtered.empty:
    st.info("No counties match these filters.")
    st.stop()
############################
# legend: fixed order, counts reflect the current filters
counts = filtered["story_type"].value_counts()
st.markdown(
    '<div class="story-legend">' + "".join(
        f'<span><i style="background:{story(k)[1]}"></i>{story(k)[0]} ({counts[k]:,})</span>'
        for k in STORY_TYPES if k in counts
    ) + "</div>",
    unsafe_allow_html=True,
)
############################
# map: same radius scaling as the county map, color = story type, hover = PD only
min_radius, max_radius = 5_000, 100_000

def scale_radius(loans):
    if pd.isna(loans) or loans_max == loans_min:
        return min_radius
    return min_radius + (loans - loans_min) / (loans_max - loans_min) * (max_radius - min_radius)

on_map = filtered.dropna(subset=["latitude", "longitude"]).rename(columns={"latitude": "lat", "longitude": "lon"})
on_map["radius"] = on_map["loans"].apply(scale_radius)
on_map["color"] = on_map["story_type"].apply(lambda k: hex_to_rgba(story(k)[1]))
on_map["pd_display"] = on_map["avg_joint_pd"].apply(lambda v: "n/a" if pd.isna(v) else f"{v:.2%}")

layer = pdk.Layer(
    "ScatterplotLayer",
    data=on_map,
    get_position=["lon", "lat"],
    get_fill_color="color",
    get_radius="radius",
    radius_min_pixels=2,
    radius_max_pixels=240,
    stroked=True,              # thin ring keeps overlapping dots separable
    get_line_color=[255, 255, 255, 200],
    line_width_min_pixels=1,
    pickable=True,
)
tooltip = {
    "html": "<b>{county_name}, {state_abbrev}</b><br/>Probability of Default: {pd_display}",
    "style": {"backgroundColor": "steelblue", "color": "white", "fontSize": "13px"},
}
view_state = pdk.ViewState(latitude=on_map["lat"].mean(), longitude=on_map["lon"].mean(),
                           zoom=3.5 if state == "All" else 5.5)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
missing = len(filtered) - len(on_map)
if missing:
    st.caption(f"{missing} count{'y has' if missing == 1 else 'ies have'} no map location "
               "(e.g. Connecticut's planning regions) and appear only in the table.")
############################
# table: highest PD first, full explanations with their paragraph breaks. An HTML table
# instead of st.dataframe, which can't wrap or break lines. The dot carries the story color;
# all text stays in ink. Paged so the browser isn't drawing ~1,700 explanations at once.
st.subheader("Explanations")
table = filtered.sort_values("avg_joint_pd", ascending=False, na_position="last")

ROWS_PER_PAGE = 25
pages = max(1, -(-len(table) // ROWS_PER_PAGE))
if pages > 1:
    # a widget label can't show its own value, so the "Page N of M" text sits beside a bare box
    box, label = st.columns([1, 5], vertical_alignment="center")
    with box:
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1,
                               label_visibility="collapsed")
    with label:
        st.markdown(f"**Page {page:,} of {pages:,}** · {len(table):,} counties, "
                    f"{ROWS_PER_PAGE} per page, highest PD first")
else:
    page = 1
chunk = table.iloc[(page - 1) * ROWS_PER_PAGE: page * ROWS_PER_PAGE]


def explanation_html(text):
    paras = [p.strip() for p in re.split(r"\n\s*\n", str(text or "")) if p.strip()]
    return "".join(f"<p>{html.escape(p)}</p>" for p in paras)


def market_html(r):
    """Price growth / price swings / drop to underwater, stacked. Plain labels, same
    words as the explanations; growth is signed so a fall reads as negative."""
    def fmt(v, signed=False):
        if pd.isna(v):
            return "n/a"
        return f"{v * 100:+.1f}%".replace("-", "\u2212") if signed else f"{v * 100:.1f}%"
    cushion = "n/a" if pd.isna(r["pct_drop_to_default"]) else f"{r['pct_drop_to_default']:.0f}%"
    # μ / σ match the PD map tooltip and the Scenario Heatmap sliders
    lines = (("Price growth (μ)", fmt(r["avg_mu_a"], signed=True)),
             ("Price swings (σ)", fmt(r["avg_sigma_a"])),
             ("Drop to underwater", cushion))
    return "".join(f'<div><span>{k}</span><b>{v}</b></div>' for k, v in lines)


rows = []
for _, r in chunk.iterrows():
    label, color, _ = story(r["story_type"])
    name = html.escape(str(r["county_name"] if pd.notna(r["county_name"]) else r["geoid_clean"]))
    pd_txt = "n/a" if pd.isna(r["avg_joint_pd"]) else f"{r['avg_joint_pd']:.2%}"
    loans_txt = "n/a" if pd.isna(r["loans"]) else f"{int(r['loans']):,}"
    rows.append(
        "<tr>"
        f'<td class="county"><span class="dot" style="background:{color}"></span>{name}'
        f'<div class="sub">{html.escape(str(r["state_abbrev"] or ""))} · {label}</div></td>'
        f'<td class="num">{pd_txt}</td>'
        f'<td class="market">{market_html(r)}</td>'
        f'<td class="num">{loans_txt}</td>'
        f"<td>{explanation_html(r['pd_explanation'])}</td>"
        "</tr>"
    )
st.markdown(
    '<table class="rationale-table"><thead><tr><th>County</th><th class="num">PD</th><th>Market</th>'
    '<th class="num">Loans</th><th>Explanation</th></tr></thead><tbody>'
    + "".join(rows) + "</tbody></table>",
    unsafe_allow_html=True,
)
