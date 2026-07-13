import streamlit as st
import pandas as pd
import pydeck as pdk
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import altair as alt
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
    </style>
""", unsafe_allow_html=True)
############################
neon = st.secrets["neon"]














@st.cache_resource
def get_engine():
    return create_engine(neon)

@st.cache_data
def load_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM public.hmda_merton_primary_residence_pd_summary_by_county_vw", engine)
    df_state = pd.read_sql("SELECT * FROM public.hmda_merton_primary_residence_pd_summary_by_state_vw", engine)
    df_national = pd.read_sql("SELECT * FROM public.hmda_merton_primary_residence_pd_summary_national_vw", engine)
    return df, df_state, df_national

df, df_state, df_national = load_data()



year_list = [2018,2019,2020,2021,2022,2023,2024,2025]
















############################
st.title("County-Level Mortgage Default Risk")
############################
@st.dialog("Data Dictionary")
def show_data_dictionary():
    st.markdown("""
    | Term | Definition |
    |---|---|
    | **General & Source Data** | |
    | `HMDA` | Home Mortgage Disclosure Act, the federal law requiring most U.S. lenders to publicly report loan-level mortgage data (lender, income, loan amount, property value, DTI, geography). This app uses purchase loan originations from 2018–2025 |
    | `FHFA HPI` | Federal Housing Finance Agency House Price Index, quarterly home price data by metro area, based on actual closed transactions. Source of the volatility (σ_V) and drift (μ_V) inputs |
    | `BLS LAUS` | Bureau of Labor Statistics Local Area Unemployment Statistics, annual unemployment rates by metro area, used to estimate income-shock risk |
    | `activity_year` | HMDA reporting year for the loan origination |
    | `state_name` | U.S. state of the property |
    | `county_name` | U.S. county of the property |
    | `latitude` / `longitude` | Geographic coordinates used for mapping |
    | `loans` | Count of loan originations in the selected area |
    | **Loan Terms** | |
    | `principal` | The amount borrowed |
    | `term` | The length of time over which the loan is repaid |
    | `apr` | Annual Percentage Rate, the cost of borrowing, expressed yearly |
    | `avg_dti` | Average debt-to-income ratio of borrowers in the area, monthly debt payments divided by gross monthly income |
    | `avg_ltv` | Average loan-to-value ratio, loan balance divided by property value at origination |
    | **Merton Model / Risk** | |
    | `avg_joint_pd` | Average joint probability of default across loans in the area, based on the Merton model (property value falling below loan balance, combined with borrower income-shock risk) |
    | `avg_sigma_a` | Average asset (home price) volatility, σ_V, derived from the FHFA HPI. Higher values mean prices swing more year to year (e.g. Las Vegas ~0.059 vs. national average ~0.037) |
    | `pct_drop_to_default` | The percentage decline in property value that would push the loan into default territory (asset value falls below loan balance) |
    """)

@st.dialog("Project Brief")
def show_project_brief():
    st.markdown("""
This project takes public housing and lending data (from federal sources like HMDA, FHFA, and the BLS) and uses it to estimate the likelihood that a mortgage will default, Meaning the homeowner stops paying and the property is worth less than what's owed. The model is based on a well-established financial formula (the Merton model), which is normally used to predict when companies will fail on their debts, but has been adapted here to work for home loans instead.

The end result is a tool that can look at any county in the country and show how risky the mortgages in that area are, based on things like local home prices, how much debt borrowers are carrying, and local job market conditions. It's built entirely from free public data, so there's no hidden "black box." Every number can be traced back to where it came from.
""")

@st.dialog("How to Guide")
def show_guide():
    st.markdown("""
Filters at the top control all visuals with the exception of year, which won't control the line chart. Top 10/Bottom 10 Chart regions are only included if there are more than 500 sales. 
                
As more filters (State/County) are used the line chart will overlay regions to see how the lowest level of detail compares to national and state. 
    """)

@st.dialog("Versions")
def show_versions():
    st.markdown("""
    | Version Number | Changes |
    |---|---|
    | `1.0` | Release Version |
    """)
st.subheader("Filters")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📖 Data Dictionary", use_container_width=True):
        show_data_dictionary()
with col2:
    if st.button("Project Brief", use_container_width=True):
        show_project_brief()
with col3:
    if st.button("How to Guide", use_container_width=True):
        show_guide()
with col4:
    if st.button("Versions", use_container_width=True):
        show_versions()
############################
year = st.selectbox("Year", sorted(year_list, reverse=True))
filtered = df[df["activity_year"] == year]
state_filtered = df_state[df_state["activity_year"] == year]
national_filtered = df_national[df_national["activity_year"] == year]
state = st.selectbox("State", ["All"] + sorted(df["state_name"].dropna().unique()))
county_options = df if state == "All" else df[df["state_name"] == state]
county = st.selectbox("County Name", ["All"] + sorted(county_options["county_name"].dropna().unique()))
#############################
#line chart visual 
st.subheader("PD Over Time")
filtered = df[df["activity_year"] == year].copy()
if state != "All":
    filtered = filtered[filtered["state_name"] == state]
# --- PD over time: intentionally NOT filtered by `year`, so it always shows all years ---
trend_df = df.copy()
if state != "All":
    trend_df = trend_df[trend_df["state_name"] == state]
if county != "All":
    trend_df = trend_df[trend_df["county_name"] == county]
yearly_pd = df_national[["activity_year", "avg_joint_pd"]].rename(columns={"avg_joint_pd": "National"})

if state != "All":
    state_yearly = df_state[df_state["state_name"] == state][["activity_year", "avg_joint_pd"]].rename(columns={"avg_joint_pd": state})
    yearly_pd = yearly_pd.merge(state_yearly, on="activity_year", how="outer")

if county != "All":
    county_yearly = trend_df[["activity_year", "avg_joint_pd"]].rename(columns={"avg_joint_pd": county})
    yearly_pd = yearly_pd.merge(county_yearly, on="activity_year", how="outer")
yearly_pd["activity_year"] = yearly_pd["activity_year"].astype(int).astype(str)

yearly_long = yearly_pd.melt(id_vars="activity_year", var_name="level", value_name="avg_joint_pd").dropna(subset=["avg_joint_pd"])

line = alt.Chart(yearly_long).mark_line(point=True).encode(
    x=alt.X("activity_year:N", title="Year"),
    y=alt.Y("avg_joint_pd:Q", title="Avg Joint PD"),
    color=alt.Color("level:N", title=None),
)

labels = alt.Chart(yearly_long).mark_text(dy=-10, fontSize=11).encode(
    x=alt.X("activity_year:N"),
    y=alt.Y("avg_joint_pd:Q"),
    text=alt.Text("avg_joint_pd:Q", format=".2%"),
    color=alt.Color("level:N", legend=None),
)
st.altair_chart(line + labels, use_container_width=True)
#################################################################################################################################################
@st.dialog("Top 10 Highest Risk Counties")
def show_top_10():
    top10 = filtered[filtered["loans"] > 500].sort_values("avg_joint_pd", ascending=False).head(10)
    
    display_df = top10[["county_name", "state_name", "avg_joint_pd", "loans"]].copy()
    display_df["avg_joint_pd"] = (display_df["avg_joint_pd"] * 100).round(2).astype(str) + "%"
    display_df["loans"] = display_df["loans"].apply(lambda x: f"{x:,}")
    display_df.columns = ["County", "State", "Avg Joint PD", "Loans"]
    display_df.index = range(1, len(display_df) + 1)
    
    st.caption(f"Showing top {len(top10)} of {len(filtered)} counties matching current filters with more than 500 originations")
    st.dataframe(display_df, use_container_width=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    if st.button("🔺 Top 10 Highest Risk Counties", use_container_width=True):
        show_top_10()
@st.dialog("bottom 10 Highest Risk Counties")
def show_bottom_10():
    bottom10 = filtered[filtered["loans"] > 500].sort_values("avg_joint_pd", ascending=False).tail(10)
    
    display_df = bottom10[["county_name", "state_name", "avg_joint_pd", "loans"]].copy()
    display_df["avg_joint_pd"] = (display_df["avg_joint_pd"] * 100).round(2).astype(str) + "%"
    display_df["loans"] = display_df["loans"].apply(lambda x: f"{x:,}")
    display_df.columns = ["County", "State", "Avg Joint PD", "Loans"]
    display_df.index = range(1, len(display_df) + 1)
    
    st.caption(f"Showing bottom {len(bottom10)} of {len(filtered)} counties matching current filters with more than 500 originations")
    st.dataframe(display_df, use_container_width=True)

with col2:
    if st.button("🔺Bottom 10 Lowest Risk Counties", use_container_width=True):
        show_bottom_10()
####################################################################################################################
# # --- Scale radius by loan volume ---
min_radius = 5_000     # meters, smallest dot
max_radius = 100_000    # meters, largest dot




loans_min = df["loans"].min()
loans_max = df["loans"].max()
def scale_radius(loans):
    if loans_max == loans_min:  # avoid divide-by-zero if all values equal
        return (min_radius + max_radius) / 2
    scaled = (loans - loans_min) / (loans_max - loans_min)
    return min_radius + scaled * (max_radius - min_radius)
filtered["radius"] = filtered["loans"].apply(scale_radius)
fig, ax = plt.subplots(figsize=(6, 0.4))
fig.subplots_adjust(bottom=0.5)
####################################################################################################################
# # --- Build cool-to-warm color scale based on avg_joint_pd ---
norm = mcolors.Normalize(vmin=0, vmax=0.12)
cmap = mpl.colormaps["coolwarm"]

def pd_to_rgba(value):
    r, g, b, a = cmap(norm(value))
    return [int(r * 255), int(g * 255), int(b * 255), 180]  # 180 = alpha for transparency

filtered["color"] = filtered["avg_joint_pd"].apply(pd_to_rgba)
filtered = filtered.rename(columns={"latitude": "lat", "longitude": "lon"})
filtered["avg_joint_pd_display"] = (filtered["avg_joint_pd"] * 100).round(2).astype(str) + "%"
filtered["pct_drop_to_default"] = (filtered["pct_drop_to_default"] ).round(2).astype(str) + "%"
filtered["loans_display"] = filtered["loans"].apply(lambda x: f"{x:,}")  # adds comma separators
# # --- Render with pydeck ---
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
cb.set_label("Avg Joint PD")

tooltip = {
    "html": """
        <b>{county_name}</b><br/>
        State: {state_name}<br/>
        Loans: {loans_display}<br/>
        Average Probability of Default: {avg_joint_pd_display}<br/>
        Sigma A: {avg_sigma_a}<br/>
        Percentage Drop-To-Default: {pct_drop_to_default}<br/>
        Average Debt-To-Income: {avg_dti}<br/>
        Average Loan To Value: {avg_ltv}
    """,
    "style": {"backgroundColor": "steelblue", "color": "white", "fontSize": "13px"}
}
layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered,
    get_position=["lon", "lat"],
    get_fill_color="color",
    get_radius="radius",   # <-- use the column instead of a fixed number
    radius_min_pixels=1,   # floor so dots stay visible when zoomed out (national view)
    radius_max_pixels=240,  # ceiling so dots don't overwhelm the map when zoomed in
    pickable=True,
)


pd_min = filtered["avg_joint_pd"].min() * 100
pd_max = filtered["avg_joint_pd"].max() * 100




view_state = pdk.ViewState(
    latitude=filtered["lat"].mean(),
    longitude=filtered["lon"].mean(),
    zoom=5,
)
st.markdown("""
<style>
div[class*="st-key-map_container"] {
    position: relative;
}
.map-legend {
    position: absolute;
    top: 16px;
    left: 20px;
    width: 250px;
    background: rgba(255,255,255,0.9);
    padding: 8px 12px;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    z-index: 999;
}
</style>
""", unsafe_allow_html=True)
with st.container(key="map_container"):
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))