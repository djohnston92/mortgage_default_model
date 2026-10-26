import streamlit as st

############################
# Entry point. Streamlit Cloud's main file for this app is merton_streamlit.py and can't be
# changed without redeploying (and possibly losing the URL), so this file only routes to the
# real pages. st.navigation replaces the pages/ folder auto-discovery, so every page is listed here.
st.navigation([
    st.Page("Probability of Default Map.py", title="Probability of Default Map", default=True),
    st.Page("pages/2_County_Rationales.py", title="County Rationales"),
    st.Page("pages/3_Scenario_Heatmap.py", title="Scenario Heatmap"),
]).run()
