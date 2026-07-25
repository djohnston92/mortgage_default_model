Mortgage Default Risk Model

A Streamlit app that estimates mortgage default probability across U.S. counties, states, and nationally. Built on public HMDA loan data and the Merton structural credit model.

Live app: https://mortgagedefaultmodel-6xpczdpbz99ccevavqewwj.streamlit.app/

What it does
  - Models probability of default (PD) using the Merton framework — treating a borrower's home equity position like a call option on the property's value
  - Aggregates results at the county, state, and national level
  - Visualizes risk concentration on an interactive map
  - Pulls live from a NeonDB backend, not a static dataset

Stack
 - Layer	Tool
 - Frontend	Streamlit
 - Database	Neon (serverless Postgres)
 - ORM	SQLAlchemy
 - Mapping	pydeck
 - Charts	Matplotlib, Altair

Data source
Home Mortgage Disclosure Act (HMDA) public loan-level data, aggregated into county/state/national PD summary views.

Notes
This is a modeling/exploration tool, not a production financial product. PD estimates are model outputs, not lending decisions or guarantees.
