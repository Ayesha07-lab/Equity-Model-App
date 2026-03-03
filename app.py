import os
import streamlit as st
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from model import run_full_model
from excel_export import export_to_excel

import os
import streamlit as st

def get_av_key():
    # First try Streamlit Cloud secrets
    if "ALPHA_VANTAGE_API_KEY" in st.secrets:
        return st.secrets["ALPHA_VANTAGE_API_KEY"].strip()
    # Then try local environment (.env or system)
    return os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

AV_KEY = get_av_key()

# Key status (never display the key itself)
if AV_KEY:
    st.caption("Alpha Vantage key loaded ✅")
else:
    st.warning("Alpha Vantage key not found. (App still works using yfinance.)")
# ---------------- Sidebar ----------------
st.sidebar.header("Core")
ticker = st.sidebar.text_input("Company Ticker", value="GOOG").strip().upper()

comps_input = st.sidebar.text_area("Competitors / Comps (comma-separated)", value="MSFT, AMZN, META")
comps_list = [x.strip().upper() for x in comps_input.split(",") if x.strip()]

forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 3, 10, 5, 1)

st.sidebar.divider()
st.sidebar.header("Scenarios (Auto + Deltas)")
fade_to_growth = st.sidebar.number_input(
    "Fade-to long-run revenue growth (Base, end-year)",
    min_value=-0.05, max_value=0.20, value=0.04, step=0.005, format="%.3f"
)

c1, c2 = st.sidebar.columns(2)
bull_growth_delta = c1.number_input("Bull growth delta", -0.10, 0.20, 0.03, 0.005, format="%.3f")
weak_growth_delta = c2.number_input("Weak growth delta", -0.20, 0.10, -0.04, 0.005, format="%.3f")

c3, c4 = st.sidebar.columns(2)
bull_margin_delta = c3.number_input("Bull margin delta", -0.10, 0.20, 0.03, 0.005, format="%.3f")
weak_margin_delta = c4.number_input("Weak margin delta", -0.20, 0.10, -0.04, 0.005, format="%.3f")

st.sidebar.divider()
st.sidebar.header("DCF")
terminal_growth = st.sidebar.number_input("Terminal growth (TGR)", -0.02, 0.08, 0.03, 0.005, format="%.3f")
blend_weight_pgm = st.sidebar.slider("Blend weight to PGM (rest to Exit Multiple)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.divider()
st.sidebar.header("Overrides (Advanced)")

with st.sidebar.expander("Tax + Drivers + Shares/Net Debt", expanded=False):
    tax_override = st.checkbox("Override tax rate", value=True)
    tax_rate = st.number_input("Tax rate", 0.0, 0.45, 0.18, 0.01, format="%.2f")

    st.caption("Drivers auto-estimated from history unless overridden.")
    da_override = st.checkbox("Override D&A % of revenue", value=False)
    da_pct_rev_override = st.number_input("D&A % revenue (override)", 0.0, 0.20, 0.04, 0.005, format="%.3f")

    capex_override = st.checkbox("Override CapEx % of revenue", value=False)
    capex_pct_rev_override = st.number_input("CapEx % revenue (override)", 0.0, 0.30, 0.05, 0.005, format="%.3f")

    wc_override = st.checkbox("Override Working Capital % of revenue", value=False)
    wc_pct_rev_override = st.number_input("WC % revenue (override)", -0.10, 0.30, 0.08, 0.005, format="%.3f")

    shares_override = st.checkbox("Override shares outstanding", value=False)
    shares_override_value = st.number_input("Shares outstanding (override)", 0.0, 1e11, 1e9, 1e8)

    net_debt_override = st.checkbox("Override net debt", value=False)
    net_debt_override_value = st.number_input("Net debt (override)", -1e12, 1e12, 0.0, 1e9)

with st.sidebar.expander("WACC (auto build OR override)", expanded=False):
    wacc_override = st.checkbox("Override WACC directly", value=False)
    wacc_override_value = st.number_input("WACC (override)", 0.01, 0.30, 0.10, 0.01, format="%.2f")

    rf = st.number_input("Risk-free rate (rf)", 0.00, 0.10, 0.04, 0.005, format="%.3f")
    erp = st.number_input("Equity risk premium (ERP)", 0.00, 0.12, 0.055, 0.005, format="%.3f")
    cod = st.number_input("Pre-tax cost of debt", 0.00, 0.20, 0.055, 0.005, format="%.3f")

    beta_override = st.checkbox("Override beta", value=False)
    beta_override_value = st.number_input("Beta (override)", 0.1, 3.0, 1.0, 0.05, format="%.2f")

    target_debt_weight_override = st.checkbox("Override target debt weight D/(D+E)", value=False)
    target_debt_weight_value = st.number_input("Target debt weight", 0.0, 0.95, 0.10, 0.01, format="%.2f")

with st.sidebar.expander("Terminal Exit Multiple (auto from comps OR override)", expanded=False):
    exit_multiple_override = st.checkbox("Override exit multiple", value=False)
    exit_multiple_override_value = st.number_input("Exit multiple (override)", 0.0, 100.0, 20.0, 0.5, format="%.2f")

run = st.sidebar.button("Run Model", type="primary")

# ---------------- Run / session state ----------------
if "results" not in st.session_state:
    st.session_state["results"] = None

if run:
    try:
        st.session_state["results"] = run_full_model(
            ticker=ticker,
            comps_list=comps_list,
            forecast_years=int(forecast_years),
            terminal_growth=float(terminal_growth),
            blend_weight_pgm=float(blend_weight_pgm),
            fade_to_growth=float(fade_to_growth),
            bull_growth_delta=float(bull_growth_delta),
            weak_growth_delta=float(weak_growth_delta),
            bull_margin_delta=float(bull_margin_delta),
            weak_margin_delta=float(weak_margin_delta),
            tax_rate=float(tax_rate),
            tax_override=bool(tax_override),
            da_override=bool(da_override),
            da_pct_rev_override=float(da_pct_rev_override),
            capex_override=bool(capex_override),
            capex_pct_rev_override=float(capex_pct_rev_override),
            wc_override=bool(wc_override),
            wc_pct_rev_override=float(wc_pct_rev_override),
            shares_override=bool(shares_override),
            shares_override_value=float(shares_override_value),
            net_debt_override=bool(net_debt_override),
            net_debt_override_value=float(net_debt_override_value),
            wacc_override=bool(wacc_override),
            wacc_override_value=float(wacc_override_value),
            rf=float(rf),
            erp=float(erp),
            beta_override=bool(beta_override),
            beta_override_value=float(beta_override_value),
            cod=float(cod),
            target_debt_weight_override=bool(target_debt_weight_override),
            target_debt_weight_value=float(target_debt_weight_value),
            exit_multiple_override=bool(exit_multiple_override),
            exit_multiple_override_value=float(exit_multiple_override_value),
        )
    except Exception as e:
        st.error(f"Model failed: {e}")
        st.stop()

results = st.session_state.get("results")
if results is None:
    st.info("Enter inputs on the left and click **Run Model**.")
    st.stop()

headline = results["headline"]

# ---------------- Headline metrics ----------------
colA, colB, colC, colD, colE = st.columns(5)
with colA:
    px = headline.get("Price", np.nan)
    st.metric("Current Price", "—" if np.isnan(px) else f"${px:,.2f}")
with colB:
    st.metric("WACC", f"{headline['WACC']*100:.2f}%")
with colC:
    st.metric("TGR", f"{headline['TGR']*100:.2f}%")
with colD:
    st.metric("EV (Base)", f"{headline['EV_Base']/1e9:,.2f} B")
with colE:
    ps = headline.get("PerShare_Base", np.nan)
    st.metric("Implied / Share (Base)", "—" if np.isnan(ps) else f"${ps:,.2f}")

if not np.isnan(px) and not np.isnan(headline.get("PerShare_Base", np.nan)):
    upside = (headline["PerShare_Base"] / px - 1.0) * 100.0
    st.caption(f"Base-case upside vs current: **{upside:,.1f}%**")

st.caption(
    f"Exit: **{headline.get('ExitBasis','')} {headline.get('ExitMultiple', np.nan):.2f}x** "
    f"(source: {headline.get('ExitSource','')}); "
    f"PGM weight used: {headline.get('BlendWeightPGM_Used', 0.5):.2f}"
)

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Scenarios", "Comps", "Forecasts", "Sensitivity", "Assumptions Trace", "Export"]
)

with tab1:
    st.subheader("Scenario Summary")
    st.dataframe(results["scenario_summary"], use_container_width=True)

with tab2:
    st.subheader("Comparable Companies")
    comps_df = results.get("comps_df")
    if comps_df is None or comps_df.empty:
        st.info("No comps returned. Add peer tickers and rerun.")
    else:
        show = comps_df.copy()
        for c in ["MarketCap", "EnterpriseValue", "Revenue_TTM", "EBITDA_TTM"]:
            show[c] = (show[c] / 1e9).round(2)
        for c in ["Gross Margin", "Op Margin", "Net Margin"]:
            show[c] = (show[c] * 100).round(2)
        st.dataframe(show, use_container_width=True)

        med_cols = ["EV/Revenue", "EV/EBITDA", "P/E (TTM)", "P/E (Fwd)"]
        med = comps_df[med_cols].median(numeric_only=True).to_frame("Median").reset_index().rename(columns={"index": "Metric"})
        st.subheader("Median Multiples")
        st.dataframe(med, use_container_width=True)

with tab3:
    st.subheader("Forecasts")
    sd = results.get("scenario_detail", {})
    for scen in ["Base", "Bull", "Weak"]:
        st.markdown(f"### {scen}")
        if scen in sd and isinstance(sd[scen].get("forecast_df"), pd.DataFrame):
            st.dataframe(sd[scen]["forecast_df"], use_container_width=True)
        else:
            st.info(f"No forecast found for {scen}.")

with tab4:
    st.subheader("Sensitivity (Enterprise Value) — Base Forecast")
    st.dataframe(results["sensitivity_df"], use_container_width=True)

with tab5:
    st.subheader("Assumptions Trace (what the model actually used)")
    used = results.get("inputs_used", {})
    rows = []
    for k, v in used.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                rows.append({"Key": f"{k}.{k2}", "Value": v2})
        else:
            rows.append({"Key": k, "Value": v})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("WACC Build (Auto)")
    st.dataframe(pd.DataFrame([results.get("wacc_build", {})]), use_container_width=True)

with tab6:
    st.subheader("Export to Excel")
    xbytes = export_to_excel(results)
    st.download_button(
        "Download Excel (.xlsx)",
        data=xbytes,
        file_name=f"{ticker}_SellSide_Model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with st.expander("Debug: result keys", expanded=False):
    st.write(list(results.keys()))
