import numpy as np
import pandas as pd
import yfinance as yf


# ------------------------
# Helpers
# ------------------------
def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).replace(",", "").strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def _nanmedian(series: pd.Series, default=np.nan):
    try:
        v = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        m = np.nanmedian(v.values)
        return default if np.isnan(m) else float(m)
    except Exception:
        return default


def _cagr(values: pd.Series):
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    if len(v) < 2:
        return np.nan
    start, end = v[0], v[-1]
    n = len(v) - 1
    if start <= 0:
        return np.nan
    return float((end / start) ** (1 / n) - 1)


def _linear_ramp(start: float, end: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([end], dtype=float)
    return np.linspace(start, end, n, dtype=float)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


# ------------------------
# Data pulls
# ------------------------
def get_info_snapshot(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.get_info() or {}

    price = _safe_float(info.get("currentPrice"))  # sometimes missing
    if np.isnan(price):
        price = _safe_float(info.get("regularMarketPrice"))

    return {
        "ticker": ticker,
        "price": price,
        "market_cap": _safe_float(info.get("marketCap")),
        "shares_out": _safe_float(info.get("sharesOutstanding")),
        "beta": _safe_float(info.get("beta")),
        "enterprise_value": _safe_float(info.get("enterpriseValue")),
    }


def get_hist_financials(ticker: str) -> pd.DataFrame:
    """
    Normalized annual history from yfinance:
    Year, Revenue, EBIT, NetIncome, EBITDA, D&A, CapEx, Cash, Debt, WC
    """
    t = yf.Ticker(ticker)

    inc = t.financials
    cf = t.cashflow
    bs = t.balance_sheet

    if inc is None or inc.empty:
        raise ValueError("Could not fetch income statement via yfinance.")

    cols = sorted(list(inc.columns))  # oldest -> newest
    years = [c.year for c in cols]

    def get_series(df, keys):
        if df is None or df.empty:
            return [np.nan] * len(cols)
        out = []
        for c in cols:
            val = np.nan
            for k in keys:
                if k in df.index:
                    val = _safe_float(df.loc[k, c])
                    break
            out.append(val)
        return out

    revenue = get_series(inc, ["Total Revenue", "TotalRevenue", "Revenue"])
    ebit = get_series(inc, ["Ebit", "EBIT", "Operating Income", "OperatingIncome"])
    net_income = get_series(inc, ["Net Income", "NetIncome", "Net Income Common Stockholders"])

    # EBITDA sometimes exists; else approximate as EBIT + D&A later
    ebitda = get_series(inc, ["Ebitda", "EBITDA"])

    da = get_series(cf, ["Depreciation And Amortization", "Depreciation", "Depreciation & Amortization"])
    capex = get_series(cf, ["Capital Expenditure", "CapitalExpenditures", "Capital Expenditures"])

    cash = get_series(bs, ["Cash And Cash Equivalents", "Cash", "Cash And Short Term Investments"])
    debt = get_series(bs, ["Total Debt", "Long Term Debt", "Short Long Term Debt"])

    curr_assets = get_series(bs, ["Total Current Assets", "Current Assets"])
    curr_liab = get_series(bs, ["Total Current Liabilities", "Current Liabilities"])
    wc = []
    for a, l in zip(curr_assets, curr_liab):
        wc.append(np.nan if (np.isnan(a) or np.isnan(l)) else (a - l))

    df = pd.DataFrame(
        {
            "Year": years,
            "Revenue": revenue,
            "EBIT": ebit,
            "NetIncome": net_income,
            "EBITDA": ebitda,
            "D&A": da,
            "CapEx": capex,
            "Cash": cash,
            "Debt": debt,
            "WC": wc,
        }
    ).dropna(subset=["Revenue"]).sort_values("Year").reset_index(drop=True)

    # Fill EBITDA if missing using EBIT + D&A when possible
    missing = df["EBITDA"].isna()
    df.loc[missing, "EBITDA"] = df.loc[missing, "EBIT"] + df.loc[missing, "D&A"]

    df["EBIT_Margin"] = df["EBIT"] / df["Revenue"]
    df["EBITDA_Margin"] = df["EBITDA"] / df["Revenue"]

    return df


# ------------------------
# Comps
# ------------------------
def fetch_comps(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        tk = tk.strip().upper()
        if not tk:
            continue
        info = yf.Ticker(tk).get_info() or {}
        mcap = _safe_float(info.get("marketCap"))
        ev = _safe_float(info.get("enterpriseValue"))
        rev = _safe_float(info.get("totalRevenue"))
        ebitda = _safe_float(info.get("ebitda"))
        pe = _safe_float(info.get("trailingPE"))
        fpe = _safe_float(info.get("forwardPE"))
        opm = _safe_float(info.get("operatingMargins"))
        npm = _safe_float(info.get("profitMargins"))
        grossm = _safe_float(info.get("grossMargins"))

        ev_rev = ev / rev if (not np.isnan(ev) and not np.isnan(rev) and rev != 0) else np.nan
        ev_ebitda = ev / ebitda if (not np.isnan(ev) and not np.isnan(ebitda) and ebitda != 0) else np.nan

        rows.append(
            {
                "Ticker": tk,
                "MarketCap": mcap,
                "EnterpriseValue": ev,
                "Revenue_TTM": rev,
                "EBITDA_TTM": ebitda,
                "EV/Revenue": ev_rev,
                "EV/EBITDA": ev_ebitda,
                "P/E (TTM)": pe,
                "P/E (Fwd)": fpe,
                "Gross Margin": grossm,
                "Op Margin": opm,
                "Net Margin": npm,
            }
        )
    return pd.DataFrame(rows)


def infer_exit_multiple(comps_df: pd.DataFrame) -> dict:
    """
    Auto pick median EV/EBITDA; fallback to EV/Revenue if EV/EBITDA not usable.
    """
    if comps_df is None or comps_df.empty:
        return {"method": "manual_default", "multiple": np.nan}

    ev_ebitda_med = _nanmedian(comps_df["EV/EBITDA"], default=np.nan)
    if not np.isnan(ev_ebitda_med) and ev_ebitda_med > 0:
        return {"method": "EV/EBITDA_median", "multiple": float(ev_ebitda_med)}

    ev_rev_med = _nanmedian(comps_df["EV/Revenue"], default=np.nan)
    if not np.isnan(ev_rev_med) and ev_rev_med > 0:
        return {"method": "EV/Revenue_median", "multiple": float(ev_rev_med)}

    return {"method": "manual_default", "multiple": np.nan}


# ------------------------
# WACC
# ------------------------
def build_wacc(
    market_cap: float,
    total_debt: float,
    tax_rate: float,
    beta: float,
    rf: float,
    erp: float,
    cost_of_debt: float,
    target_debt_weight: float | None = None,
) -> dict:
    """
    Auto WACC build (CAPM + after-tax debt).
    If target_debt_weight is provided, use it; otherwise weight by market cap and debt.
    """
    tax_rate = _clamp(tax_rate, 0.0, 0.5)

    ke = rf + beta * erp
    kd = cost_of_debt
    kd_at = kd * (1 - tax_rate)

    if target_debt_weight is None or np.isnan(target_debt_weight):
        E = market_cap
        D = total_debt
        if np.isnan(E) or E <= 0:
            E = 0.0
        if np.isnan(D) or D < 0:
            D = 0.0
        V = E + D
        if V <= 0:
            wd = 0.0
            we = 1.0
        else:
            wd = D / V
            we = E / V
    else:
        wd = _clamp(float(target_debt_weight), 0.0, 0.95)
        we = 1.0 - wd

    wacc = we * ke + wd * kd_at
    return {
        "cost_of_equity": float(ke),
        "cost_of_debt": float(kd),
        "after_tax_cost_of_debt": float(kd_at),
        "weight_debt": float(wd),
        "weight_equity": float(we),
        "wacc": float(wacc),
    }


# ------------------------
# Forecast + DCF
# ------------------------
def build_forecast(
    hist_df: pd.DataFrame,
    years: int,
    rev_growth_start: float,
    rev_growth_end: float,
    ebit_margin_start: float,
    ebit_margin_end: float,
    da_pct_rev: float,
    capex_pct_rev: float,
    wc_pct_rev: float,
    tax_rate: float,
):
    last_year = int(hist_df["Year"].iloc[-1])
    yrs = [last_year + i for i in range(1, years + 1)]

    rev0 = float(hist_df["Revenue"].iloc[-1])
    growth_path = _linear_ramp(rev_growth_start, rev_growth_end, years)

    revenue = []
    cur = rev0
    for g in growth_path:
        cur = cur * (1 + g)
        revenue.append(cur)
    revenue = np.array(revenue, dtype=float)

    ebit_margin_path = _linear_ramp(ebit_margin_start, ebit_margin_end, years)
    ebit = revenue * ebit_margin_path
    nopat = ebit * (1 - _clamp(tax_rate, 0.0, 0.5))

    da = revenue * da_pct_rev
    capex = revenue * capex_pct_rev

    wc = revenue * wc_pct_rev
    wc_prev = float(hist_df["WC"].iloc[-1]) if not np.isnan(hist_df["WC"].iloc[-1]) else wc[0]
    change_wc = np.array([wc[0] - wc_prev] + [wc[i] - wc[i - 1] for i in range(1, len(wc))], dtype=float)

    fcff = nopat + da - capex - change_wc

    return pd.DataFrame(
        {
            "Year": yrs,
            "Revenue": revenue,
            "Rev_Growth": growth_path,
            "EBIT_Margin": ebit_margin_path,
            "EBIT": ebit,
            "NOPAT": nopat,
            "D&A": da,
            "CapEx": capex,
            "WC": wc,
            "ChangeWC": change_wc,
            "FCFF": fcff,
        }
    )


def pv_of_fcff(forecast_df: pd.DataFrame, wacc: float) -> float:
    fcff = forecast_df["FCFF"].values.astype(float)
    disc = np.array([(1 / ((1 + wacc) ** (i + 1))) for i in range(len(fcff))], dtype=float)
    return float(np.sum(fcff * disc))


def terminal_value_pgm(fcff_last: float, wacc: float, tg: float) -> float:
    if wacc <= tg:
        return np.nan
    return float((fcff_last * (1 + tg)) / (wacc - tg))


def terminal_value_exit_multiple(
    forecast_df: pd.DataFrame,
    exit_multiple: float,
    multiple_basis: str,
):
    """
    multiple_basis: 'EV/EBITDA' or 'EV/Revenue'
    Uses final forecast year EBITDA proxy or Revenue.
    EBITDA proxy = EBIT + D&A in final year.
    """
    last = forecast_df.iloc[-1]
    if multiple_basis == "EV/Revenue":
        base = float(last["Revenue"])
    else:
        base = float(last["EBIT"] + last["D&A"])
    if base <= 0 or np.isnan(exit_multiple):
        return np.nan
    return float(exit_multiple * base)


def dcf_enterprise_value(
    forecast_df: pd.DataFrame,
    wacc: float,
    tg: float,
    use_exit_multiple: bool,
    exit_multiple: float,
    multiple_basis: str,
    blend_weight_pgm: float,
):
    wacc = float(wacc)
    tg = float(tg)
    blend_weight_pgm = _clamp(float(blend_weight_pgm), 0.0, 1.0)

    pv_fcff = pv_of_fcff(forecast_df, wacc=wacc)

    fcff_last = float(forecast_df["FCFF"].iloc[-1])
    tv_pgm = terminal_value_pgm(fcff_last, wacc=wacc, tg=tg)

    tv_exit = np.nan
    if use_exit_multiple:
        tv_exit = terminal_value_exit_multiple(forecast_df, exit_multiple, multiple_basis)

    # If exit multiple is not usable, fall back to PGM 100%
    if use_exit_multiple and (np.isnan(tv_exit) or tv_exit <= 0):
        blend_weight_pgm = 1.0

    tv_blended = (
        (blend_weight_pgm * tv_pgm) + ((1.0 - blend_weight_pgm) * tv_exit)
        if use_exit_multiple
        else tv_pgm
    )

    # discount terminal value to present
    n = len(forecast_df)
    pv_tv = float(tv_blended / ((1 + wacc) ** n))

    ev = float(pv_fcff + pv_tv)

    return {
        "PV_FCFF": pv_fcff,
        "TV_PGM": tv_pgm,
        "TV_Exit": tv_exit,
        "TV_Blended": tv_blended,
        "PV_TerminalValue": pv_tv,
        "EnterpriseValue": ev,
        "blend_weight_pgm": blend_weight_pgm,
    }


def sensitivity_ev(
    forecast_df: pd.DataFrame,
    wacc_grid: list[float],
    tg_grid: list[float],
    use_exit_multiple: bool,
    exit_multiple: float,
    multiple_basis: str,
    blend_weight_pgm: float,
) -> pd.DataFrame:
    out = []
    for w in wacc_grid:
        row = {"WACC": w}
        for g in tg_grid:
            if w <= g:
                row[f"g={g:.2%}"] = np.nan
            else:
                ev = dcf_enterprise_value(
                    forecast_df,
                    wacc=w,
                    tg=g,
                    use_exit_multiple=use_exit_multiple,
                    exit_multiple=exit_multiple,
                    multiple_basis=multiple_basis,
                    blend_weight_pgm=blend_weight_pgm,
                )["EnterpriseValue"]
                row[f"g={g:.2%}"] = ev
        out.append(row)
    return pd.DataFrame(out)


# ------------------------
# Orchestrator (auto-first, override-always)
# ------------------------
def run_full_model(
    ticker: str,
    comps_list: list[str],
    # Scenario controls
    forecast_years: int,
    bull_growth_delta: float,
    weak_growth_delta: float,
    bull_margin_delta: float,
    weak_margin_delta: float,
    fade_to_growth: float,
    # Tax
    tax_rate: float,
    tax_override: bool,
    # Drivers overrides
    da_override: bool,
    da_pct_rev_override: float,
    capex_override: bool,
    capex_pct_rev_override: float,
    wc_override: bool,
    wc_pct_rev_override: float,
    # Shares / net debt overrides
    shares_override: bool,
    shares_override_value: float,
    net_debt_override: bool,
    net_debt_override_value: float,
    # WACC controls
    wacc_override: bool,
    wacc_override_value: float,
    rf: float,
    erp: float,
    beta_override: bool,
    beta_override_value: float,
    cod: float,
    target_debt_weight_override: bool,
    target_debt_weight_value: float,
    # Terminal controls
    terminal_growth: float,
    exit_multiple_override: bool,
    exit_multiple_override_value: float,
    blend_weight_pgm: float,
):
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker is required.")

    info = get_info_snapshot(ticker)
    hist = get_hist_financials(ticker)

    # ---- Tax auto default
    if not tax_override:
        # effective tax approximation: clamp your input if user doesn't override
        tax = _clamp(float(tax_rate), 0.10, 0.30)
    else:
        tax = _clamp(float(tax_rate), 0.0, 0.5)

    # ---- Driver autos from history
    rev_cagr = _cagr(hist["Revenue"].tail(min(5, len(hist))))
    if np.isnan(rev_cagr):
        rev_cagr = 0.08

    ebit_m_start_auto = _nanmedian(hist["EBIT_Margin"].tail(min(3, len(hist))), default=0.20)
    ebit_m_start_auto = _clamp(ebit_m_start_auto, 0.02, 0.60)

    # Fade growth: start ~ historical, end = fade_to_growth (user-set)
    base_g_start = float(rev_cagr)
    base_g_end = float(fade_to_growth)

    # Margin target: slight improvement from start, capped
    base_m_end = _clamp(ebit_m_start_auto + 0.01, 0.02, 0.65)

    da_pct_auto = _nanmedian((hist["D&A"] / hist["Revenue"]).tail(min(3, len(hist))), default=0.04)
    capex_pct_auto = _nanmedian((hist["CapEx"] / hist["Revenue"]).tail(min(3, len(hist))), default=0.05)
    wc_pct_auto = _nanmedian((hist["WC"] / hist["Revenue"]).tail(min(3, len(hist))), default=0.08)

    da_pct = float(da_pct_rev_override) if da_override else float(da_pct_auto)
    capex_pct = float(capex_pct_rev_override) if capex_override else float(capex_pct_auto)
    wc_pct = float(wc_pct_rev_override) if wc_override else float(wc_pct_auto)

    # Clean driver bounds
    da_pct = _clamp(da_pct, 0.0, 0.20)
    capex_pct = _clamp(capex_pct, 0.0, 0.30)
    wc_pct = _clamp(wc_pct, -0.10, 0.30)

    # ---- Shares + Net debt
    shares = info["shares_out"]
    if shares_override:
        shares = float(shares_override_value)

    # net debt from balance sheet latest debt-cash if possible
    last_cash = float(hist["Cash"].iloc[-1]) if not np.isnan(hist["Cash"].iloc[-1]) else np.nan
    last_debt = float(hist["Debt"].iloc[-1]) if not np.isnan(hist["Debt"].iloc[-1]) else np.nan
    net_debt_auto = (last_debt - last_cash) if (not np.isnan(last_debt) and not np.isnan(last_cash)) else np.nan

    net_debt = net_debt_auto
    if net_debt_override:
        net_debt = float(net_debt_override_value)

    # ---- Comps + exit multiple auto
    comps_df = fetch_comps(comps_list) if comps_list else pd.DataFrame()
    exit_infer = infer_exit_multiple(comps_df)

    multiple_basis = "EV/EBITDA" if exit_infer["method"].startswith("EV/EBITDA") else "EV/Revenue"
    exit_multiple = exit_infer["multiple"]

    if exit_multiple_override:
        exit_multiple = float(exit_multiple_override_value)
        # If user overrides, we assume they mean EV/EBITDA unless they’re clearly using EV/Revenue;
        # keep basis as inferred unless it was empty, then default to EV/EBITDA.
        if multiple_basis not in {"EV/EBITDA", "EV/Revenue"}:
            multiple_basis = "EV/EBITDA"

    use_exit_multiple = True  # always on; will gracefully fall back to PGM if unusable

    # ---- WACC
    beta = info["beta"]
    if beta_override:
        beta = float(beta_override_value)
    if np.isnan(beta) or beta <= 0:
        beta = 1.0

    market_cap = info["market_cap"]
    total_debt_for_wacc = last_debt

    target_dw = float(target_debt_weight_value) if target_debt_weight_override else np.nan

    wacc_build = build_wacc(
        market_cap=market_cap,
        total_debt=total_debt_for_wacc,
        tax_rate=tax,
        beta=beta,
        rf=float(rf),
        erp=float(erp),
        cost_of_debt=float(cod),
        target_debt_weight=target_dw if target_debt_weight_override else None,
    )

    wacc = float(wacc_override_value) if wacc_override else float(wacc_build["wacc"])
    wacc = _clamp(wacc, 0.01, 0.30)

    # ---- Build scenarios (auto-first + deltas)
    scenarios = {
        "Bull": {
            "g_start": base_g_start + float(bull_growth_delta),
            "g_end": base_g_end + float(bull_growth_delta) * 0.5,
            "m_start": ebit_m_start_auto + float(bull_margin_delta),
            "m_end": base_m_end + float(bull_margin_delta),
        },
        "Base": {
            "g_start": base_g_start,
            "g_end": base_g_end,
            "m_start": ebit_m_start_auto,
            "m_end": base_m_end,
        },
        "Weak": {
            "g_start": base_g_start + float(weak_growth_delta),
            "g_end": base_g_end + float(weak_growth_delta) * 0.5,
            "m_start": ebit_m_start_auto + float(weak_margin_delta),
            "m_end": base_m_end + float(weak_margin_delta),
        },
    }

    # Clamp scenario growth & margin to sane bounds
    for s in scenarios.values():
        s["g_start"] = _clamp(s["g_start"], -0.20, 0.50)
        s["g_end"] = _clamp(s["g_end"], -0.05, 0.15)
        s["m_start"] = _clamp(s["m_start"], 0.01, 0.70)
        s["m_end"] = _clamp(s["m_end"], 0.01, 0.75)

    scenario_rows = []
    scenario_detail = {}

    for name, p in scenarios.items():
        fc = build_forecast(
            hist_df=hist,
            years=int(forecast_years),
            rev_growth_start=float(p["g_start"]),
            rev_growth_end=float(p["g_end"]),
            ebit_margin_start=float(p["m_start"]),
            ebit_margin_end=float(p["m_end"]),
            da_pct_rev=float(da_pct),
            capex_pct_rev=float(capex_pct),
            wc_pct_rev=float(wc_pct),
            tax_rate=float(tax),
        )

        dcf = dcf_enterprise_value(
            forecast_df=fc,
            wacc=wacc,
            tg=float(terminal_growth),
            use_exit_multiple=use_exit_multiple,
            exit_multiple=float(exit_multiple) if not np.isnan(exit_multiple) else np.nan,
            multiple_basis=multiple_basis,
            blend_weight_pgm=float(blend_weight_pgm),
        )

        ev = float(dcf["EnterpriseValue"])
        equity = np.nan if np.isnan(net_debt) else (ev - float(net_debt))

        per_share = np.nan
        if not np.isnan(equity) and not np.isnan(shares) and shares > 0:
            per_share = equity / shares

        scenario_rows.append(
            {
                "Scenario": name,
                "RevGrowth_Start": float(p["g_start"]),
                "RevGrowth_End": float(p["g_end"]),
                "EBITMargin_Start": float(p["m_start"]),
                "EBITMargin_End": float(p["m_end"]),
                "WACC": wacc,
                "TGR": float(terminal_growth),
                "EV": ev,
                "EquityValue": equity,
                "PerShare": per_share,
                "PV_FCFF": dcf["PV_FCFF"],
                "PV_TerminalValue": dcf["PV_TerminalValue"],
                "TV_PGM": dcf["TV_PGM"],
                "TV_Exit": dcf["TV_Exit"],
                "TV_Blended": dcf["TV_Blended"],
            }
        )

        scenario_detail[name] = {
            "forecast_df": fc,
            "dcf_details": dcf,
        }

    scen_df = pd.DataFrame(scenario_rows)
    scen_df["ScenarioRank"] = scen_df["Scenario"].map({"Bull": 0, "Base": 1, "Weak": 2})
    scen_df = scen_df.sort_values("ScenarioRank").drop(columns=["ScenarioRank"]).reset_index(drop=True)

    # Headline = Base
    base = scen_df[scen_df["Scenario"] == "Base"].iloc[0]
    headline = {
        "Price": info["price"],
        "MarketCap": info["market_cap"],
        "SharesOut": shares,
        "NetDebt": net_debt,
        "WACC": wacc,
        "TGR": float(terminal_growth),
        "ExitMultiple": float(exit_multiple) if not np.isnan(exit_multiple) else np.nan,
        "ExitMultipleBasis": multiple_basis,
        "ExitMultipleSource": exit_infer["method"] if not exit_multiple_override else "OVERRIDE",
        "BlendWeightPGM": float(blend_weight_pgm),
        "EV_Base": float(base["EV"]),
        "Equity_Base": float(base["EquityValue"]) if not np.isnan(base["EquityValue"]) else np.nan,
        "PerShare_Base": float(base["PerShare"]) if not np.isnan(base["PerShare"]) else np.nan,
    }

    # Sensitivity on Base forecast
    base_fc = scenario_detail["Base"]["forecast_df"]
    wacc_grid = [round(x, 4) for x in [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02] if x > 0]
    tg_grid = [round(x, 4) for x in [terminal_growth - 0.01, terminal_growth, terminal_growth + 0.01]]

    sens_df = sensitivity_ev(
        forecast_df=base_fc,
        wacc_grid=wacc_grid,
        tg_grid=tg_grid,
        use_exit_multiple=use_exit_multiple,
        exit_multiple=float(exit_multiple) if not np.isnan(exit_multiple) else np.nan,
        multiple_basis=multiple_basis,
        blend_weight_pgm=float(blend_weight_pgm),
    )

    # Inputs trace (for transparency)
    inputs_used = {
        "ticker": ticker,
        "forecast_years": forecast_years,
        "fade_to_growth": fade_to_growth,
        "tax_rate": tax,
        "tax_override": tax_override,
        "da_pct_rev": da_pct,
        "da_override": da_override,
        "capex_pct_rev": capex_pct,
        "capex_override": capex_override,
        "wc_pct_rev": wc_pct,
        "wc_override": wc_override,
        "shares": shares,
        "shares_override": shares_override,
        "net_debt": net_debt,
        "net_debt_override": net_debt_override,
        "wacc": wacc,
        "wacc_override": wacc_override,
        "rf": rf,
        "erp": erp,
        "beta": beta,
        "beta_override": beta_override,
        "cost_of_debt": cod,
        "target_debt_weight_override": target_debt_weight_override,
        "target_debt_weight_value": target_debt_weight_value,
        "terminal_growth": terminal_growth,
        "exit_multiple": exit_multiple,
        "exit_multiple_override": exit_multiple_override,
        "exit_multiple_basis": multiple_basis,
        "blend_weight_pgm": blend_weight_pgm,
        "scenario_deltas": {
            "bull_growth_delta": bull_growth_delta,
            "weak_growth_delta": weak_growth_delta,
            "bull_margin_delta": bull_margin_delta,
            "weak_margin_delta": weak_margin_delta,
        },
    }

    return {
        "info": info,
        "hist_df": hist,
        "comps_df": comps_df,
        "wacc_build": wacc_build,
        "scenario_summary": scen_df,
        "scenario_detail": scenario_detail,
        "sensitivity_df": sens_df,
        "headline": headline,
        "inputs_used": inputs_used,
    }