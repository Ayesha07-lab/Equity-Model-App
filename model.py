import numpy as np
import pandas as pd
import yfinance as yf


# ---------------- Helpers ----------------
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


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _linear_ramp(start: float, end: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([end], dtype=float)
    return np.linspace(start, end, n, dtype=float)


# ---------------- Data pulls ----------------
def get_info_snapshot(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.get_info() or {}

    price = _safe_float(info.get("currentPrice"))
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
    ebitda = get_series(inc, ["Ebitda", "EBITDA"])

    da = get_series(cf, ["Depreciation And Amortization", "Depreciation", "Depreciation & Amortization"])
    capex = get_series(cf, ["Capital Expenditure", "CapitalExpenditures", "Capital Expenditures"])

    cash = get_series(bs, ["Cash And Cash Equivalents", "Cash", "Cash And Short Term Investments"])
    debt = get_series(bs, ["Total Debt", "Long Term Debt", "Short Long Term Debt"])

    curr_assets = get_series(bs, ["Total Current Assets", "Current Assets"])
    curr_liab = get_series(bs, ["Total Current Liabilities", "Current Liabilities"])
    wc = [np.nan if (np.isnan(a) or np.isnan(l)) else (a - l) for a, l in zip(curr_assets, curr_liab)]

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

    # Fill EBITDA if missing using EBIT + D&A
    missing = df["EBITDA"].isna()
    df.loc[missing, "EBITDA"] = df.loc[missing, "EBIT"] + df.loc[missing, "D&A"]

    df["EBIT_Margin"] = df["EBIT"] / df["Revenue"]
    df["EBITDA_Margin"] = df["EBITDA"] / df["Revenue"]
    return df


# ---------------- Comps ----------------
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
    if comps_df is None or comps_df.empty:
        return {"basis": "EV/EBITDA", "source": "NO_COMPS", "multiple": np.nan}

    ev_ebitda_med = _nanmedian(comps_df["EV/EBITDA"], default=np.nan)
    if not np.isnan(ev_ebitda_med) and ev_ebitda_med > 0:
        return {"basis": "EV/EBITDA", "source": "MEDIAN_COMPS", "multiple": float(ev_ebitda_med)}

    ev_rev_med = _nanmedian(comps_df["EV/Revenue"], default=np.nan)
    if not np.isnan(ev_rev_med) and ev_rev_med > 0:
        return {"basis": "EV/Revenue", "source": "MEDIAN_COMPS", "multiple": float(ev_rev_med)}

    return {"basis": "EV/EBITDA", "source": "COMPS_INVALID", "multiple": np.nan}


# ---------------- WACC build ----------------
def build_wacc(
    market_cap: float,
    total_debt: float,
    tax_rate: float,
    beta: float,
    rf: float,
    erp: float,
    cod: float,
    target_debt_weight: float | None,
) -> dict:
    tax_rate = _clamp(tax_rate, 0.0, 0.5)
    beta = 1.0 if (np.isnan(beta) or beta <= 0) else float(beta)

    ke = float(rf + beta * erp)
    kd = float(cod)
    kd_at = float(kd * (1 - tax_rate))

    if target_debt_weight is None or np.isnan(target_debt_weight):
        E = 0.0 if (np.isnan(market_cap) or market_cap <= 0) else float(market_cap)
        D = 0.0 if (np.isnan(total_debt) or total_debt < 0) else float(total_debt)
        V = E + D
        if V <= 0:
            wd, we = 0.0, 1.0
        else:
            wd, we = D / V, E / V
    else:
        wd = _clamp(float(target_debt_weight), 0.0, 0.95)
        we = 1.0 - wd

    wacc = float(we * ke + wd * kd_at)
    return {
        "rf": float(rf),
        "erp": float(erp),
        "beta": float(beta),
        "cost_of_equity": float(ke),
        "cost_of_debt": float(kd),
        "after_tax_cost_of_debt": float(kd_at),
        "weight_debt": float(wd),
        "weight_equity": float(we),
        "wacc": float(wacc),
    }


# ---------------- Forecast + DCF ----------------
def build_forecast(
    hist_df: pd.DataFrame,
    years: int,
    g_start: float,
    g_end: float,
    m_start: float,
    m_end: float,
    da_pct_rev: float,
    capex_pct_rev: float,
    wc_pct_rev: float,
    tax_rate: float,
) -> pd.DataFrame:
    last_year = int(hist_df["Year"].iloc[-1])
    yrs = [last_year + i for i in range(1, years + 1)]

    rev0 = float(hist_df["Revenue"].iloc[-1])
    growth_path = _linear_ramp(g_start, g_end, years)

    revenue = []
    cur = rev0
    for g in growth_path:
        cur = cur * (1 + g)
        revenue.append(cur)
    revenue = np.array(revenue, dtype=float)

    margin_path = _linear_ramp(m_start, m_end, years)
    ebit = revenue * margin_path
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
            "EBIT_Margin": margin_path,
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
    basis: str,
) -> float:
    last = forecast_df.iloc[-1]
    if basis == "EV/Revenue":
        base = float(last["Revenue"])
    else:
        base = float(last["EBIT"] + last["D&A"])  # EBITDA proxy
    if base <= 0 or np.isnan(exit_multiple) or exit_multiple <= 0:
        return np.nan
    return float(exit_multiple * base)


def dcf_enterprise_value(
    forecast_df: pd.DataFrame,
    wacc: float,
    tg: float,
    exit_multiple: float,
    exit_basis: str,
    blend_weight_pgm: float,
) -> dict:
    wacc = float(wacc)
    tg = float(tg)
    blend_weight_pgm = _clamp(float(blend_weight_pgm), 0.0, 1.0)

    pv_fcff = pv_of_fcff(forecast_df, wacc=wacc)

    fcff_last = float(forecast_df["FCFF"].iloc[-1])
    tv_pgm = terminal_value_pgm(fcff_last, wacc=wacc, tg=tg)

    tv_exit = terminal_value_exit_multiple(forecast_df, float(exit_multiple), exit_basis)

    # If exit multiple invalid, fall back to 100% PGM
    use_blend = True
    if np.isnan(tv_exit) or tv_exit <= 0:
        blend_weight_pgm = 1.0
        use_blend = False

    tv_blended = (blend_weight_pgm * tv_pgm) + ((1.0 - blend_weight_pgm) * tv_exit) if use_blend else tv_pgm

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
        "blend_weight_pgm_used": blend_weight_pgm,
    }


def sensitivity_ev(
    forecast_df: pd.DataFrame,
    wacc_grid: list[float],
    tg_grid: list[float],
    exit_multiple: float,
    exit_basis: str,
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
                    exit_multiple=exit_multiple,
                    exit_basis=exit_basis,
                    blend_weight_pgm=blend_weight_pgm,
                )["EnterpriseValue"]
                row[f"g={g:.2%}"] = ev
        out.append(row)
    return pd.DataFrame(out)


# ---------------- The ONE signature app.py uses ----------------
def run_full_model(
    ticker: str,
    comps_list: list[str],
    forecast_years: int,
    terminal_growth: float,
    blend_weight_pgm: float,
    # scenario deltas
    fade_to_growth: float,
    bull_growth_delta: float,
    weak_growth_delta: float,
    bull_margin_delta: float,
    weak_margin_delta: float,
    # overrides: tax + driver ratios
    tax_rate: float,
    tax_override: bool,
    da_override: bool,
    da_pct_rev_override: float,
    capex_override: bool,
    capex_pct_rev_override: float,
    wc_override: bool,
    wc_pct_rev_override: float,
    # overrides: shares + net debt
    shares_override: bool,
    shares_override_value: float,
    net_debt_override: bool,
    net_debt_override_value: float,
    # WACC build / override
    wacc_override: bool,
    wacc_override_value: float,
    rf: float,
    erp: float,
    beta_override: bool,
    beta_override_value: float,
    cod: float,
    target_debt_weight_override: bool,
    target_debt_weight_value: float,
    # terminal exit multiple override
    exit_multiple_override: bool,
    exit_multiple_override_value: float,
) -> dict:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker is required.")

    info = get_info_snapshot(ticker)
    hist = get_hist_financials(ticker)

    # Tax
    tax = _clamp(float(tax_rate), 0.0, 0.5) if tax_override else _clamp(float(tax_rate), 0.10, 0.30)

    # Autos from history
    rev_cagr = _cagr(hist["Revenue"].tail(min(5, len(hist))))
    if np.isnan(rev_cagr):
        rev_cagr = 0.08

    m_start_auto = _nanmedian(hist["EBIT_Margin"].tail(min(3, len(hist))), default=0.20)
    m_start_auto = _clamp(m_start_auto, 0.02, 0.60)

    g_start_base = float(rev_cagr)
    g_end_base = float(fade_to_growth)

    m_end_base = _clamp(m_start_auto + 0.01, 0.02, 0.65)

    da_pct_auto = _nanmedian((hist["D&A"] / hist["Revenue"]).tail(min(3, len(hist))), default=0.04)
    capex_pct_auto = _nanmedian((hist["CapEx"] / hist["Revenue"]).tail(min(3, len(hist))), default=0.05)
    wc_pct_auto = _nanmedian((hist["WC"] / hist["Revenue"]).tail(min(3, len(hist))), default=0.08)

    da_pct = float(da_pct_rev_override) if da_override else float(da_pct_auto)
    capex_pct = float(capex_pct_rev_override) if capex_override else float(capex_pct_auto)
    wc_pct = float(wc_pct_rev_override) if wc_override else float(wc_pct_auto)

    da_pct = _clamp(da_pct, 0.0, 0.20)
    capex_pct = _clamp(capex_pct, 0.0, 0.30)
    wc_pct = _clamp(wc_pct, -0.10, 0.30)

    # Shares + Net debt
    shares = float(shares_override_value) if shares_override else info["shares_out"]

    last_cash = float(hist["Cash"].iloc[-1]) if not np.isnan(hist["Cash"].iloc[-1]) else np.nan
    last_debt = float(hist["Debt"].iloc[-1]) if not np.isnan(hist["Debt"].iloc[-1]) else np.nan
    net_debt_auto = (last_debt - last_cash) if (not np.isnan(last_debt) and not np.isnan(last_cash)) else np.nan
    net_debt = float(net_debt_override_value) if net_debt_override else net_debt_auto

    # Comps + exit multiple auto/override
    comps_df = fetch_comps(comps_list) if comps_list else pd.DataFrame()
    exit_auto = infer_exit_multiple(comps_df)
    exit_basis = exit_auto["basis"]
    exit_multiple = float(exit_auto["multiple"]) if not np.isnan(exit_auto["multiple"]) else np.nan

    if exit_multiple_override:
        exit_multiple = float(exit_multiple_override_value)

    # WACC build / override
    beta = float(beta_override_value) if beta_override else info["beta"]
    if np.isnan(beta) or beta <= 0:
        beta = 1.0

    target_dw = float(target_debt_weight_value) if target_debt_weight_override else np.nan
    wacc_build = build_wacc(
        market_cap=info["market_cap"],
        total_debt=last_debt,
        tax_rate=tax,
        beta=beta,
        rf=float(rf),
        erp=float(erp),
        cod=float(cod),
        target_debt_weight=target_dw if target_debt_weight_override else None,
    )
    wacc = float(wacc_override_value) if wacc_override else float(wacc_build["wacc"])
    wacc = _clamp(wacc, 0.01, 0.30)

    # Scenarios
    scenarios = {
        "Bull": {
            "g_start": g_start_base + float(bull_growth_delta),
            "g_end": g_end_base + float(bull_growth_delta) * 0.5,
            "m_start": m_start_auto + float(bull_margin_delta),
            "m_end": m_end_base + float(bull_margin_delta),
        },
        "Base": {
            "g_start": g_start_base,
            "g_end": g_end_base,
            "m_start": m_start_auto,
            "m_end": m_end_base,
        },
        "Weak": {
            "g_start": g_start_base + float(weak_growth_delta),
            "g_end": g_end_base + float(weak_growth_delta) * 0.5,
            "m_start": m_start_auto + float(weak_margin_delta),
            "m_end": m_end_base + float(weak_margin_delta),
        },
    }

    # Clamp scenario values
    for s in scenarios.values():
        s["g_start"] = _clamp(s["g_start"], -0.20, 0.50)
        s["g_end"] = _clamp(s["g_end"], -0.05, 0.20)
        s["m_start"] = _clamp(s["m_start"], 0.01, 0.75)
        s["m_end"] = _clamp(s["m_end"], 0.01, 0.80)

    scenario_rows = []
    scenario_detail = {}

    for name, p in scenarios.items():
        fc = build_forecast(
            hist_df=hist,
            years=int(forecast_years),
            g_start=float(p["g_start"]),
            g_end=float(p["g_end"]),
            m_start=float(p["m_start"]),
            m_end=float(p["m_end"]),
            da_pct_rev=float(da_pct),
            capex_pct_rev=float(capex_pct),
            wc_pct_rev=float(wc_pct),
            tax_rate=float(tax),
        )

        dcf = dcf_enterprise_value(
            forecast_df=fc,
            wacc=wacc,
            tg=float(terminal_growth),
            exit_multiple=float(exit_multiple) if not np.isnan(exit_multiple) else np.nan,
            exit_basis=exit_basis,
            blend_weight_pgm=float(blend_weight_pgm),
        )

        ev = float(dcf["EnterpriseValue"])
        equity = np.nan if np.isnan(net_debt) else (ev - float(net_debt))

        per_share = np.nan
        if not np.isnan(equity) and not np.isnan(shares) and shares and shares > 0:
            per_share = equity / shares

        scenario_rows.append(
            {
                "Scenario": name,
                "WACC": wacc,
                "TGR": float(terminal_growth),
                "RevGrowth_Start": float(p["g_start"]),
                "RevGrowth_End": float(p["g_end"]),
                "EBITMargin_Start": float(p["m_start"]),
                "EBITMargin_End": float(p["m_end"]),
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

        scenario_detail[name] = {"forecast_df": fc, "dcf_details": dcf}

    scen_df = pd.DataFrame(scenario_rows)
    scen_df["Rank"] = scen_df["Scenario"].map({"Bull": 0, "Base": 1, "Weak": 2})
    scen_df = scen_df.sort_values("Rank").drop(columns=["Rank"]).reset_index(drop=True)

    base_row = scen_df[scen_df["Scenario"] == "Base"].iloc[0]

    headline = {
        "Ticker": ticker,
        "Price": info["price"],
        "MarketCap": info["market_cap"],
        "SharesOut": shares,
        "NetDebt": net_debt,
        "WACC": wacc,
        "TGR": float(terminal_growth),
        "ExitMultiple": exit_multiple,
        "ExitBasis": exit_basis,
        "ExitSource": "OVERRIDE" if exit_multiple_override else exit_auto["source"],
        "BlendWeightPGM_Requested": float(blend_weight_pgm),
        "BlendWeightPGM_Used": float(scenario_detail["Base"]["dcf_details"]["blend_weight_pgm_used"]),
        "EV_Base": float(base_row["EV"]),
        "Equity_Base": float(base_row["EquityValue"]) if not np.isnan(base_row["EquityValue"]) else np.nan,
        "PerShare_Base": float(base_row["PerShare"]) if not np.isnan(base_row["PerShare"]) else np.nan,
    }

    # Sensitivity (Base)
    base_fc = scenario_detail["Base"]["forecast_df"]
    wacc_grid = [round(x, 4) for x in [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02] if x > 0]
    tg_grid = [round(x, 4) for x in [terminal_growth - 0.01, terminal_growth, terminal_growth + 0.01]]

    sens_df = sensitivity_ev(
        forecast_df=base_fc,
        wacc_grid=wacc_grid,
        tg_grid=tg_grid,
        exit_multiple=float(exit_multiple) if not np.isnan(exit_multiple) else np.nan,
        exit_basis=exit_basis,
        blend_weight_pgm=float(blend_weight_pgm),
    )

    inputs_used = {
        "tax_rate_used": tax,
        "driver_da_pct_rev_used": da_pct,
        "driver_capex_pct_rev_used": capex_pct,
        "driver_wc_pct_rev_used": wc_pct,
        "exit_multiple_used": exit_multiple,
        "exit_basis_used": exit_basis,
        "wacc_used": wacc,
        "wacc_build": wacc_build,
        "overrides": {
            "tax_override": tax_override,
            "da_override": da_override,
            "capex_override": capex_override,
            "wc_override": wc_override,
            "shares_override": shares_override,
            "net_debt_override": net_debt_override,
            "wacc_override": wacc_override,
            "beta_override": beta_override,
            "target_debt_weight_override": target_debt_weight_override,
            "exit_multiple_override": exit_multiple_override,
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