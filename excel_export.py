from io import BytesIO
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


def _dict_to_df(d: dict) -> pd.DataFrame:
    rows = []
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                rows.append({"Key": f"{k}.{k2}", "Value": v2})
        else:
            rows.append({"Key": k, "Value": v})
    return pd.DataFrame(rows)


def _add_sheet(wb: Workbook, name: str, df: pd.DataFrame):
    ws = wb.create_sheet(title=name[:31])
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)
    ws.freeze_panes = "A2"


def export_to_excel(results: dict) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)

    _add_sheet(wb, "Headline", _dict_to_df(results.get("headline", {})))
    _add_sheet(wb, "Inputs_Used", _dict_to_df(results.get("inputs_used", {})))

    if "hist_df" in results and isinstance(results["hist_df"], pd.DataFrame):
        _add_sheet(wb, "Historical", results["hist_df"])

    if "scenario_summary" in results and isinstance(results["scenario_summary"], pd.DataFrame):
        _add_sheet(wb, "Scenario_Summary", results["scenario_summary"])

    sd = results.get("scenario_detail", {})
    if isinstance(sd, dict):
        for scen in ["Base", "Bull", "Weak"]:
            if scen in sd and "forecast_df" in sd[scen]:
                _add_sheet(wb, f"Forecast_{scen}", sd[scen]["forecast_df"])

    sens = results.get("sensitivity_df")
    if isinstance(sens, pd.DataFrame) and not sens.empty:
        _add_sheet(wb, "Sensitivity", sens)

    comps = results.get("comps_df")
    if isinstance(comps, pd.DataFrame) and not comps.empty:
        _add_sheet(wb, "Comps", comps)

    wacc = results.get("wacc_build", {})
    if isinstance(wacc, dict) and wacc:
        _add_sheet(wb, "WACC_Build", _dict_to_df(wacc))

    bio = BytesIO()
    wb.save(bio)
    return bio.getvalue()