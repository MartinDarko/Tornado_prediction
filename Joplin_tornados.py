#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joplin Missouri is the site of the largest recorded tornado in history, an EF5. This script
uses publicly available data to calculate the likelihood of a tornado in Joplin in the next 24 hours.

Data sources:
- NWS API (api.weather.gov): points -> forecastHourly, forecastGridData, and active alerts near a point
- SPC Outlooks (ArcGIS REST): Day 1 categorical and probabilistic tornado polygons

How it works (high level):
1) Resolve the forecast endpoints for a lat/lon (Joplin) via /points.
2) Pull next-24h hourly forecast; extract thunder mentions, max PoP, and wind gusts.
3) Check active NWS alerts at that point (Tornado Watch/Warning lift the score).
4) Query SPC polygons for the point to get categorical risk (MRGL/SLGT/ENH/MDT/HIGH)
   and probabilistic tornado percentage (e.g., 2%, 5%, 10%, 15%, 30%).
5) Combine into a 0–100 score with a transparent, explainable heuristic.

You can adapt lat/lon to any location.
"""

import datetime as dt
import math
import re
import sys
from typing import Dict, Any, Optional

import requests


# -----------------------------
# lat and long coordinates  for Joplin
# -----------------------------
LAT = 37.0842
LON = -94.5133

# User-Agent is required by api.weather.gov policy; include contact info
HEADERS = {
    "User-Agent": "TornadoHeuristicDemo/1.0 (your_email@example.com)",
    "Accept": "application/geo+json"
}

# SPC MapServer base and layer IDs (per NOAA ArcGIS service)
SPC_BASE = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/SPC_wx_outlks/MapServer"
LAYER_DAY1_CATEGORICAL = 1            # MRGL/SLGT/ENH/MDT/HIGH
LAYER_DAY1_PROB_TORNADO = 3           # e.g., 2%, 5%, 10%, 15%, 30%, 45%

# -----------------------------
# Utility helpers
# -----------------------------
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def next_24h_window() -> tuple[dt.datetime, dt.datetime]:
    start = now_utc()
    end = start + dt.timedelta(hours=24)
    return start, end

def fmt_pct(x: float) -> str:
    return f"{x:.0f}%"

def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# -----------------------------
# Step 1: Resolve NWS endpoints
# -----------------------------
def get_points_metadata(lat: float, lon: float) -> Dict[str, Any]:
    """
    Calls api.weather.gov/points/{lat},{lon} to discover forecast URLs.
    """
    url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Step 2: Hourly forecast (24h)
# -----------------------------
def get_hourly_forecast(hourly_url: str) -> list[Dict[str, Any]]:
    """
    Fetch hourly forecast periods from the provided URL.
    """
    r = requests.get(hourly_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("properties", {}).get("periods", [])

def summarize_hourly(periods_24h: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    From next-24h periods, summarize:
      - thunder_hours: count hours whose shortForecast mentions thunder/storms
      - max_pop: maximum Probability of Precipitation (%)
      - max_gust_mph: maximum wind gust in mph (if available)
    """
    thunder_re = re.compile(r"(thunder|t-storm|storm)", re.I)

    thunder_hours = 0
    max_pop = 0
    max_gust_mph = 0

    for p in periods_24h:
        # NWS hourly fields commonly include: shortForecast, probabilityOfPrecipitation.value, windGust
        sf = p.get("shortForecast", "") or ""
        if thunder_re.search(sf):
            thunder_hours += 1

        pop = safe_get(p, "probabilityOfPrecipitation", "value", default=0) or 0
        # probability may be None; clamp to [0,100]
        try:
            pop = max(0, min(100, int(pop)))
        except Exception:
            pop = 0
        max_pop = max(max_pop, pop)

        gust = p.get("windGust")  # sometimes "windGust" is a string like "25 mph"; sometimes None
        gust_mph = 0
        if isinstance(gust, str):
            m = re.search(r"(\d+)", gust)
            if m:
                gust_mph = int(m.group(1))
        elif isinstance(gust, (int, float)):
            gust_mph = float(gust)
        # Some feeds use "windSpeed" and no gusts; you could also parse that as a fallback.

        max_gust_mph = max(max_gust_mph, int(gust_mph))

    return {
        "thunder_hours": thunder_hours,
        "max_pop": max_pop,
        "max_gust_mph": max_gust_mph,
    }

# --------------------------------------
# Step 3: NWS active alerts at the point
# --------------------------------------
def get_active_alerts(lat: float, lon: float) -> list[Dict[str, Any]]:
    """
    Query NWS active alerts for the given point. We'll look for Tornado Watch/Warning.
    """
    # The alerts API supports filtering; here we pull all active for the point and scan.
    url = f"https://api.weather.gov/alerts/active?point={lat:.4f},{lon:.4f}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("features", []) or []

def classify_tornado_alerts(features: list[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Return flags for Tornado Watch / Tornado Warning / PDS (particularly dangerous situation).
    """
    has_watch = False
    has_warning = False
    has_pds = False

    for f in features:
        event = safe_get(f, "properties", "event", default="") or ""
        headline = safe_get(f, "properties", "headline", default="") or ""
        desc = safe_get(f, "properties", "description", default="") or ""
        txt = f"{event} {headline} {desc}".lower()

        if "tornado warning" in txt:
            has_warning = True
        if "tornado watch" in txt:
            has_watch = True
        if "particularly dangerous situation" in txt or "pds" in txt:
            has_pds = True

    return {
        "has_watch": has_watch,
        "has_warning": has_warning,
        "has_pds": has_pds,
    }

# ---------------------------------------------------------
# Step 4: SPC Day 1 Categorical & Probabilistic Tornado
# ---------------------------------------------------------
def _spc_point_query(layer_id: int, lat: float, lon: float) -> Dict[str, Any]:
    """
    ArcGIS REST 'query' with a point geometry and spatialRel=intersects, return GeoJSON.
    """
    url = f"{SPC_BASE}/{layer_id}/query"
    params = {
        "f": "geoJSON",
        "geometry": f"{lon},{lat}",  # x,y = lon,lat
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "false",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_spc_categorical(lat: float, lon: float) -> Optional[str]:
    """
    Returns one of: 'TSTM','MRGL','SLGT','ENH','MDT','HIGH' if point is inside a Day 1 categorical polygon.
    """
    data = _spc_point_query(LAYER_DAY1_CATEGORICAL, lat, lon)
    feats = data.get("features", [])
    if not feats:
        return None
    # Common categorical label fields across SPC feeds include 'LABEL' or 'Type'—we'll try both.
    props = feats[0].get("properties", {})
    label = props.get("LABEL") or props.get("LABEL2") or props.get("Type") or props.get("label") or ""
    # Normalize common text
    label = label.upper().replace(" ", "")
    # Map a few variants
    if "TSTM" in label: return "TSTM"
    for v in ["MRGL", "SLGT", "ENH", "MDT", "HIGH"]:
        if v in label:
            return v
    return None

def get_spc_prob_tornado(lat: float, lon: float) -> Optional[int]:
    """
    Returns probabilistic tornado percentage (e.g., 2,5,10,15,30,45) for Day 1 if present.
    """
    data = _spc_point_query(LAYER_DAY1_PROB_TORNADO, lat, lon)
    feats = data.get("features", [])
    if not feats:
        return None
    props = feats[0].get("properties", {})
    # Fields often include 'LABEL' like '5%'; extract the integer
    label = (props.get("LABEL") or props.get("label") or "")
    m = re.search(r"(\d+)\s*%", str(label))
    if m:
        return int(m.group(1))
    # Sometimes percentage might be in another field; scan all property values
    for v in props.values():
        m = re.search(r"(\d+)\s*%", str(v))
        if m:
            return int(m.group(1))
    return None

# ---------------------------------------------------------
# Step 5: Combine into a heuristic 0–100 score
# ---------------------------------------------------------
CAT_WEIGHTS = {
    None: 0,      # no categorical polygon at this point
    "TSTM": 5,    # general thunder
    "MRGL": 15,
    "SLGT": 25,
    "ENH": 40,
    "MDT": 60,
    "HIGH": 80,
}

def score_likelihood(
    spc_cat: Optional[str],
    spc_prob_tor: Optional[int],
    alerts: Dict[str, bool],
    hourly_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simple, explainable scoring:
      - If Tornado Warning: 90 (min) — immediate high risk.
      - If PDS Warning: 98 (min).
      - Else if Watch: raise floor to 70.
      - Base from SPC: categorical weight + 1.2 * prob% (if present).
      - Weather flavor: +2 per thunder hour (cap 10 hours), + bonus if gusts >35 mph, +0.2 * max PoP.

    Final score clipped to [0, 100].
    """
    has_watch = alerts.get("has_watch", False)
    has_warning = alerts.get("has_warning", False)
    has_pds = alerts.get("has_pds", False)

    thunder_hours = min(int(hourly_summary.get("thunder_hours", 0)), 10)
    max_pop = int(hourly_summary.get("max_pop", 0))
    max_gust = int(hourly_summary.get("max_gust_mph", 0))

    # Base from SPC
    score = CAT_WEIGHTS.get(spc_cat, 0)
    if spc_prob_tor is not None:
        score += 1.2 * spc_prob_tor  # give real weight to the SPC tornado % polygon

    # Hourly flavor
    score += 2.0 * thunder_hours
    if max_gust > 35:
        score += min(max_gust - 35, 30)  # cap the gust bonus
    score += 0.2 * max_pop

    # Alerts dominate
    if has_watch:
        score = max(score, 70.0)
    if has_warning:
        score = max(score, 90.0)
    if has_pds:
        score = max(score, 98.0)

    score = max(0.0, min(100.0, score))

    return {
        "score_0_100": round(score, 1),
        "components": {
            "spc_categorical": spc_cat,
            "spc_prob_tornado_pct": spc_prob_tor,
            "alerts": alerts,
            "hourly_summary": hourly_summary,
        }
    }

# ---------------------------------------------------------
# Orchestration
# ---------------------------------------------------------
def main(lat: float = LAT, lon: float = LON):
    # 1) Resolve forecast URLs for this point
    points = get_points_metadata(lat, lon)
    hourly_url = safe_get(points, "properties", "forecastHourly")
    if not hourly_url:
        raise RuntimeError("Could not resolve forecastHourly URL from NWS points endpoint.")

    # 2) Pull hourly forecast and slice next 24 hours
    periods = get_hourly_forecast(hourly_url)

    start_utc, end_utc = next_24h_window()
    next24 = []
    for p in periods:
        # Parse start time of each hourly period
        ts = p.get("startTime")
        if not ts:
            continue
        t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if start_utc <= t < end_utc:
            next24.append(p)

    hourly_summary = summarize_hourly(next24)

    # 3) Active alerts at the point
    active = get_active_alerts(lat, lon)
    alert_flags = classify_tornado_alerts(active)

    # 4) SPC Day 1 categorical + probabilistic tornado polygons
    spc_cat = get_spc_categorical(lat, lon)          # e.g., SLGT
    spc_prob = get_spc_prob_tornado(lat, lon)        # e.g., 5 (meaning 5%)

    # 5) Combine into a score
    result = score_likelihood(spc_cat, spc_prob, alert_flags, hourly_summary)

    # Pretty print for interview demo
    print("=== Tornado Likelihood (Heuristic) — Next 24h ===")
    print(f"Location: {lat:.4f}, {lon:.4f} (Joplin, MO)")
    print(f"SPC categorical: {spc_cat or 'None'} | SPC tornado prob: {spc_prob or 0}%")
    print(f"Alerts — Watch: {alert_flags['has_watch']}  Warning: {alert_flags['has_warning']}  PDS: {alert_flags['has_pds']}")
    print(f"Hourly summary — thunder_hours: {hourly_summary['thunder_hours']}, "
          f"max PoP: {hourly_summary['max_pop']}%, max gust: {hourly_summary['max_gust_mph']} mph")
    print(f"Likelihood score (0–100): {result['score_0_100']}")
    return result

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
