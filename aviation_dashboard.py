import datetime
import json
import os
import re
import requests
import concurrent.futures
import logging
import pygrib
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CACHE_DIR = "./workspace_cache"
HISTORY_FILE = "history.json"
STATIONS = ["kdab", "kxmr", "kmlb", "kfpr", "kpbi"]
MODELS = ["gfs", "rap", "hrrr"]

# Fixed coordinates for target locations
STN_COORDS = {
    "kxmr": {"lat": 28.468, "lon": -80.556},
    "kdab": {"lat": 29.180, "lon": -81.058},
    "kmlb": {"lat": 28.103, "lon": -80.645},
    "kfpr": {"lat": 27.498, "lon": -80.373},
    "kpbi": {"lat": 26.683, "lon": -80.095}
}

# Known fixed message ordering within each HREFLD GRIB2 file.
# Confirmed by CDL metadata variable ordering:
#   msg 1 = above_25, msg 2 = above_50, msg 3 = above_100, msg 4 = above_200
# This positional map is used as the primary strategy; key-based introspection
# is attempted first per message and overrides the positional value if it succeeds.
MSG_INDEX_THRESHOLDS = {1: "p25", 2: "p50", 3: "p100", 4: "p200"}
THRESHOLD_MAP = {25: "p25", 50: "p50", 100: "p100", 200: "p200"}


def purge_workspace(cache_dir=CACHE_DIR):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    else:
        for f in os.listdir(cache_dir):
            try:
                os.unlink(os.path.join(cache_dir, f))
            except Exception:
                pass
    return cache_dir


def pressure_to_height_ft(pres_hpa):
    """Converts barometric pressure to standard atmosphere height in feet."""
    return 145366.45 * (1.0 - (pres_hpa / 1013.25) ** 0.190284)


def _get_probability_upper_limit(grb):
    """
    Attempts to extract (probabilityType, upperLimitValue) from a pygrib message
    using three strategies in order of reliability:

      1. scaledValueOfUpperLimit / scaleFactorOfUpperLimit  — raw PDT 4.9 keys,
         most universally available across eccodes versions.
      2. probabilityUpperLimit convenience alias  — available in newer builds.
      3. Returns (None, None) if both fail, signalling the caller to use the
         positional fallback instead.

    probabilityType == 1 means "above upper limit" per GRIB2 Code Table 4.9.
    """
    # Strategy 1: raw scaled keys (most compatible)
    try:
        ptype = int(grb.probabilityType)
        scale_val = float(grb.scaledValueOfUpperLimit)
        scale_fac = int(grb.scaleFactorOfUpperLimit)
        upper = scale_val / (10 ** scale_fac) if scale_fac != 0 else scale_val
        return ptype, upper
    except Exception:
        pass

    # Strategy 2: convenience alias key
    try:
        ptype = int(grb.probabilityType)
        upper = float(grb.probabilityUpperLimit)
        return ptype, upper
    except Exception:
        pass

    return None, None


def extract_lightning_from_file(filepath, lat, lon):
    """
    Opens a downloaded SPC Post HREFLD GRIB2 file, finds the nearest Lambert
    Conformal grid point to (lat, lon), and returns all 4 lightning strike
    probability thresholds: p25, p50, p100, p200.

    Matching strategy (per message):
      - Try key-based probability metadata first (_get_probability_upper_limit).
      - Fall back to known fixed message-index ordering if keys are unavailable.
        The CDL metadata confirms consistent ordering across all files:
        msg 1=above_25, msg 2=above_50, msg 3=above_100, msg 4=above_200.
    """
    threshold_results = {"p25": 0, "p50": 0, "p100": 0, "p200": 0}

    try:
        grbs = pygrib.open(filepath)

        # Build lat/lon grid from first message
        sample = grbs[1]
        lats, lons = sample.latlons()

        # Normalise longitudes to -180..+180 if the grid uses 0-360
        if np.any(lons > 180):
            lons = lons - 360.0

        # Nearest-neighbour index in the 185x129 Lambert Conformal grid
        dist = (lats - lat) ** 2 + (lons - lon) ** 2
        y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)

        grbs.seek(0)
        for msg_idx, grb in enumerate(grbs, start=1):
            pixel_value = float(grb.values[y_idx, x_idx])
            if np.isnan(pixel_value):
                pixel_value = 0.0
            val = int(round(pixel_value))

            # --- Primary: key-based matching ---
            ptype, upper = _get_probability_upper_limit(grb)
            if ptype is not None and upper is not None:
                if ptype == 1:
                    key = THRESHOLD_MAP.get(int(round(upper)))
                    if key:
                        threshold_results[key] = val
                        logging.debug(f"Key match: msg {msg_idx} upper={upper} -> {key}={val}%")
                    else:
                        logging.debug(f"Unrecognised upper limit value: {upper}")
                continue

            # --- Fallback: positional ordering ---
            key = MSG_INDEX_THRESHOLDS.get(msg_idx)
            if key:
                threshold_results[key] = val
                logging.debug(f"Positional fallback: msg {msg_idx} -> {key}={val}%")

        grbs.close()

    except Exception as e:
        logging.error(f"Pygrib extraction failed for {filepath}: {e}")

    return threshold_results


def fetch_href_lightning_point(session, stn, lat, lon, date_str, cycle, f_hour):
    """
    Downloads one HREFLD GRIB2 file for the given forecast hour and extracts
    the lightning probability values at the nearest grid point to (lat, lon).
    Cleans up the local file after extraction.
    """
    base_url = (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/"
        f"spc_post.{date_str}/ltgdensity"
    )
    f_hour_padded = f"{int(f_hour):03d}"
    file_name = f"spc_post.t{cycle}z.hrefld_4hr.f{f_hour_padded}.grib2"
    url = f"{base_url}/{file_name}"
    local_path = os.path.join(CACHE_DIR, f"{stn}_{file_name}")

    try:
        with session.get(url, timeout=7, stream=True) as r:
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                vals = extract_lightning_from_file(local_path, lat, lon)
                if os.path.exists(local_path):
                    os.remove(local_path)
                return stn, vals
            else:
                logging.debug(f"HTTP {r.status_code} for {url}")
    except Exception as e:
        logging.debug(f"Failed download or parse for {file_name}: {e}")

    return stn, {"p25": 0, "p50": 0, "p100": 0, "p200": 0}


def fetch_href_lightning(time_keys):
    """
    Fetches HREFLD lightning probability data for all stations across all
    valid-time rows in time_keys.

    Cycle discovery checks in descending order (18Z -> 12Z -> 06Z -> 00Z)
    so the most recently initialised run is always preferred.

    Forecast hours are derived from the valid-time string ("DD/HH") relative
    to the active cycle init time, avoiding the idx+1 offset bug.
    """
    href_data = {
        stn: {row: {"p25": 0, "p50": 0, "p100": 0, "p200": 0} for row in time_keys}
        for stn in STATIONS
    }

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    date_str = now_utc.strftime("%Y%m%d")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50)
    session.mount("https://", adapter)

    # Descending cycle order so the freshest available run is picked first
    active_cycle = "00"
    for cycle in ["18", "12", "06", "00"]:
        test_url = (
            f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/"
            f"spc_post.{date_str}/ltgdensity/"
            f"spc_post.t{cycle}z.hrefld_4hr.f004.grib2"
        )
        try:
            if session.head(test_url, timeout=3).status_code == 200:
                active_cycle = cycle
                logging.info(f"Connected to live SPC Post inventory for cycle: {active_cycle}Z")
                break
        except Exception:
            continue

    # Anchor forecast-hour calculations to the active cycle's init time
    cycle_init_utc = now_utc.replace(
        hour=int(active_cycle), minute=0, second=0, microsecond=0
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures_map = {}
        for row_key in time_keys:
            # Derive forecast hour from valid-time string "DD/HH"
            try:
                d_part, h_part = map(int, row_key.split("/"))
                valid_dt = now_utc.replace(
                    day=d_part, hour=h_part, minute=0, second=0, microsecond=0
                )
                # Handle month boundary: if valid day < today's day, it's next month
                if valid_dt < cycle_init_utc:
                    valid_dt += datetime.timedelta(days=30)
                f_hour_int = int((valid_dt - cycle_init_utc).total_seconds() / 3600)
            except Exception:
                continue

            if f_hour_int < 1 or f_hour_int > 48:
                continue

            for stn, coords in STN_COORDS.items():
                future = executor.submit(
                    fetch_href_lightning_point,
                    session, stn, coords["lat"], coords["lon"],
                    date_str, active_cycle, f_hour_int
                )
                futures_map[future] = (stn, row_key)

        for future in concurrent.futures.as_completed(futures_map):
            stn, row_key = futures_map[future]
            try:
                _, vals = future.result()
                href_data[stn][row_key] = vals
            except Exception:
                pass

    return href_data


def parse_time_series_bufkit(bufkit_text):
    """
    Parses a BUFKit sounding text file and returns a dict keyed by
    valid-time string "DD/HH" containing derived meteorological parameters.
    """
    hourly_data = {}
    blocks = bufkit_text.split("STID = ")

    for block in blocks:
        if not block.strip():
            continue
        time_match = re.search(r"TIME\s*=\s*(\d{6})/(\d{4})", block)
        if not time_match:
            continue

        date_part, time_part = time_match.groups()
        try:
            valid_hour_key = f"{int(date_part[4:6]):02d}/{int(time_part[0:2]):02d}"
        except (ValueError, IndexError):
            continue

        lines = block.splitlines()
        profile_layers = []
        pres_idx, tmpc_idx, dwpt_idx, sknt_idx = 0, 1, 3, 5
        header_names = []
        in_profile = False

        for line in lines:
            cleaned = line.strip()
            if not cleaned:
                continue
            if "PRES" in cleaned or "TMPC" in cleaned or "SKNT" in cleaned:
                in_profile = True
                header_names.extend(cleaned.split())
                try:
                    if "PRES" in header_names:
                        pres_idx = header_names.index("PRES")
                    if "TMPC" in header_names:
                        tmpc_idx = header_names.index("TMPC")
                    if "DWPT" in header_names:
                        dwpt_idx = header_names.index("DWPT")
                    if "SKNT" in header_names:
                        sknt_idx = header_names.index("SKNT")
                except ValueError:
                    pass
                continue

            if in_profile:
                if "STID" in cleaned or "STNM" in cleaned:
                    break
                parts = cleaned.split()
                if len(parts) > max(pres_idx, tmpc_idx, dwpt_idx, sknt_idx):
                    try:
                        if not parts[0].replace(".", "", 1).replace("-", "", 1).isdigit():
                            continue
                        pres = float(parts[pres_idx])
                        tmpc = float(parts[tmpc_idx])
                        dwpt = float(parts[dwpt_idx])
                        sknt = float(parts[sknt_idx])
                        if 100.0 <= pres <= 1050.0:
                            profile_layers.append({
                                "pres": pres,
                                "hght": pressure_to_height_ft(pres),
                                "tmpc": tmpc,
                                "dwpt": dwpt,
                                "depr": tmpc - dwpt,
                                "sknt": sknt,
                            })
                    except (ValueError, IndexError):
                        continue

        if not profile_layers:
            continue
        profile_layers.sort(key=lambda x: x["pres"], reverse=True)

        def get_height_of_isotherm(target_temp):
            for i in range(len(profile_layers) - 1):
                t1, t2 = profile_layers[i]["tmpc"], profile_layers[i + 1]["tmpc"]
                h1, h2 = profile_layers[i]["hght"], profile_layers[i + 1]["hght"]
                if (t1 >= target_temp >= t2) or (t1 <= target_temp <= t2):
                    if t1 == t2:
                        return h1 / 1000.0
                    fraction = (target_temp - t1) / (t2 - t1)
                    return (h1 + fraction * (h2 - h1)) / 1000.0
            return profile_layers[-1]["hght"] / 1000.0

        cloud_layers = []
        active_cloud = None
        for layer in profile_layers:
            if layer["depr"] <= 2.0:
                if active_cloud is None:
                    active_cloud = {
                        "base": layer["hght"],
                        "top": layer["hght"],
                        "min_temp": layer["tmpc"],
                    }
                else:
                    active_cloud["top"] = layer["hght"]
                    active_cloud["min_temp"] = min(active_cloud["min_temp"], layer["tmpc"])
            elif active_cloud is not None:
                cloud_layers.append(active_cloud)
                active_cloud = None
        if active_cloud:
            cloud_layers.append(active_cloud)

        pbl_winds = [l["sknt"] for l in profile_layers if l["pres"] >= 850.0]
        mean_wind = sum(pbl_winds) / len(pbl_winds) if pbl_winds else 0.0
        sfc_depr = profile_layers[0]["depr"] if profile_layers else 10.0
        vis = (
            0.25 if sfc_depr <= 0.5
            else (1.0 if sfc_depr <= 1.0 else (3.0 if sfc_depr <= 2.0 else 10.0))
        )
        valid_ceilings = [c for c in cloud_layers if c["base"] >= 100.0]
        ceiling_val = round(valid_ceilings[0]["base"]) if valid_ceilings else 24000.0

        hourly_data[valid_hour_key] = {
            "mom_mean": round(mean_wind, 1),
            "mom_max": round(max(pbl_winds) if pbl_winds else 0.0, 1),
            "shear": "WS020/25022KT" if (max(pbl_winds) if pbl_winds else 0) > 35 else None,
            "vis": vis,
            "ceiling": ceiling_val,
            "hght_0c": round(get_height_of_isotherm(0.0), 1),
            "hght_5c": round(get_height_of_isotherm(-5.0), 1),
            "hght_10c": round(get_height_of_isotherm(-10.0), 1),
            "hght_20c": round(get_height_of_isotherm(-20.0), 1),
            "cloud_top": round(
                max([c["top"] for c in cloud_layers], default=0.0) / 1000.0, 1
            ),
            "cloud_thick": round(
                max([max(0.0, c["top"] - c["base"]) for c in cloud_layers], default=0.0)
                / 1000.0,
                1,
            ),
        }
    return hourly_data


def fetch_station_model(session, stn, model):
    download_id = "xmr" if stn == "kxmr" else stn
    model_prefix = "gfs3" if model == "gfs" else model
    url = (
        f"http://www.meteo.psu.edu/bufkit/data/{model.upper()}/latest/"
        f"{model_prefix}_{download_id}.buf"
    )
    try:
        response = session.get(url, timeout=15)
        if response.status_code == 200:
            return stn, model, parse_time_series_bufkit(response.text)
    except Exception as e:
        logging.error(f"Error fetching {stn} {model}: {e}")
    return stn, model, {}


def generate_aviation_dashboard(stations, models, current_sounding_matrix, time_rows):
    href_lightning = fetch_href_lightning(time_rows)

    history_runs = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history_runs = json.load(f)
        except Exception:
            history_runs = []

    current_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    current_entry = {
        "timestamp": current_timestamp,
        "data": current_sounding_matrix,
        "href_lightning": href_lightning,
    }

    if not history_runs or history_runs[0]["timestamp"] != current_timestamp:
        history_runs.insert(0, current_entry)
    history_runs = history_runs[:5]

    with open(HISTORY_FILE, "w") as f:
        json.dump(history_runs, f, indent=2)
    logging.info("Dashboard matrix completely compiled and written to history.json.")


def run_pipeline():
    logging.info("Starting complete structural iteration run...")
    purge_workspace()
    sounding_matrix = {stn: {mdl: {} for mdl in MODELS} for stn in STATIONS}

    temp_time_rows_set = set()
    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(fetch_station_model, session, s, m)
                for s in STATIONS
                for m in MODELS
            ]
            for future in concurrent.futures.as_completed(futures):
                stn, model, data = future.result()
                if data:
                    sounding_matrix[stn][model] = data
                    temp_time_rows_set.update(data.keys())

    time_rows = sorted(list(temp_time_rows_set))
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    trimmed_rows = []
    for row in time_rows:
        try:
            d_part, h_part = map(int, row.split("/"))
            if d_part < now_utc.day:
                continue
            if d_part == now_utc.day and h_part < now_utc.hour:
                continue
            trimmed_rows.append(row)
        except Exception:
            trimmed_rows.append(row)
    if trimmed_rows:
        time_rows = trimmed_rows

    generate_aviation_dashboard(STATIONS, MODELS, sounding_matrix, time_rows)


if __name__ == "__main__":
    run_pipeline()
