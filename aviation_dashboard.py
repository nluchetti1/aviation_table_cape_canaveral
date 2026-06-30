import datetime
import json
import math
import os
import re
import requests
import concurrent.futures
import logging
import pygrib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CACHE_DIR = "./workspace_cache"
HISTORY_FILE = "history.json"
STATIONS = ["kdab", "kxmr", "kmlb", "kfpr", "kpbi"]
MODELS = ["gfs", "rap", "hrrr"]

STN_COORDS = {
    "kxmr": {"lat": 28.468, "lon": -80.556},
    "kdab": {"lat": 29.180, "lon": -81.058},
    "kmlb": {"lat": 28.103, "lon": -80.645},
    "kfpr": {"lat": 27.498, "lon": -80.373},
    "kpbi": {"lat": 26.683, "lon": -80.095}
}

THRESHOLD_MAP = {25: "p25", 50: "p50", 100: "p100", 200: "p200"}
MSG_INDEX_THRESHOLDS = {1: "p25", 2: "p50", 3: "p100", 4: "p200"}

# Bounding box used for the HREF lightning density spatial plots (covers peninsular FL)
FL_DOMAIN = {"lat_min": 24.3, "lat_max": 31.2, "lon_min": -87.7, "lon_max": -79.5}

# Spatial plot PNGs are written here (relative path, served alongside index.html)
MAPS_DIR = "./maps"

# Global cache for static grid indices to maximize ThreadPool performance
_GRID_INDEX_CACHE = {}


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


def prune_stale_maps(current_href_maps, maps_dir=MAPS_DIR):
    """Deletes spatial-map PNGs on disk that aren't part of the current run's href_maps
    (the maps/ folder only ever needs to hold the latest run's images)."""
    if not os.path.exists(maps_dir):
        return

    referenced = set()
    for row_maps in (current_href_maps or {}).values():
        for rel_path in (row_maps or {}).values():
            referenced.add(os.path.basename(rel_path))

    for f in os.listdir(maps_dir):
        if f not in referenced:
            try:
                os.unlink(os.path.join(maps_dir, f))
            except Exception:
                pass


def pressure_to_height_ft(pres_hpa):
    return 145366.45 * (1.0 - (pres_hpa / 1013.25) ** 0.190284)


def extract_lightning_from_file(filepath, lat, lon, stn):
    """
    Extracts 4 lightning strike probability thresholds from SPC HREF GRIB2.
    Uses string representation tokens to bypass version incompatibilities with eccodes.
    """
    global _GRID_INDEX_CACHE
    threshold_results = {"p25": 0, "p50": 0, "p100": 0, "p200": 0}

    try:
        grbs = pygrib.open(filepath)
        
        # Grid Point Index Optimization
        if stn in _GRID_INDEX_CACHE:
            y_idx, x_idx = _GRID_INDEX_CACHE[stn]
        else:
            sample = grbs[1]
            lats, lons = sample.latlons()
            lons_normalized = np.where(lons > 180, lons - 360.0, lons)

            dist = (lats - lat) ** 2 + (lons_normalized - lon) ** 2
            y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)
            _GRID_INDEX_CACHE[stn] = (y_idx, x_idx)
            logging.info(f"Verified grid index for {stn.upper()} at [y={y_idx}, x={x_idx}]")

        grbs.seek(0)
        for msg_idx, grb in enumerate(grbs, start=1):
            raw_val = grb.values[y_idx, x_idx]
            pixel_value = 0.0 if np.isnan(float(raw_val)) else float(raw_val)
            val = int(round(pixel_value))

            msg_str = str(grb).lower()
            
            if "upperlimit=25" in msg_str or "prob > 0.25" in msg_str or "probability=25" in msg_str:
                threshold_results["p25"] = val
            elif "upperlimit=50" in msg_str or "prob > 0.50" in msg_str or "probability=50" in msg_str:
                threshold_results["p50"] = val
            elif "upperlimit=100" in msg_str or "prob > 1.0" in msg_str or "probability=100" in msg_str:
                threshold_results["p100"] = val
            elif "upperlimit=200" in msg_str or "prob > 2.0" in msg_str or "probability=200" in msg_str:
                threshold_results["p200"] = val
            else:
                pos_key = MSG_INDEX_THRESHOLDS.get(msg_idx)
                if pos_key:
                    threshold_results[pos_key] = val

            if val > 0:
                logging.debug(f"  [HIT] {stn.upper()} scored {val}% for {msg_str.split(':')[0]}")

        grbs.close()
    except Exception as e:
        logging.error(f"Pygrib extraction failed for {stn.upper()}: {e}")

    return threshold_results


def _fig_to_png_file(fig, filename):
    os.makedirs(MAPS_DIR, exist_ok=True)
    out_path = os.path.join(MAPS_DIR, filename)
    fig.savefig(out_path, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    # Relative path for use directly as an <img src="..."> in index.html
    return f"maps/{filename}"


def generate_spatial_threshold_maps(filepath, file_prefix):
    """
    Builds a Florida-domain spatial plot (PNG, written to MAPS_DIR) for each of the 4
    HREF lightning exceedance thresholds (p25/p50/p100/p200) from a single GRIB2 file.
    `file_prefix` should uniquely identify the run/forecast-hour, e.g. "20260630_00z_f012".
    Returns a dict like {"p25": "maps/20260630_00z_f012_p25.png", ...} (missing keys if a
    given threshold message wasn't found or plotting failed).
    """
    maps = {}
    try:
        grbs = pygrib.open(filepath)
        sample = grbs[1]
        lats, lons = sample.latlons()
        lons_n = np.where(lons > 180, lons - 360.0, lons)

        domain_mask = (
            (lats >= FL_DOMAIN["lat_min"]) & (lats <= FL_DOMAIN["lat_max"]) &
            (lons_n >= FL_DOMAIN["lon_min"]) & (lons_n <= FL_DOMAIN["lon_max"])
        )
        ys, xs = np.where(domain_mask)
        if len(ys) == 0:
            grbs.close()
            return maps

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        sub_lats = lats[y0:y1 + 1, x0:x1 + 1]
        sub_lons = lons_n[y0:y1 + 1, x0:x1 + 1]

        proj = ccrs.PlateCarree()
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_1_states_provinces_lines",
            scale="50m", facecolor="none"
        )
        counties = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_2_counties",
            scale="10m", facecolor="none"
        )

        grbs.seek(0)
        for msg_idx, grb in enumerate(grbs, start=1):
            msg_str = str(grb).lower()

            if "upperlimit=25" in msg_str or "prob > 0.25" in msg_str or "probability=25" in msg_str:
                thresh_key = "p25"
            elif "upperlimit=50" in msg_str or "prob > 0.50" in msg_str or "probability=50" in msg_str:
                thresh_key = "p50"
            elif "upperlimit=100" in msg_str or "prob > 1.0" in msg_str or "probability=100" in msg_str:
                thresh_key = "p100"
            elif "upperlimit=200" in msg_str or "prob > 2.0" in msg_str or "probability=200" in msg_str:
                thresh_key = "p200"
            else:
                thresh_key = MSG_INDEX_THRESHOLDS.get(msg_idx)

            if not thresh_key or thresh_key in maps:
                continue

            try:
                raw_vals = grb.values[y0:y1 + 1, x0:x1 + 1]
                vals = np.nan_to_num(np.asarray(raw_vals, dtype=float), nan=0.0)
                # Normalize fractional probabilities (0-1) up to percent (0-100)
                if vals.max() <= 1.0:
                    vals = vals * 100.0

                fig = plt.figure(figsize=(4.0, 4.2), dpi=100)
                ax = fig.add_subplot(1, 1, 1, projection=proj)
                ax.set_extent(
                    [FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"], FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]],
                    crs=proj
                )

                # Base map styling
                ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
                ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f5f9", zorder=0)
                ax.add_feature(counties, edgecolor="#cbd5e1", linewidth=0.35, zorder=1)
                ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1e293b", linewidth=0.9, zorder=3)
                ax.add_feature(states_provinces, edgecolor="#475569", linewidth=0.8, zorder=3)
                ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1e293b", linewidth=0.8, zorder=3)

                masked_vals = np.ma.masked_less_equal(vals, 0.0)
                mesh = ax.pcolormesh(
                    sub_lons, sub_lats, masked_vals, cmap="hot_r", vmin=0, vmax=100,
                    shading="auto", transform=proj, zorder=2, alpha=0.85
                )

                # Station markers for orientation
                for stn_id, coords in STN_COORDS.items():
                    ax.plot(
                        coords["lon"], coords["lat"], marker="^", markersize=5,
                        color="#2563eb", markeredgecolor="white", markeredgewidth=0.6,
                        transform=proj, zorder=4
                    )

                gl = ax.gridlines(draw_labels=False, linewidth=0.4, color="#94a3b8", alpha=0.5, linestyle="--")

                ax.set_title(f"HREF ≥ {thresh_key[1:]} Flash Density", fontsize=9, fontweight="bold", color="#1e293b")
                cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
                cbar.set_label("Exceedance Probability (%)", fontsize=7)
                cbar.ax.tick_params(labelsize=6)

                maps[thresh_key] = _fig_to_png_file(fig, f"{file_prefix}_{thresh_key}.png")
            except Exception as plot_err:
                logging.error(f"Spatial plot render failed for {thresh_key}: {plot_err}")

        grbs.close()
    except Exception as e:
        logging.error(f"Spatial map extraction failed for {filepath}: {e}")

    return maps


def fetch_href_spatial_map(session, date_str, cycle, f_hour_int):
    """Downloads the HREF lightning GRIB2 once per forecast hour (domain-wide, not
    station-specific) and renders the Florida spatial threshold maps from it."""
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/spc_post.{date_str}/ltgdensity"
    file_name = f"spc_post.t{cycle}z.hrefld_4hr.f{f_hour_int:03d}.grib2"
    url = f"{base_url}/{file_name}"
    local_path = os.path.join(CACHE_DIR, f"spatial_{file_name}")

    try:
        with session.get(url, timeout=10, stream=True) as r:
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_prefix = f"{date_str}_{cycle}z_f{f_hour_int:03d}"
                maps = generate_spatial_threshold_maps(local_path, file_prefix)

                if os.path.exists(local_path):
                    os.remove(local_path)
                return maps
    except Exception as e:
        logging.debug(f"Spatial map download break for {file_name}: {e}")

    if os.path.exists(local_path):
        try: os.remove(local_path)
        except Exception: pass
    return {}


def fetch_href_lightning_point(session, stn, lat, lon, date_str, cycle, f_hour_int):
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/spc_post.{date_str}/ltgdensity"
    file_name = f"spc_post.t{cycle}z.hrefld_4hr.f{f_hour_int:03d}.grib2"
    url = f"{base_url}/{file_name}"
    local_path = os.path.join(CACHE_DIR, f"{stn}_{file_name}")

    try:
        with session.get(url, timeout=7, stream=True) as r:
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                vals = extract_lightning_from_file(local_path, lat, lon, stn)
                
                if os.path.exists(local_path):
                    os.remove(local_path)
                return stn, vals
    except Exception as e:
        logging.debug(f"Download break for {file_name}: {e}")

    if os.path.exists(local_path):
        try: os.remove(local_path)
        except Exception: pass
    return stn, {"p25": 0, "p50": 0, "p100": 0, "p200": 0}


def fetch_href_lightning(time_keys):
    href_data = {
        stn: {row: {"p25": 0, "p50": 0, "p100": 0, "p200": 0} for row in time_keys}
        for stn in STATIONS
    }

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=3)
    session.mount("https://", adapter)

    active_cycle = None
    active_date_str = None

    # HREF Lightning density files ONLY initialize for 00Z and 12Z cycles
    for days_back in [0, 1]:
        check_date = now_utc - datetime.timedelta(days=days_back)
        date_str = check_date.strftime("%Y%m%d")
        
        for cycle in ["12", "00"]:
            test_url = (
                f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/"
                f"spc_post.{date_str}/ltgdensity/spc_post.t{cycle}z.hrefld_4hr.f004.grib2"
            )
            try:
                if session.head(test_url, timeout=3).status_code == 200:
                    active_cycle = cycle
                    active_date_str = date_str
                    break
            except Exception:
                continue
        if active_cycle:
            break

    if not active_cycle:
        logging.warning("No active 00Z or 12Z SPC HREF lightning directories found on NOMADS.")
        return href_data, {}

    logging.info(f"Targeting Valid NOMADS Initialization Run: {active_date_str} at {active_cycle}Z")

    cycle_init_utc = datetime.datetime.strptime(f"{active_date_str}{active_cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures_map = {}
        map_futures_map = {}
        row_to_fhour = {}
        for row_key in time_keys:
            try:
                d_part, h_part = map(int, row_key.split("/"))
                valid_dt = cycle_init_utc.replace(day=d_part, hour=h_part, minute=0, second=0, microsecond=0)
                
                if valid_dt < cycle_init_utc:
                    valid_dt += datetime.timedelta(days=28)
                    valid_dt = valid_dt.replace(day=d_part)

                f_hour_int = int(round((valid_dt - cycle_init_utc).total_seconds() / 3600))
            except Exception:
                continue

            # Core HREF limit configuration
            if f_hour_int < 1 or f_hour_int > 48:
                continue

            row_to_fhour[row_key] = f_hour_int

            for stn, coords in STN_COORDS.items():
                future = executor.submit(
                    fetch_href_lightning_point,
                    session, stn, coords["lat"], coords["lon"],
                    active_date_str, active_cycle, f_hour_int
                )
                futures_map[future] = (stn, row_key)

            # One spatial-map render per forecast hour (domain-wide, not per station)
            map_future = executor.submit(
                fetch_href_spatial_map, session, active_date_str, active_cycle, f_hour_int
            )
            map_futures_map[map_future] = row_key

        for future in concurrent.futures.as_completed(futures_map):
            stn, row_key = futures_map[future]
            try:
                _, vals = future.result()
                href_data[stn][row_key] = vals
            except Exception:
                pass

        href_maps = {}
        for future in concurrent.futures.as_completed(map_futures_map):
            row_key = map_futures_map[future]
            try:
                href_maps[row_key] = future.result()
            except Exception:
                href_maps[row_key] = {}

    # Streamlined runner-safe Audit Log
    logging.info("=============================================")
    logging.info("         HREF LIGHTNING AUDIT LOG            ")
    logging.info("=============================================")
    total_signals = 0
    for stn in STATIONS:
        stn_hits = 0
        max_p25 = 0
        for r_key, thresh_vals in href_data[stn].items():
            if any(v > 0 for v in thresh_vals.values()):
                stn_hits += 1
                total_signals += 1
                max_p25 = max(max_p25, thresh_vals["p25"])
        
        if stn_hits > 0:
            logging.info(f"  {stn.upper()} -> Processed {stn_hits} intervals with active lightning signals (Max p25: {max_p25}%)")
        else:
            logging.info(f"  {stn.upper()} -> All 48 forecast intervals returned flat 0%")
            
    logging.info(f"Audit Complete: Cleanly tracked {total_signals} total non-zero cell vectors.")
    logging.info(f"Spatial threshold maps rendered for {sum(1 for v in href_maps.values() if v)} of {len(href_maps)} forecast hours.")
    logging.info("=============================================")

    return href_data, href_maps


def parse_time_series_bufkit(bufkit_text):
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
        pres_idx, tmpc_idx, dwpt_idx, sknt_idx, drct_idx = 0, 1, 3, 5, 4
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
                    if "PRES" in header_names: pres_idx = header_names.index("PRES")
                    if "TMPC" in header_names: tmpc_idx = header_names.index("TMPC")
                    if "DWPT" in header_names: dwpt_idx = header_names.index("DWPT")
                    if "SKNT" in header_names: sknt_idx = header_names.index("SKNT")
                    if "DRCT" in header_names: drct_idx = header_names.index("DRCT")
                except ValueError:
                    pass
                continue

            if in_profile:
                if "STID" in cleaned or "STNM" in cleaned:
                    break
                parts = cleaned.split()
                if len(parts) > max(pres_idx, tmpc_idx, dwpt_idx, sknt_idx, drct_idx):
                    try:
                        if not parts[0].replace(".", "", 1).replace("-", "", 1).isdigit():
                            continue
                        pres = float(parts[pres_idx])
                        tmpc = float(parts[tmpc_idx])
                        dwpt = float(parts[dwpt_idx])
                        sknt = float(parts[sknt_idx])
                        try:
                            drct = float(parts[drct_idx])
                        except (ValueError, IndexError):
                            drct = None
                        if 100.0 <= pres <= 1050.0:
                            # Meteorological wind vector components (u: east+, v: north+).
                            # "FROM" direction convention -> components point opposite the heading.
                            if drct is not None and 0.0 <= drct <= 360.0:
                                u_comp = -sknt * math.sin(math.radians(drct))
                                v_comp = -sknt * math.cos(math.radians(drct))
                            else:
                                u_comp, v_comp = None, None
                            profile_layers.append({
                                "pres": pres,
                                "hght": pressure_to_height_ft(pres),
                                "tmpc": tmpc,
                                "dwpt": dwpt,
                                "depr": tmpc - dwpt,
                                "sknt": sknt,
                                "drct": drct,
                                "u": u_comp,
                                "v": v_comp,
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

        def get_wind_component_at_agl(target_agl_ft, sfc_hght):
            """Linearly interpolate u/v wind components to a target height (ft AGL)."""
            for i in range(len(profile_layers) - 1):
                l1, l2 = profile_layers[i], profile_layers[i + 1]
                if l1["u"] is None or l1["v"] is None or l2["u"] is None or l2["v"] is None:
                    continue
                agl1 = l1["hght"] - sfc_hght
                agl2 = l2["hght"] - sfc_hght
                if (agl1 <= target_agl_ft <= agl2) or (agl2 <= target_agl_ft <= agl1):
                    if agl1 == agl2:
                        return l1["u"], l1["v"]
                    fraction = (target_agl_ft - agl1) / (agl2 - agl1)
                    u = l1["u"] + fraction * (l2["u"] - l1["u"])
                    v = l1["v"] + fraction * (l2["v"] - l1["v"])
                    return u, v
            # Fall back to the last available wind-bearing layer
            for layer in reversed(profile_layers):
                if layer["u"] is not None and layer["v"] is not None:
                    return layer["u"], layer["v"]
            return None, None

        def calc_shear_0_6km():
            """0-6 km AGL bulk shear magnitude (kt), using true wind vector components."""
            if not profile_layers:
                return None
            sfc_layer = profile_layers[0]
            if sfc_layer["u"] is None or sfc_layer["v"] is None:
                return None
            sfc_hght = sfc_layer["hght"]
            u_sfc, v_sfc = sfc_layer["u"], sfc_layer["v"]
            target_agl_ft = 6000.0 * 3.280839895  # 6 km converted to feet
            u_6km, v_6km = get_wind_component_at_agl(target_agl_ft, sfc_hght)
            if u_6km is None or v_6km is None:
                return None
            shear_kt = math.hypot(u_6km - u_sfc, v_6km - v_sfc)
            return round(shear_kt, 1)

        cloud_layers = []
        active_cloud = None
        for layer in profile_layers:
            if layer["depr"] <= 2.0:
                if active_cloud is None:
                    active_cloud = {"base": layer["hght"], "top": layer["hght"], "min_temp": layer["tmpc"]}
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
        max_pbl = max(pbl_winds) if pbl_winds else 0.0
        sfc_depr = profile_layers[0]["depr"] if profile_layers else 10.0
        vis = 0.25 if sfc_depr <= 0.5 else (1.0 if sfc_depr <= 1.0 else (3.0 if sfc_depr <= 2.0 else 10.0))
        valid_ceilings = [c for c in cloud_layers if c["base"] >= 100.0]
        ceiling_val = round(valid_ceilings[0]["base"]) if valid_ceilings else 24000.0

        hourly_data[valid_hour_key] = {
            "mom_mean": round(mean_wind, 1),
            "mom_max": round(max_pbl, 1),
            "shear": calc_shear_0_6km(),
            "vis": vis,
            "ceiling": ceiling_val,
            "hght_0c": round(get_height_of_isotherm(0.0), 1),
            "hght_5c": round(get_height_of_isotherm(-5.0), 1),
            "hght_10c": round(get_height_of_isotherm(-10.0), 1),
            "hght_20c": round(get_height_of_isotherm(-20.0), 1),
            "cloud_top": round(max([c["top"] for c in cloud_layers], default=0.0) / 1000.0, 1),
            "cloud_thick": round(max([max(0.0, c["top"] - c["base"]) for c in cloud_layers], default=0.0) / 1000.0, 1),
        }
    return hourly_data


def fetch_station_model(session, stn, model):
    download_id = "xmr" if stn == "kxmr" else stn
    model_prefix = "gfs3" if model == "gfs" else model
    url = f"http://www.meteo.psu.edu/bufkit/data/{model.upper()}/latest/{model_prefix}_{download_id}.buf"
    try:
        response = session.get(url, timeout=15)
        if response.status_code == 200:
            return stn, model, parse_time_series_bufkit(response.text)
    except Exception as e:
        logging.error(f"Error fetching {stn} {model}: {e}")
    return stn, model, {}


def generate_aviation_dashboard(stations, models, current_sounding_matrix, time_rows):
    href_lightning, href_maps = fetch_href_lightning(time_rows)

    history_runs = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                existing = json.load(f)
            # Tolerate the legacy flat-array history.json format from before href_maps_latest existed.
            history_runs = existing.get("runs", []) if isinstance(existing, dict) else existing
        except Exception:
            history_runs = []

    current_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    current_entry = {
        "timestamp": current_timestamp,
        "data": current_sounding_matrix,
        # HREF lightning point/percentage data DOES participate in dprog/dt history.
        "href_lightning": href_lightning,
    }

    if not history_runs or history_runs[0]["timestamp"] != current_timestamp:
        history_runs.insert(0, current_entry)
    history_runs = history_runs[:5]

    # The HREF spatial PNG maps are NOT part of dprog/dt history — they always reflect
    # only the latest run and get fully overwritten (and pruned) each pipeline pass.
    href_maps_latest = {
        "timestamp": current_timestamp,
        "href_maps": href_maps,
    }

    payload = {
        "runs": history_runs,
        "href_maps_latest": href_maps_latest,
    }

    with open(HISTORY_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    prune_stale_maps(href_maps)
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
            if d_part < now_utc.day and now_utc.day - d_part < 25: 
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
