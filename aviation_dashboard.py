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

# Cape Canaveral / KSC launch pads. These are derived from raw model isobaric GRIB2
# (GFS/RAP/HRRR) rather than BUFKIT, since the pads have no dedicated BUFKIT profiles.
LAUNCH_PADS = {
    "lc39a": {"lat": 28.608, "lon": -80.604, "label": "LC-39A (KSC)"},
    "lc39b": {"lat": 28.627, "lon": -80.621, "label": "LC-39B (KSC)"},
    "lc37":  {"lat": 28.532, "lon": -80.565, "label": "LC-37B (CCSFS)"},
    "slc40": {"lat": 28.562, "lon": -80.577, "label": "SLC-40 (CCSFS)"},
    "slc41": {"lat": 28.583, "lon": -80.583, "label": "SLC-41 (CCSFS)"},
    "lc36":  {"lat": 28.470, "lon": -80.538, "label": "LC-36 (CCSFS)"},
}

THRESHOLD_MAP = {25: "p25", 50: "p50", 100: "p100", 200: "p200"}
MSG_INDEX_THRESHOLDS = {1: "p25", 2: "p50", 3: "p100", 4: "p200"}

# ---- RRFS / REFS configuration -------------------------------------------------
# RRFS (deterministic) and REFS (ensemble) are pre-operational until 2026-08-31 12z.
# We pull from the public AWS Open-Data bucket (no auth, HTTP range-request friendly)
# rather than NOMADS, using each file's .idx sidecar to byte-range only the ~21 isobaric
# levels we need instead of downloading the whole ~40MB CONUS file.
#
# The rrfs_public/ tree is the "operationally-representative" set per the AWS registry:
#   rrfs_public/rrfs.YYYYMMDD/CC/rrfs.tCCz.prslev.3km.fFFF.conus.grib2      (deterministic)
#   rrfs_public/refs.YYYYMMDD/CC/ensprod/ ...                              (ensemble products)
RRFS_ENABLED = True          # master switch for the RRFS deterministic pad column
REFS_ENABLED = True          # master switch for the REFS ensemble-average pad column
RRFS_AWS_ROOT = "https://noaa-rrfs-pds.s3.amazonaws.com"
RRFS_CYCLE_HOURS = [0, 6, 12, 18]   # cycles that run to full length
RRFS_MAX_FH = 48             # RRFS/REFS run to 60h; cap at 48 to match the rest of the board
RRFS_LATENCY_H = 4           # approx hours before a cycle's files are complete on AWS
# REFS averages ('avrg') post ~6-hourly. Since the app runs hourly, each run just picks up
# the latest available cycle; the 6-hourly cadence is fine (rows repeat until the next cycle).

# The exact REFS ensemble-mean filename ordering has drifted across the pre-op feed. We probe
# these candidate patterns (formatted with cycle `c` and forecast-hour ints) once per run and
# cache whichever resolves, so every subsequent hour reuses the confirmed pattern.
REFS_FILENAME_CANDIDATES = [
    "refs.t{c}z.mean.f{f2}.conus.grib2",
    "refs.t{c}z.conus.mean.f{f2}.grib2",
    "refs.t{c}z.mean.f{f3}.conus.grib2",
    "refs.t{c}z.conus.prslev.mean.f{f2}.grib2",
    "rrfsce.t{c}z.conus.mean.f{f2}.grib2",
]
_REFS_RESOLVED_PATTERN = None   # set once we confirm a working pattern this run




# Bounding box — zoomed into the Space Coast launch corridor rather than all of FL
FL_DOMAIN = {"lat_min": 25.5, "lat_max": 30.5, "lon_min": -83.5, "lon_max": -79.5}

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


def prune_stale_maps(current_href_maps, blank_basemap_path=None, maps_dir=MAPS_DIR):
    """Deletes spatial-map PNGs on disk that aren't part of the current run's href_maps
    (the maps/ folder only ever needs to hold the latest run's images, plus the
    always-present blank basemap fallback)."""
    if not os.path.exists(maps_dir):
        return

    referenced = set()
    if blank_basemap_path:
        referenced.add(os.path.basename(blank_basemap_path))
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


def generate_blank_basemap():
    """
    Renders a single 'no data available' Florida basemap (coastlines, counties, state
    borders, station markers, no overlay) used as a fallback whenever a given forecast
    hour/threshold has no spatial map yet (download failure, hour outside the HREF
    0-48h window, etc). Overwritten each run; lives at maps/blank_basemap.png.
    """
    try:
        proj = ccrs.PlateCarree()
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_1_states_provinces_lines",
            scale="50m", facecolor="none"
        )
        counties = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_2_counties",
            scale="10m", facecolor="none"
        )

        fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(
            [FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"], FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]],
            crs=proj
        )

        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f5f9", zorder=0)
        ax.add_feature(counties, edgecolor="#cbd5e1", linewidth=0.35, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1e293b", linewidth=0.9, zorder=3)
        ax.add_feature(states_provinces, edgecolor="#475569", linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1e293b", linewidth=0.8, zorder=3)

        for stn_id, coords in STN_COORDS.items():
            ax.plot(
                coords["lon"], coords["lat"], marker="^", markersize=6,
                color="#2563eb", markeredgecolor="white", markeredgewidth=0.8,
                transform=proj, zorder=5
            )
            ax.text(
                coords["lon"] + 0.06, coords["lat"] + 0.05,
                stn_id.upper(), fontsize=6, fontweight="bold", color="#1e3a5f",
                transform=proj, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none")
            )

        ax.gridlines(draw_labels=False, linewidth=0.4, color="#94a3b8", alpha=0.5, linestyle="--")
        ax.set_title("No Active Signal", fontsize=9, fontweight="bold", color="#94a3b8")

        out_path = os.path.join(MAPS_DIR, "blank_basemap.png")
        os.makedirs(MAPS_DIR, exist_ok=True)
        fig.savefig(out_path, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return "maps/blank_basemap.png"
    except Exception as e:
        logging.error(f"Blank basemap generation failed: {e}")
        return None


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

                fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
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

                # Station markers + labels for quick orientation
                for stn_id, coords in STN_COORDS.items():
                    ax.plot(
                        coords["lon"], coords["lat"], marker="^", markersize=6,
                        color="#2563eb", markeredgecolor="white", markeredgewidth=0.8,
                        transform=proj, zorder=5
                    )
                    ax.text(
                        coords["lon"] + 0.06, coords["lat"] + 0.05,
                        stn_id.upper(), fontsize=6, fontweight="bold", color="#1e3a5f",
                        transform=proj, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none")
                    )

                ax.gridlines(draw_labels=False, linewidth=0.4, color="#94a3b8", alpha=0.5, linestyle="--")

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

    # Build the full 1-48h valid-time list directly from the cycle init, independent of
    # the sounding matrix time keys (which only cover what the models happen to provide).
    # This guarantees HREF data always spans the full 48h window including tomorrow's
    # diurnal maximum, not just whatever hours the soundings happened to cover.
    all_href_time_keys = {}  # row_key -> f_hour_int
    for f_hour_int in range(1, 49):
        valid_dt = cycle_init_utc + datetime.timedelta(hours=f_hour_int)
        # Zero-pad the day to exactly match the sounding-matrix key format ("%d/%H"),
        # otherwise "1/06" and "01/06" collide as two separate rows for the same hour.
        row_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
        all_href_time_keys[row_key] = f_hour_int

    # Also include any sounding-matrix time keys that fall inside the HREF window but
    # may use a slightly different day/hour representation (e.g. near month boundaries).
    for row_key in time_keys:
        try:
            d_part, h_part = map(int, row_key.split("/"))
            valid_dt = cycle_init_utc.replace(day=d_part, hour=h_part, minute=0, second=0, microsecond=0)
            if valid_dt < cycle_init_utc:
                valid_dt += datetime.timedelta(days=28)
                valid_dt = valid_dt.replace(day=d_part)
            f_hour_int = int(round((valid_dt - cycle_init_utc).total_seconds() / 3600))
            if 1 <= f_hour_int <= 48:
                norm_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
                all_href_time_keys[norm_key] = f_hour_int
        except Exception:
            continue

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures_map = {}
        map_futures_map = {}
        for row_key, f_hour_int in all_href_time_keys.items():
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


def compute_profile_variables(profile_layers):
    """
    Given a list of profile layers (each a dict with pres/hght/tmpc/dwpt/depr/sknt/u/v),
    compute the full set of aviation + launch variables. Shared by both the BUFKIT
    station path and the raw-GRIB launch-pad path so the math stays identical.
    Returns the per-hour data dict, or None if the profile is unusable.
    """
    if not profile_layers:
        return None
    profile_layers = sorted(profile_layers, key=lambda x: x["pres"], reverse=True)

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
        for layer in reversed(profile_layers):
            if layer["u"] is not None and layer["v"] is not None:
                return layer["u"], layer["v"]
        return None, None

    def calc_shear_0_6km():
        """0-6 km AGL bulk shear magnitude (kt), using true wind vector components."""
        sfc_layer = profile_layers[0]
        if sfc_layer["u"] is None or sfc_layer["v"] is None:
            return None
        sfc_hght = sfc_layer["hght"]
        u_sfc, v_sfc = sfc_layer["u"], sfc_layer["v"]
        target_agl_ft = 6000.0 * 3.280839895  # 6 km converted to feet
        u_6km, v_6km = get_wind_component_at_agl(target_agl_ft, sfc_hght)
        if u_6km is None or v_6km is None:
            return None
        return round(math.hypot(u_6km - u_sfc, v_6km - v_sfc), 1)

    def _layer_rh(layer):
        """Relative humidity (%) for a layer. Uses stored 'rh' if present (raw GRIB paths),
        otherwise derives it from temperature/dewpoint (BUFKIT path) via the Magnus formula."""
        rh = layer.get("rh")
        if rh is not None:
            return rh
        t = layer.get("tmpc")
        td = layer.get("dwpt")
        if t is None or td is None:
            return None
        a, b = 17.625, 243.04
        try:
            gt = (a * t) / (b + t)
            gd = (a * td) / (b + td)
            return 100.0 * math.exp(gd - gt)
        except Exception:
            return None

    # A layer is "in cloud" when RH >= 95%. This is a tighter, more realistic ceiling
    # criterion than dewpoint-depression <= 2C, which was flagging MVFR too aggressively.
    RH_CLOUD_THRESHOLD = 95.0
    cloud_layers = []
    active_cloud = None
    for layer in profile_layers:
        rh = _layer_rh(layer)
        is_cloud = (rh is not None and rh >= RH_CLOUD_THRESHOLD)
        if is_cloud:
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

    return {
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

        result = compute_profile_variables(profile_layers)
        if result is not None:
            hourly_data[valid_hour_key] = result
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


# ---------------------------------------------------------------------------
# Launch-pad soundings derived from raw isobaric GRIB2 (GFS / RAP / HRRR)
# ---------------------------------------------------------------------------

# Isobaric levels to request from the NOMADS GRIB filter, in hPa. GFS carries the
# full mandatory+standard set; RAP/HRRR carry 25 hPa spacing but we request the same
# nominal list and just use whatever comes back.
PAD_LEVELS_HPA = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
                  550, 500, 450, 400, 350, 300, 250, 200, 150, 100]


def _rh_to_dewpoint_c(temp_c, rh_pct):
    """Magnus-formula dewpoint (°C) from temperature (°C) and relative humidity (%)."""
    if rh_pct is None or rh_pct <= 0:
        return temp_c - 30.0  # very dry fallback
    rh = max(1.0, min(100.0, rh_pct))
    a, b = 17.625, 243.04
    gamma = math.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def _nomads_grib_url(model, date_str, cycle, f_hour_int):
    """Build a NOMADS GRIB-filter URL that subsets to isobaric T/RH/HGT/UGRD/VGRD +
    surface pressure over a small Cape Canaveral bounding box (keeps downloads tiny)."""
    lev_params = "".join(f"&lev_{lv}_mb=on" for lv in PAD_LEVELS_HPA)
    var_params = "&var_TMP=on&var_RH=on&var_HGT=on&var_UGRD=on&var_VGRD=on&var_PRES=on"
    region = "&subregion=&leftlon=-81.2&rightlon=-80.0&toplat=29.2&bottomlat=28.0"

    if model == "hrrr":
        # Use the pressure-level HRRR filter (filter_hrrr_2d.pl is SURFACE fields only and
        # cannot serve the wrfprs 3D isobaric file we need for a sounding).
        base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_sub.pl"
        f_name = f"hrrr.t{cycle}z.wrfprsf{f_hour_int:02d}.grib2"
        dir_part = f"&dir=%2Fhrrr.{date_str}%2Fconus"
    elif model == "rap":
        base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl"
        f_name = f"rap.t{cycle}z.awp130pgrbf{f_hour_int:02d}.grib2"
        dir_part = f"&dir=%2Frap.{date_str}"
    else:  # gfs
        base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        f_name = f"gfs.t{cycle}z.pgrb2.0p25.f{f_hour_int:03d}"
        dir_part = f"&dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"

    return f"{base}?file={f_name}{lev_params}{var_params}{region}{dir_part}"


def build_pad_profiles_from_grib(filepath, pad_coords, debug=False):
    """
    Extract a vertical column at each pad's nearest grid cell from a raw isobaric
    GRIB2 file and assemble profile_layers dicts (matching the BUFKIT schema) so the
    shared compute_profile_variables() can run on them.
    Returns {pad_id: profile_layers_list}. When debug=True, logs a summary of every
    distinct (shortName, typeOfLevel) seen and how many isobaric fields matched, so a
    first live run reveals exactly what NOMADS returned vs what the parser expects.
    """
    per_pad_levels = {pid: {} for pid in pad_coords}
    seen_short_types = {}   # (shortName, typeOfLevel) -> count      [debug]
    matched_counts = {"t": 0, "rh": 0, "hgt": 0, "u": 0, "v": 0}   # [debug]
    isobaric_levels_seen = set()                                    # [debug]
    total_msgs = 0                                                  # [debug]
    try:
        grbs = pygrib.open(filepath)
        # Cache lat/lon grid + nearest-cell index per pad from the first message.
        grid_lats, grid_lons = None, None
        pad_ij = {}

        for grb in grbs:
            total_msgs += 1
            try:
                level = grb.level
                short = getattr(grb, "shortName", "")
                type_lvl = getattr(grb, "typeOfLevel", "")
            except Exception:
                continue

            if debug:
                key = (short, type_lvl)
                seen_short_types[key] = seen_short_types.get(key, 0) + 1

            if type_lvl != "isobaricInhPa" or level not in PAD_LEVELS_HPA:
                continue

            if debug:
                isobaric_levels_seen.add(level)

            if grid_lats is None:
                grid_lats, grid_lons = grb.latlons()
                glons = np.where(grid_lons > 180, grid_lons - 360.0, grid_lons)
                for pid, c in pad_coords.items():
                    dist = (grid_lats - c["lat"]) ** 2 + (glons - c["lon"]) ** 2
                    pad_ij[pid] = np.unravel_index(np.argmin(dist), dist.shape)

            vals = grb.values
            field = None
            if short in ("t", "TMP"): field = "t"
            elif short in ("r", "RH"): field = "rh"
            elif short in ("gh", "HGT"): field = "hgt"
            elif short in ("u", "UGRD", "10u"): field = "u"
            elif short in ("v", "VGRD", "10v"): field = "v"
            if field is None:
                continue

            if debug:
                matched_counts[field] += 1

            for pid, (iy, ix) in pad_ij.items():
                per_pad_levels[pid].setdefault(level, {})[field] = float(vals[iy, ix])

        grbs.close()
    except Exception as e:
        logging.error(f"Pad GRIB parse failed for {filepath}: {e}")
        return {}

    if debug:
        logging.info(f"[PAD DEBUG] {os.path.basename(filepath)}: {total_msgs} total GRIB messages")
        logging.info(f"[PAD DEBUG]   distinct (shortName, typeOfLevel) seen: "
                     + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(seen_short_types.items())))
        logging.info(f"[PAD DEBUG]   isobaric levels matched (hPa): {sorted(isobaric_levels_seen, reverse=True)}")
        logging.info(f"[PAD DEBUG]   fields matched to parser: {matched_counts}")
        if sum(matched_counts.values()) == 0:
            logging.warning("[PAD DEBUG]   >>> ZERO fields matched. shortNames above don't match the "
                            "parser's expected set (t/r/gh/u/v). Update the field mapping to match.")

    pad_profiles = {}
    for pid, levels in per_pad_levels.items():
        layers = []
        for pres, f in levels.items():
            if "t" not in f:
                continue
            tmpc = f["t"] - 273.15 if f["t"] > 100 else f["t"]  # K -> C guard
            rh = f.get("rh")
            dwpt = _rh_to_dewpoint_c(tmpc, rh)
            u = f.get("u")
            v = f.get("v")
            sknt = math.hypot(u, v) * 1.943844 if (u is not None and v is not None) else 0.0
            # GRIB geopotential height (gpm) -> feet if present, else barometric fallback
            hght_ft = f["hgt"] * 3.280839895 if "hgt" in f else pressure_to_height_ft(pres)
            layers.append({
                "pres": pres,
                "hght": hght_ft,
                "tmpc": tmpc,
                "dwpt": dwpt,
                "depr": tmpc - dwpt,
                "rh": rh,  # native GRIB RH (%), used directly for RH>=95% cloud detection
                "sknt": sknt,
                "drct": None,
                "u": u * 1.943844 if u is not None else None,  # m/s -> kt
                "v": v * 1.943844 if v is not None else None,
            })
        if layers:
            pad_profiles[pid] = layers
    return pad_profiles


def fetch_pad_model(session, model, date_str, cycle, f_hour_int, row_key, debug=False):
    """Download one raw GRIB2 subset and build pad profiles/variables for a single
    model forecast hour. Returns (row_key, model, {pad_id: variables_dict}).
    When debug=True, logs the request URL, HTTP status, and downloaded byte size."""
    url = _nomads_grib_url(model, date_str, cycle, f_hour_int)
    local_path = os.path.join(CACHE_DIR, f"pad_{model}_{cycle}z_f{f_hour_int:03d}.grib2")
    out = {}
    try:
        with session.get(url, timeout=25, stream=True) as r:
            if debug:
                logging.info(f"[PAD DEBUG] {model.upper()} f{f_hour_int:03d} HTTP {r.status_code}")
                logging.info(f"[PAD DEBUG]   URL: {url}")
            if r.status_code != 200:
                if debug:
                    logging.warning(f"[PAD DEBUG]   >>> Non-200 status. Check the NOMADS filter path/"
                                    f"filename for {model.upper()}. First 300 chars of body:")
                    try:
                        logging.warning(f"[PAD DEBUG]   {r.text[:300]}")
                    except Exception:
                        pass
                return row_key, model, out
            with open(local_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=16384):
                    fh.write(chunk)

        if debug:
            sz = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            logging.info(f"[PAD DEBUG]   downloaded {sz} bytes")

        pad_profiles = build_pad_profiles_from_grib(local_path, LAUNCH_PADS, debug=debug)
        for pid, layers in pad_profiles.items():
            result = compute_profile_variables(layers)
            if result is not None:
                out[pid] = result
        if debug:
            sample_pid = next(iter(pad_profiles), None)
            n_layers = len(pad_profiles[sample_pid]) if sample_pid else 0
            logging.info(f"[PAD DEBUG]   built {len(pad_profiles)} pad profiles, "
                         f"~{n_layers} levels each, {len(out)} produced variable sets")
    except Exception as e:
        logging.debug(f"Pad fetch break {model} f{f_hour_int:03d}: {e}")
        if debug:
            logging.warning(f"[PAD DEBUG]   >>> Exception during {model.upper()} f{f_hour_int:03d}: {e}")
    finally:
        if os.path.exists(local_path):
            try: os.remove(local_path)
            except Exception: pass
    return row_key, model, out


def determine_model_cycle(session, model):
    """Find the most recent available cycle for a given model on NOMADS by probing
    directory listings for the last few candidate cycles."""
    now = datetime.datetime.now(datetime.timezone.utc)
    if model == "gfs":
        cycle_hours, latency_h = [0, 6, 12, 18], 5
    else:  # rap, hrrr are hourly
        cycle_hours, latency_h = list(range(24)), 2

    for back in range(0, 30):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in cycle_hours:
            continue
        if (now - cand).total_seconds() / 3600.0 < latency_h:
            continue
        date_str = cand.strftime("%Y%m%d")
        cycle = f"{cand.hour:02d}"
        # Probe one representative file
        probe_url = _nomads_grib_url(model, date_str, cycle, 1)
        try:
            resp = session.head(probe_url, timeout=8)
            if resp.status_code == 200:
                return date_str, cycle
            resp = session.get(probe_url, timeout=8, stream=True)
            if resp.status_code == 200:
                resp.close()
                return date_str, cycle
        except Exception:
            continue
    return None, None


def fetch_all_pad_soundings():
    """Build the pad sounding matrix {pad_id: {model: {row_key: variables}}} across
    all three models for the 1-48h window (GFS) / 1-48h (RAP/HRRR capped to availability)."""
    pad_matrix = {pid: {m: {} for m in MODELS} for pid in LAUNCH_PADS}

    with requests.Session() as session:
        for model in MODELS:
            date_str, cycle = determine_model_cycle(session, model)
            if not cycle:
                logging.warning(f"No available {model.upper()} cycle found for pad soundings.")
                continue
            cycle_init = datetime.datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)

            # All three are requested hourly across the 48h window. GFS carries hourly
            # native output through f120 on NOMADS (3-hourly only kicks in after f120),
            # so within 48h we get a full hourly series that matches the BUFKIT airports
            # and avoids sparse every-third-row gaps in the merged table.
            max_fh = 48
            step = 1
            f_hours = list(range(step, max_fh + 1, step))

            logging.info(f"Fetching {model.upper()} pad columns: {date_str} {cycle}z, {len(f_hours)} hours")
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                for idx, fh in enumerate(f_hours):
                    valid_dt = cycle_init + datetime.timedelta(hours=fh)
                    row_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
                    # Emit verbose diagnostics only on the first forecast hour of each
                    # model so the log shows exactly what NOMADS returned without spam.
                    dbg = (idx == 0)
                    futures.append(executor.submit(
                        fetch_pad_model, session, model, date_str, cycle, fh, row_key, dbg
                    ))
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        row_key, mdl, pad_vals = fut.result()
                        for pid, vars_dict in pad_vals.items():
                            pad_matrix[pid][mdl][row_key] = vars_dict
                    except Exception:
                        pass

            # Per-model summary: how many forecast hours produced usable pad data.
            sample_pad = next(iter(LAUNCH_PADS))
            hours_ok = len(pad_matrix[sample_pad].get(model, {}))
            if hours_ok == 0:
                logging.warning(f"[PAD DEBUG] {model.upper()} produced ZERO usable pad-hours — "
                                f"see the [PAD DEBUG] lines above for HTTP status / shortName mismatch.")
            else:
                logging.info(f"{model.upper()} pad soundings: {hours_ok}/{len(f_hours)} forecast hours produced data.")

    return pad_matrix


# ---------------------------------------------------------------------------
# RRFS (deterministic) + REFS (ensemble mean) pad columns via AWS Open Data
# ---------------------------------------------------------------------------

def _parse_grib_idx(idx_text):
    """Parse a GRIB2 .idx sidecar into a list of (msg_num, byte_start, shortName, level).
    Each idx line looks like: '1:0:d=2026070100:REFC:entire atmosphere:...'
    We only need the byte offsets so we can range-request specific messages."""
    entries = []
    lines = [ln for ln in idx_text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        parts = ln.split(":")
        if len(parts) < 5:
            continue
        try:
            msg_num = int(parts[0])
            byte_start = int(parts[1])
        except ValueError:
            continue
        short = parts[3].strip()
        level = parts[4].strip()
        # Byte end = start of next message - 1 (or EOF for the last message)
        byte_end = None
        if i + 1 < len(lines):
            nxt = lines[i + 1].split(":")
            try:
                byte_end = int(nxt[1]) - 1
            except (ValueError, IndexError):
                byte_end = None
        entries.append({"msg": msg_num, "start": byte_start, "end": byte_end,
                        "short": short, "level": level})
    return entries


def _range_download_grib(session, grib_url, idx_entries, wanted_levels_hpa, debug=False):
    """Given parsed idx entries, byte-range download only the isobaric TMP/RH/HGT/UGRD/VGRD
    messages at the wanted levels and concatenate them into a local temp GRIB2 file."""
    # Match idx level strings like "500 mb" and variable names.
    wanted_vars = ("TMP", "RH", "HGT", "UGRD", "VGRD")
    wanted_level_strs = {f"{lv} mb" for lv in wanted_levels_hpa}

    ranges = []
    for e in idx_entries:
        if e["short"] not in wanted_vars:
            continue
        if e["level"] not in wanted_level_strs:
            continue
        if e["end"] is None:
            ranges.append((e["start"], ""))  # open-ended to EOF
        else:
            ranges.append((e["start"], e["end"]))

    if not ranges:
        if debug:
            logging.warning("[RRFS DEBUG]   idx parsed but no matching isobaric TMP/RH/HGT/U/V "
                            "messages at wanted levels — check idx var/level naming.")
        return None

    local_path = os.path.join(CACHE_DIR, f"rrfs_col_{abs(hash(grib_url)) % 10_000_000}.grib2")
    try:
        with open(local_path, "wb") as fh:
            # Group into a single multi-range request where possible; fall back to per-range.
            for start, end in ranges:
                hdr = {"Range": f"bytes={start}-{end}"}
                r = session.get(grib_url, headers=hdr, timeout=25)
                if r.status_code in (200, 206):
                    fh.write(r.content)
        if os.path.getsize(local_path) == 0:
            os.remove(local_path)
            return None
        return local_path
    except Exception as e:
        if debug:
            logging.warning(f"[RRFS DEBUG]   range download failed: {e}")
        if os.path.exists(local_path):
            try: os.remove(local_path)
            except Exception: pass
        return None


def _rrfs_determine_cycle(session, model_kind):
    """Find the most recent RRFS/REFS cycle available on AWS by probing .idx existence.
    model_kind: 'rrfs' (deterministic) or 'refs' (ensemble). Returns (date_str, cycle).
    For REFS, also resolves and caches which filename pattern actually carries isobaric
    temperature data (guards against picking a precip-only product like 'avrg')."""
    global _REFS_RESOLVED_PATTERN
    now = datetime.datetime.now(datetime.timezone.utc)
    for back in range(0, 30):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in RRFS_CYCLE_HOURS:
            continue
        if (now - cand).total_seconds() / 3600.0 < RRFS_LATENCY_H:
            continue
        date_str = cand.strftime("%Y%m%d")
        cycle = f"{cand.hour:02d}"

        if model_kind == "refs":
            # Try each candidate filename; accept the first whose idx contains isobaric TMP.
            # Probe several forecast hours (some ensemble products don't emit f01), so we
            # don't reject a valid pattern just because its earliest hour is missing.
            base = f"{RRFS_AWS_ROOT}/rrfs_a/refs.{date_str}/{cycle}/enspost"
            resolved = False
            for pat in REFS_FILENAME_CANDIDATES:
                for probe_fh in (1, 6, 8, 12):
                    fn = pat.format(c=cycle, f2=f"{probe_fh:02d}", f3=f"{probe_fh:03d}")
                    probe = f"{base}/{fn}.idx"
                    try:
                        r = session.get(probe, timeout=10)
                        if r.status_code == 200 and "TMP" in r.text and "mb" in r.text:
                            _REFS_RESOLVED_PATTERN = pat
                            logging.info(f"[RRFS DEBUG] REFS resolved filename pattern: {pat} "
                                         f"(confirmed at f{probe_fh:02d})")
                            resolved = True
                            break
                    except Exception:
                        continue
                if resolved:
                    return date_str, cycle
            continue  # this cycle had no working REFS mean file; try older cycle
        else:
            probe = _rrfs_grib_url(model_kind, date_str, cycle, 1) + ".idx"
            try:
                r = session.get(probe, timeout=10)
                if r.status_code == 200 and len(r.text) > 50:
                    return date_str, cycle
            except Exception:
                continue
    return None, None


def _rrfs_grib_url(model_kind, date_str, cycle, f_hour_int):
    """Build the AWS S3 URL for an RRFS deterministic or REFS ensemble-mean prslev file.
    For REFS, uses the module-cached resolved filename pattern (falls back to the first
    candidate if not yet resolved)."""
    if model_kind == "refs":
        pat = _REFS_RESOLVED_PATTERN or REFS_FILENAME_CANDIDATES[0]
        f_name = pat.format(c=cycle, f2=f"{f_hour_int:02d}", f3=f"{f_hour_int:03d}")
        return f"{RRFS_AWS_ROOT}/rrfs_a/refs.{date_str}/{cycle}/enspost/{f_name}"
    else:
        f_name = f"rrfs.t{cycle}z.prslev.3km.f{f_hour_int:03d}.conus.grib2"
        return f"{RRFS_AWS_ROOT}/rrfs_public/rrfs.{date_str}/{cycle}/{f_name}"


def fetch_rrfs_pad_hour(session, model_kind, date_str, cycle, f_hour_int, row_key, all_coords, debug=False):
    """Fetch one RRFS/REFS forecast hour from AWS via idx byte-range, extract columns at
    every site in all_coords (pads + airports), and compute variables.
    Returns (row_key, model_kind, {site_id: variables})."""
    grib_url = _rrfs_grib_url(model_kind, date_str, cycle, f_hour_int)
    idx_url = grib_url + ".idx"
    out = {}
    local_path = None
    try:
        idx_resp = session.get(idx_url, timeout=15)
        if debug:
            logging.info(f"[RRFS DEBUG] {model_kind.upper()} f{f_hour_int:03d} idx HTTP {idx_resp.status_code}")
            logging.info(f"[RRFS DEBUG]   idx URL: {idx_url}")
        if idx_resp.status_code != 200:
            if debug:
                logging.warning(f"[RRFS DEBUG]   >>> idx not found. Check {model_kind.upper()} "
                                f"AWS path/filename. GRIB URL was: {grib_url}")
            return row_key, model_kind, out

        idx_entries = _parse_grib_idx(idx_resp.text)
        if debug:
            uniq_vars = sorted({e["short"] for e in idx_entries})
            logging.info(f"[RRFS DEBUG]   idx has {len(idx_entries)} messages; distinct vars: {uniq_vars[:25]}")

        local_path = _range_download_grib(session, grib_url, idx_entries, PAD_LEVELS_HPA, debug=debug)
        if not local_path:
            return row_key, model_kind, out

        if debug:
            sz = os.path.getsize(local_path)
            logging.info(f"[RRFS DEBUG]   range-downloaded {sz} bytes of isobaric fields")

        site_profiles = build_pad_profiles_from_grib(local_path, all_coords, debug=debug)
        for sid, layers in site_profiles.items():
            result = compute_profile_variables(layers)
            if result is not None:
                out[sid] = result
    except Exception as e:
        logging.debug(f"RRFS fetch break {model_kind} f{f_hour_int:03d}: {e}")
        if debug:
            logging.warning(f"[RRFS DEBUG]   >>> exception: {e}")
    finally:
        if local_path and os.path.exists(local_path):
            try: os.remove(local_path)
            except Exception: pass
    return row_key, model_kind, out


def fetch_all_rrfs_refs_soundings():
    """Build {site_id: {'rrfs'|'refs': {row_key: variables}}} from AWS for both systems,
    for BOTH the launch pads and the BUFKIT airport points (airports have no BUFKIT RRFS/
    REFS profiles yet, so we point-extract them from the raw grid the same way)."""
    kinds = []
    if RRFS_ENABLED: kinds.append("rrfs")
    if REFS_ENABLED: kinds.append("refs")

    # Combined site set: launch pads + the 5 airport stations, all point-extracted.
    all_coords = {}
    for pid, c in LAUNCH_PADS.items():
        all_coords[pid] = {"lat": c["lat"], "lon": c["lon"]}
    for sid, c in STN_COORDS.items():
        all_coords[sid] = {"lat": c["lat"], "lon": c["lon"]}

    if not kinds:
        return {sid: {} for sid in all_coords}

    matrix = {sid: {k: {} for k in kinds} for sid in all_coords}

    with requests.Session() as session:
        for kind in kinds:
            date_str, cycle = _rrfs_determine_cycle(session, kind)
            if not cycle:
                logging.warning(f"No available {kind.upper()} cycle found on AWS (pre-op feed may be down).")
                continue
            cycle_init = datetime.datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)
            # RRFS deterministic and REFS avrg both provide hourly forecast output, so we
            # request every hour 1-48. Any hour that happens to be missing simply 404s on
            # the .idx probe and is skipped, so the exact availability never has to be hardcoded.
            f_hours = list(range(1, RRFS_MAX_FH + 1))

            logging.info(f"Fetching {kind.upper()} columns from AWS ({len(all_coords)} sites): {date_str} {cycle}z, {len(f_hours)} hours")
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for idx, fh in enumerate(f_hours):
                    valid_dt = cycle_init + datetime.timedelta(hours=fh)
                    row_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
                    dbg = (idx == 0)  # verbose only on first hour
                    futures.append(executor.submit(
                        fetch_rrfs_pad_hour, session, kind, date_str, cycle, fh, row_key, all_coords, dbg
                    ))
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        row_key, mk, site_vals = fut.result()
                        for sid, vd in site_vals.items():
                            matrix[sid][mk][row_key] = vd
                    except Exception:
                        pass

            sample = next(iter(all_coords))
            hours_ok = len(matrix[sample].get(kind, {}))
            if hours_ok == 0:
                logging.warning(f"[RRFS DEBUG] {kind.upper()} produced ZERO usable site-hours — "
                                f"see [RRFS DEBUG] lines above for the idx/URL mismatch.")
            else:
                logging.info(f"{kind.upper()} soundings: {hours_ok}/{len(f_hours)} forecast hours produced data.")

    return matrix


def generate_aviation_dashboard(stations, models, current_sounding_matrix, time_rows, pad_matrix=None):
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

    # Merge launch-pad (raw-GRIB) soundings into the same station-keyed data block so the
    # frontend treats them identically to the BUFKIT stations (just extra dropdown entries).
    combined_data = dict(current_sounding_matrix)
    if pad_matrix:
        for pid, model_data in pad_matrix.items():
            combined_data[pid] = model_data

    current_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    current_entry = {
        "timestamp": current_timestamp,
        "data": combined_data,
        # HREF lightning point/percentage data DOES participate in dprog/dt history.
        "href_lightning": href_lightning,
    }

    if not history_runs or history_runs[0]["timestamp"] != current_timestamp:
        history_runs.insert(0, current_entry)
    history_runs = history_runs[:5]

    # The HREF spatial PNG maps are NOT part of dprog/dt history — they always reflect
    # only the latest run and get fully overwritten (and pruned) each pipeline pass.
    blank_basemap_path = generate_blank_basemap()
    href_maps_latest = {
        "timestamp": current_timestamp,
        "href_maps": href_maps,
        "blank_map": blank_basemap_path,
    }

    payload = {
        "runs": history_runs,
        "href_maps_latest": href_maps_latest,
    }

    with open(HISTORY_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    prune_stale_maps(href_maps, blank_basemap_path)
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

    # Fetch launch-pad soundings from raw GRIB2 (additive; independent of BUFKIT stations).
    try:
        pad_matrix = fetch_all_pad_soundings()
        pad_hours = sum(len(m.get("hrrr", {})) for m in pad_matrix.values())
        logging.info(f"Launch-pad soundings assembled ({pad_hours} HRRR pad-hours across {len(pad_matrix)} pads).")
    except Exception as e:
        logging.error(f"Launch-pad sounding fetch failed, continuing without pads: {e}")
        pad_matrix = None

    # Fetch RRFS (deterministic) + REFS (ensemble avg) columns from AWS and merge them in.
    # These are point-extracted for BOTH the launch pads (into pad_matrix) and the BUFKIT
    # airport stations (into sounding_matrix), since airports have no BUFKIT RRFS/REFS yet.
    if RRFS_ENABLED or REFS_ENABLED:
        try:
            rrfs_matrix = fetch_all_rrfs_refs_soundings()
            if pad_matrix is None:
                pad_matrix = {pid: {} for pid in LAUNCH_PADS}
            for sid, kinds in rrfs_matrix.items():
                is_pad = sid in LAUNCH_PADS
                target = pad_matrix if is_pad else sounding_matrix
                if not is_pad and sid not in target:
                    target[sid] = {}
                target.setdefault(sid, {})
                for kind, rows in kinds.items():
                    if rows:
                        target[sid][kind] = rows
            r_hours = sum(len(k.get("rrfs", {})) for k in rrfs_matrix.values())
            e_hours = sum(len(k.get("refs", {})) for k in rrfs_matrix.values())
            logging.info(f"RRFS/REFS columns merged (RRFS {r_hours}, REFS {e_hours} site-hours across pads + airports).")
        except Exception as e:
            logging.error(f"RRFS/REFS fetch failed, continuing without them: {e}")

    generate_aviation_dashboard(STATIONS, MODELS, sounding_matrix, time_rows, pad_matrix=pad_matrix)


if __name__ == "__main__":
    run_pipeline()
