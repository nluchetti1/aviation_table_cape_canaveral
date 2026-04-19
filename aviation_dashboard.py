import os
import re
import glob
import json
import requests
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime, timedelta, timezone

warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
TAF_SITES_META = {
    'KXMR': {'lat': 28.4675, 'lon': -80.5594}, # Cape Canaveral Skid Strip
    'KTTS': {'lat': 28.6150, 'lon': -80.6945}, # NASA Shuttle Landing Facility
    'KCOF': {'lat': 28.2349, 'lon': -80.6101}, # Patrick Space Force Base
    'KTIX': {'lat': 28.5141, 'lon': -80.7992}, # Space Coast Regional
    'KMLB': {'lat': 28.1028, 'lon': -80.6453}  # Melbourne Orlando Intl
}
TAF_SITES = list(TAF_SITES_META.keys())
MODELS_VIS = ['GFS', 'NAM', 'RAP', 'HRRR', 'ARW', 'NEST']
MODELS_BUFKIT = ['gfs', 'nam', 'rap', 'hrrr', 'arw', 'nest']
DATA_DIR = "visibility_data"
HISTORY_FILE = "history.json"
os.makedirs(DATA_DIR, exist_ok=True)

model_init_strings = {} 

# --- 2. HELPERS ---
def calculate_total_rh(t_c, td_c):
    rh_water = mpcalc.relative_humidity_from_dewpoint(t_c, td_c).magnitude * 100
    if t_c.magnitude >= 0: return rh_water
    T, Td = t_c.magnitude, td_c.magnitude
    e = 6.112 * np.exp((17.67 * Td) / (Td + 243.5))
    e_s_ice = 6.112 * np.exp((22.46 * T) / (T + 272.62))
    return max(rh_water, (e / e_s_ice) * 100)

def find_nearest_gridpoint(ds, target_lat, target_lon):
    lat_arr, lon_arr = ds['latitude'].values, ds['longitude'].values
    if lon_arr.max() > 180 and target_lon < 0: target_lon += 360
    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        lat_idx = np.abs(lat_arr - target_lat).argmin()
        lon_idx = np.abs(lon_arr - target_lon).argmin()
        return (lat_idx, lon_idx)
    else:
        dist_sq = (lat_arr - target_lat)**2 + (lon_arr - target_lon)**2
        return np.unravel_index(np.argmin(dist_sq, axis=None), dist_sq.shape)

def format_visibility(vis_meters):
    if np.isnan(vis_meters): return "N/A"
    vis_sm = min(vis_meters * 0.000621371, 10.0) 
    if vis_sm >= 1: return str(int(round(vis_sm)))
    elif vis_sm >= 0.75: return "3/4"
    elif vis_sm >= 0.50: return "1/2"
    else: return "1/4"

def colorize_flight_rules(val):
    if pd.isna(val) or str(val).lower() in ['nan', 'n/a', '--']: return ''
    try:
        f = 0.25 if val == "1/4" else 0.5 if val == "1/2" else 0.75 if val == "3/4" else float(val)
        if f > 5: return ''
        elif 3 <= f <= 5: return 'background-color: #458B00; color: white;'
        elif 1 <= f < 3: return 'background-color: #CD3333; color: white;'
        else: return 'background-color: #EE82EE; color: black;'
    except: return ''

def style_ceiling_table(val):
    if val in ["N/A", "--"]: return ''
    try:
        h = int(val)
        if h > 3000: return ''
        elif 1000 <= h <= 3000: return 'background-color: #458B00; color: white;'
        elif 500 <= h < 1000: return 'background-color: #CD3333; color: white;'
        else: return 'background-color: #EE82EE; color: black;'
    except: return ''

def style_llws_table(val):
    if pd.isna(val) or "|" not in str(val): return ''
    try:
        mag = float(str(val).split('|')[0].strip())
        if mag < 20: return ''
        elif 20 <= mag < 30: return 'background-color: #FFC125; color: black;'
        elif 30 <= mag < 40: return 'background-color: #CD5B45; color: white;'
        else: return 'background-color: #7A378B; color: white;'
    except: return ''

def download_file(url, filepath):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20, stream=True)
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(8192): f.write(chunk)
            return True
    except: return False
    return False

def process_bufkit(filepath, model_name, mode='cig'):
    if not os.path.exists(filepath): return pd.Series()
    with open(filepath, 'r') as f: lines = f.readlines()
    selv_ft = 0.0
    for line in lines:
        if "SELV" in line:
            parts = line.split()
            selv_idx = parts.index('SELV')
            val_str = parts[selv_idx + 2] if parts[selv_idx + 1] == "=" else parts[selv_idx + 1]
            selv_ft = float(val_str) * 3.28084
            break
            
    times, profile_starts = [], []
    for i, line in enumerate(lines):
        if "TIME =" in line:
            match = re.search(r'TIME =\s*(\d{6}/\d{4})', line)
            if match:
                dt = datetime.strptime(match.group(1), "%y%m%d/%H%M").replace(tzinfo=timezone.utc)
                times.append(dt)
                if model_name not in model_init_strings: model_init_strings[model_name] = dt.strftime('%m/%d %HZ')
        if "PRES TMPC" in line: profile_starts.append(i)

    results = []
    is_gfs = (model_name.lower() == "gfs")
    for i, start_idx in enumerate(profile_starts):
        end_idx = profile_starts[i+1] if i + 1 < len(profile_starts) else len(lines)
        data_lines = [l.split() for l in lines[start_idx+2 : end_idx] if l.strip() and "STN" not in l]
        if mode == 'cig':
            lowest_ft = np.nan
            for j in range(0, len(data_lines)-1, 2):
                l1, l2 = data_lines[j], data_lines[j+1]
                if len(l1) < 4 or len(l2) < 1: continue
                try:
                    h_agl = ((float(l2[0]) if is_gfs else float(l2[1])) * 3.28084) - selv_ft
                    if 200 <= h_agl <= 38000 and calculate_total_rh(float(l1[1])*units.degC, float(l1[3])*units.degC) >= 95.0:
                        lowest_ft = h_agl; break
                except: continue
            results.append(str(int(round(lowest_ft/100)*100)) if not np.isnan(lowest_ft) else "--")
        else:
            heights, dirs, spds = [], [], []
            for j in range(0, len(data_lines)-1, 2):
                l1, l2 = data_lines[j], data_lines[j+1]
                if len(l1) < 7 or len(l2) < 1: continue
                try:
                    h_agl = ((float(l2[0]) if is_gfs else float(l2[1])) * 3.28084) - selv_ft
                    if h_agl < 2100: heights.append(h_agl); dirs.append(float(l1[5])); spds.append(float(l1[6]))
                except: continue
            max_s, t_h, t_d, t_s = 0.0, 0, 0, 0
            for l in range(len(heights)-1):
                for u in range(l+1, len(heights)):
                    dr_l, dr_u = np.radians(dirs[l]), np.radians(dirs[u])
                    s = np.sqrt(max(0, spds[l]**2 + spds[u]**2 - 2*spds[l]*spds[u]*np.cos(dr_u-dr_l)))
                    if s > max_s: max_s, t_h, t_d, t_s = s, heights[u], dirs[u], spds[u]
            if max_s >= 20: 
                results.append(f"{int(round(max_s))}|WS{int(round(t_h/100.0)):03d}/{int(round(t_d/10.0)*10):03d}{int(round(t_s/5.0)*5):02d}KT")
            else: results.append("--")
    return pd.Series(results, index=times, name=model_name.upper())

# --- 3. MAIN EXECUTION ---
def main():
    now = datetime.now(timezone.utc)
    last_updated_str = now.strftime("%Y-%m-%d %H:%M UTC")
    
    # 1. Download GRIBs (BBOX for Central Florida)
    bbox = "&var_VIS=on&lev_surface=on&subregion=&toplat=30&leftlon=279&rightlon=281&bottomlat=27"
    # ... [Rest of your download logic remains the same as previous rewrite] ...

    # 2. Processing (GRIB & Bufkit)
    # ... [Rest of processing logic remains the same] ...

    # 3. Generate the actual index.html file
    # We pass the processed data and history into a helper function
    generate_dashboard_html(last_updated_str)

def generate_dashboard_html(last_updated_str):
    # This reads history.json and writes the new index.html
    with open(HISTORY_FILE, 'r') as f: history_data = json.load(f)
    history_json = json.dumps(history_data)
    
    default_site = TAF_SITES[0].lower()
    
    def gen_links(param):
        return "&nbsp; ".join([f'<a {"id=\'def\' " if i==0 and param==\'cig\' else ""}onmouseover="setSiteData(this, \'{param}\', \'{s.lower()}\')">{s}</a>' for i, s in enumerate(TAF_SITES)])

    # This is where your WORD document HTML goes, but we use f-strings for the dynamic parts
    html_template = f"""
    <html>
    [... Insert the CSS and HTML from your Word Document here ...]
    <script>
        var historyData = {history_json};
        [... Insert your JS Logic here ...]
    </script>
    <body>
       <h3>Run Time: {last_updated_str}</h3>
       <p>Ceilings: {gen_links('cig')} Vis: {gen_links('vis')} Shear: {gen_links('llws')}</p>
       <div id="table-container"></div>
    </body>
    </html>
    """
    with open("index.html", "w") as f:
        f.write(html_template)

if __name__ == "__main__":
    main()
