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
        else: # LLWS
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

# --- 3. MAIN LOGIC ---
def main():
    now = datetime.now(timezone.utc)
    last_updated_str = now.strftime("%Y-%m-%d %H:%M UTC")
    
    # Cycles
    cyc_6hr = 18 if now.hour < 4 else 0 if now.hour < 10 else 6 if now.hour < 16 else 12 if now.hour < 22 else 18
    cyc_6_d = (now - timedelta(days=1)).strftime("%Y%m%d") if now.hour < 4 else now.strftime("%Y%m%d")
    hrly_time = now - timedelta(hours=2)
    cyc_h_s, cyc_h_d = f"{hrly_time.hour:02d}", hrly_time.strftime("%Y%m%d")

    # BBOX Adjusted for Florida
    bbox = "&var_VIS=on&lev_surface=on&subregion=&toplat=30&leftlon=279&rightlon=281&bottomlat=27"
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/"
    nomads_scripts = {'GFS':'filter_gfs_0p25_1hr.pl','NAM':'filter_nam.pl','RAP':'filter_rap.pl','HRRR':'filter_hrrr_2d.pl','ARW':'filter_hiresconus.pl','NEST':'filter_nam_conusnest.pl'}

    for model, script in nomads_scripts.items():
        cyc_s, cyc_d = (cyc_h_s, cyc_h_d) if model in ['HRRR', 'RAP'] else (f"{cyc_6hr:02d}", cyc_6_d)
        if model == 'GFS': dir_path, file_tpl = f'gfs.{cyc_d}/{cyc_s}/atmos', f'gfs.t{cyc_s}z.pgrb2.0p25.f{{hr}}'
        elif model == 'NAM': dir_path, file_tpl = f'nam.{cyc_d}', f'nam.t{cyc_s}z.awphys{{hr}}.tm00.grib2'
        elif model == 'RAP': dir_path, file_tpl = f'rap.{cyc_d}', f'rap.t{cyc_s}z.awp130pgrbf{{hr}}.grib2'
        elif model == 'HRRR': dir_path, file_tpl = f'hrrr.{cyc_d}/conus', f'hrrr.t{cyc_s}z.wrfsfcf{{hr}}.grib2'
        elif model == 'ARW': dir_path, file_tpl = f'hiresw.{cyc_d}', f'hiresw.t{cyc_s}z.arw_5km.f{{hr}}.conus.grib2'
        elif model == 'NEST': dir_path, file_tpl = f'nam.{cyc_d}', f'nam.t{cyc_s}z.conusnest.hiresf{{hr}}.tm00.grib2'

        for hr in range(1, 24): 
            hr_str = f"{hr:03d}" if model == 'GFS' else f"{hr:02d}"
            url = f"{base_url}{script}?dir=%2F{dir_path.replace('/', '%2F')}&file={file_tpl.format(hr=hr_str)}{bbox}"
            download_file(url, os.path.join(DATA_DIR, f"{model.lower()}.f{hr_str}.grib2"))

    buf_urls = {'nam': "http://www.meteo.psu.edu/bufkit/data/latest/nam_{site}.buf", 'gfs': "http://www.meteo.psu.edu/bufkit/data/GFS/latest/gfs3_{site}.buf", 'rap': "http://www.meteo.psu.edu/bufkit/data/RAP/latest/rap_{site}.buf", 'hrrr': "https://www.meteo.psu.edu/bufkit/data/HRRR/latest/hrrr_{site}.buf", 'nest': "https://www.meteo.psu.edu/bufkit/data/NAMNEST/latest/namnest_{site}.buf", 'arw': "https://www.meteo.psu.edu/bufkit/data/HIRESW/latest/hiresw_{site}.buf"}
    for s in TAF_SITES:
        for m, u in buf_urls.items(): download_file(u.format(site=s.lower()), os.path.join(DATA_DIR, f"{m}_{s.lower()}.buf"))

    v_dfs, c_dfs, l_dfs = {s: pd.DataFrame() for s in TAF_SITES}, {s: pd.DataFrame() for s in TAF_SITES}, {s: pd.DataFrame() for s in TAF_SITES}
    for m in MODELS_VIS:
        files = sorted(glob.glob(os.path.join(DATA_DIR, f"{m.lower()}*.grib2")))
        if not files: continue
        try:
            ds = xr.open_mfdataset(files, engine='cfgrib', combine='nested', concat_dim='valid_time', coords="minimal", compat="override")
            model_init_strings[m.lower()] = pd.to_datetime(np.atleast_1d(ds['time'].values)[0]).strftime('%m/%d %HZ')
            for s, co in TAF_SITES_META.items():
                idx = find_nearest_gridpoint(ds, co['lat'], co['lon'])
                v_dfs[s] = v_dfs[s].join(pd.DataFrame({m: [format_visibility(v) for v in ds['vis'].values[:, idx[0], idx[1]]]}, index=pd.to_datetime(ds.valid_time.values)), how='outer')
        except: pass

    for s in TAF_SITES:
        for m in MODELS_BUFKIT:
            path = os.path.join(DATA_DIR, f"{m}_{s.lower()}.buf")
            c_dfs[s] = c_dfs[s].join(process_bufkit(path, m, 'cig'), how='outer')
            l_dfs[s] = l_dfs[s].join(process_bufkit(path, m, 'llws'), how='outer')

    cur_ts = pd.Timestamp.utcnow().tz_localize(None).replace(minute=0, second=0, microsecond=0)
    html_tbls = {}
    def to_st_html(df, pt):
        if df.empty: return "<p>N/A</p>"
        d = df.copy(); d.index = pd.to_datetime(d.index).tz_localize(None)
        d = d[~d.index.duplicated()].sort_index(); d = d[d.index >= cur_ts].head(24)
        d.columns = [f"{c} [{model_init_strings.get(c.lower(), '??')}]" for c in d.columns]
        d.index = d.index.strftime('%d/%H'); d.index.name = "Time (UTC)"
        st = d.style.map(colorize_flight_rules) if pt=='vis' else d.style.map(style_ceiling_table) if pt=='cig' else d.style.format(lambda v: f"{str(v).split('|')[0]}kt | {str(v).split('|')[-1]}" if '|' in str(v) else v).map(style_llws_table)
        return st.to_html(escape=False)

    for s in TAF_SITES:
        ss = s.lower()
        html_tbls[f"vis_{ss}"] = to_st_html(v_dfs[s], 'vis')
        html_tbls[f"cig_{ss}"] = to_st_html(c_dfs[s], 'cig')
        html_tbls[f"llws_{ss}"] = to_st_html(l_dfs[s], 'llws')

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f: full_history = json.load(f)
    else: full_history = []
    full_history.insert(0, {"timestamp": last_updated_str, "tables": html_tbls})
    full_history = full_history[:5]
    with open(HISTORY_FILE, 'w') as f: json.dump(full_history, f)

    # Final HTML Factory
    history_json = json.dumps(full_history)
    default_site = TAF_SITES[0].lower()
    
    def gen_links(param):
        links = []
        for i, s in enumerate(TAF_SITES):
            id_tag = 'id="def" ' if (i == 0 and param == "cig") else ""
            links.append(f'<a {id_tag}onmouseover="setSiteData(this, \'{param}\', \'{s.lower()}\')">{s}</a>')
        return "&nbsp; ".join(links)

    dashboard_html = f"""
    <html><head><style>
    body {{ font-family: sans-serif; margin: 8px; background-color: #ffffff; color: #000000; transition: background-color 0.3s, color 0.3s; }}
    a {{ color: #0000ee; text-decoration: none; padding: 3px 6px; border-radius: 4px; cursor: pointer; }}
    .active-link {{ background-color: #007acc !important; color: #ffffff !important; font-weight: bold; }}
    .main-container {{ display: flex; justify-content: center; gap: 30px; margin-top: 40px; }}
    .vertical-run-controls {{ display: flex; flex-direction: column; gap: 10px; background-color: #f0f0f0; padding: 15px; border-radius: 8px; border: 1px solid #ccc; min-width: 120px; }}
    table {{ border-collapse: collapse; margin: 0 auto; background-color: white; }}
    th, td {{ border: 1px solid #999; padding: 4px 8px; text-align: center; font-size: 14px; min-width: 70px; }}
    th {{ background-color: #6495ED; color: white; }}
    .legend-container {{ background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 8px; padding: 15px; width: 320px; font-size: 13px; display: flex; flex-direction: column; }}
    .legend-grid {{ display: flex; justify-content: space-between; text-align: center; margin-bottom: 10px; }}
    .legend-item {{ margin-bottom: 8px; padding: 6px; border-radius: 4px; font-weight: bold; }}
    body.dark-mode {{ background-color: #1e1e1e; color: #e0e0e0; }}
    body.dark-mode td, body.dark-mode th {{ border: 1px solid #555; background-color: #444; color: white; }}
    </style>
    <script>
    var historyData = {history_json};
    var currentRunIdx = 0; var cParam = 'cig'; var cSite = '{default_site}';
    function setRun(idx) {{ currentRunIdx = parseInt(idx); update(); }}
    function setSiteData(el, p, s) {{
        cParam = p; cSite = s;
        var links = document.getElementsByTagName('a');
        for(var i=0; i<links.length; i++) links[i].classList.remove('active-link');
        if(el) el.classList.add('active-link');
        update();
    }}
    function update() {{
        var run = historyData[currentRunIdx];
        document.getElementById('table-container').innerHTML = run.tables[cParam + '_' + cSite] || "<p>N/A</p>";
        document.getElementById('ts').innerText = "Run Time: " + run.timestamp;
    }}
    function toggleTheme() {{ document.body.classList.toggle('dark-mode'); }}
    window.onload = function() {{ setSiteData(document.getElementById('def'), 'cig', '{default_site}'); }};
    </script></head><body>
    <button style="position: absolute; top: 15px; right: 15px;" onclick="toggleTheme()">Toggle Dark Mode</button>
    <div class="main-container"><div style="text-align: center;">
    <h3 id="ts">Run Time: {last_updated_str}</h3>
    <p>Ceilings: {gen_links('cig')} &nbsp;&nbsp;&nbsp; Vis: {gen_links('vis')} &nbsp;&nbsp;&nbsp; Shear: {gen_links('llws')}</p>
    <div style="display: flex; gap: 20px; align-items: flex-start;">
        <div class="vertical-run-controls">
            <b>Model Run</b>
            <label><input type="radio" name="r" onclick="setRun(0)" checked> Current Run</label>
            <label><input type="radio" name="r" onclick="setRun(1)"> Run -1</label>
            <label><input type="radio" name="r" onclick="setRun(2)"> Run -2</label>
            <label><input type="radio" name="r" onclick="setRun(3)"> Run -3</label>
            <label><input type="radio" name="r" onclick="setRun(4)"> Run -4</label>
        </div>
        <div id="table-container" style="min-width: 600px; overflow-x: auto;"></div>
        <div class="legend-container">
            <h3>Legend</h3><hr>
            <div class="legend-grid">
                <div><h4>Flight Cat</h4>
                    <div class="legend-item" style="background-color: #458B00; color: white;">MVFR</div>
                    <div class="legend-item" style="background-color: #CD3333; color: white;">IFR</div>
                    <div class="legend-item" style="background-color: #EE82EE; color: black;">LIFR</div>
                </div>
                <div><h4>LLWS</h4>
                    <div class="legend-item" style="background-color: #FFC125; color: black;">Marginal</div>
                    <div class="legend-item" style="background-color: #CD5B45; color: white;">Moderate</div>
                    <div class="legend-item" style="background-color: #7A378B; color: white;">High</div>
                </div>
            </div>
            <p style="font-size: 11px; font-style: italic;">* VFR (>3000ft / >5sm) is uncolored.</p>
        </div>
    </div></div></div></body></html>
    """
    with open("index.html", "w") as f: f.write(dashboard_html)
    print("Update Complete.")

if __name__ == "__main__":
    main()
