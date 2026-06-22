import datetime
import json
import os
import re
import requests


def purge_workspace(cache_dir="./workspace_cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    else:
        for f in os.listdir(cache_dir):
            try:
                os.unlink(os.path.join(cache_dir, f))
            except:
                pass
    return cache_dir


def pressure_to_height_ft(pres_hpa):
    """Converts barometric pressure to standard atmosphere height in feet."""
    return 145366.45 * (1.0 - (pres_hpa / 1013.25) ** 0.190284)


def parse_time_series_bufkit(bufkit_text):
    """
    Parses vertical profile sounding chunks out of a raw Bufkit file.
    Tracks header placements across lines to support varying GFS/RAP/HRRR schemas.
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
        dd = int(date_part[4:6])
        hh = int(time_part[0:2])
        valid_hour_key = f"{dd:02d}/{hh:02d}"
        
        lines = block.splitlines()
        profile_layers = []
        in_profile = False
        
        # Safe baseline structural index guesses
        pres_idx, tmpc_idx, dwpt_idx, sknt_idx = 0, 1, 3, 5
        header_names = []
        
        for line in lines:
            cleaned = line.strip()
            
            # Catch headers even if split across multiple lines
            if "PRES" in cleaned or "TMPC" in cleaned or "SKNT" in cleaned:
                in_profile = True
                header_names.extend(cleaned.split())
                try:
                    if "PRES" in header_names: pres_idx = header_names.index("PRES")
                    if "TMPC" in header_names: tmpc_idx = header_names.index("TMPC")
                    if "DWPT" in header_names: dwpt_idx = header_names.index("DWPT")
                    if "SKNT" in header_names: sknt_idx = header_names.index("SKNT")
                except ValueError:
                    pass
                continue
                
            if in_profile:
                # Stop parsing if we hit station metadata at the bottom of the profile stack
                if "STID" in cleaned or "STNM" in cleaned or not cleaned:
                    break
                
                parts = cleaned.split()
                if len(parts) > max(pres_idx, tmpc_idx, dwpt_idx, sknt_idx):
                    try:
                        # Skip description/header lines that aren't numeric profile data
                        if not parts[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                            continue
                            
                        pres = float(parts[pres_idx])
                        tmpc = float(parts[tmpc_idx])
                        dwpt = float(parts[dwpt_idx])
                        sknt = float(parts[sknt_idx])
                        
                        if 100.0 <= pres <= 1050.0:
                            hght_ft = pressure_to_height_ft(pres)
                            profile_layers.append({
                                "pres": pres,
                                "hght": hght_ft,
                                "tmpc": tmpc,
                                "dwpt": dwpt,
                                "depr": tmpc - dwpt,
                                "sknt": sknt
                            })
                    except ValueError:
                        continue
                        
        if not profile_layers:
            continue
            
        # Organize profile surface upward (Highest pressure down to lowest pressure)
        profile_layers = sorted(profile_layers, key=lambda x: x["pres"], reverse=True)
        
        # Calculate dynamic isotherm heights via linear interpolation (converted to kft)
        def get_height_of_isotherm(target_temp):
            for i in range(len(profile_layers) - 1):
                t1, t2 = profile_layers[i]["tmpc"], profile_layers[i+1]["tmpc"]
                h1, h2 = profile_layers[i]["hght"], profile_layers[i+1]["hght"]
                if (t1 >= target_temp >= t2) or (t1 <= target_temp <= t2):
                    if t1 == t2:
                        return h1 / 1000.0
                    fraction = (target_temp - t1) / (t2 - t1)
                    return (h1 + fraction * (h2 - h1)) / 1000.0
            return profile_layers[-1]["hght"] / 1000.0

        hght_0c = get_height_of_isotherm(0.0)
        hght_5c = get_height_of_isotherm(-5.0)
        hght_10c = get_height_of_isotherm(-10.0)
        hght_20c = get_height_of_isotherm(-20.0)
        
        # Isolate cloud layers using strict Dewpoint Depression <= 2.0C (RH >= 95% proxy)
        cloud_layers = []
        active_cloud = None
        
        for layer in profile_layers:
            if layer["depr"] <= 2.0:
                if active_cloud is None:
                    active_cloud = {"base": layer["hght"], "top": layer["hght"], "min_temp": layer["tmpc"]}
                else:
                    active_cloud["top"] = layer["hght"]
                    active_cloud["min_temp"] = min(active_cloud["min_temp"], layer["tmpc"])
            else:
                if active_cloud is not None:
                    cloud_layers.append(active_cloud)
                    active_cloud = None
        if active_cloud is not None:
            cloud_layers.append(active_cloud)
            
        highest_cloud_top = 0.0
        max_layer_thickness = 0.0
        lowest_ceiling = 24000.0  # Clear air standard default
        
        if cloud_layers:
            lowest_ceiling = max(0.0, cloud_layers[0]["base"])
            highest_cloud_top = max(0.0, max([c["top"] for c in cloud_layers])) / 1000.0
            max_layer_thickness = max([max(0.0, c["top"] - c["base"]) for c in cloud_layers]) / 1000.0

        # Boundary Layer Kinematics (Sfc to 850hPa core layer)
        pbl_winds = [l["sknt"] for l in profile_layers if l["pres"] >= 850.0]
        if not pbl_winds:
            pbl_winds = [0.0]
            
        mean_wind = sum(pbl_winds) / len(pbl_winds)
        max_wind = max(pbl_winds)
        
        # Clean visibility heuristic based on true surface moisture trends
        sfc_depr = profile_layers[0]["depr"] if profile_layers else 10.0
        if sfc_depr <= 0.5:
            derived_vis = 0.25  # Dense Fog
        elif sfc_depr <= 1.0:
            derived_vis = 1.0   # Fog
        elif sfc_depr <= 2.0:
            derived_vis = 3.0   # Mist
        else:
            derived_vis = 10.0  # Unrestricted VFR

        hourly_data[valid_hour_key] = {
            "mom_mean": round(mean_wind, 1),
            "mom_max": round(max_wind, 1),
            "shear": "WS020/25022KT" if max_wind > 35 else None,
            "vis": round(derived_vis, 2),
            "ceiling": round(lowest_ceiling),
            "hght_0c": round(hght_0c, 1),
            "hght_5c": round(hght_5c, 1),
            "hght_10c": round(hght_10c, 1),
            "hght_20c": round(hght_20c, 1),
            "cloud_top": round(highest_cloud_top, 1),
            "cloud_thick": round(max_layer_thickness, 1)
        }
        
    return hourly_data


def query_sounding_stations():
    print("=== Sourcing Live Sounding Stations & Evaluating Data ===")
    models = ["gfs", "rap", "hrrr"]
    frontend_stations = ["kdab", "kxmr", "kmlb", "kfpr", "kpbi"]
    sounding_data = {stn: {mdl: {} for mdl in models} for stn in frontend_stations}

    for stn in frontend_stations:
        download_id = "xmr" if stn == "kxmr" else stn
        for model in models:
            model_prefix = "gfs3" if model == "gfs" else model
            url = f"http://www.meteo.psu.edu/bufkit/data/{model.upper()}/latest/{model_prefix}_{download_id}.buf"
            
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    parsed_series = parse_time_series_bufkit(response.text)
                    if parsed_series:
                        sounding_data[stn][model] = parsed_series
                        print(f" Successfully loaded true profile data for {stn.upper()} [{model.upper()}]")
                        continue
                print(f" Warning: Empty data or status {response.status_code} for {stn.upper()} [{model.upper()}]")
            except Exception as e:
                print(f" Connection Error accessing URL for {stn.upper()} [{model.upper()}]: {str(e)}")
                
            sounding_data[stn][model] = {}
                        
    return frontend_stations, models, sounding_data


def generate_aviation_dashboard(stations, models, current_sounding_matrix):
    history_file = "history.json"
    history_runs = []
    
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history_runs = json.load(f)
        except:
            history_runs = []
            
    current_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    current_entry = {"timestamp": current_timestamp, "data": current_sounding_matrix}
    
    if not history_runs or history_runs[0]["timestamp"] != current_timestamp:
        history_runs.insert(0, current_entry)
    history_runs = history_runs[:5]
    
    with open(history_file, "w") as f:
        json.dump(history_runs, f, indent=2)

    time_rows_set = set()
    for stn in stations:
        for model in models:
            if stn in current_sounding_matrix and model in current_sounding_matrix[stn]:
                time_rows_set.update(current_sounding_matrix[stn][model].keys())
                
    time_rows = sorted(list(time_rows_set))
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    trimmed_rows = []
    for row in time_rows:
        try:
            d_part, h_part = map(int, row.split("/"))
            if d_part < now_utc.day: continue
            if d_part == now_utc.day and h_part < now_utc.hour: continue
            trimmed_rows.append(row)
        except:
            trimmed_rows.append(row)
    if trimmed_rows:
        time_rows = trimmed_rows

    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Aviation & Launch Commit Weather Grid</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 15px; font-size: 0.85rem; }
        
        .control-bar { display: flex; align-items: center; justify-content: space-between; background: #e2e8f0; padding: 10px; border-radius: 4px; margin-bottom: 12px; gap: 15px; }
        .nav-links { line-height: 1.8; }
        .nav-links span { font-weight: bold; margin-right: 5px; color: #334155; }
        .nav-links a { margin: 0 3px; text-decoration: none; color: #2563eb; padding: 2px 5px; border-radius: 3px; transition: all 0.1s ease; }
        .nav-links a.active { color: #ffffff; background: #2563eb; font-weight: bold; }
        
        .station-selector { display: flex; align-items: center; gap: 6px; font-size: 0.9rem; font-weight: bold; background: white; padding: 4px 8px; border-radius: 4px; border: 1px solid #cbd5e1; }
        .station-selector select { font-size: 0.85rem; font-weight: bold; color: #1e3a8a; padding: 2px 4px; cursor: pointer; border: 1px solid #cbd5e1; border-radius: 3px; }
        
        .dprog-bar { background: #f1f5f9; border: 1px solid #cbd5e1; padding: 8px; border-radius: 4px; margin-bottom: 12px; display: flex; align-items: center; gap: 10px; }
        .dprog-title { font-weight: bold; color: #475569; font-size: 0.82rem; text-transform: uppercase; }
        
        .main-layout { display: flex; gap: 20px; align-items: flex-start; position: relative; }
        .table-container { max-height: 72vh; overflow-y: auto; border: 1px solid #cbd5e1; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        table { border-collapse: collapse; background: white; width: 580px; table-layout: fixed; }
        th { background: #2563eb; color: white; padding: 8px; border: 1px solid #cbd5e1; font-size: 0.8rem; text-align: center; position: sticky; top: 0; z-index: 10; }
        td { border: 1px solid #cbd5e1; padding: 6px; text-align: center; font-family: monospace; font-size: 0.85rem; white-space: nowrap; position: relative; }
        .time-col { background: #3b82f6; color: white; font-weight: bold; width: 95px; position: sticky; left: 0; }
        
        .side-panel { background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px; padding: 15px; width: 360px; }
        .legend-title { font-weight: bold; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; margin-bottom: 10px; text-align: center; font-size: 0.95rem; color: #1e293b; }
        .legend-item { display: flex; justify-content: space-between; margin-bottom: 4px; padding: 5px; border-radius: 3px; font-size: 0.8rem; }
        
        #hover-popup-card {
            position: absolute;
            z-index: 500;
            background: #ffffff;
            color: #1e293b;
            border: 2px solid #3b82f6;
            border-radius: 6px;
            padding: 12px;
            width: 260px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            display: none;
            pointer-events: none;
            font-size: 0.8rem;
            line-height: 1.4;
        }
        .popup-header { font-weight: bold; border-bottom: 1px solid #e2e8f0; margin-bottom: 6px; padding-bottom: 4px; color: #2563eb; text-transform: uppercase; }
        .status-banner { font-weight: bold; font-size: 0.9rem; color: #1e3a8a; margin-bottom: 12px; text-transform: uppercase; background: #dbeafe; padding: 6px; border-radius: 4px; text-align: center; }
    </style>
</head>
<body>

    <div class="control-bar">
        <div class="nav-links">
            <span>Aviation:</span>
            <a href="#" class="active" data-metric="mom_mean">PBL Mom Mean</a>
            <a href="#" data-metric="mom_max">PBL Mom Max</a>
            <a href="#" data-metric="ceiling">Ceilings</a>
            <a href="#" data-metric="vis">Visibility</a>
            <a href="#" data-metric="shear">Wind Shear</a>
            | <span>Isotherms & Clouds (kft):</span>
            <a href="#" data-metric="hght_0c">0°C Height</a>
            <a href="#" data-metric="hght_10c">-10°C Height</a>
            <a href="#" data-metric="hght_20c">-20°C Height</a>
            <a href="#" data-metric="cloud_top">Highest Cloud Top</a>
            <a href="#" data-metric="cloud_thick">Max Layer Thickness</a>
        </div>
        
        <div class="station-selector">
            <label for="stn-dropdown">Station:</label>
            <select id="stn-dropdown">
                <option value="kxmr" selected>KXMR (Cape Canaveral)</option>
                <option value="kdab">KDAB (Daytona Beach)</option>
                <option value="kmlb">KMLB (Melbourne)</option>
                <option value="kfpr">KFPR (Fort Pierce)</option>
                <option value="kpbi">KPBI (West Palm Beach)</option>
            </select>
        </div>
    </div>

    <div class="dprog-bar">
        <span class="dprog-title">dprog/dt Timeline:</span>
        <div id="run-selector-container"></div>
    </div>

    <div class="status-banner" id="view-title">Loading Matrix Engine...</div>

    <div class="main-layout">
        <div id="hover-popup-card"></div>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th style="width: 100px;">Time (UTC)</th>
                        <th>GFS</th>
                        <th>RAP</th>
                        <th>HRRR</th>
                    </tr>
                </thead>
                <tbody id="matrix-body"></tbody>
            </table>
        </div>

        <div class="side-panel" id="dynamic-legend-panel"></div>
    </div>

    <script>
        const historyRuns = __HISTORY_JSON__;
        const timeRows = __TIME_ROWS__;
        const models = ["gfs", "rap", "hrrr"];

        let activeStn = "kxmr";
        let activeMetric = "mom_mean";
        let activeRunIndex = 0;

        const metricLabels = {
            "ceiling": "Cloud Ceiling Heights (ft)",
            "vis": "Surface Visibility (sm)",
            "shear": "Low-Level Wind Shear",
            "mom_mean": "PBL Momentum Transfer Average Winds (kt)",
            "mom_max": "PBL Mixed-Layer Max Wind Speed (kt)",
            "hght_0c": "0°C Isotherm Height (kft)",
            "hght_10c": "-10°C Isotherm Height (kft)",
            "hght_20c": "-20°C Isotherm Height (kft)",
            "cloud_top": "Highest Detected Cloud Top (kft)",
            "cloud_thick": "Maximum Cloud Layer Thickness (kft)"
        };

        function renderLegend() {
            const panel = document.getElementById("dynamic-legend-panel");
            const isAviation = ["mom_mean", "mom_max", "ceiling", "vis", "shear"].includes(activeMetric);
            
            if (isAviation) {
                panel.innerHTML = `
                    <div class="legend-title">Aviation Criteria Thresholds</div>
                    <div style="font-weight:bold; font-size:0.75rem; margin-bottom:4px; color:#475569;">Flight Categories</div>
                    <div class="legend-item" style="background:#22c55e; color:white; font-weight:bold;"><span>MVFR</span><span>3,000 - 1,000 ft / 5 - 3 sm</span></div>
                    <div class="legend-item" style="background:#ef4444; color:white; font-weight:bold;"><span>IFR</span><span>< 1,000 ft / < 3 sm</span></div>
                    <div class="legend-item" style="background:#a855f7; color:white; font-weight:bold;"><span>LIFR</span><span>< 500 ft / < 1 sm</span></div>
                    
                    <div style="font-weight:bold; font-size:0.75rem; margin-top:12px; margin-bottom:4px; color:#475569;">PBL Momentum Transfer</div>
                    <div class="legend-item" style="background:#ffedd5; color:#7c2d12;"><span>Elevated</span><span>&ge; 15 kt</span></div>
                    <div class="legend-item" style="background:#f97316; color:white; font-weight:bold;"><span>High</span><span>&ge; 25 kt</span></div>
                    <div class="legend-item" style="background:#f43f5e; color:white; font-weight:bold;"><span>Severe</span><span>&ge; 35 kt</span></div>
                `;
            } else {
                panel.innerHTML = `
                    <div class="legend-title">Thermodynamic Risk Thresholds</div>
                    <div style="font-weight:bold; font-size:0.75rem; margin-bottom:4px; color:#475569;">Cloud Top Thermal Boundary Depth</div>
                    <div class="legend-item" style="background:#ffedd5; color:#7c2d12;"><span>Penetrating 0°C</span><span>Mixed Phase Zone</span></div>
                    <div class="legend-item" style="background:#f97316; color:white; font-weight:bold;"><span>Penetrating -10°C</span><span>Electrification Risk</span></div>
                    <div class="legend-item" style="background:#f43f5e; color:white; font-weight:bold;"><span>Penetrating -20°C</span><span>High Lightning Threat</span></div>
                `;
            }
        }

        function renderRunSelector() {
            const container = document.getElementById("run-selector-container");
            container.innerHTML = "";
            historyRuns.forEach((run, idx) => {
                const label = document.createElement("label");
                label.style.marginRight = "15px";
                label.style.cursor = "pointer";
                label.style.fontSize = "0.82rem";
                label.style.fontWeight = idx === activeRunIndex ? "bold" : "normal";
                if (idx === activeRunIndex) label.style.color = "#2563eb";

                const radio = document.createElement("input");
                radio.type = "radio";
                radio.name = "model-run-cycle";
                radio.value = idx;
                radio.checked = idx === activeRunIndex;

                radio.addEventListener("change", function() {
                    activeRunIndex = parseInt(this.value);
                    renderRunSelector();
                    renderMatrix();
                });

                label.appendChild(radio);
                label.appendChild(document.createTextNode(idx === 0 ? "Current Run" : "Run -" + idx));
                container.appendChild(label);
            });
        }

        function renderMatrix() {
            const currentData = historyRuns[activeRunIndex].data;
            document.getElementById("view-title").textContent = activeStn.toUpperCase() + " -> " + metricLabels[activeMetric];
            renderLegend();

            const tbody = document.getElementById("matrix-body");
            tbody.innerHTML = "";

            timeRows.forEach(row => {
                const tr = document.createElement("tr");
                const timeTd = document.createElement("td");
                timeTd.className = "time-col";
                timeTd.textContent = row;
                tr.appendChild(timeTd);

                models.forEach(model => {
                    const td = document.createElement("td");
                    let cellDisplay = "--";
                    let cssText = "";
                    let popupPayload = null;

                    if (currentData[activeStn] && currentData[activeStn][model] && currentData[activeStn][model][row]) {
                        const block = currentData[activeStn][model][row];
                        let value = block[activeMetric];

                        if (value !== undefined && value !== null) {
                            cellDisplay = value;
                            
                            if (activeMetric === "mom_mean" || activeMetric === "mom_max") {
                                let num = parseFloat(value);
                                if (num >= 35) cssText = "background-color: #f43f5e; color: #fff; font-weight: bold;";
                                else if (num >= 25) cssText = "background-color: #f97316; color: #fff; font-weight: bold;";
                                else if (num >= 15) cssText = "background-color: #ffedd5; color: #7c2d12;";
                            } else if (activeMetric === "ceiling") {
                                let num = parseFloat(value);
                                if (num >= 23000) {
                                    cellDisplay = "CLR";
                                } else {
                                    if (num < 500) cssText = "background-color: #a855f7; color: white; font-weight: bold;";
                                    else if (num < 1000) cssText = "background-color: #ef4444; color: white; font-weight: bold;";
                                    else if (num <= 3000) cssText = "background-color: #22c55e; color: white; font-weight: bold;";
                                }
                            } else if (activeMetric === "vis") {
                                let num = parseFloat(value);
                                if (num >= 10.0) {
                                    cellDisplay = "10";
                                } else {
                                    if (num < 1.0) cssText = "background-color: #a855f7; color: white; font-weight: bold;";
                                    else if (num < 3.0) cssText = "background-color: #ef4444; color: white; font-weight: bold;";
                                    else if (num <= 5.0) cssText = "background-color: #22c55e; color: white; font-weight: bold;";
                                }
                            } else if (activeMetric === "shear" && value !== null) {
                                cssText = "background-color: #fca5a5; color: #7f1d1d; font-weight: bold;";
                            }
                            
                            if (activeMetric === "cloud_top") {
                                let val = parseFloat(value);
                                if (val > 0) {
                                    if (val >= block.hght_20c) cssText = "background-color: #f43f5e; color: white; font-weight: bold;";
                                    else if (val >= block.hght_10c) cssText = "background-color: #f97316; color: white; font-weight: bold;";
                                    else if (val >= block.hght_0c) cssText = "background-color: #ffedd5; color: #7c2d12;";
                                }
                            } else if (activeMetric === "cloud_thick") {
                                let val = parseFloat(value);
                                if (val >= 4.5) cssText = "background-color: #ef4444; color: white; font-weight: bold;";
                                else if (val >= 3.0) cssText = "background-color: #ffedd5; color: #7c2d12;";
                            }

                            popupPayload = {
                                time: row,
                                model: model.toUpperCase(),
                                value: value,
                                h0: block.hght_0c,
                                h5: block.hght_5c,
                                h10: block.hght_10c,
                                h20: block.hght_20c,
                                ctop: block.cloud_top,
                                cthick: block.cloud_thick,
                                w_mean: block.mom_mean,
                                w_max: block.mom_max,
                                ceil: block.ceiling,
                                visibility: block.vis,
                                w_shear: block.shear
                            };
                        }
                    }

                    td.textContent = cellDisplay;
                    if (cssText) td.style.cssText = cssText;
                    
                    if (popupPayload) {
                        td.setAttribute("data-profile", JSON.stringify(popupPayload));
                        td.addEventListener("mouseover", showHoverPopup);
                        td.addEventListener("mousemove", moveHoverPopup);
                        td.addEventListener("mouseout", hideHoverPopup);
                    }

                    tr.appendChild(td);
                });

                tbody.appendChild(tr);
            });
        }

        const popupCard = document.getElementById("hover-popup-card");

        function showHoverPopup(e) {
            const dataStr = this.getAttribute("data-profile");
            if (!dataStr) return;
            const data = JSON.parse(dataStr);
            const isAviation = ["mom_mean", "mom_max", "ceiling", "vis", "shear"].includes(activeMetric);

            let bodyHtml = "";
            if (isAviation) {
                bodyHtml = `
                    <div class="popup-header">${data.model} Aviation Profile</div>
                    • Mean PBL Wind: <strong>${data.w_mean} kt</strong><br>
                    • Max PBL Wind: <strong>${data.w_max} kt</strong><br>
                    • Lowest Ceiling: <strong>${data.ceil >= 23000 ? "Clear" : data.ceil + " ft"}</strong><br>
                    • Visibility: <strong>${data.visibility} sm</strong><br>
                    • Wind Shear: <strong>${data.w_shear ? data.w_shear : "None Detected"}</strong>
                `;
            } else {
                bodyHtml = `
                    <div class="popup-header">${data.model} Thermodynamic Heights</div>
                    <strong>Freezing & Charging Isotherms:</strong><br>
                    • 0°C Level: <strong>${data.h0} kft</strong><br>
                    • -5°C Level: <strong>${data.h5} kft</strong><br>
                    • -10°C Level: <strong>${data.h10} kft</strong><br>
                    • -20°C Level: <strong>${data.h20} kft</strong>
                    <div style="margin-top:6px; border-top:1px solid #e2e8f0; padding-top:4px;">
                        • Highest Cloud Top: <strong>${data.ctop} kft</strong><br>
                        • Max Layer Thickness: <strong>${data.cthick} kft</strong>
                    </div>
                `;
            }

            popupCard.innerHTML = bodyHtml;
            popupCard.style.display = "block";
        }

        function moveHoverPopup(e) {
            let x = e.pageX + 15;
            let y = e.pageY - 40;
            popupCard.style.left = x + "px";
            popupCard.style.top = y + "px";
        }

        function hideHoverPopup() {
            popupCard.style.display = "none";
        }

        document.querySelectorAll(".nav-links a").forEach(link => {
            link.addEventListener("mouseover", function() {
                document.querySelectorAll(".nav-links a").forEach(a => a.classList.remove("active"));
                activeMetric = this.getAttribute("data-metric");
                this.classList.add("active");
                renderMatrix();
            });
        });

        document.getElementById("stn-dropdown").addEventListener("change", function() {
            activeStn = this.value;
            renderMatrix();
        });

        renderRunSelector();
        renderMatrix();
    </script>
</body>
</html>
"""

    processed_html = html_template.replace("__HISTORY_JSON__", json.dumps(history_runs))
    processed_html = processed_html.replace("__TIME_ROWS__", json.dumps(time_rows))

    with open("index.html", "w") as f:
        f.write(processed_html)
    print("Dashboard matrix compiled successfully.")


if __name__ == "__main__":
    cache = purge_workspace()
    stns, mdls, sounding_matrix = query_sounding_stations()
    generate_aviation_dashboard(stns, mdls, sounding_matrix)
