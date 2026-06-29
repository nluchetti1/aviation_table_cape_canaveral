<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Aviation & Launch Commit Weather Grid</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 15px; font-size: 0.85rem; }\n        .control-bar { display: flex; align-items: center; justify-content: space-between; background: #e2e8f0; padding: 10px; border-radius: 4px; margin-bottom: 12px; gap: 15px; }\n        .nav-links { line-height: 1.8; }\n        .nav-links span { font-weight: bold; margin-right: 5px; color: #334155; }\n        .nav-links a { margin: 0 3px; text-decoration: none; color: #2563eb; padding: 2px 5px; border-radius: 3px; transition: all 0.1s ease; cursor: pointer; }\n        .nav-links a.active { color: #ffffff; background: #2563eb; font-weight: bold; }\n        .station-selector { display: flex; align-items: center; gap: 6px; font-size: 0.9rem; font-weight: bold; background: white; padding: 4px 8px; border-radius: 4px; border: 1px solid #cbd5e1; }\n        .station-selector select { font-size: 0.9rem; font-weight: bold; padding: 2px; color: #1e293b; border: 1px solid #94a3b8; border-radius: 3px; cursor: pointer; }\n        .update-tag { font-size: 0.75rem; color: #64748b; font-style: italic; }\n        table { border-collapse: collapse; background: #ffffff; width: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 4px; overflow: hidden; }\n        th, td { border: 1px solid #e2e8f0; padding: 6px 8px; text-align: center; font-size: 0.8rem; }\n        th { background-color: #f1f5f9; color: #334155; font-weight: bold; }\n        .time-col { background: #f8fafc; font-weight: bold; font-size: 0.85rem; color: #0f172a; white-space: nowrap; width: 70px; }\n        .model-hdr { background-color: #cbd5e1; font-weight: bold; color: #1e293b; border-bottom: 2px solid #94a3b8; }\n        \n        /* Color Matrix Threshold States */\n        .cell-vfr { background-color: #f0fdf4 !important; color: #166534 !important; }\n        .cell-mvfr { background-color: #eff6ff !important; color: #1e40af !important; font-weight: 500; }\n        .cell-ifr { background-color: #fef2f2 !important; color: #991b1b !important; font-weight: bold; }\n        .cell-lifr { background-color: #ffa6a6 !important; color: #7f0000 !important; font-weight: bold; }\n        \n        /* Wind & Shear States */\n        .cell-wind-alert { border: 2px solid #dc2626 !important; font-weight: bold; background-color: #fef2f2; }\n        .cell-shear-alert { background-color: #fae8ff !important; color: #86198f !important; font-weight: bold; border: 2px dashed #c084fc !important; }\n        \n        /* Lightning Probability Heat Map */\n        .ltg-none { background-color: #ffffff; color: #cbd5e1; }\n        .ltg-vlow { background-color: #fef9c3 !important; color: #713f12 !important; font-weight: bold; }\n        .ltg-low { background-color: #fef08a !important; color: #854d0e !important; font-weight: bold; }\n        .ltg-med { background-color: #fed7aa !important; color: #9a3412 !important; font-weight: bold; }\n        .ltg-high { background-color: #fca5a5 !important; color: #991b1b !important; font-weight: bold; box-shadow: inset 0 0 4px #ef4444; }\n        \n        /* Interactive Interactive Elements */\n        .hover-cell { cursor: help; transition: background 0.05s ease; }\n        .hover-cell:hover { filter: brightness(0.93); border: 1px solid #475569; }\n        #hover-popup-card { position: absolute; z-index: 1000; display: none; background: #1e293b; color: #f8fafc; border-radius: 6px; padding: 10px 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.25); font-size: 0.8rem; pointer-events: none; max-width: 280px; border: 1px solid #475569; line-height: 1.4; }\n        .popup-header { font-weight: bold; color: #38bdf8; border-bottom: 1px solid #475569; margin-bottom: 6px; padding-bottom: 4px; font-size: 0.85rem; }\n    </style>
</head>
<body>

    <div class="control-bar">
        <div class="station-selector">
            <span>Target Terminal Location:</span>
            <select id="station-picker" onchange="changeStation(this.value)">
                <option value="KXMR">Cape Canaveral Space Force Station (KXMR)</option>
                <option value="KDAB">Daytona Beach International (KDAB)</option>
                <option value="KMLB">Melbourne Orlando International (KMLB)</option>
                <option value="KFPR">St. Lucie County International (KFPR)</option>
                <option value="KPBI">Palm Beach International (KPBI)</option>
            </select>
        </div>
        <div class="nav-links" id="history-nav"><span>Available Forecast Run Cycles:</span></div>
        <div class="update-tag" id="run-age-label">Awaiting run initialization...</div>
    </div>

    <div id="hover-popup-card"></div>

    <table id="matrix-display-table">
        <thead>
            <tr id="top-header-row"></tr>
            <tr id="sub-header-row"></tr>
        </thead>
        <tbody id="matrix-body"></tbody>
    </table>

    <script>
        let historyRuns = [];
        let currentRunIndex = 0;
        let currentStation = "KXMR";
        const popupCard = document.getElementById("hover-popup-card");

        async function initDashboard() {
            try {
                const r = await fetch("history.json?t=" + new Date().getTime());
                historyRuns = await r.json();
                buildNavigation();
                loadRun(0);
            } catch(e) {
                document.getElementById("run-age-label").innerText = "Critical Error: history.json unresolved.";
                console.error(e);
            }
        }

        function buildNavigation() {
            const nav = document.getElementById("history-nav");
            // Clear existing links except label
            nav.innerHTML = "<span>Available Forecast Run Cycles:</span>";
            historyRuns.forEach((run, idx) => {
                const a = document.createElement("a");
                a.innerText = run.timestamp;
                a.className = idx === currentRunIndex ? "active" : "";
                a.onclick = () => loadRun(idx);
                nav.appendChild(a);
            });
        }

        function changeStation(val) {
            currentStation = val;
            buildMatrix();
        }

        function loadRun(idx) {
            currentRunIndex = idx;
            buildNavigation();
            document.getElementById("run-age-label").innerText = `Cycle Generated: ${historyRuns[idx].timestamp}`;
            buildMatrix();
        }

        function evaluateFlightRules(ceil, vis) {
            if (ceil < 500 || vis < 1) return { label: `LIFR (${ceil} ft)`, cls: "cell-lifr" };
            if (ceil < 1000 || vis < 3) return { label: `IFR (${ceil} ft)`, cls: "cell-ifr" };
            if (ceil <= 3000 || vis <= 5) return { label: `MVFR (${ceil} ft)`, cls: "cell-mvfr" };
            return { label: "VFR (Clear/Unrestricted)", cls: "cell-vfr" };
        }

        function buildMatrix() {
            if (!historyRuns.length) return;
            const runData = historyRuns[currentRunIndex];
            const stn = currentStation;
            
            const topRow = document.getElementById("top-header-row");
            const subRow = document.getElementById("sub-header-row");
            const body = document.getElementById("matrix-body");

            topRow.innerHTML = `<th rowspan="2" class="time-col" style="vertical-align: middle;">Valid (UTC)<br>Day/Hour</th>`;
            subRow.innerHTML = "";
            body.innerHTML = "";

            // Dynamic header generation for target models
            const models = ["gfs", "rap", "hrrr"];
            models.forEach(m => {
                topRow.innerHTML += `<th colspan="3" class="model-hdr">${m.toUpperCase()} Bufkit Diagnostic Profiles</th>`;
                subRow.innerHTML += `<th>Aviation Profile</th><th>Mom. Winds</th><th>Isotherms / Clouds</th>`;
            });
            topRow.innerHTML += `<th class="model-hdr" style="background:#0f172a; color:#fff;">HREF Probability</th>`;
            subRow.innerHTML += `<th style="background:#334155; color:#fff;">Lightning 4-Hr Density</th>`;

            // Compile global continuous time track
            let allTimes = new Set();
            models.forEach(m => {
                if (runData.data[stn] && runData.data[stn][m]) {
                    Object.keys(runData.data[stn][m]).forEach(t => allTimes.add(t));
                }
            });
            let sortedTimes = Array.from(allTimes).sort();

            sortedTimes.forEach(rowKey => {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td class="time-col">${rowKey}Z</td>`;

                models.forEach(m => {
                    const mData = (runData.data[stn] && runData.data[stn][m]) ? runData.data[stn][m][rowKey] : null;

                    if (!mData) {
                        tr.innerHTML += `<td class="ltg-none">-</td><td class="ltg-none">-</td><td class="ltg-none">-</td>`;
                        return;
                    }

                    // 1. Aviation Profile Cell Calculation
                    let avObj = evaluateFlightRules(mData.ceiling, mData.vis);
                    let avClass = avObj.cls;
                    if (mData.shear) avClass += " cell-shear-alert";

                    let cell1Id = `pop_${stn}_${m}_${rowKey.replace('/','_')}_av`;
                    tr.innerHTML += `<td id="${cell1Id}" class="hover-cell ${avClass}">${avObj.label}</td>`;

                    // 2. Momentum Boundary Layer Wind Cell Calculation
                    let wClass = (mData.mom_max >= 25 || mData.mom_mean >= 15) ? "cell-wind-alert" : "cell-vfr";
                    let cell2Id = `pop_${stn}_${m}_${rowKey.replace('/','_')}_wind`;
                    tr.innerHTML += `<td id="${cell2Id}" class="hover-cell ${wClass}">Mean: ${mData.mom_mean} kt<br>Max: ${mData.mom_max} kt</td>`;

                    // 3. Thermodynamic Isotherm Levels Cell Calculation
                    let cell3Id = `pop_${stn}_${m}_${rowKey.replace('/','_')}_thermo`;
                    tr.innerHTML += `<td id="${cell3Id}" class="hover-cell cell-vfr">0°C: ${mData.hght_0c} kft<br>Top: ${mData.cloud_top} kft</td>`;

                    // Attach hover interactions safely
                    setTimeout(() => {
                        setupCellHover(cell1Id, { type: "av", model: m.toUpperCase(), ceil: mData.ceiling, visibility: mData.vis, w_shear: mData.shear });
                        setupCellHover(cell2Id, { type: "wind", model: m.toUpperCase(), w_mean: mData.mom_mean, w_max: mData.mom_max });
                        setupCellHover(cell3Id, { type: "thermo", model: m.toUpperCase(), h0: mData.hght_0c, h5: mData.hght_5c, h10: mData.hght_10c, h20: mData.hght_20c, ctop: mData.cloud_top, cthick: mData.cloud_thick });
                    }, 0);
                });

                // --- FIXED HREF LIGHTNING RENDERING MATRIX BLOCK ---
                const stnLower = stn.toLowerCase(); // Map structural case disconnect
                let ltgObj = (runData.href_lightning && runData.href_lightning[stnLower]) ? runData.href_lightning[stnLower][rowKey] : null;

                let p25 = 0, p50 = 0, p100 = 0, p200 = 0;
                if (ltgObj) {
                    p25  = ltgObj.p25  || 0;
                    p50  = ltgObj.p50  || 0;
                    p100 = ltgObj.p100 || 0;
                    p200 = ltgObj.p200 || 0;
                }

                let ltgClass = "ltg-none";
                let ltgLabel = "0%";

                if (p25 > 0) {
                    ltgLabel = `${p25}%`;
                    if (p25 >= 30) ltgClass = "ltg-high";
                    else if (p25 >= 15) ltgClass = "ltg-med";
                    else if (p25 >= 5) ltgClass = "ltg-low";
                    else ltgClass = "ltg-vlow";
                }

                let ltgCellId = `pop_${stn}_href_${rowKey.replace('/','_')}`;
                tr.innerHTML += `<td id="${ltgCellId}" class="hover-cell ${ltgClass}">${ltgLabel}</td>`;

                setTimeout(() => {
                    setupCellHover(ltgCellId, { type: "ltg", p25: p25, p50: p50, p100: p100, p200: p200 });
                }, 0);

                body.appendChild(tr);
            });
        }

        function setupCellHover(id, data) {
            const el = document.getElementById(id);
            if (!el) return;
            el.addEventListener("mouseenter", (e) => showHoverPopup(e, data));
            el.addEventListener("mousemove", moveHoverPopup);
            el.addEventListener("mouseleave", hideHoverPopup);
        }

        function showHoverPopup(e, data) {
            if (data.type === "ltg") {
                popupCard.innerHTML = `
                    <div class="popup-header">HREF Lightning Probability</div>
                    <strong>Grid Radius Thresholds:</strong><br>
                    • Strike within 25km: <strong>${data.p25}%</strong><br>
                    • Strike within 50km: <strong>${data.p50}%</strong><br>
                    • Strike within 100km: <strong>${data.p100}%</strong><br>
                    • Strike within 200km: <strong>${data.p200}%</strong>
                `;
            } else {
                popupCard.innerHTML = data.type === "av" ? `
                    <div class="popup-header">${data.model} Profile Details</div>
                    • Lowest Ceiling Layer: <strong>${data.ceil >= 24000 ? "Clear (unrestricted)" : data.ceil + " ft"}</strong><br>
                    • Horizontal Visibility: <strong>${data.visibility} sm</strong><br>
                    • Microscale Wind Shear: <strong>${data.w_shear ? data.w_shear : "None detected"}</strong>
                ` : data.type === "wind" ? `
                    <div class="popup-header">${data.model} Boundary Layer Winds</div>
                    • Mean Profile Wind Speed: <strong>${data.w_mean} kt</strong><br>
                    • Max Profile Column Velocity: <strong>${data.w_max} kt</strong>
                ` : `
                    <div class="popup-header">${data.model} Isotherm Profiles</div>
                    • 0°C Level height: <strong>${data.h0} kft</strong><br>
                    • -5°C (Charging zone base): <strong>${data.h5} kft</strong><br>
                    • -10°C level height: <strong>${data.h10} kft</strong><br>
                    • -20°C level height: <strong>${data.h20} kft</strong>
                    <div style="margin-top:6px; border-top:1px solid #475569; padding-top:4px;">
                        • Profile Maximum Cloud Top: <strong>${data.ctop} kft</strong><br>
                        • Aggregated Cloud Layer Thickness: <strong>${data.cthick} kft</strong>
                    </div>
                `;
            }
            popupCard.style.display = "block";
            moveHoverPopup(e);
        }

        function moveHoverPopup(e) { 
            popupCard.style.left = (e.pageX + 15) + "px"; 
            popupCard.style.top = (e.pageY - 40) + "px"; 
        }
        
        function hideHoverPopup() { 
            popupCard.style.display = "none"; 
        }

        window.onload = initDashboard;
    </script>
</body>
</html>
