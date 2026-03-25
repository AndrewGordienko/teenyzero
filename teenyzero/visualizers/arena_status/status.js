function fmtInt(value) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? Math.round(num).toLocaleString() : "0";
}

function fmtNumber(value, digits = 1) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? num.toFixed(digits) : "0.0";
}

function fmtPct(value) {
    const num = Number(value || 0);
    return `${(num * 100).toFixed(1)}%`;
}

function fmtDate(ts) {
    if (!ts) return "n/a";
    return new Date(ts * 1000).toLocaleString();
}

function prettyStatus(value) {
    return String(value || "idle")
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

function render(state) {
    const summary = document.getElementById("summary");
    const details = document.getElementById("details");
    const external = document.getElementById("external-results");
    const matchMeta = document.getElementById("match-meta");
    const breakdown = document.getElementById("promotion-breakdown");
    const profile = state.runtime_profile_settings || {};

    const cards = [
        ["Arena Status", prettyStatus(state.status), "evaluation loop heartbeat and current phase"],
        ["Profile", state.runtime_profile || "local", "active runtime hardware profile"],
        ["Heartbeat", fmtDate(state.heartbeat_at), "latest arena state update"],
        ["Champion", state.champion_id || "n/a", "current self-play gate winner feeding best-model self-play"],
        ["Champion Elo", fmtNumber(state.champion_rating, 0), "internal arena rating after promotion and benchmark updates"],
        ["Candidate Elo", fmtNumber(state.candidate_rating, 0), "latest checkpoint rating from promotion matches"],
        ["Last Cycle", fmtInt(state.last_evaluated_cycle), "most recent training cycle evaluated by the arena"],
    ];

    summary.innerHTML = cards.map(([label, value, note]) => `
        <article class="metric-card">
            <span class="metric-label">${label}</span>
            <div class="metric-value ${String(value).length > 16 ? "metric-value--compact" : ""}">${value}</div>
            <div class="metric-note">${note}</div>
        </article>
    `).join("");

    const match = state.latest_match || {};
    matchMeta.textContent = match.games
        ? `${match.player_a} vs ${match.player_b} on ${fmtDate(match.played_at)}`
        : "No promotion match has been logged yet.";

    breakdown.innerHTML = [
        ["Score", fmtPct(match.score || 0), `${fmtInt(match.games || 0)} games in the latest gate`],
        ["Wins", fmtInt(match.wins || 0), "candidate wins as white or black"],
        ["Draws", fmtInt(match.draws || 0), "shared points still count in the promotion score"],
        ["Losses", fmtInt(match.losses || 0), "candidate losses in the latest gate"],
    ].map(([label, value, note]) => `
        <div class="info-card">
            <div class="info-label">${label}</div>
            <div class="info-value">${value}</div>
            <div class="info-note">${note}</div>
        </div>
    `).join("");

    const externalResults = state.external_results || [];
    external.innerHTML = externalResults.length
        ? externalResults.map((item) => `
            <div class="result-row">
                <div>
                    <div>${item.label || item.id}</div>
                    <div class="result-meta">
                        W ${fmtInt(item.wins)} / D ${fmtInt(item.draws)} / L ${fmtInt(item.losses)}
                        ${item.rating_anchor ? ` · Anchor ${fmtInt(item.rating_anchor)}` : ""}
                    </div>
                </div>
                <div class="result-value">${fmtPct(item.score)}</div>
            </div>
        `).join("")
        : `<div class="result-row"><div><div>No external results yet</div><div class="result-meta">Set \`TEENYZERO_STOCKFISH_PATH\` to benchmark against Stockfish levels.</div></div><div class="result-value">n/a</div></div>`;

    details.innerHTML = [
        ["Champion Archive", state.champion_archive_path || "n/a"],
        ["Stockfish Available", state.stockfish_available ? "yes" : "no"],
        ["Stockfish Path", state.stockfish_path || "n/a"],
        ["Promotion Threshold", fmtPct(state.promotion_threshold || 0)],
        ["Promotion Games", fmtInt(state.promotion_games || 0)],
        ["Arena Search", state.arena_simulations ? `${fmtInt(state.arena_simulations)} sims / move` : "n/a"],
        ["Inference Precision", String(profile.inference_precision || "fp32").toUpperCase()],
        ["External Games", state.baseline_games ? fmtInt(state.baseline_games) : "n/a"],
        ["Model Shape", profile.model_res_blocks ? `${fmtInt(profile.model_res_blocks)} blocks x ${fmtInt(profile.model_channels || 0)} channels` : "n/a"],
        ["Candidate", state.candidate_id || "n/a"],
        ["Last Error", state.last_error || "none"],
    ].map(([key, value]) => `
        <div class="detail-row">
            <div class="detail-key">${key}</div>
            <div class="detail-value">${value}</div>
        </div>
    `).join("");
}

function seriesRange(values) {
    const finite = values.filter((value) => Number.isFinite(value));
    if (!finite.length) return [0, 1];
    const min = Math.min(...finite);
    const max = Math.max(...finite);
    if (min === max) return [Math.min(0, min - 1), max + 1];
    return [Math.min(min, 0), max];
}

function axisMarkup(width, height, padding, minValue, maxValue, xLabels) {
    const usableHeight = height - padding.top - padding.bottom;
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const y = padding.top + usableHeight * ratio;
        const value = maxValue - ratio * (maxValue - minValue);
        return `
            <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#e6eaee" stroke-width="1" />
            <text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#66727f">${value.toFixed(maxValue >= 100 ? 0 : 2)}</text>
        `;
    }).join("");
    return `
        ${ticks}
        <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="#cfd6dd" stroke-width="1" />
        <text x="${padding.left}" y="${height - 6}" font-size="10" fill="#66727f">${xLabels[0]}</text>
        <text x="${width - padding.right}" y="${height - 6}" text-anchor="end" font-size="10" fill="#66727f">${xLabels[1]}</text>
    `;
}

function ensureTooltip(svgId) {
    const svg = document.getElementById(svgId);
    let tooltip = svg.parentElement.querySelector(".chart-tooltip");
    if (!tooltip) {
        tooltip = document.createElement("div");
        tooltip.className = "chart-tooltip";
        svg.parentElement.appendChild(tooltip);
    }
    return tooltip;
}

function attachTooltip(svgId, pointCount, formatter) {
    const svg = document.getElementById(svgId);
    const tooltip = ensureTooltip(svgId);
    if (!pointCount) {
        tooltip.style.opacity = "0";
        return;
    }

    const width = 520;
    const padding = { left: 44, right: 14 };
    const usableWidth = width - padding.left - padding.right;

    svg.onmousemove = (event) => {
        const rect = svg.getBoundingClientRect();
        const localX = ((event.clientX - rect.left) / rect.width) * width;
        const clamped = Math.max(padding.left, Math.min(width - padding.right, localX));
        const ratio = (clamped - padding.left) / Math.max(1, usableWidth);
        const index = Math.max(0, Math.min(pointCount - 1, Math.round(ratio * Math.max(1, pointCount - 1))));
        tooltip.innerHTML = formatter(index);
        tooltip.style.opacity = "1";
        tooltip.style.left = `${Math.min(rect.width - 196, Math.max(12, ((clamped / width) * rect.width) + 12))}px`;
    };

    svg.onmouseleave = () => {
        tooltip.style.opacity = "0";
    };
}

function renderLineChart(svgId, series, labels, formatter) {
    const svg = document.getElementById(svgId);
    const width = 520;
    const height = 200;
    const padding = { top: 14, right: 14, bottom: 24, left: 44 };
    svg.innerHTML = "";
    if (!series.length || !series.some((item) => item.values.length)) return;

    const allValues = series.flatMap((item) => item.values);
    const [minValue, maxValue] = seriesRange(allValues);
    const usableWidth = width - padding.left - padding.right;
    const usableHeight = height - padding.top - padding.bottom;
    const span = Math.max(1e-6, maxValue - minValue);

    const lines = series.map((item) => {
        const segments = [];
        let current = [];
        item.values.forEach((value, index) => {
            if (!Number.isFinite(value)) {
                if (current.length) {
                    segments.push(current);
                    current = [];
                }
                return;
            }
            const x = padding.left + (index * usableWidth) / Math.max(1, item.values.length - 1);
            const y = padding.top + usableHeight - ((value - minValue) / span) * usableHeight;
            current.push(`${x},${y}`);
        });
        if (current.length) {
            segments.push(current);
        }
        return segments.map((points) => `<polyline fill="none" stroke="${item.color}" stroke-width="3" points="${points.join(" ")}" />`).join("");
    }).join("");

    svg.innerHTML = axisMarkup(width, height, padding, minValue, maxValue, labels) + lines;
    attachTooltip(svgId, series[0].values.length, formatter);
}

function renderHistory(history) {
    const recent = history.slice(-32);
    const legend = document.getElementById("external-legend");
    const labels = recent.length
        ? [`Cycle ${recent[0].cycle}`, `Cycle ${recent[recent.length - 1].cycle}`]
        : ["Oldest", "Latest"];

    renderLineChart(
        "rating-chart",
        [
            { values: recent.map((item) => Number(item.champion_rating || 0)), color: "#111111" },
            { values: recent.map((item) => Number(item.candidate_rating || 0)), color: "#2155d6" },
        ],
        labels,
        (index) => {
            const item = recent[index];
            return `
                <strong>Cycle ${item.cycle}</strong><br>
                Champion: ${fmtNumber(item.champion_rating, 0)}<br>
                Candidate: ${fmtNumber(item.candidate_rating, 0)}
            `;
        },
    );

    renderLineChart(
        "promotion-chart",
        [{ values: recent.map((item) => Number(item.promotion_score || 0)), color: "#1a9b53" }],
        labels,
        (index) => {
            const item = recent[index];
            return `
                <strong>Cycle ${item.cycle}</strong><br>
                Score: ${fmtPct(item.promotion_score)}<br>
                Promoted: ${item.promoted ? "yes" : "no"}
            `;
        },
    );

    const benchmarkEntries = [];
    for (const item of recent) {
        for (const result of item.external_results || []) {
            if (!result.id || benchmarkEntries.some((entry) => entry.id === result.id)) {
                continue;
            }
            benchmarkEntries.push({
                id: result.id,
                label: result.label || result.id,
            });
        }
    }
    const palette = ["#111111", "#2155d6", "#1a9b53", "#c58f12", "#b04f73"];
    const benchmarkSeries = benchmarkEntries.map((entry, index) => ({
        label: entry.label,
        color: palette[index % palette.length],
        values: recent.map((item) => {
            const match = (item.external_results || []).find((result) => result.id === entry.id);
            return match ? Number(match.score || 0) : Number.NaN;
        }),
    }));
    legend.innerHTML = benchmarkSeries.length
        ? benchmarkSeries.map((entry) => `
            <span class="chart-legend-item">
                <i class="chart-legend-swatch" style="background:${entry.color}"></i>
                ${entry.label}
            </span>
        `).join("")
        : `<span class="chart-legend-item">No external benchmarks logged yet</span>`;
    renderLineChart(
        "external-chart",
        benchmarkSeries,
        labels,
        (index) => {
            const item = recent[index];
            const rows = benchmarkEntries.map((entry) => {
                const match = (item.external_results || []).find((result) => result.id === entry.id);
                const value = match ? fmtPct(match.score) : "n/a";
                return `${entry.label}: ${value}`;
            }).join("<br>");
            return `
                <strong>Cycle ${item.cycle}</strong><br>
                ${rows || "No external benchmarks logged"}
            `;
        },
    );
}

function renderLoadError(message) {
    const summary = document.getElementById("summary");
    const legend = document.getElementById("external-legend");
    summary.innerHTML = `
        <article class="metric-card">
            <span class="metric-label">Arena Error</span>
            <div class="metric-value metric-value--compact">Unable to load arena data</div>
            <div class="metric-note">${message}</div>
        </article>
    `;
    if (legend) {
        legend.innerHTML = "";
    }
    for (const id of ["rating-chart", "promotion-chart", "external-chart"]) {
        const svg = document.getElementById(id);
        if (svg) {
            svg.innerHTML = "";
        }
    }
}

async function fetchJson(url) {
    const response = await fetch(url);
    const text = await response.text();
    let payload;
    try {
        payload = JSON.parse(text);
    } catch (error) {
        throw new Error(`${url} returned invalid JSON: ${error.message}`);
    }
    if (!response.ok) {
        throw new Error(payload.error || `${url} failed with status ${response.status}`);
    }
    return payload;
}

async function update() {
    try {
        const [state, history] = await Promise.all([
            fetchJson("/api/arena_status"),
            fetchJson("/api/arena_history"),
        ]);
        render(state);
        renderHistory(history);
    } catch (error) {
        console.error("Failed to update arena dashboard:", error);
        renderLoadError(error.message);
    }
}

update();
setInterval(update, 5000);
