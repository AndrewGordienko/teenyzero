function prettyStatus(value) {
    return String(value || "idle")
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

function fmtNumber(value, digits = 1) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? num.toFixed(digits) : "0.0";
}

function fmtLoss(value) {
    const num = Number(value || 0);
    if (!Number.isFinite(num)) return "0.0000";
    if (num === 0) return "0.0000";
    if (Math.abs(num) < 0.001) return num.toExponential(2);
    return num.toFixed(4);
}

function fmtInt(value) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? Math.round(num).toLocaleString() : "0";
}

function fmtDate(ts) {
    if (!ts) return "n/a";
    return new Date(ts * 1000).toLocaleString();
}

function fmtDuration(seconds) {
    const value = Number(seconds || 0);
    if (!Number.isFinite(value)) return "0.0s";
    if (value >= 60) return `${(value / 60).toFixed(1)}m`;
    if (value >= 1) return `${value.toFixed(1)}s`;
    return `${(value * 1000).toFixed(0)}ms`;
}

function stageElapsed(state) {
    const started = Number(state.stage_started_at || 0);
    const heartbeat = Number(state.heartbeat_at || 0);
    if (!started || !heartbeat) return 0;
    return Math.max(0, heartbeat - started);
}

function buildLiveHistory(state, history) {
    const next = history.slice();
    if (state.status === "waiting") {
        return next;
    }

    next.push({
        finished_at: Number(state.heartbeat_at || 0),
        loss: Number(state.running_loss || state.last_loss || 0),
        policy_loss: Number(state.running_policy_loss || state.last_policy_loss || 0),
        value_loss: Number(state.running_value_loss || state.last_value_loss || 0),
        duration_s: Number(state.last_train_duration_s || 0),
        scan_duration_s: Number(state.last_scan_duration_s || 0),
        window_build_duration_s: Number(state.last_window_build_duration_s || 0),
        train_duration_s: Number(state.train_elapsed_s || state.last_train_phase_duration_s || 0),
        checkpoint_duration_s: Number(state.last_checkpoint_duration_s || 0),
        prune_duration_s: Number(state.last_prune_duration_s || 0),
        samples_per_s: Number(state.samples_per_s || state.last_samples_per_s || 0),
        batches_per_s: Number(state.batches_per_s || state.last_batches_per_s || 0),
        avg_batch_time_ms: Number(state.avg_batch_time_ms || state.last_avg_batch_time_ms || 0),
        is_live: true,
    });
    return next;
}

function normalizeHistory(history) {
    return history.map((item) => {
        const trainDuration = Number(item.train_duration_s || 0);
        return {
            ...item,
            train_duration_s: trainDuration > 0 ? trainDuration : Number(item.duration_s || 0),
            scan_duration_s: Number(item.scan_duration_s || 0),
            window_build_duration_s: Number(item.window_build_duration_s || 0),
            checkpoint_duration_s: Number(item.checkpoint_duration_s || 0),
            prune_duration_s: Number(item.prune_duration_s || 0),
            samples_per_s: Number(item.samples_per_s || 0),
            avg_batch_time_ms: Number(item.avg_batch_time_ms || 0),
        };
    });
}

function render(state) {
    const summary = document.getElementById("summary");
    const progress = document.getElementById("progress");
    const details = document.getElementById("details");
    const stageCards = document.getElementById("stage-cards");
    const throughputCards = document.getElementById("throughput-cards");
    const profile = state.runtime_profile_settings || {};

    const nextTrainTarget = Number(state.train_increment || 0);
    const newSamples = Number(state.new_samples_since_last_train || 0);
    const bufferSamples = Number(state.buffer_sample_count || 0);
    const minReady = Number(state.min_samples_ready || 0);
    const batchesDone = Number(state.completed_batches || 0);
    const batchesTotal = Number(state.total_batches || 0);
    const filesDone = Number(state.loaded_files || 0);
    const filesTotal = Number(state.total_window_files || 0);
    const batchSize = Number(state.train_batch_size || profile.train_batch_size || 0);
    const gradAccum = Math.max(1, Number(profile.train_grad_accum_steps || 1));
    const effectiveBatch = batchSize * gradAccum;
    const loaderWorkers = Number(state.train_num_workers ?? profile.train_num_workers ?? 0);
    const modelBlocks = Number(profile.model_res_blocks || 0);
    const modelChannels = Number(profile.model_channels || 0);
    const inputHistory = Number(profile.input_history_length || 0);
    const inputPlanes = Number(profile.input_planes || 0);

    const cards = [
        ["Status", prettyStatus(state.status), "trainer loop state"],
        ["Profile", state.runtime_profile || "local", "active runtime hardware profile"],
        ["Heartbeat", fmtDate(state.heartbeat_at), "latest trainer state update"],
        ["Buffer Samples", fmtInt(state.buffer_sample_count), "all replay samples currently available"],
        ["New Samples", fmtInt(state.new_samples_since_last_train), "new samples since last train cycle"],
        ["Train Cycles", fmtInt(state.training_cycles), "completed training updates"],
        ["Loss", fmtLoss(state.running_loss || state.last_loss), "latest or in-progress total loss"],
    ];

    summary.innerHTML = cards.map(([label, value, note]) => `
        <article class="metric-card">
            <span class="metric-label">${label}</span>
            <div class="metric-value ${String(value).length > 18 ? "metric-value--compact" : ""}">${value}</div>
            <div class="metric-note">${note}</div>
        </article>
    `).join("");

    const progressCards = [
        [
            "Next Train Trigger",
            `${fmtInt(newSamples)} / ${fmtInt(nextTrainTarget)}`,
            Math.max(0, Math.min(100, nextTrainTarget ? (newSamples / nextTrainTarget) * 100 : 0)),
            `${fmtInt(Math.max(0, nextTrainTarget - newSamples))} samples remaining`,
        ],
        [
            "Buffer Readiness",
            `${fmtInt(bufferSamples)} / ${fmtInt(minReady)}`,
            Math.max(0, Math.min(100, minReady ? (bufferSamples / minReady) * 100 : 0)),
            "minimum replay size before training starts",
        ],
        [
            "Current Cycle Progress",
            batchesTotal > 0 ? `${fmtInt(batchesDone)} / ${fmtInt(batchesTotal)}` : `${fmtInt(filesDone)} / ${fmtInt(filesTotal)}`,
            Math.max(
                0,
                Math.min(
                    100,
                    batchesTotal > 0
                        ? (batchesDone / Math.max(1, batchesTotal)) * 100
                        : (filesDone / Math.max(1, filesTotal || 1)) * 100,
                ),
            ),
            batchesTotal > 0 ? "training batches completed" : "replay files loaded into the active window",
        ],
    ];

    progress.innerHTML = progressCards.map(([title, value, pct, note]) => `
        <section class="panel progress-card">
            <h2>${title}</h2>
            <div class="progress-value">${value}</div>
            <div class="progress-bar"><div class="progress-fill" style="width:${pct.toFixed(1)}%"></div></div>
            <div class="progress-meta">
                <span>${note}</span>
                <span>${pct.toFixed(0)}%</span>
            </div>
        </section>
    `).join("");

    const stageInfo = [
        [
            "Current Stage",
            prettyStatus(state.status),
            `${fmtDuration(stageElapsed(state))} elapsed in the active pipeline stage`,
        ],
        [
            "Last Cycle",
            fmtDuration(state.last_train_duration_s),
            `${fmtDuration(state.last_scan_duration_s)} scan, ${fmtDuration(state.last_window_build_duration_s)} window, ${fmtDuration(state.last_train_phase_duration_s)} train`,
        ],
        [
            "Window Build",
            `${fmtInt(state.loaded_files)} / ${fmtInt(state.total_window_files)}`,
            `${fmtDuration(state.window_load_elapsed_s)} load time at ${fmtInt(state.window_samples_per_s)} samples/s`,
        ],
    ];

    stageCards.innerHTML = stageInfo.map(([label, value, note]) => `
        <div class="info-card">
            <div class="info-label">${label}</div>
            <div class="info-value">${value}</div>
            <div class="info-note">${note}</div>
        </div>
    `).join("");

    const throughputInfo = [
        [
            "Samples / Sec",
            fmtInt(state.samples_per_s || state.last_samples_per_s),
            "effective training throughput over the active or last epoch",
        ],
        [
            "Batches / Sec",
            fmtNumber(state.batches_per_s || state.last_batches_per_s, 2),
            "optimizer step rate through the current replay window",
        ],
        [
            "Batch Time",
            `${fmtNumber(state.avg_batch_time_ms || state.last_avg_batch_time_ms, 1)} ms`,
            `${fmtInt(state.trained_samples || state.last_window_samples)} samples seen in the active or last epoch`,
        ],
    ];

    throughputCards.innerHTML = throughputInfo.map(([label, value, note]) => `
        <div class="info-card">
            <div class="info-label">${label}</div>
            <div class="info-value">${value}</div>
            <div class="info-note">${note}</div>
        </div>
    `).join("");

    const sections = [
        [
            "Replay Buffer",
            [
                ["Buffer Samples", fmtInt(state.buffer_sample_count)],
                ["New Since Last Train", fmtInt(state.new_samples_since_last_train)],
                ["Train Increment", fmtInt(state.train_increment)],
                ["Min Ready", fmtInt(state.min_samples_ready)],
                ["Replay Window", fmtInt(state.replay_window_samples)],
                ["Buffer Files", fmtInt(state.buffer_file_count)],
            ],
        ],
        [
            "Current Cycle",
            [
                ["Window Files", `${fmtInt(state.loaded_files)} / ${fmtInt(state.total_window_files)}`],
                ["Window Samples", fmtInt(state.loaded_window_samples)],
                ["Batches", `${fmtInt(state.completed_batches)} / ${fmtInt(state.total_batches)}`],
                ["Samples / Sec", fmtInt(state.samples_per_s || state.last_samples_per_s)],
                ["Avg Batch Time", `${fmtNumber(state.avg_batch_time_ms || state.last_avg_batch_time_ms, 1)} ms`],
                ["Current Stage", prettyStatus(state.status)],
            ],
        ],
        [
            "Last Completed Train",
            [
                ["Train End", fmtDate(state.last_train_finished_at)],
                ["Total Duration", fmtDuration(state.last_train_duration_s)],
                ["Policy Loss", fmtLoss(state.last_policy_loss)],
                ["Value Loss", fmtLoss(state.last_value_loss)],
                ["Checkpoint", fmtDuration(state.last_checkpoint_duration_s)],
                ["Latest Model", state.latest_model_path || "n/a"],
            ],
        ],
        [
            "Runtime Plan",
            [
                ["Device", state.device || "n/a"],
                ["Optimizer", String(profile.train_optimizer || "n/a").toUpperCase()],
                ["Precision", String(profile.train_precision || "fp32").toUpperCase()],
                ["Compile", profile.train_compile ? "enabled" : "off"],
                ["Batch / Accum", batchSize ? `${fmtInt(batchSize)} x ${fmtInt(gradAccum)}` : "n/a"],
                ["Effective Batch", effectiveBatch ? fmtInt(effectiveBatch) : "n/a"],
                ["Loader Workers", fmtInt(loaderWorkers)],
                ["Model", modelBlocks ? `${fmtInt(modelBlocks)} blocks x ${fmtInt(modelChannels)} channels` : "n/a"],
                ["Input Stack", inputHistory ? `${fmtInt(inputHistory)} positions / ${fmtInt(inputPlanes)} planes` : "n/a"],
                ["Self-Play Budget", profile.selfplay_simulations ? `${fmtInt(profile.selfplay_simulations)} sims x ${fmtInt(profile.selfplay_workers || 0)} workers` : "n/a"],
                ["Arena Budget", profile.arena_simulations ? `${fmtInt(profile.arena_simulations)} sims x ${fmtInt(profile.arena_promotion_games || 0)} games` : "n/a"],
            ],
        ],
    ];

    details.innerHTML = sections.map(([title, rows]) => `
        <section class="detail-section">
            <h3 class="detail-section-title">${title}</h3>
            <div class="detail-list">
                ${rows.map(([key, value]) => `
                    <div class="detail-row">
                        <div class="detail-key">${key}</div>
                        <div class="detail-value">${value}</div>
                    </div>
                `).join("")}
            </div>
        </section>
    `).join("");
}

function renderLoadError(message) {
    const summary = document.getElementById("summary");
    const progress = document.getElementById("progress");
    const details = document.getElementById("details");
    const stageCards = document.getElementById("stage-cards");
    const throughputCards = document.getElementById("throughput-cards");

    summary.innerHTML = `
        <article class="metric-card">
            <span class="metric-label">Dashboard Error</span>
            <div class="metric-value metric-value--compact">Training data unavailable</div>
            <div class="metric-note">${message}</div>
        </article>
    `;
    progress.innerHTML = "";
    details.innerHTML = "";
    stageCards.innerHTML = "";
    throughputCards.innerHTML = "";
    for (const id of ["loss-chart", "duration-chart", "throughput-chart", "batch-chart"]) {
        const svg = document.getElementById(id);
        if (svg) {
            svg.innerHTML = "";
        }
    }
}

function seriesRange(series) {
    const values = series.flatMap((item) => item.values);
    if (!values.length) {
        return [0, 1];
    }

    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    if (minValue === maxValue) {
        return [Math.min(0, minValue), maxValue + 1];
    }
    return [Math.min(0, minValue), maxValue];
}

function axisMarkup(width, height, padding, minValue, maxValue, xLabels) {
    const usableHeight = height - padding.top - padding.bottom;
    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const y = padding.top + usableHeight * ratio;
        const value = maxValue - ratio * (maxValue - minValue);
        return `
            <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#e6eaee" stroke-width="1" />
            <text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#66727f">${value.toFixed(maxValue >= 10 ? 0 : 2)}</text>
        `;
    }).join("");

    return `
        ${yTicks}
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
    if (pointCount <= 0) {
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
        const scaledX = (clamped / width) * rect.width;
        const tooltipLeft = Math.min(rect.width - 196, Math.max(12, scaledX + 12));
        tooltip.style.left = `${tooltipLeft}px`;
    };

    svg.onmouseleave = () => {
        tooltip.style.opacity = "0";
    };
}

function renderMultiLineChart(svgId, series, xLabels = ["Oldest", "Latest"]) {
    const svg = document.getElementById(svgId);
    const width = 520;
    const height = 180;
    const padding = { top: 14, right: 14, bottom: 24, left: 44 };
    svg.innerHTML = "";

    if (!series.length || !series.some((item) => item.values.length)) {
        return;
    }

    const [minValue, maxValue] = seriesRange(series);
    const valueSpan = Math.max(1e-6, maxValue - minValue);
    const usableWidth = width - padding.left - padding.right;
    const usableHeight = height - padding.top - padding.bottom;

    const lines = series.map((item) => {
        if (!item.values.length) {
            return "";
        }
        const points = item.values.map((value, index) => {
            const x = padding.left + (index * usableWidth) / Math.max(1, item.values.length - 1);
            const y = padding.top + usableHeight - ((value - minValue) / valueSpan) * usableHeight;
            return `${x},${y}`;
        }).join(" ");
        return `<polyline fill="none" stroke="${item.color}" stroke-width="3" points="${points}" />`;
    }).join("");

    svg.innerHTML = axisMarkup(width, height, padding, minValue, maxValue, xLabels) + lines;
}

function renderStackedDurationChart(svgId, history, xLabels = ["Oldest", "Latest"]) {
    const svg = document.getElementById(svgId);
    const width = 520;
    const height = 180;
    const padding = { top: 14, right: 14, bottom: 24, left: 44 };
    svg.innerHTML = "";

    if (!history.length) {
        return;
    }

    const stages = [
        ["scan_duration_s", "#2155d6"],
        ["window_build_duration_s", "#c58f12"],
        ["train_duration_s", "#111111"],
        ["checkpoint_duration_s", "#1a9b53"],
        ["prune_duration_s", "#b04f73"],
    ];
    const totals = history.map((item) => stages.reduce((sum, [key]) => sum + Number(item[key] || 0), 0));
    const maxTotal = Math.max(1, ...totals);
    const usableWidth = width - padding.left - padding.right;
    const usableHeight = height - padding.top - padding.bottom;
    const barWidth = usableWidth / Math.max(history.length, 1) - 4;

    const bars = history.map((item, index) => {
        let stackHeight = 0;
        const x = padding.left + index * (barWidth + 4);
        return stages.map(([key, color]) => {
            const value = Number(item[key] || 0);
            const segmentHeight = (value / maxTotal) * usableHeight;
            const y = height - padding.bottom - stackHeight - segmentHeight;
            stackHeight += segmentHeight;
            return `<rect x="${x}" y="${y}" width="${Math.max(2, barWidth)}" height="${Math.max(0, segmentHeight)}" rx="3" fill="${color}" opacity="${item.is_live ? 0.65 : 0.95}" />`;
        }).join("");
    }).join("");

    svg.innerHTML = axisMarkup(width, height, padding, 0, maxTotal, xLabels) + bars;
}

function cycleLabelForIndex(state, recent, index) {
    const hasLive = Boolean(recent[recent.length - 1]?.is_live);
    if (hasLive && index === recent.length - 1) {
        return "Live";
    }
    const completedCount = recent.length - (hasLive ? 1 : 0);
    const startCycle = Math.max(1, Number(state.training_cycles || completedCount) - completedCount + 1);
    return `Cycle ${startCycle + index}`;
}

function renderHistory(state, history) {
    const recent = normalizeHistory(history.slice(-24));
    const xLabels = recent.length
        ? [cycleLabelForIndex(state, recent, 0), cycleLabelForIndex(state, recent, recent.length - 1)]
        : ["Oldest", "Latest"];
    renderMultiLineChart("loss-chart", [
        { values: recent.map((item) => Number(item.loss || 0)), color: "#2155d6" },
        { values: recent.map((item) => Number(item.policy_loss || 0)), color: "#111111" },
        { values: recent.map((item) => Number(item.value_loss || 0)), color: "#1a9b53" },
    ], xLabels);
    renderStackedDurationChart("duration-chart", recent, xLabels);
    renderMultiLineChart("throughput-chart", [
        { values: recent.map((item) => Number(item.samples_per_s || 0)), color: "#2155d6" },
    ], xLabels);
    renderMultiLineChart("batch-chart", [
        { values: recent.map((item) => Number(item.avg_batch_time_ms || 0)), color: "#111111" },
    ], xLabels);

    attachTooltip("loss-chart", recent.length, (index) => {
        const item = recent[index];
        return `
            <strong>${cycleLabelForIndex(state, recent, index)}</strong>
            Total Loss: ${fmtLoss(item.loss)}<br>
            Policy Loss: ${fmtLoss(item.policy_loss)}<br>
            Value Loss: ${fmtLoss(item.value_loss)}
        `;
    });
    attachTooltip("duration-chart", recent.length, (index) => {
        const item = recent[index];
        return `
            <strong>${cycleLabelForIndex(state, recent, index)}</strong>
            Scan: ${fmtDuration(item.scan_duration_s)}<br>
            Window: ${fmtDuration(item.window_build_duration_s)}<br>
            Train: ${fmtDuration(item.train_duration_s)}<br>
            Checkpoint: ${fmtDuration(item.checkpoint_duration_s)}<br>
            Prune: ${fmtDuration(item.prune_duration_s)}
        `;
    });
    attachTooltip("throughput-chart", recent.length, (index) => {
        const item = recent[index];
        return `
            <strong>${cycleLabelForIndex(state, recent, index)}</strong>
            Samples / Sec: ${fmtInt(item.samples_per_s)}<br>
            Duration: ${fmtDuration(item.train_duration_s)}
        `;
    });
    attachTooltip("batch-chart", recent.length, (index) => {
        const item = recent[index];
        return `
            <strong>${cycleLabelForIndex(state, recent, index)}</strong>
            Avg Batch Time: ${fmtNumber(item.avg_batch_time_ms, 1)} ms<br>
            Samples / Sec: ${fmtInt(item.samples_per_s)}
        `;
    });
}

async function resetTraining() {
    const confirmed = window.confirm(
        "Reset the model, replay buffer, and training history? This will stop and restart the trainer automatically."
    );
    if (!confirmed) {
        return;
    }

    const response = await fetch("/api/training/reset", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
    });
    const payload = await response.json();
    if (!response.ok) {
        window.alert(payload.error || "Failed to reset training state.");
        return;
    }
    window.alert(
        `Reset complete. Removed ${payload.removed_replay_files} replay files and `
        + `${payload.removed_archive_models || 0} archived models. `
        + `Trainer restarted: ${payload.trainer_restarted ? "yes" : "no"}. `
        + `Arena restarted: ${payload.arena_restarted ? "yes" : "no"}.`
    );
    window.location.reload();
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
            fetchJson("/api/training_status"),
            fetchJson("/api/training_history"),
        ]);
        const liveHistory = buildLiveHistory(state, history);
        render(state);
        renderHistory(state, liveHistory);
    } catch (error) {
        console.error("Failed to update training dashboard:", error);
        renderLoadError(error.message);
    }
}

document.getElementById("reset-training")?.addEventListener("click", resetTraining);
update();
setInterval(update, 5000);
