const timingKeys = [
    ["selection", "Selection"],
    ["leaf_eval", "Leaf Eval"],
    ["inference_wait", "Inference Wait"],
    ["inference_forward", "Model Forward"],
    ["policy_mask", "Policy Mask"],
    ["encode", "Board Encode"],
    ["backprop", "Backprop"],
];

const pieceMap = {
    p: "♟", r: "♜", n: "♞", b: "♝", q: "♛", k: "♚",
    P: "♙", R: "♖", N: "♘", B: "♗", Q: "♕", K: "♔",
};

const flippedWorkers = new Set();
const throughputSamples = [];

function fmtNumber(value, digits = 1) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? num.toFixed(digits) : "0.0";
}

function fmtInt(value) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? Math.round(num).toLocaleString() : "0";
}

function pct(value) {
    return `${fmtNumber((value || 0) * 100, 1)}%`;
}

function safeWorkers(data) {
    const sortParts = (id) => String(id).split(":").map((part) => {
        const value = Number(part);
        return Number.isFinite(value) ? value : 0;
    });
    const compareIds = (left, right) => {
        const leftParts = sortParts(left);
        const rightParts = sortParts(right);
        const limit = Math.max(leftParts.length, rightParts.length);
        for (let index = 0; index < limit; index += 1) {
            const delta = (leftParts[index] || 0) - (rightParts[index] || 0);
            if (delta !== 0) return delta;
        }
        return String(left).localeCompare(String(right));
    };
    return Object.entries(data)
        .filter(([key]) => key !== "__cluster__")
        .map(([id, stats]) => ({ id, ...stats }))
        .sort((a, b) => compareIds(a.id, b.id));
}

function aggregateWorkers(workers) {
    const empty = {
        avgMs: Object.fromEntries(timingKeys.map(([key]) => [key, 0])),
        avgBatchMs: Object.fromEntries([["total", 0], ...timingKeys.map(([key]) => [key, 0])]),
        avgTotalMs: 0,
        avgSlotPlyIntervalMs: 0,
        avgSimulations: 0,
        avgTotalSimulations: 0,
        avgCacheHitRate: 0,
        totalGames: 0,
        totalPositions: 0,
    };

    if (!workers.length) return empty;

    for (const worker of workers) {
        const avg = worker.search?.avg_ms || {};
        for (const [key] of timingKeys) {
            empty.avgMs[key] += Number(avg[key] || 0);
            empty.avgBatchMs[key] += Number(worker.search?.batch_ms?.[key] || 0);
        }
        empty.avgTotalMs += Number(avg.total || 0);
        empty.avgBatchMs.total += Number(worker.search?.batch_ms?.total || 0);
        empty.avgSlotPlyIntervalMs += Number(worker.search?.slot_ply_interval_ms || 0);
        empty.avgSimulations += Number(worker.search?.avg_simulations || 0);
        empty.avgTotalSimulations += Number(worker.search?.simulations_completed_total || 0);
        empty.avgCacheHitRate += Number(worker.search?.avg_cache_hit_rate || 0);
        empty.totalGames += Number(worker.total_games || 0);
        empty.totalPositions += Number(worker.total_positions_saved || 0);
    }

    const n = workers.length;
    for (const [key] of timingKeys) {
        empty.avgMs[key] /= n;
    }
    empty.avgTotalMs /= n;
    empty.avgBatchMs.total /= n;
    empty.avgSlotPlyIntervalMs /= n;
    empty.avgSimulations /= n;
    empty.avgTotalSimulations /= n;
    empty.avgCacheHitRate /= n;
    return empty;
}

function timingWithCluster(aggregate, cluster) {
    const combined = { ...aggregate.avgMs };
    const inference = cluster.inference || {};
    const clusterForward = Number(inference.avg_forward_ms || 0);

    if (clusterForward > combined.inference_forward) {
        combined.inference_forward = clusterForward;
    }

    return combined;
}

function aggregateClusterTotals(cluster) {
    const totals = cluster.totals || {};
    const combined = {
        games: 0,
        positions: 0,
        gameTimeMs: 0,
        whiteWins: 0,
        draws: 0,
        blackWins: 0,
    };

    for (const value of Object.values(totals)) {
        combined.games += Number(value.games || 0);
        combined.positions += Number(value.positions || 0);
        combined.gameTimeMs += Number(value.game_time_ms || 0);
        combined.whiteWins += Number(value.white_wins || 0);
        combined.draws += Number(value.draws || 0);
        combined.blackWins += Number(value.black_wins || 0);
    }

    return combined;
}

function positionsPerSecond(clusterTotals) {
    const now = Date.now();
    throughputSamples.push({
        ts: now,
        positions: Number(clusterTotals.positions || 0),
    });

    while (throughputSamples.length > 1 && (now - throughputSamples[0].ts) > 15000) {
        throughputSamples.shift();
    }

    if (throughputSamples.length < 2) return 0;

    const oldest = throughputSamples[0];
    const newest = throughputSamples[throughputSamples.length - 1];
    const dtMs = Math.max(1, newest.ts - oldest.ts);
    const dPositions = Math.max(0, newest.positions - oldest.positions);
    return (dPositions * 1000.0) / dtMs;
}

function renderHeroResults(clusterTotals) {
    const target = document.getElementById("hero-results");
    target.innerHTML = `
        <div class="result-item">
            <span class="result-label">Games</span>
            <span class="result-value">${fmtInt(clusterTotals.games)}</span>
        </div>
        <div class="result-item">
            <span class="result-label">Positions</span>
            <span class="result-value">${fmtInt(clusterTotals.positions)}</span>
        </div>
        <div class="result-item">
            <span class="result-label">White Wins</span>
            <span class="result-value">${fmtInt(clusterTotals.whiteWins)}</span>
        </div>
        <div class="result-item">
            <span class="result-label">Draws</span>
            <span class="result-value">${fmtInt(clusterTotals.draws)}</span>
        </div>
        <div class="result-item">
            <span class="result-label">Black Wins</span>
            <span class="result-value">${fmtInt(clusterTotals.blackWins)}</span>
        </div>
    `;
}

function renderSummary(workers, cluster, aggregate, combinedTiming, clusterTotals) {
    const target = document.getElementById("summary");
    const actorMode = String(cluster.config?.actor_mode || "");
    const moveNote = actorMode === "inprocess"
        ? "normalized batch time; not wall-clock slot progress"
        : "average end-to-end move time";
    const livePositionsPerSecond = positionsPerSecond(clusterTotals);
    const cards = [
        {
            label: "Move Total",
            value: `${fmtNumber(aggregate.avgTotalMs)} ms`,
            note: moveNote,
        },
        {
            label: "Slot Ply Interval",
            value: `${fmtNumber(actorMode === "inprocess" ? aggregate.avgSlotPlyIntervalMs : aggregate.avgTotalMs)} ms`,
            note: actorMode === "inprocess"
                ? "actual wall-clock time for one slot to advance one ply"
                : "same as move total outside in-process batching",
        },
        {
            label: "Leaf Eval",
            value: `${fmtNumber(combinedTiming.leaf_eval)} ms`,
            note: "largest MCTS stage",
        },
        {
            label: "Inference Wait",
            value: `${fmtNumber(combinedTiming.inference_wait)} ms`,
            note: "worker waiting on server",
        },
        {
            label: "Model Forward",
            value: `${fmtNumber(combinedTiming.inference_forward)} ms`,
            note: "network forward pass",
        },
        {
            label: "Selection",
            value: `${fmtNumber(combinedTiming.selection)} ms`,
            note: "tree traversal time",
        },
        {
            label: "Game Time",
            value: `${fmtNumber(clusterTotals.games > 0 ? (clusterTotals.gameTimeMs / clusterTotals.games) / 1000.0 : 0)} s`,
            note: "average time to play a game",
        },
        {
            label: "Positions / Sec",
            value: fmtNumber(livePositionsPerSecond),
            note: "live saved-position throughput over the last 15 seconds",
        },
    ];

    target.innerHTML = cards.map((card) => `
        <article class="metric-card">
            <span class="metric-label">${card.label}</span>
            <div class="metric-value">${card.value}</div>
            <div class="metric-note">${card.note}</div>
        </article>
    `).join("");
}

function renderBoard(fen) {
    const boardPart = (fen || "").split(" ")[0] || "8/8/8/8/8/8/8/8";
    const squares = [];
    const rows = boardPart.split("/");

    rows.forEach((row, rankIndex) => {
        let fileIndex = 0;
        for (const char of row) {
            if (Number.isInteger(Number(char)) && char !== "0") {
                for (let i = 0; i < Number(char); i += 1) {
                    const isLight = (rankIndex + fileIndex) % 2 === 0;
                    squares.push(`<div class="square ${isLight ? "light" : "dark"}"></div>`);
                    fileIndex += 1;
                }
            } else {
                const isLight = (rankIndex + fileIndex) % 2 === 0;
                const pieceClass = char === char.toUpperCase() ? "white" : "black";
                squares.push(
                    `<div class="square ${isLight ? "light" : "dark"}"><span class="piece ${pieceClass}">${pieceMap[char] || ""}</span></div>`,
                );
                fileIndex += 1;
            }
        }
    });

    return squares.join("");
}

function workerFrontMarkup(worker) {
    const search = worker.search || {};
    const last = search.last_ms || {};
    const isSlot = String(worker.id).includes(":");
    const slotId = isSlot ? String(worker.id).split(":").slice(-1)[0] : worker.id;
    const label = isSlot ? `Slot ${slotId}` : `Worker ${worker.id}`;
    return `
        <div class="worker-top">
            <div class="worker-meta">
                <div class="worker-id-tag">${label}</div>
            </div>
        </div>

        <div class="kv">
            <div class="kv-item"><span class="k">Move Count</span><span class="v">${fmtInt(worker.move_count)}</span></div>
            <div class="kv-item"><span class="k">Turn Number</span><span class="v">${fmtInt(worker.turn_number)}</span></div>
            <div class="kv-item"><span class="k">Games Played</span><span class="v">${fmtInt(worker.total_games)}</span></div>
            <div class="kv-item"><span class="k">Positions Saved</span><span class="v">${fmtInt(worker.total_positions_saved)}</span></div>
            <div class="kv-item"><span class="k">Avg Move Time</span><span class="v">${fmtNumber(search.avg_ms?.total || 0)} ms</span></div>
            <div class="kv-item"><span class="k">Last Move Time</span><span class="v">${fmtNumber(last.total || 0)} ms</span></div>
            <div class="kv-item"><span class="k">Slot Ply Interval</span><span class="v">${fmtNumber(search.slot_ply_interval_ms || last.total || 0)} ms</span></div>
            <div class="kv-item"><span class="k">Batch Active Games</span><span class="v">${fmtInt(search.batch_active_games || 1)}</span></div>
            <div class="kv-item"><span class="k">Batch Simulations</span><span class="v">${fmtInt(search.simulations_completed_total || 0)}</span></div>
            <div class="kv-item"><span class="k">Inference Wait</span><span class="v">${fmtNumber(search.avg_ms?.inference_wait || 0)} ms</span></div>
            <div class="kv-item"><span class="k">Model Forward</span><span class="v">${fmtNumber(search.avg_ms?.inference_forward || 0)} ms</span></div>
            <div class="kv-item"><span class="k">Legal Moves</span><span class="v">${fmtInt(worker.legal_moves)}</span></div>
            <div class="kv-item"><span class="k">Leaf Eval</span><span class="v">${fmtNumber(search.avg_ms?.leaf_eval || 0)} ms</span></div>
            <div class="kv-item"><span class="k">Avg Simulations</span><span class="v">${fmtNumber(search.avg_simulations || 0, 1)}</span></div>
            <div class="kv-item"><span class="k">Cache Hit Rate</span><span class="v">${pct(search.avg_cache_hit_rate || 0)}</span></div>
        </div>

        <div class="fen mono">${worker.fen || "Waiting for FEN..."}</div>
    `;
}

function workerBackMarkup(worker) {
    const isSlot = String(worker.id).includes(":");
    const slotId = isSlot ? String(worker.id).split(":").slice(-1)[0] : worker.id;
    const label = isSlot ? `Slot ${slotId}` : `Worker ${worker.id}`;
    return `
        <div class="worker-top">
            <div class="worker-meta">
                <div class="worker-id-tag">${label}</div>
            </div>
        </div>

        <div class="board">${renderBoard(worker.fen)}</div>
    `;
}

function createWorkerCard(worker) {
    const wrapper = document.createElement("div");
    wrapper.className = "worker-flip";

    const card = document.createElement("article");
    card.className = "worker-card";
    card.dataset.workerId = worker.id;
    if (flippedWorkers.has(worker.id)) {
        card.classList.add("is-flipped");
    }

    const front = document.createElement("section");
    front.className = "worker-face panel front";
    front.innerHTML = workerFrontMarkup(worker);

    const back = document.createElement("section");
    back.className = "worker-face panel back";
    back.innerHTML = workerBackMarkup(worker);

    card.appendChild(front);
    card.appendChild(back);
    card.addEventListener("click", () => {
        const workerId = card.dataset.workerId;
        if (flippedWorkers.has(workerId)) {
            flippedWorkers.delete(workerId);
        } else {
            flippedWorkers.add(workerId);
        }
        card.classList.toggle("is-flipped");
    });
    wrapper.appendChild(card);
    return wrapper;
}

function renderWorkers(workers) {
    const target = document.getElementById("workers");
    if (!workers.length) {
        target.innerHTML = `<article class="panel"><div class="metric-note">Cluster monitor is running, but no self-play workers are publishing telemetry. Start scripts/run_actors.py to populate this view.</div></article>`;
        return;
    }

    const existing = new Map(
        Array.from(target.querySelectorAll(".worker-card")).map((card) => [card.dataset.workerId, card.parentElement]),
    );

    target.innerHTML = "";
    for (const worker of workers) {
        const existingWrapper = existing.get(worker.id);
        if (existingWrapper) {
            const card = existingWrapper.querySelector(".worker-card");
            const front = existingWrapper.querySelector(".front");
            const back = existingWrapper.querySelector(".back");
            front.innerHTML = workerFrontMarkup(worker);
            back.innerHTML = workerBackMarkup(worker);
            card.classList.toggle("is-flipped", flippedWorkers.has(worker.id));
            target.appendChild(existingWrapper);
            existing.delete(worker.id);
        } else {
            target.appendChild(createWorkerCard(worker));
        }
    }
}

async function update() {
    try {
        const response = await fetch("/api/stats");
        const data = await response.json();
        const workers = safeWorkers(data);
        const cluster = data.__cluster__ || {};
        const aggregate = aggregateWorkers(workers);
        const combinedTiming = timingWithCluster(aggregate, cluster);
        const clusterTotals = aggregateClusterTotals(cluster);

        renderHeroResults(clusterTotals);
        renderSummary(workers, cluster, aggregate, combinedTiming, clusterTotals);
        renderWorkers(workers);
    } catch (error) {
        const target = document.getElementById("hero-results");
        if (target) {
            target.innerHTML = "";
        }
    }
}

update();
setInterval(update, 1000);
