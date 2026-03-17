from pathlib import Path

from flask import Flask, jsonify, render_template

DASHBOARDS_ROOT = Path(__file__).resolve().parents[1]

app = Flask(
    __name__,
    template_folder=str(DASHBOARDS_ROOT),
    static_folder=str(DASHBOARDS_ROOT),
    static_url_path="/static",
)
shared_stats = {}


@app.route("/")
def index():
    return render_template("cluster_monitor/dashboard.html")


@app.route("/api/stats")
def get_stats():
    normalized = {
        str(key): value
        for key, value in dict(shared_stats).items()
    }
    return jsonify(normalized)


def run_dashboard(stats_ref):
    global shared_stats
    shared_stats = stats_ref
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=False, use_reloader=False)
