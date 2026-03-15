from flask import Flask, render_template, jsonify
import time

app = Flask(__name__)
shared_stats = {} # Injected by run_actors.py

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/stats")
def get_stats():
    # Convert shared_stats to a regular dict for JSON serialization
    return jsonify(dict(shared_stats))

def run_dashboard(stats_ref):
    global shared_stats
    shared_stats = stats_ref
    # Running on 5002 to not clash with your chess play app
    app.run(host='0.0.0.0', port=5002, debug=False)