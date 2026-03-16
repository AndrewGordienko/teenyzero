from flask import Flask, render_template, jsonify

app = Flask(__name__, static_folder="static", template_folder="templates")
shared_stats = {} # Injected by run_actors.py

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/stats")
def get_stats():
    # Manager dict keys are mixed ints/strings, which breaks Flask JSON sorting.
    normalized = {
        str(key): value
        for key, value in dict(shared_stats).items()
    }
    return jsonify(normalized)

def run_dashboard(stats_ref):
    global shared_stats
    shared_stats = stats_ref
    # Running on 5002 to not clash with your chess play app
    app.run(host='0.0.0.0', port=5002, debug=False)
