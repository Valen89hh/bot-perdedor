import os
import sys
import json

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import CONFIG

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

RESULTS_DIR = CONFIG["results_dir"]


def _load_results(dataset="test"):
    path = os.path.join(RESULTS_DIR, f"evaluation_{dataset}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/candles/<dataset>")
def candles(dataset):
    results = _load_results(dataset)
    if not results:
        return jsonify({"error": "No results found. Run evaluate first."}), 404
    return jsonify(results["prices"])


@app.route("/api/trades/<dataset>")
def trades(dataset):
    results = _load_results(dataset)
    if not results:
        return jsonify({"error": "No results found."}), 404
    return jsonify(results["trade_history"])


@app.route("/api/portfolio/<dataset>")
def portfolio(dataset):
    results = _load_results(dataset)
    if not results:
        return jsonify({"error": "No results found."}), 404
    return jsonify({
        "values": results["portfolio_history"],
        "timestamps": results["timestamps"][:len(results["portfolio_history"])],
    })


@app.route("/api/metrics/<dataset>")
def metrics(dataset):
    results = _load_results(dataset)
    if not results:
        return jsonify({"error": "No results found."}), 404
    return jsonify({
        "metrics": results["metrics"],
        "buy_hold_return_pct": results["buy_hold_return_pct"],
    })


@app.route("/api/datasets")
def datasets():
    available = []
    for name in ["train", "val", "test"]:
        path = os.path.join(RESULTS_DIR, f"evaluation_{name}.json")
        if os.path.exists(path):
            available.append(name)
    return jsonify(available)


def run_server(host=None, port=None):
    host = host or CONFIG["dashboard_host"]
    port = port or CONFIG["dashboard_port"]
    print(f"\nDashboard: http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    run_server()
