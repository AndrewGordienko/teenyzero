import socket
import subprocess
import sys
from pathlib import Path

from teenyzero.visualizers.app import app


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    actor_process = None
    trainer_process = None

    if not _port_in_use(5002):
        run_actors = _project_root() / "scripts" / "run_actors.py"
        actor_process = subprocess.Popen(
            [sys.executable, str(run_actors)],
            cwd=str(_project_root()),
        )

    run_trainer = _project_root() / "scripts" / "run_trainer.py"
    trainer_process = subprocess.Popen(
        [sys.executable, str(run_trainer)],
        cwd=str(_project_root()),
    )

    try:
        app.run(debug=False, port=5001, host="0.0.0.0")
    finally:
        if actor_process is not None and actor_process.poll() is None:
            actor_process.terminate()
        if trainer_process is not None and trainer_process.poll() is None:
            trainer_process.terminate()
