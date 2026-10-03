"""Production unified runner for Black Box AI Agent Observability & Debugging System.

Runs both FastAPI backend and Streamlit dashboard concurrently in production,
handling signals, port binding, and process lifecycles cleanly.
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Black Box Production Runner")
    parser.add_argument("--mode", choices=["all", "ui", "api"], default="all", help="Service mode to start")
    args = parser.parse_args()

    port = os.getenv("PORT", "8501")
    api_port = os.getenv("API_PORT", "8000")

    processes = []

    def terminate_all(signum=None, frame=None):
        print("\n[Black Box] Shutting down production services...")
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, terminate_all)
    signal.signal(signal.SIGTERM, terminate_all)

    # 1. Start FastAPI API Server
    if args.mode in ("all", "api"):
        print(f"[Black Box] Starting FastAPI API server on port {api_port}...")
        api_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(api_port),
            "--log-level",
            "info",
        ]
        api_proc = subprocess.Popen(api_cmd)
        processes.append(api_proc)

    # Allow FastAPI to initialize before Streamlit checks
    time.sleep(1.5)

    # 2. Start Streamlit Dashboard
    if args.mode in ("all", "ui"):
        print(f"[Black Box] Starting Streamlit Dashboard on port {port}...")
        ui_cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(port),
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]
        ui_proc = subprocess.Popen(ui_cmd)
        processes.append(ui_proc)

    print("[Black Box] Production services running. Press Ctrl+C to terminate.")
    try:
        # Keep runner alive and monitor subprocesses
        while True:
            for p in processes:
                code = p.poll()
                if code is not None:
                    print(f"[Black Box] Process {p.args} exited with code {code}")
                    terminate_all()
            time.sleep(1)
    except KeyboardInterrupt:
        terminate_all()


if __name__ == "__main__":
    main()
