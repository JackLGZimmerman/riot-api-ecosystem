#!/usr/bin/env python3
"""Cancel all active Prefect runs for a deployment, then wait for concurrency to clear."""
from __future__ import annotations

import os
import sys
import time

import requests

DEFAULT_DEPLOYMENT = "riot-pipeline/riot-pipeline"
CANCEL_MESSAGE = "Cancelled during restart cleanup before launching a new deployment run."
MAX_WAIT_S = 60
POLL_INTERVAL_S = 2
ACTIVE_STATES = ["RUNNING", "PENDING", "SCHEDULED", "CANCELLING"]


def main(deployment_name: str) -> None:
    base = os.getenv("PREFECT_API_URL", "http://localhost:4200/api").rstrip("/")

    response = requests.get(f"{base}/deployments/name/{deployment_name}", timeout=30)
    if response.status_code == 404:
        print(f"Deployment {deployment_name!r} not found — nothing to cancel.")
        return
    response.raise_for_status()
    deployment_id = response.json()["id"]

    def fetch_active_runs() -> list[dict]:
        resp = requests.post(
            f"{base}/flow_runs/filter",
            json={"flow_runs": {"state": {"type": {"any_": ACTIVE_STATES}}}},
            timeout=60,
        )
        resp.raise_for_status()
        return [r for r in resp.json() if r.get("deployment_id") == deployment_id]

    def active_slots() -> int:
        resp = requests.get(f"{base}/deployments/{deployment_id}", timeout=30)
        resp.raise_for_status()
        return (resp.json().get("global_concurrency_limit") or {}).get("active_slots") or 0

    deadline = time.monotonic() + MAX_WAIT_S
    while True:
        for run in fetch_active_runs():
            resp = requests.post(
                f"{base}/flow_runs/{run['id']}/set_state",
                json={
                    "state": {"type": "CANCELLED", "name": "Cancelled", "message": CANCEL_MESSAGE},
                    "force": True,
                },
                timeout=60,
            )
            resp.raise_for_status()
            if resp.json().get("status") != "ACCEPT":
                sys.exit(f"Failed to cancel run {run['id']}: {resp.text}")

        if not fetch_active_runs() and active_slots() == 0:
            return

        if time.monotonic() >= deadline:
            sys.exit("Non-terminal Prefect runs remain after cleanup.")

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEPLOYMENT
    main(name)
