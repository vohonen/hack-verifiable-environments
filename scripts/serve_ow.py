#!/usr/bin/env python
"""Serve a model on OpenWeights as a vLLM OpenAI-compatible API and hold it up until told to stop.

The openweights client is not in this repo's venv; use its own tool venv:

    OWPY="$(uv tool dir)/openweights/bin/python"
    $OWPY scripts/serve_ow.py --model Qwen/Qwen3-4B --endpoint-file /tmp/qwen3-4b.json &
    # wait for the endpoint file, then point calibrate.py at it:
    .venv/bin/python scripts/calibrate.py --base-url "$(jq -r .base_url /tmp/qwen3-4b.json)" \\
        --api-key "$(jq -r .api_key /tmp/qwen3-4b.json)" --model Qwen/Qwen3-4B ...
    touch /tmp/qwen3-4b.json.stop        # or SIGTERM: either tears the deployment down

Why not ``ow.api.deploy(model).up()``: its readiness probe goes through the openai client of
the openweights venv, which behind the Claude sandbox cannot reach ``*.proxy.runpod.net``
(curl and this repo's venv can), so ``up()`` retries for an hour and then cancels the job.
This script creates the same API job (same parameters, same job id, so a rerun attaches
to a live deployment instead of renting a second pod), waits for a worker, probes
readiness with curl, and refreshes the job's 15-minute lease itself once a minute.

The API job is cancelled on exit. OpenWeights also expires an API job 15 minutes after the
lease was last refreshed, so a killed shell costs at most that.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id; vLLM loads it with --dtype auto")
    p.add_argument("--max-model-len", type=int, default=8192, help="an hvta episode is under 2k prompt + 4k response")
    p.add_argument("--max-num-seqs", type=int, default=64, help="concurrent sequences; match calibrate.py --concurrency")
    p.add_argument("--hardware", default="1x H100S",
                   help="OpenWeights allowed_hardware entry; its GPU names are H100S (SXM), H100N (NVL), A100, A100S, H200")
    p.add_argument("--endpoint-file", required=True,
                   help="JSON {base_url, api_key, model, job_id, pod_id}, written once the API answers; <file>.stop ends the deployment")
    p.add_argument("--ready-timeout", type=int, default=1800, help="seconds to wait for a worker and the first completion")
    return p.parse_args()


def ready(base_url: str) -> bool:
    """curl, not the openai client: see the module docstring."""
    r = subprocess.run(["curl", "-sf", "-m", "10", f"{base_url}/models"], capture_output=True)
    return r.returncode == 0


def main() -> int:
    args = parse_args()
    if not os.environ.get("OPENWEIGHTS_API_KEY"):
        sys.exit("OPENWEIGHTS_API_KEY is unset (set -a; . ./.env; set +a)")
    from openweights import OpenWeights  # noqa: PLC0415 - only importable from the openweights tool venv

    def on_signal(signum, _frame):
        raise SystemExit(128 + signum)  # unwinds into the finally below

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    ow = OpenWeights()
    api = ow.api.deploy(args.model, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                        allowed_hardware=[args.hardware])
    job_id = api.job_id
    stop_file = args.endpoint_file + ".stop"
    stop = threading.Event()

    def refresh_lease() -> None:
        while not stop.is_set():
            try:
                until = datetime.now(timezone.utc) + timedelta(minutes=15)
                ow._supabase.table("jobs").update({"timeout": until.isoformat()}).eq("id", job_id).execute()
            except Exception as exc:  # noqa: BLE001 - a missed refresh is not fatal, the lease has slack
                print(f"lease refresh failed: {exc!r}", flush=True)
            stop.wait(60)

    threading.Thread(target=refresh_lease, daemon=True).start()
    print(f"api job {job_id} on {args.hardware}; waiting for a worker", flush=True)
    try:
        deadline = time.time() + args.ready_timeout
        pod_id = None
        while time.time() < deadline:
            job = ow.jobs.retrieve(job_id)
            if job.status in ("failed", "canceled"):
                sys.exit(f"api job {job_id} is {job.status}")
            if job.status == "in_progress" and job.worker_id:
                worker = ow._supabase.table("worker").select("pod_id").eq("id", job.worker_id).single().execute().data
                pod_id = worker["pod_id"]
                break
            time.sleep(5)
        if pod_id is None:
            sys.exit(f"no worker took api job {job_id} within {args.ready_timeout}s")
        base_url = f"https://{pod_id}-8000.proxy.runpod.net/v1"
        api_key = (job.params or {}).get("api_key", "api_key")
        print(f"pod {pod_id}; waiting for vLLM at {base_url}", flush=True)
        while not ready(base_url):
            if time.time() > deadline:
                sys.exit(f"vLLM at {base_url} did not answer within {args.ready_timeout}s")
            time.sleep(10)
        with open(args.endpoint_file, "w") as f:
            json.dump({"base_url": base_url, "api_key": api_key, "model": args.model, "job_id": job_id, "pod_id": pod_id}, f)
        print(f"ready: {base_url}  (touch {stop_file} to tear down)", flush=True)
        while not os.path.exists(stop_file):
            time.sleep(5)
        print("stop file seen", flush=True)
    finally:
        stop.set()
        print(f"cancelling api job {job_id}", flush=True)
        ow.jobs.cancel(job_id)
        for path in (args.endpoint_file, stop_file):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
