#!/usr/bin/env python3
"""
Synthetic data collector for future assistant training.
Generates mock inference/harness records only (no real data).
"""

import argparse
import json
import os
import random
import statistics
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Default model/device/precision sets (synthetic only)
DEFAULT_MODELS = [
    "ov/yolov8n",
    "ov/clip-vit-b32",
    "ov/bert-base-uncased",
]
DEFAULT_DEVICES = ["CPU", "GPU", "VPU"]
DEFAULT_PRECISIONS = ["FP32", "FP16", "INT8"]


def require_synthetic_only() -> None:
    if os.getenv("SYNTHETIC_ONLY") != "1":
        print("Refusing to collect: set SYNTHETIC_ONLY=1 for synthetic-only data.", file=sys.stderr)
        sys.exit(1)


def ensure_dirs(output_root: Path, date_str: str) -> Dict[str, Path]:
    raw_dir = output_root / "raw" / date_str
    summary_dir = output_root / "summary" / date_str
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "summary": summary_dir}


def synthetic_record(model: str, device: str, precision: str, seed: int) -> Dict:
    random.seed(seed)
    latency = round(random.uniform(5.0, 45.0), 3)
    status = "success" if random.random() > 0.05 else "error"
    errors: List[str] = []
    if status == "error":
        errors.append(random.choice(["Timeout", "SyntheticValidationError", "ResourceExhausted"]))

    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "inference",
        "model_name": model,
        "device": device,
        "precision": precision,
        "input_signature": {"shape": [1, 3, 224, 224], "dtype": "float32", "channels": 3},
        "synthetic_input_seed": seed,
        "synthetic_input_params": {"pattern": "noise", "scale": round(random.random(), 3)},
        "preproc_steps": ["resize", "normalize"],
        "postproc_steps": ["decode", "argmax"],
        "latency_ms": latency,
        "errors": errors,
        "status": status,
        "notes": "synthetic-only; no real inputs stored",
    }


def write_ndjson(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def write_metrics(path: Path, records: List[Dict]) -> None:
    latencies = [r["latency_ms"] for r in records if r["status"] == "success"]
    errors = sum(1 for r in records if r["status"] == "error")
    total = len(records)
    p50 = statistics.median(latencies) if latencies else 0.0
    p90 = statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else p50
    p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else p90

    lines = [
        "metric,value",
        f"total_records,{total}",
        f"errors,{errors}",
        f"latency_p50_ms,{p50:.3f}",
        f"latency_p90_ms,{p90:.3f}",
        f"latency_p99_ms,{p99:.3f}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect synthetic inference data (no real data).")
    parser.add_argument("--output-root", type=Path, default=Path("data/training"), help="Base output directory.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model names to simulate.")
    parser.add_argument("--devices", nargs="+", default=DEFAULT_DEVICES, help="Devices to simulate.")
    parser.add_argument("--precisions", nargs="+", default=DEFAULT_PRECISIONS, help="Precisions to simulate.")
    parser.add_argument("--samples", type=int, default=20, help="Samples per (model,device,precision).")
    parser.add_argument("--seed", type=int, default=1337, help="Global RNG seed.")
    return parser.parse_args()


def main() -> None:
    require_synthetic_only()
    args = parse_args()
    random.seed(args.seed)

    date_str = datetime.utcnow().strftime("%Y%m%d")
    dirs = ensure_dirs(args.output_root, date_str)

    all_records: List[Dict] = []
    for model in args.models:
        for device in args.devices:
            for precision in args.precisions:
                for i in range(args.samples):
                    seed = hash((args.seed, model, device, precision, i)) % (2**31)
                    rec = synthetic_record(model, device, precision, seed)
                    all_records.append(rec)

    # Write NDJSON per device/precision
    for device in args.devices:
        for precision in args.precisions:
            subset = [r for r in all_records if r["device"] == device and r["precision"] == precision]
            out_file = dirs["raw"] / f"inference_{device}_{precision}.ndjson"
            write_ndjson(out_file, subset)

    metrics_file = dirs["summary"] / "metrics.csv"
    write_metrics(metrics_file, all_records)

    print(f"[+] Wrote {len(all_records)} synthetic records to {dirs['raw']}")
    print(f"[+] Metrics: {metrics_file}")


if __name__ == "__main__":
    main()

