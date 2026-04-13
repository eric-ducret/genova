"""
client.py — CLI client for submitting variant analysis jobs from the login node.

The pipeline runs on the compute node (via SLURM + launch.sh).
This script sends the CSV and phenotype to the FastAPI server running on that node
and streams the result back as plain text.

Usage:
    python client.py \\
        --server http://<node>:8000 \\
        --csv variants.csv \\
        --phenotype "Obesity, intellectual disability, hyperphagia" \\
        --output report.txt
"""

import argparse
import sys

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit a variant analysis job to the pipeline server."
    )
    p.add_argument(
        "--server",
        required=True,
        metavar="URL",
        help="Base URL of the FastAPI server, e.g. http://node001:8000",
    )
    p.add_argument(
        "--csv",
        required=True,
        metavar="FILE",
        help="Path to the variants CSV (or .xlsx) file.",
    )
    p.add_argument(
        "--phenotype",
        required=True,
        metavar="TEXT",
        help='Free-text patient phenotype, e.g. "Obesity, intellectual disability"',
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Write result to this file (default: print to stdout).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Request timeout in seconds (default: 3600 = 1 hour).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Health check ──────────────────────────────────────────────────────────
    health_url = args.server.rstrip("/") + "/health"
    try:
        r = requests.get(health_url, timeout=10)
        r.raise_for_status()
        print(f"[client] Server OK — {r.json()}", file=sys.stderr)
    except Exception as e:
        print(f"[client] ERROR: server not reachable at {health_url}: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Submit job ────────────────────────────────────────────────────────────
    analyze_url = args.server.rstrip("/") + "/analyze"
    print(f"[client] Submitting {args.csv} to {analyze_url} ...", file=sys.stderr)

    try:
        with open(args.csv, "rb") as f:
            response = requests.post(
                analyze_url,
                files={"csv_file": (args.csv, f)},
                data={"patient_report": args.phenotype},
                timeout=args.timeout,
            )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print(
            f"[client] ERROR: request timed out after {args.timeout}s. "
            "Increase --timeout or check server logs.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[client] ERROR: HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[client] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    result = response.text

    # ── Output ────────────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[client] Result saved to {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
