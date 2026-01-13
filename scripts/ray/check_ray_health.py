#!/usr/bin/env python3
"""Check Ray cluster health and print status.

This script can be used to:
1. Perform a one-time health check of the Ray cluster
2. Run continuous monitoring with periodic checks
3. Integrate into sbatch scripts for health monitoring during jobs

Usage:
    # One-time check
    python check_ray_health.py --address 10.0.0.1:6379 --expected-nodes 2 --expected-gpus 8

    # Continuous monitoring (check every 60 seconds)
    python check_ray_health.py --address 10.0.0.1:6379 --continuous --interval 60

    # Exit on first failure (useful for background monitoring)
    python check_ray_health.py --address 10.0.0.1:6379 --continuous --exit-on-failure
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime


def log(msg: str) -> None:
    """Print with timestamp and immediate flush."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def check_health(
    address: str,
    expected_nodes: int = 1,
    expected_gpus: float = 0.0,
    verbose: bool = False,
) -> bool:
    """Check Ray cluster health and return True if healthy.

    Args:
        address: Ray head address (e.g., "10.0.0.1:6379")
        expected_nodes: Minimum number of nodes expected
        expected_gpus: Minimum number of GPUs expected
        verbose: Print detailed resource information

    Returns:
        True if cluster meets expectations, False otherwise
    """
    try:
        import ray
    except ImportError:
        log("ERROR: ray package not installed")
        return False

    try:
        ray.init(address=address, ignore_reinit_error=True)

        # Get cluster resources and nodes
        resources = ray.cluster_resources()
        nodes = ray.nodes()

        # Count alive nodes
        alive_nodes = [n for n in nodes if n.get("Alive", False)]
        dead_nodes = [n for n in nodes if not n.get("Alive", False)]

        # Get resource counts
        available_gpus = resources.get("GPU", 0.0)
        available_cpus = resources.get("CPU", 0.0)

        # Determine health status
        nodes_ok = len(alive_nodes) >= expected_nodes
        gpus_ok = available_gpus >= expected_gpus
        healthy = nodes_ok and gpus_ok

        # Build status message
        status_icon = "✓" if healthy else "✗"
        status_msg = (
            f"[Ray Health] {status_icon} "
            f"nodes={len(alive_nodes)}/{expected_nodes} "
            f"GPUs={available_gpus}/{expected_gpus} "
            f"CPUs={available_cpus}"
        )

        if dead_nodes:
            status_msg += f" (dead_nodes={len(dead_nodes)})"

        log(status_msg)

        if verbose:
            log(f"  Full resources: {resources}")
            for i, node in enumerate(alive_nodes):
                node_id = node.get("NodeID", "unknown")[:8]
                node_resources = node.get("Resources", {})
                log(f"  Node {i}: {node_id} resources={node_resources}")

            if dead_nodes:
                log("  Dead nodes:")
                for node in dead_nodes:
                    node_id = node.get("NodeID", "unknown")[:8]
                    log(f"    - {node_id}")

        if not healthy:
            if not nodes_ok:
                log(f"  WARNING: Expected {expected_nodes} nodes, got {len(alive_nodes)}")
            if not gpus_ok:
                log(f"  WARNING: Expected {expected_gpus} GPUs, got {available_gpus}")

        return healthy

    except Exception as e:
        log(f"ERROR: Failed to connect to Ray at {address}: {e}")
        return False
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


def run_continuous_monitoring(
    address: str,
    expected_nodes: int,
    expected_gpus: float,
    interval: int,
    exit_on_failure: bool,
    verbose: bool,
) -> None:
    """Run continuous health monitoring.

    Args:
        address: Ray head address
        expected_nodes: Minimum number of nodes expected
        expected_gpus: Minimum number of GPUs expected
        interval: Seconds between health checks
        exit_on_failure: Exit with code 1 on first failure
        verbose: Print detailed information
    """
    log(f"Starting continuous Ray health monitoring (interval={interval}s)")
    log(f"  Address: {address}")
    log(f"  Expected: {expected_nodes} nodes, {expected_gpus} GPUs")
    log(f"  Exit on failure: {exit_on_failure}")

    failure_count = 0
    check_count = 0

    try:
        while True:
            check_count += 1
            healthy = check_health(
                address=address,
                expected_nodes=expected_nodes,
                expected_gpus=expected_gpus,
                verbose=verbose,
            )

            if not healthy:
                failure_count += 1
                log(f"  Health check #{check_count} FAILED (total failures: {failure_count})")

                if exit_on_failure:
                    log("Exiting due to --exit-on-failure flag")
                    sys.exit(1)

            time.sleep(interval)

    except KeyboardInterrupt:
        log(f"Monitoring stopped. Total checks: {check_count}, failures: {failure_count}")
        sys.exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Ray cluster health status.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--address",
        required=True,
        help="Ray head address (e.g., 10.0.0.1:6379)",
    )
    parser.add_argument(
        "--expected-nodes",
        type=int,
        default=1,
        help="Minimum number of nodes expected (default: 1)",
    )
    parser.add_argument(
        "--expected-gpus",
        type=float,
        default=0.0,
        help="Minimum number of GPUs expected (default: 0)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous monitoring instead of one-time check",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between checks in continuous mode (default: 60)",
    )
    parser.add_argument(
        "--exit-on-failure",
        action="store_true",
        help="In continuous mode, exit with code 1 on first failure",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed resource and node information",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.continuous:
        run_continuous_monitoring(
            address=args.address,
            expected_nodes=args.expected_nodes,
            expected_gpus=args.expected_gpus,
            interval=args.interval,
            exit_on_failure=args.exit_on_failure,
            verbose=args.verbose,
        )
    else:
        healthy = check_health(
            address=args.address,
            expected_nodes=args.expected_nodes,
            expected_gpus=args.expected_gpus,
            verbose=args.verbose,
        )
        sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
