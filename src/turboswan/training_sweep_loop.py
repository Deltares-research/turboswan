"""Repeat a stepped turboswan.training_sweep until it finishes; all other flags pass through to it.

Local stand-in for the Airflow sensor, e.g.:
    python -m turboswan.training_sweep_loop --executor local-process --poll-interval 5
"""

import argparse
import time

from turboswan import training_sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--poll-interval", type=float, default=5, help="Seconds between steps (default: 5).")
    parser.add_argument(
        "--sweep-id",
        help="Sweep id passed to every step (default: a timestamp generated once for this loop).",
    )
    own, sweep_args = parser.parse_known_args()
    sweep_id = own.sweep_id or training_sweep.default_sweep_id()
    sweep_args += ["--sweep-id", sweep_id]
    print(f"sweep id: {sweep_id}")
    try:
        while (code := training_sweep.main(sweep_args)) == training_sweep.NOT_FINISHED:
            time.sleep(own.poll_interval)
    except KeyboardInterrupt:
        # trials are detached, so they would otherwise outlive this loop
        print(f"\nInterrupted; cancelling pending jobs (rerun with --sweep-id {sweep_id} to resume).")
        return training_sweep.main([*sweep_args, "--cancel"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
