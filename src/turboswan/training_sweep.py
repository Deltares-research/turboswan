"""Run a TurboSWAN parameter sweep from a sweep TOML, tracked as one MLflow parent run.

local and ray run the whole sweep in this process. local-process and slurm run 
one step per call, so rerunning with the same --sweep-id resumes it;
the exit code is 0 once the sweep is finished and NOT_FINISHED while it is still
running (use training_sweep_loop to repeat the call locally).
"""

import argparse
import datetime
import logging
import pathlib
import re

from turboswan import training
from turboswan.mlops.sweeps.blocking.driver import run_sweep
from turboswan.mlops.sweeps.blocking.executors import RayExecutor, SequentialExecutor
from turboswan.mlops.sweeps.configuration import load_sweep
from turboswan.mlops.sweeps.stepped.driver import cancel_sweep, step_sweep
from turboswan.mlops.sweeps.stepped.executors import LocalProcessExecutor, SlurmExecutor

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
SWEEPS_DIR = PROJECT_DIR / "configs" / "sweeps"

DEFAULT_SWEEP_CONFIG = {
    "local": SWEEPS_DIR / "demo.toml",
    "ray": SWEEPS_DIR / "demo_optuna.toml",
    "slurm": SWEEPS_DIR / "demo_stepped.toml",
    "local-process": SWEEPS_DIR / "demo_stepped.toml",
}
STEPPED_EXECUTORS = ("slurm", "local-process")
NOT_FINISHED = 75  # EX_TEMPFAIL; keeps an unfinished sweep distinct from a crash (exit 1)

# Flags that only make sense for some executors; rejected elsewhere instead of ignored.
EXECUTOR_FLAGS = {
    "cpus_per_trial": ("--cpus-per-trial", ("ray",)),
    "gpus_per_trial": ("--gpus-per-trial", ("ray",)),
    "slurm_host": ("--slurm-host", STEPPED_EXECUTORS),
    "state_path": ("--state-path", STEPPED_EXECUTORS),
}


def default_sweep_id() -> str:
    return f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executor",
        choices=tuple(DEFAULT_SWEEP_CONFIG),
        default="local",
        help="Where trials run; ray uses the active Ray cluster, the slurm executors step "
        "a resumable state file (default: %(default)s).",
    )
    parser.add_argument(
        "--sweep-config",
        type=pathlib.Path,
        help="Path to the sweep TOML file (default by --executor, from configs/sweeps/: "
        "local demo, ray demo_optuna, stepped executors demo_stepped).",
    )
    parser.add_argument(
        "--sweep-id",
        help="Airflow dag_run id or other sweep tag; suffixed to run.model_version so "
        "concurrent sweeps get separate output directories. Default: current UTC timestamp, "
        "except for slurm executors, where it keys the state file and is required so "
        "reusing it resumes the sweep.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Trials in flight at once; overrides strategy.max_concurrent. An adaptive "
        "strategy proposes each extra in-flight trial without that trial's score.",
    )
    parser.add_argument(
        "--resume-study",
        metavar="STUDY_NAME",
        help="optuna only: continue this study (turboswan-<hash>-NNN, see the optuna_study_name "
        "tag on a sweep's parent run) instead of starting a new one. Its hash must match the "
        "current search. For slurm executors it applies only when the sweep is first created.",
    )
    parser.add_argument("--cpus-per-trial", type=int, help="ray only (default: 2).")
    parser.add_argument(
        "--gpus-per-trial",
        type=float,
        help="ray only; fractional values are allowed, e.g. 0.5 to pack two trials per GPU (default: 0).",
    )
    parser.add_argument("--slurm-host", help="SSH host to submit to (required for --executor slurm).")
    parser.add_argument(
        "--state-path",
        type=pathlib.Path,
        help="stepped executors only; sweep state file (default: ~/turboswan/sweeps/<sweep-id>.json).",
    )
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="stepped executors only; cancel the sweep's pending jobs and exit.",
    )
    args = parser.parse_args(argv)

    if args.cancel and args.executor not in STEPPED_EXECUTORS:
        parser.error("--cancel only applies to --executor slurm or local-process")
    for dest, (flag, executors) in EXECUTOR_FLAGS.items():
        if getattr(args, dest) is not None and args.executor not in executors:
            parser.error(f"{flag} only applies to --executor {' or '.join(executors)}")
    if args.executor == "slurm" and not args.slurm_host:
        parser.error("--executor slurm requires --slurm-host")
    if args.sweep_id is None:
        if args.executor in STEPPED_EXECUTORS:
            parser.error("--sweep-id is required for --executor slurm or local-process (it keys the resumable state)")
        args.sweep_id = default_sweep_id()

    sweep_config_path = (args.sweep_config or DEFAULT_SWEEP_CONFIG[args.executor]).resolve()
    sweep, strategy, search_space = load_sweep(sweep_config_path)
    base_config, recipe_toml = training.load_config_source(
        sweep_config_path.parent / sweep.base_config_toml
    )
    if args.max_concurrent:
        strategy = strategy.model_copy(update={"max_concurrent": args.max_concurrent})
    sweep_id = re.sub(r"[^A-Za-z0-9]+", "-", args.sweep_id).strip("-")
    base_config["run"]["model_version"] = f"{base_config['run']['model_version']}-{sweep_id}"

    if args.executor in STEPPED_EXECUTORS:
        state_path = args.state_path or pathlib.Path.home() / "turboswan" / "sweeps" / f"{sweep_id}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        if args.executor == "slurm":
            executor = SlurmExecutor(host=args.slurm_host)
        else:
            executor = LocalProcessExecutor(state_path.with_suffix(""))
        if args.cancel:
            print(f"cancelled {cancel_sweep(state_path, executor)} pending job(s)")
            return 0
        finished = step_sweep(
            sweep, strategy, search_space, base_config,
            sweep_id=sweep_id, state_path=state_path, executor=executor,
            resume_study=args.resume_study, recipe_toml=recipe_toml,
        )
        return 0 if finished else NOT_FINISHED

    if args.executor == "ray":
        import ray

        if base_config.get("run", {}).get("mode", "local") == "local":
            logging.warning("local MLflow tracking only works when Ray runs on this machine")
        ray.init()
        logging.info(ray.cluster_resources())
        executor = RayExecutor(
            cpus_per_trial=2 if args.cpus_per_trial is None else args.cpus_per_trial,
            gpus_per_trial=0 if args.gpus_per_trial is None else args.gpus_per_trial,
        )
    else:
        executor = SequentialExecutor()

    run_sweep(
        sweep, strategy, search_space, base_config,
        sweep_id=sweep_id, executor=executor, recipe_toml=recipe_toml,
        resume_study=args.resume_study,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
