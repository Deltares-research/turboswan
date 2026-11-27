"""Executors that run trials as detached jobs, which a stepped sweep polls."""

import logging
from typing import Protocol

# remove with LocalProcessExecutor
import json
import os
import pathlib
import subprocess
import sys
import psutil

log = logging.getLogger(__name__)


class SteppedExecutor(Protocol):
	"""Job ids are owned by the caller, so the executor itself holds no state between steps."""

	def connect(self) -> None: ...

	def submit(self, config_index: int, **kwargs) -> str: ...

	def poll(self, jobs: dict[int, str]) -> dict[int, dict]: ...

	def cancel(self, job_id: str) -> None: ...


class SlurmExecutor:
	"""Submit trials as Slurm batch jobs on a remote HPC over SSH."""

	def __init__(self, host: str):
		self.host = host
		raise NotImplementedError("SlurmExecutor is not yet implemented. Use 'local-process' for testing.")

	def connect(self) -> None:
		return

	def submit(self, config_index: int, **kwargs) -> str:
		return

	def poll(self, jobs: dict[int, str]) -> dict[int, dict]:
		return

	def cancel(self, job_id: str) -> None:
		return


def _alive(pid: int) -> bool:
	try:
		# an exited child stays a zombie while the in-process step loop is its parent
		return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
	except psutil.NoSuchProcess:
		return False


class LocalProcessExecutor:
	"""Run each trial as a detached local process; the PID is the job id."""

	def __init__(self, workdir: pathlib.Path):
		self.workdir = workdir

	def connect(self) -> None:
		self.workdir.mkdir(parents=True, exist_ok=True)

	def submit(self, config_index: int, **kwargs) -> str:
		args_path = self.workdir / f"{config_index}.json"
		args_path.write_text(json.dumps({"config_index": config_index, **kwargs}))
		detach = (
			{"creationflags": subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP}
			if os.name == "nt"
			else {"start_new_session": True}
		)
		# on Windows each trial streams to its own console window; elsewhere it inherits this terminal
		proc = subprocess.Popen(
			[sys.executable, "-m", "turboswan.mlops.sweeps.trial", str(args_path)], **detach
		)
		return str(proc.pid)

	def poll(self, jobs: dict[int, str]) -> dict[int, dict]:
		done = {}
		for config_index, pid in jobs.items():
			# check the process before the file, so an exit in between is never missed
			if _alive(int(pid)):
				continue
			result = self.workdir / f"{config_index}.result.json"
			done[config_index] = (
				json.loads(result.read_text())
				if result.exists()
				else {"error": f"process {pid} exited without a result"}
			)
		return done

	def cancel(self, job_id: str) -> None:
		try:
			psutil.Process(int(job_id)).terminate()
		except psutil.NoSuchProcess:
			pass


__all__ = ["LocalProcessExecutor", "SlurmExecutor", "SteppedExecutor"]
