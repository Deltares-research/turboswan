"""Executors that run trials while the driver waits on them."""

import logging
from collections.abc import Callable
from typing import Protocol

log = logging.getLogger(__name__)


class BlockingExecutor(Protocol):
	"""A failed trial's result is ``{"error": repr(exception)}``; the exception never leaves the executor."""

	def submit(self, config_index: int, fn: Callable[..., dict], /, **kwargs) -> None: ...

	def wait_next(self) -> tuple[int, dict]: ...

	def cancel_all(self) -> None: ...


class SequentialExecutor:
	"""Run each submitted trial locally, one at a time."""

	def __init__(self):
		self._queued = []

	def submit(self, config_index: int, fn: Callable[..., dict], /, **kwargs) -> None:
		self._queued.append((config_index, fn, kwargs))

	def wait_next(self) -> tuple[int, dict]:
		config_index, fn, kwargs = self._queued.pop(0)
		try:
			return config_index, fn(**kwargs)
		except Exception as error:
			log.error("config_index=%d failed: %r", config_index, error, exc_info=True)
			return config_index, {"error": repr(error)}

	def cancel_all(self) -> None:
		self._queued.clear()


class RayExecutor:
	"""Submit independent trials to the active Ray cluster."""

	def __init__(self, *, cpus_per_trial: int = 2, gpus_per_trial: float = 0):
		self.cpus_per_trial = cpus_per_trial
		self.gpus_per_trial = gpus_per_trial
		self._pending = []
		self._submitted = {}

	def submit(self, config_index: int, fn: Callable[..., dict], /, **kwargs) -> None:
		import ray

		# fn must be a module-level function; ray.remote rejects functools.partial.
		trial = ray.remote(num_cpus=self.cpus_per_trial, num_gpus=self.gpus_per_trial)(fn)
		ref = trial.remote(**kwargs)
		self._pending.append(ref)
		self._submitted[ref] = config_index

	def wait_next(self) -> tuple[int, dict]:
		import ray

		[done], self._pending = ray.wait(self._pending, num_returns=1)
		config_index = self._submitted.pop(done)
		try:
			return config_index, ray.get(done)
		except Exception as error:
			log.error("config_index=%d failed: %r", config_index, error, exc_info=True)
			return config_index, {"error": repr(error)}

	def cancel_all(self) -> None:
		import ray

		for ref in self._pending:
			try:
				ray.cancel(ref)
			except Exception as error:
				log.warning("cancel failed for config_index=%s: %r", self._submitted.get(ref), error)
		self._pending.clear()
		self._submitted.clear()


__all__ = ["BlockingExecutor", "RayExecutor", "SequentialExecutor"]
