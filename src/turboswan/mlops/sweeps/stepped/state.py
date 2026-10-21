"""State a stepped sweep persists between steps."""

import pathlib

from pydantic import BaseModel


class PendingTrial(BaseModel):
	# exists so we don't have to keep 2 dicts in sync
	job_id: str
	overrides: dict


class SweepState(BaseModel):
	"""Everything a sweep step needs to pick up where the previous one stopped; must stay JSON serializable."""

	parent_run_id: str
	study_name: str | None = None
	prior_best: float | None = None
	first_index: int
	next_index: int
	pending: dict[int, PendingTrial] = {}
	results: list[dict] = []
	finished: bool = False

	def record(self, config_index: int, result: dict) -> None:
		"""Move a trial from pending to results."""
		self.pending.pop(config_index)
		self.results.append(result)

	def save(self, path: pathlib.Path) -> None:
		# write-then-rename so a killed pod never leaves a truncated state file
		tmp = path.with_suffix(".tmp")
		tmp.write_text(self.model_dump_json(indent=2))
		tmp.replace(path)


__all__ = ["PendingTrial", "SweepState"]
