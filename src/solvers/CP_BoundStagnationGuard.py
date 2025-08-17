import time


class BoundStagnationGuard:
    """
    Terminates the search if BestObjectiveBound has not changed sufficiently (relative)
    for longer than `no_improvement_seconds`.
    - Change criterion: relative change >= `relative_change` (e.g. 0.01 = 1%)
    - Sense-agnostic (min/max) via abs.
    - Warmup period is respected.
    """
    def __init__(
        self, solver, logger, no_improvement_seconds: int,
        warmup_seconds: int = 60, relative_change: float = 0.01):

        self.solver = solver
        self.logger = logger
        self.no_improve_s = int(no_improvement_seconds)
        self.warmup_s = int(warmup_seconds)
        self.relative_change = float(relative_change)

        now = time.monotonic()
        self._start = now
        self._last_improve_time = None
        self._last_bound = None

    def __call__(self, _msg: str):
        now = time.monotonic()

        # 1) Skip checks during warmup
        if now - self._start < self.warmup_s:
            return

        # 2) Read bound & initialize on first tick after warmup
        cur = self.solver.BestObjectiveBound()
        if self._last_bound is None:
            self._last_bound = cur
            self._last_improve_time = now
            return

        # 3) Check relative change (sense-agnostic via abs)
        delta = abs(cur - self._last_bound)
        if self._last_bound == 0.0:
            changed = (delta > 0)
        else:
            rel_delta = delta / abs(self._last_bound)
            changed = rel_delta >= self.relative_change

        if changed:
            # Improvement detected -> reset timer
            self._last_bound = cur
            self._last_improve_time = now
            return

        # 4) No sufficient change: check timeout
        if self._last_improve_time is not None and (now - self._last_improve_time) >= self.no_improve_s:
            self.logger.info(
                f"Stopping search: no ≥{self.relative_change:.4g} relative bound change "
                + f"for ≥{self.no_improve_s}s after warmup (bound={cur})."
            )
            self.solver.stop_search()
