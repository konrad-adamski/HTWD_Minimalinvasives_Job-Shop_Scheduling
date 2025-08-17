import time

class BoundGuard:
    """
    Stops the search if the best bound does not change enough
    within a given wall-clock time.
    Triggered via solver.best_bound_callback.
    """
    def __init__(self, solver, logger,
                 no_improvement_seconds: int,
                 warmup_seconds: int = 0,
                 relative_change: float = 0.01):
        self.solver = solver
        self.logger = logger
        self.no_improve_s = no_improvement_seconds
        self.warmup_s = warmup_seconds
        self.relative_change = relative_change

        self._start = time.monotonic()
        self._last_improve_time = None
        self._last_bound = None

    def __call__(self, cur: float):
        now = time.monotonic()
        # Warmup phase
        if now - self._start < self.warmup_s:
            return

        if self._last_bound is None:
            self._last_bound = cur
            self._last_improve_time = now
            return

        # relative change
        delta = abs(cur - self._last_bound)
        if self._last_bound == 0.0:
            changed = (delta > 0)
        else:
            rel_delta = delta / abs(self._last_bound)
            changed = rel_delta >= self.relative_change

        if changed:
            self._last_bound = cur
            self._last_improve_time = now
            return

        if self._last_improve_time and (now - self._last_improve_time) >= self.no_improve_s:

            self.logger.callback_info(
                f"Stopping search: best_bound no ≥{self.relative_change:.4g} "
                f"relative change for ≥{self.no_improve_s}s (bound={cur})."
            )
            self.solver.stop_search()
