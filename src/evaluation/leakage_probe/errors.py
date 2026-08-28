"""Domain errors raised by leakage-probe evaluation."""


class ProbeExtractionError(RuntimeError):
    """Failure to construct a scientifically valid probe dataset."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class ProbePartitionError(ValueError):
    """Failure to construct the fixed inner probe partition."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class ProbeFitError(RuntimeError):
    """Failure while fitting or evaluating one probe candidate."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)

