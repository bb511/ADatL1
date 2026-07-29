"""Compatibility import for the shared training/evaluation CAP metric."""

from src.callbacks.metrics.cap.metric import ApproximationCapacity, PosteriorConsistency

__all__ = ["ApproximationCapacity", "PosteriorConsistency"]
