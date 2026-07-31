import importlib.util
import platform


def _package_available(package_name: str) -> bool:
    """Check whether a package is importable in this environment."""
    try:
        return importlib.util.find_spec(package_name) is not None
    except (ImportError, ValueError):
        return False


_IS_WINDOWS = platform.system() == "Windows"

_SH_AVAILABLE = not _IS_WINDOWS and _package_available("sh")

_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _package_available("deepspeed")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _package_available("fairscale")

_NEPTUNE_AVAILABLE = _package_available("neptune")
_COMET_AVAILABLE = _package_available("comet_ml")
_MLFLOW_AVAILABLE = _package_available("mlflow")
