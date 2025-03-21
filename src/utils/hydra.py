import functools
from hydra.utils import get_method as original_get_method

def get_method(path: str, **kwargs):
    """Feed a partially-instantiated callable through hydra."""
    cl = original_get_method(path)
    return functools.partial(cl, **kwargs)
