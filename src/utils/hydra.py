import functools
from hydra.utils import get_method as original_get_method
from hydra.utils import get_class as original_get_class
from hydra.utils import get_object as original_get_object

def get_method(path: str, **kwargs):
    """Feed a partially-instantiated callable through hydra."""
    cl = original_get_method(path)
    return functools.partial(cl, **kwargs)

def get_class(path: str, **kwargs):
    """Feed a partially-instantiated class through hydra."""
    cl = original_get_class(path)
    return functools.partial(cl, **kwargs)

def get_object(path: str, **kwargs):
    """Feed an object through hydra with additional arguments."""
    cl = original_get_object(path)
    return cl(**kwargs)