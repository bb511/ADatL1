import inspect

def get_function_parameters(func):
    """
    Given a function, return a list of all the parameter names that can be passed to the function.

    :param func: The function to inspect
    :return: List of parameter names
    """
    # Get the signature of the function
    signature = inspect.signature(func)
    
    # Extract parameter names
    params = [param.name for param in signature.parameters.values()]
    
    return params
