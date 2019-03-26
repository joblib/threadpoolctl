

def _format_docstring(*args, **kwargs):
    def decorator(o):
        o.__doc__ = o.__doc__.format(*args, **kwargs)
        return o

    return decorator
