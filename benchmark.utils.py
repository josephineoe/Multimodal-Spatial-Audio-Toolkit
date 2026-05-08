# benchmark_utils.py
import time
from functools import wraps

profiling_data = {}

def profile(func_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            t1 = time.perf_counter()
            profiling_data.setdefault(func_name, []).append((t0, t1 - t0))
            return result
        return wrapper
    return decorator