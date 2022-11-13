from datetime import datetime

def runtime_analysis_decorator(function):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        # print(f"START {function.__name__}")
        res = function(*args, **kwargs)
        end = datetime.now()
        print(F"END {function.__name__} after {end - start}")
        return res
    return wrapper
