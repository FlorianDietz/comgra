from datetime import datetime


FUNCTION_NAME_TO_TOTAL_TIME = {}


def runtime_analysis_decorator(function):
    function_name = function.__name__
    def wrapper(*args, **kwargs):
        start = datetime.now()
        # print(f"START {function.__name__}")
        res = function(*args, **kwargs)
        end = datetime.now()
        duration = end - start
        if function_name in FUNCTION_NAME_TO_TOTAL_TIME:
            FUNCTION_NAME_TO_TOTAL_TIME[function_name] += duration
        else:
            FUNCTION_NAME_TO_TOTAL_TIME[function_name] = duration
        total_duration_in_this_function = FUNCTION_NAME_TO_TOTAL_TIME[function_name]
        # print(F"END {function_name} after {duration}, total duration {total_duration_in_this_function}")
        return res
    return wrapper


def print_total_runtimes():
    names_and_times = [(k, v.total_seconds()) for k, v in FUNCTION_NAME_TO_TOTAL_TIME.items()]
    names_and_times.sort(key=lambda a: a[1], reverse=True)
    for name, time in names_and_times:
        print(f"{name:50} -   {time}")
