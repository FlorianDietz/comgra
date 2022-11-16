import collections
from datetime import datetime, timedelta


PRINT_EACH_TIME = False

FUNCTION_NAME_TO_TOTAL_TIME = collections.defaultdict(lambda: timedelta(seconds=0))
FUNCTION_NAME_TO_TOTAL_NUM_CALLS = collections.defaultdict(lambda: 0)

_CURRENT_INDENT = 0
_CALL_STACK = []


def runtime_analysis_decorator(function):
    function_name = function.__name__

    def wrapper(*args, **kwargs):
        global _CURRENT_INDENT
        global _CALL_STACK
        start = datetime.now()
        if PRINT_EACH_TIME:
            print("  " * _CURRENT_INDENT + f"START {function.__name__}")
        _CURRENT_INDENT += 1
        _CALL_STACK.append(function.__name__)
        res = function(*args, **kwargs)
        _CURRENT_INDENT -= 1
        _CALL_STACK.pop()
        end = datetime.now()
        duration = end - start
        FUNCTION_NAME_TO_TOTAL_TIME[function_name] += duration
        FUNCTION_NAME_TO_TOTAL_NUM_CALLS[function_name] += 1
        # Subtract the time spent in a function from its parent function.
        # This is to avoid double-counting, since the parent has its own time tracking.
        # This is particularly bad if a function is recursive,
        # because it will look many times more expensive than it is.
        if _CALL_STACK:
            parent = _CALL_STACK[-1]
            FUNCTION_NAME_TO_TOTAL_TIME[parent] -= duration
        total_duration_in_this_function = FUNCTION_NAME_TO_TOTAL_TIME[function_name]
        if PRINT_EACH_TIME:
            print(
                "  " * _CURRENT_INDENT + f"END {function_name} after {duration}, "
                                         f"total duration {total_duration_in_this_function}"
            )
        return res
    return wrapper


def print_total_runtimes():
    names_and_times_and_num_calls = [(k, v.total_seconds(), FUNCTION_NAME_TO_TOTAL_NUM_CALLS[k])
                                     for k, v in FUNCTION_NAME_TO_TOTAL_TIME.items()]
    names_and_times_and_num_calls.sort(key=lambda a: a[1], reverse=True)
    for name, time, num_calls in names_and_times_and_num_calls:
        print(f"{name:>50} -  {num_calls:>10}  -  {time:>15}")
