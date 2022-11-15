from datetime import datetime


PRINT_EACH_TIME = False

FUNCTION_NAME_TO_TOTAL_TIME = {}
FUNCTION_NAME_TO_TOTAL_NUM_CALLS = {}

CURRENT_INDENT = 0

def runtime_analysis_decorator(function):
    function_name = function.__name__
    def wrapper(*args, **kwargs):
        global CURRENT_INDENT
        start = datetime.now()
        if PRINT_EACH_TIME:
            print("  " * CURRENT_INDENT + f"START {function.__name__}")
        CURRENT_INDENT += 1
        res = function(*args, **kwargs)
        CURRENT_INDENT -= 1
        end = datetime.now()
        duration = end - start
        if function_name in FUNCTION_NAME_TO_TOTAL_TIME:
            FUNCTION_NAME_TO_TOTAL_TIME[function_name] += duration
            FUNCTION_NAME_TO_TOTAL_NUM_CALLS[function_name] += 1
        else:
            FUNCTION_NAME_TO_TOTAL_TIME[function_name] = duration
            FUNCTION_NAME_TO_TOTAL_NUM_CALLS[function_name] = 1
        total_duration_in_this_function = FUNCTION_NAME_TO_TOTAL_TIME[function_name]
        if PRINT_EACH_TIME:
            print("  " * CURRENT_INDENT + f"END {function_name} after {duration}, total duration {total_duration_in_this_function}")
        return res
    return wrapper


def print_total_runtimes():
    names_and_times_and_num_calls = [(k, v.total_seconds(), FUNCTION_NAME_TO_TOTAL_NUM_CALLS[k])
                                     for k, v in FUNCTION_NAME_TO_TOTAL_TIME.items()]
    names_and_times_and_num_calls.sort(key=lambda a: a[1], reverse=True)
    for name, time, num_calls in names_and_times_and_num_calls:
        print(f"{name:>50} -  {num_calls:>10}  -  {time:>15}")
