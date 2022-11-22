import collections
from datetime import datetime, timedelta
from typing import List, Any, Tuple, Dict, Set

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


def the(a):
    lngth = len(a)
    if lngth != 1:
        raise ValueError(f"Item should have exactly one element, but has {lngth}")
    return list(a)[0]


class WildcardForComparison:
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, WildcardForComparison)


class PseudoDb:
    def __init__(self, attributes):
        self.attributes: List[str] = attributes
        self.record_set: Dict[Tuple, Any] = {}

    def merge(self, other: 'PseudoDb'):
        # Sanity checks
        assert self.attributes == other.attributes
        my_recs = {a[0] for a in self.get_matches({})[0]}
        other_recs = {a[0] for a in other.get_matches({})[0]}
        assert all(len(a) == len(self.attributes) for a in my_recs)
        assert all(len(a) == len(self.attributes) for a in other_recs)
        merged_recs = my_recs | other_recs
        assert len(my_recs) + len(other_recs) == len(merged_recs), "The recordings overlap."
        # Update
        self.record_set.update(other.record_set)

    def serialize(self):
        frequent_values = {}
        encoded_records = []
        for key, val in self.record_set.items():
            encoded_keys = []
            for attr, attr_val in zip(self.attributes, key):
                if attr_val not in frequent_values:
                    frequent_values[attr_val] = attr_val
                encoded_keys.append(frequent_values[attr_val])
            encoded_records.append([encoded_keys, val])
        frequent_values = {v: k for k, v in frequent_values.items()}
        res = [self.attributes, frequent_values, encoded_records]
        return res

    def deserialize(self, json_data):
        attributes, frequent_values, encoded_records = tuple(json_data)
        self.attributes = attributes
        self.record_set = {
            tuple([frequent_values[a] for a in val[0]]): val[1] for val in encoded_records
        }
        return self

    def add_record(self, attr_values, result):
        assert len(attr_values) == len(self.attributes)
        assert attr_values not in self.record_set, attr_values
        self.record_set[attr_values] = result

    def get_matches(self, filters: Dict[str, Any]):
        filters_with_indices = {self.attributes.index(k): v for k, v in filters.items()}
        list_of_matches = []
        possible_attribute_values = {k: set() for k in self.attributes}
        for attr_values, result in self.record_set.items():
            matches = True
            for idx, filter_value in filters_with_indices.items():
                if attr_values[idx] != filter_value or isinstance(attr_values[idx], WildcardForComparison):
                    matches = False
                    break
            if matches:
                list_of_matches.append((attr_values, result))
                for attr_name, attr_val in zip(self.attributes, attr_values):
                    possible_attribute_values[attr_name].add(attr_val)
        return list_of_matches, possible_attribute_values
