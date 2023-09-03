import collections
from datetime import datetime, timedelta
from typing import List, Any, Tuple, Dict, Set, Optional

from plotly import express as px

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
        try:
            res = function(*args, **kwargs)
        except Exception as e:
            raise
        finally:
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


class PseudoDb:
    def __init__(self, attributes):
        self.attributes: List[str] = attributes
        self.record_set: Optional[Dict[Tuple, Any]] = {}
        # Set later, when create_index() is used, to replace record_set
        self.index_attributes: Optional[List[str]] = None
        self.index_tree: Optional[Dict] = None

    def merge(self, other: 'PseudoDb'):
        # Sanity checks
        assert self.attributes == other.attributes
        my_recs = set(self.record_set.keys())
        other_recs = set(other.record_set.keys())
        assert all(len(a) == len(self.attributes) for a in my_recs)
        assert all(len(a) == len(self.attributes) for a in other_recs)
        merged_recs = my_recs | other_recs
        if len(my_recs) + len(other_recs) != len(merged_recs):
            print(f"The recordings overlap:\n{len(my_recs)=}\n{len(other_recs)=}\n{len(merged_recs)=}")
            assert False
        # Update
        self.record_set.update(other.record_set)

    def serialize(self):
        frequent_values = {}
        encoded_records = []
        for key, val in self.record_set.items():
            encoded_keys = []
            for attr, attr_val in zip(self.attributes, key):
                if attr_val not in frequent_values:
                    frequent_values[attr_val] = len(frequent_values)
                encoded_keys.append(frequent_values[attr_val])
            encoded_records.append([encoded_keys, val])
        frequent_values = {str(v): k for k, v in frequent_values.items()}
        res = [self.attributes, frequent_values, encoded_records]
        return res

    def deserialize(self, json_data):
        attributes, frequent_values, encoded_records = tuple(json_data)
        frequent_values = {int(k): v for k, v in frequent_values.items()}
        self.attributes = attributes
        self.record_set = {
            tuple([frequent_values[a] for a in val[0]]): val[1] for val in encoded_records
        }
        return self

    def add_record(self, attr_values, result):
        assert len(attr_values) == len(self.attributes)
        assert attr_values not in self.record_set, attr_values
        self.record_set[attr_values] = result

    @runtime_analysis_decorator
    def create_index(self, index_attributes):
        assert all(a in self.attributes for a in index_attributes)
        self.index_attributes = list(index_attributes)
        index_attribute_indices = [self.attributes.index(a) for a in self.index_attributes]
        self.index_tree = {}
        for key, val in self.record_set.items():
            tree = self.index_tree
            for idx in index_attribute_indices:
                tree = tree.setdefault(key[idx], {})
            tree[key] = val
        self.record_set = None

    @runtime_analysis_decorator
    def get_matches(self, filters: Dict[str, Any]) -> Tuple[List[Tuple[Tuple[Any, ...], Any]], Dict[str, Set]]:
        filters_with_indices = {self.attributes.index(k): v for k, v in filters.items()}
        # Filter using an index.
        # While doing so, up to max_num_mismatches_to_allow many attributes may be mismatched
        # while going through the index.
        # (This is important for determining possible_attribute_values later).
        max_num_mismatches_to_allow = 1
        if self.index_attributes is None:
            assert self.index_tree is None
            record_set = self.record_set
        else:
            assert self.record_set is None
            index_attribute_indices = [self.attributes.index(a) for a in self.index_attributes]
            trees_to_process = [([], self.index_tree)]
            for idx in index_attribute_indices:
                next_level_of_trees_to_process = []
                has_filter = (idx in filters_with_indices)
                if has_filter:
                    filter_value = filters_with_indices[idx]
                    for mismatches, tree in trees_to_process:
                        if filter_value in tree:
                            next_level_of_trees_to_process.append((mismatches, tree[filter_value]))
                        if len(mismatches) < max_num_mismatches_to_allow:
                            for a, subtree in tree.items():
                                if a != filter_value or ((a is None) != (filter_value is None)):
                                    next_level_of_trees_to_process.append((mismatches + [idx], subtree))
                else:
                    for mismatches, tree in trees_to_process:
                        for subtree in tree.values():
                            next_level_of_trees_to_process.append((mismatches, subtree))
                trees_to_process = next_level_of_trees_to_process
            record_set = {}
            for mismatches, d in trees_to_process:
                record_set.update(d)
        # Search for records
        list_of_matches = []
        possible_attribute_values = {k: set() for k in self.attributes}
        for attr_values, result in record_set.items():
            num_mismatches = 0
            for idx, filter_value in filters_with_indices.items():
                if attr_values[idx] != filter_value:
                    num_mismatches += 1
                    if num_mismatches == 2:
                        break
            if num_mismatches == 0:
                list_of_matches.append((attr_values, result))
            if num_mismatches < 2:
                for idx, attr_name, in enumerate(self.attributes):
                    # If the record either matches, or doesn't match but only because of the attribute in question,
                    # then that attribute value is a legal value for selection.
                    if num_mismatches == 0 or \
                            (num_mismatches == 1 and idx in filters_with_indices and attr_values[idx] != filters_with_indices[idx]):
                        possible_attribute_values[attr_name].add(attr_values[idx])
        return list_of_matches, possible_attribute_values


def number_to_hex(number):
    if number < -1:
        number = -1
    elif number > 1:
        number = 1
    colorscale = px.colors.sequential.Viridis
    return colorscale[int((number + 1) / 2 * (len(colorscale) - 1))]
