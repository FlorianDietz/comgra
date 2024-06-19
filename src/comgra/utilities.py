import collections
import colorsys
import copy
import dataclasses
import numbers
from datetime import datetime, timedelta
import math
import sys
import traceback
from typing import List, Any, Tuple, Dict, Set, Optional

from comgra import utilities_initialization_config

from plotly import express as px

DEBUG_MODE = utilities_initialization_config.DEBUG_MODE
PRINT_EACH_TIME = utilities_initialization_config.PRINT_EACH_TIME

_WARNINGS_ISSUED = {}

FUNCTION_NAME_TO_TOTAL_TIME = collections.defaultdict(lambda: timedelta(seconds=0))
FUNCTION_NAME_TO_TOTAL_NUM_CALLS = collections.defaultdict(lambda: 0)

_CURRENT_INDENT = 0
_CALL_STACK = []


def warn_once(msg):
    if msg not in _WARNINGS_ISSUED:
        _WARNINGS_ISSUED[msg] = True
        print(msg)


def get_error_message_details(exception=None):
    """
    Get a nicely formatted string for an error message collected with sys.exc_info().
    """
    if exception is None:
        exception = sys.exc_info()
    exc_type, exc_obj, exc_trace = exception
    printed_traceback_items = traceback.format_exception(exc_type, exc_obj, exc_trace)
    return ''.join(printed_traceback_items)

def runtime_analysis_decorator(function):
    if not DEBUG_MODE:
        return function
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


def recursive_pseudo_hash(a):
    """
    Return a pseudo-hash that should be unique for any element fed into it.
    Works together with recursive_equality_check().
    No guarantees of thoroughness are made.
    Just needs to be "good enough" for hashing the datastructures used to determine if two graphs are the same,
    to avoid duplicate effort in the GUI.
    """
    hash_max_value = int((2 ** sys.hash_info.width) / 2) - 1
    res = 0
    def add_sub_hash(e):
        nonlocal res, hash_max_value
        res = (res * 17 + recursive_pseudo_hash(e)) % hash_max_value
    if dataclasses.is_dataclass(a):
        a = dataclasses.asdict(a)
        add_sub_hash(1)
    if isinstance(a, dict):
        add_sub_hash(123)
        for k, v in sorted(list(a.items()), key=lambda kv: kv[0]):
            add_sub_hash(1)
            add_sub_hash(k)
            add_sub_hash(v)
    elif isinstance(a, list):
        add_sub_hash(234)
        for b in a:
            add_sub_hash(b)
    elif isinstance(a, tuple):
        add_sub_hash(345)
        for b in a:
            add_sub_hash(b)
    elif a is None or isinstance(a, (int, float, str)):
        res = hash(a)
    else:
        assert False, type(a)
    return res

def recursive_equality_check(a, b, location_list, compare_instead_of_asserting=False):
    if compare_instead_of_asserting:
        try:
            recursive_equality_check(a, b, location_list, compare_instead_of_asserting=False)
            return True
        except ValueError:
            return False
    if dataclasses.is_dataclass(a):
        if not dataclasses.is_dataclass(b):
            raise ValueError((location_list, a, b))
        a = dataclasses.asdict(a)
        b = dataclasses.asdict(b)
    if isinstance(a, dict):
        if not isinstance(b, dict):
            raise ValueError((location_list, a, b))
        if len(a) != len(b):
            raise ValueError((location_list, a, b))
        for k1, v1 in a.items():
            if k1 not in b:
                raise ValueError((location_list + [k1], a, b))
            v2 = b[k1]
            recursive_equality_check(v1, v2, location_list + [k1])
    elif isinstance(a, list):
        if not isinstance(b, list):
            raise ValueError((location_list, a, b))
        if len(a) != len(b):
            raise ValueError((location_list, a, b))
        for i, (v1, v2) in enumerate(zip(a, b)):
            recursive_equality_check(v1, v2, location_list + [i])
    elif a != b:
        raise ValueError((location_list, a, b))

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

    def add_record_value(self, attr_values, val):
        assert len(attr_values) == len(self.attributes)
        assert attr_values not in self.record_set, attr_values
        self.record_set[attr_values] = {'val': val}

    def add_record_redirection(self, attr_values, redirection):
        assert len(attr_values) == len(self.attributes)
        assert attr_values not in self.record_set, attr_values
        assert isinstance(redirection, Tuple) and len(self.attributes) == len(redirection), redirection
        self.record_set[attr_values] = {'redirect': redirection}

    @runtime_analysis_decorator
    def create_index(self, index_attributes: List, filter_values_to_ignore: Dict[str, Set]):
        assert all(a in self.attributes for a in index_attributes)
        assert all(a in self.attributes for a in filter_values_to_ignore)
        index_in_key_to_forbidden_values = {
            self.attributes.index(k): v for k, v in filter_values_to_ignore.items()
        }
        self.index_attributes = list(index_attributes)
        index_attribute_indices = [self.attributes.index(a) for a in self.index_attributes]
        self.index_tree = {}
        for key, val in self.record_set.items():
            # Skip any keys that have forbidden values in them according to filter_values_to_ignore
            if any(key[idx] in vals for idx, vals in index_in_key_to_forbidden_values.items()):
                continue
            # Recursively go through the tree structure
            tree = self.index_tree
            for idx in index_attribute_indices:
                tree = tree.setdefault(key[idx], {})
            # Resolve redirection before saving the value at the leaf node
            if 'val' in val:
                val = val['val']
            elif 'redirect' in val:
                redirected_key = tuple(val['redirect'])
                val = self.record_set[redirected_key]['val']
            else:
                assert False, val
            tree[key] = val
        self.record_set = None

    @runtime_analysis_decorator
    def get_matches(
            self,
            filters: Dict[str, Any],
    ) -> Tuple[List[Tuple[Tuple[Any, ...], Any]], Dict[str, Set]]:
        """
        Returns a list of items that match the filters, as well as a helper:
        It stores for each attribute a list of valid values that the attribute could be changed to
        while keeping the remaining filters unchanged, so that there would still be
        at least one match.
        """
        filters_with_indices = {self.attributes.index(k): v for k, v in filters.items()}
        index_attribute_indices = [self.attributes.index(a) for a in self.index_attributes]
        # Filter using an index.
        # While doing so, one attribute may be mismatched (has_a_mismatch).
        # This is used to determine possible_attribute_values.
        assert self.record_set is None
        trees_to_process = [(False, self.index_tree)]
        for idx in index_attribute_indices:
            next_level_of_trees_to_process = []
            has_filter = (idx in filters_with_indices)
            filter_value = filters_with_indices.get(idx, None)
            for has_a_mismatch, tree in trees_to_process:
                for a, subtree in tree.items():
                    if not has_filter or filter_value == a:
                        next_level_of_trees_to_process.append((has_a_mismatch, subtree))
                    elif not has_a_mismatch:
                        next_level_of_trees_to_process.append((True, subtree))
            trees_to_process = next_level_of_trees_to_process
        records_with_at_most_one_mismatch_in_the_indexed_attributes: Dict[Tuple, Any] = {}
        for _, result in trees_to_process:
            records_with_at_most_one_mismatch_in_the_indexed_attributes.update(result)
        # Search for records
        list_of_matches = []
        possible_attribute_values = {k: set() for k in self.attributes}
        for attr_values, result in records_with_at_most_one_mismatch_in_the_indexed_attributes.items():
            # Get the total number of mismatches
            # (this includes attributes that are not part of the index)
            num_mismatches = 0
            for idx, filter_value in filters_with_indices.items():
                if attr_values[idx] != filter_value:
                    num_mismatches += 1
                    if num_mismatches == 2:
                        break
            if num_mismatches == 0:
                list_of_matches.append((attr_values, result))
            if num_mismatches < 2:
                debug = 0
                for idx, attr_name, in enumerate(self.attributes):
                    possible_attribute_values[attr_name].add(attr_values[idx])
                    if idx in filters_with_indices and attr_values[idx] != filters_with_indices[idx]:
                        debug += 1
                assert num_mismatches == debug, \
                    (attr_values, filters_with_indices)
        return list_of_matches, possible_attribute_values


def number_to_hex(number):
    if math.isnan(number):
        return "#000000"
    if number < -1:
        number = -1
    elif number > 1:
        number = 1
    colorscale = px.colors.sequential.Viridis
    return colorscale[int((number + 1) / 2 * (len(colorscale) - 1))]


#
# Color hierarchy
#

def generate_hierarchically_organized_colors_for_groups(
        list_of_hierarchies, min_saturation=0.2, max_saturation=0.8,
        min_luminance=0.2, max_luminance=0.8
):
    """
    Determine what color to use for each level of the hierarchy.
    This is based on this paper: "Hierarchical Qualitative Color Palettes"
    https://mtennekes.github.io/downloads/publications/hiercolor_infovis2013.pdf
    Saturation increases with depth, luminance falls off with depth.
    """
    graph = {}
    max_depth = 0
    for visualization_layout_hierarchy in list_of_hierarchies:
        node = graph
        depth = 0
        for level in visualization_layout_hierarchy:
            if level not in node:
                node[level] = {}
            node = node[level]
            depth += 1
            max_depth = max(max_depth, depth)
    color_hierarchy = {}

    def rec(node_key, node_dict, min_hue, max_hue):
        n_children = len(node_dict)
        depth = len(node_key)
        mean_hue = (max_hue + min_hue) / 2
        hue_gap_factor_for_recursion = 0.75
        if depth != 0:  # Ignore the root, which is not a real group
            assert 1 <= depth <= max_depth, f"{depth}, {max_depth}"
            this_node_hue = mean_hue
            depth_factor = 0 if max_depth == 1 else (depth - 1) / (max_depth - 1)
            this_node_saturation = min_saturation + depth_factor * (max_saturation - min_saturation)
            this_node_saturation = max(min_saturation, min(max_saturation, this_node_saturation))
            this_node_luminance = max_luminance - depth_factor * (max_luminance - min_luminance)
            this_node_luminance = max(min_luminance, min(max_luminance, this_node_luminance))
            rgb = colorsys.hls_to_rgb(this_node_hue, this_node_luminance, this_node_saturation)
            for a in rgb:
                assert 0 <= a <= 255, [rgb, this_node_hue, this_node_luminance, this_node_saturation]
            hex_code = "#{0:02x}{1:02x}{2:02x}".format(*[max(0, min(255, int(a * 255))) for a in rgb])
            color_hierarchy[tuple(node_key)] = hex_code
        if n_children == 0:
            return
        min_hue = min_hue + (mean_hue - min_hue) * (1 - hue_gap_factor_for_recursion)
        max_hue = max_hue - (max_hue - mean_hue) * (1 - hue_gap_factor_for_recursion)
        hue_range_split = [min_hue + i / n_children * (max_hue - min_hue) for i in range(n_children + 1)]
        for min_hue, max_hue, (child_key_component, child_node_dict) in \
                zip(hue_range_split[:-1], hue_range_split[1:], node_dict.items()):
            child_node_key = node_key + [child_key_component]
            rec(child_node_key, child_node_dict, min_hue, max_hue)

    rec([], graph, 0.0, 1.0)
    return color_hierarchy

def map_to_distinct_colors(vals, colors=None):
    if colors is None:
        # Check here for more colors to use if this turns out to be insufficient:
        # https://plotly.com/python/discrete-color/
        # Note that plotly does not have a way to easily convert a continuous color scale
        # to an arbitrary number of discrete colors.
        # So we have to use modulo instead and just hope this won't come up too often.
        colors = px.colors.qualitative.Plotly
    vals = sorted(list(vals))
    return {
        k: colors[vals.index(k) % len(colors)]
        for k in vals
    }
