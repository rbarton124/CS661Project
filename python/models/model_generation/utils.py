# --- IMPORTS
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ValidationError, validate_call
# --- END IMPORTS

# --- INIT AND TYPE ENFORCING FUNCTIONS

class TypedStruct(BaseModel):
    class Config:
        arbitrary_types_allowed = True

def enforce_types(func: Callable):
    return validate_call(config=dict(arbitrary_types_allowed=True, validate_return=True))(func)

def assert_or_assign(obj, attribute, expected):
    if hasattr(obj, attribute) and getattr(obj, attribute) is not None:
        assert getattr(obj, attribute) == expected, f"attribute {attribute} must equal {expected}"
    else:
        setattr(obj, attribute, expected)

def assert_eq(val_1, val_2, error_msg="Assert equal failed"):
    assert val_1 == val_2, error_msg
# --- END INIT AND TYPE ENFORCING FUNCTIONS

# --- DEBUG VARIABLES
_DEBUG_KEEP_MSG_LEN = 5
_DEBUG_QUEUE = deque(maxlen=_DEBUG_KEEP_MSG_LEN)
_DEBUG_ENABLED_GROUPS = set(["DEFAULT"])
_DEBUG_DISABLED_GROUPS = set()
_DEBUG_NEW_GROUPS_ENABLE = True
# --- END DEBUG VARIABLES

# --- DEBUG GROUP SETTINGS
@enforce_types
def DEBUG_SET(group: str, enabled: bool) -> None:
    upper_name = group.upper()
    if enabled:
        _DEBUG_ENABLED_GROUPS.add(upper_name)
        _DEBUG_DISABLED_GROUPS.discard(upper_name)
    else:
        _DEBUG_DISABLED_GROUPS.add(upper_name)
        _DEBUG_ENABLED_GROUPS.discard(upper_name)

@enforce_types
def DEBUG_ENABLE(group: str) -> None:
    DEBUG_SET(group, True)

@enforce_types
def DEBUG_DISABLE(group: str) -> None:
    DEBUG_SET(group, False)

@enforce_types
def DEBUG_ENABLE_NEW_GROUPS() -> None:
    global _DEBUG_NEW_GROUPS_ENABLE
    _DEBUG_NEW_GROUPS_ENABLE = True

@enforce_types
def DEBUG_DISABLE_NEW_GROUPS() -> None:
    global _DEBUG_NEW_GROUPS_ENABLE
    _DEBUG_NEW_GROUPS_ENABLE = False

@enforce_types
def DEBUG_ENABLE_ALL() -> None:
    global _DEBUG_NEW_GROUPS_ENABLE
    _DEBUG_ENABLED_GROUPS.update(_DEBUG_DISABLED_GROUPS)
    _DEBUG_DISABLED_GROUPS.clear()
    _DEBUG_NEW_GROUPS_ENABLE = True

@enforce_types
def DEBUG_DISABLE_ALL() -> None:
    global _DEBUG_NEW_GROUPS_ENABLE
    _DEBUG_DISABLED_GROUPS.update(_DEBUG_ENABLED_GROUPS)
    _DEBUG_ENABLED_GROUPS.clear()
    _DEBUG_NEW_GROUPS_ENABLE = False

@enforce_types
def DEBUG_CLEAR_ALL() -> None:
    global _DEBUG_NEW_GROUPS_ENABLE
    global _DEBUG_ENABLED_GROUPS
    _DEBUG_ENABLED_GROUPS = set(["DEFAULT"])
    _DEBUG_NEW_GROUPS_ENABLE = True
    _DEBUG_DISABLED_GROUPS.clear()
    _DEBUG_QUEUE.clear()

@enforce_types
def DEBUG_SHOW_GROUPS() -> None:
    print("")
    print(f"Enabled groups: {_DEBUG_ENABLED_GROUPS}")
    print(f"Disabled groups: {_DEBUG_DISABLED_GROUPS}")
    print("")

@enforce_types
def DEBUG_DELETE_GROUP(group: str) -> None:
    upper_name = group.upper()
    if upper_name in _DEBUG_ENABLED_GROUPS:
        _DEBUG_ENABLED_GROUPS.remove(upper_name)
    elif upper_name in _DEBUG_DISABLED_GROUPS:
        _DEBUG_DISABLED_GROUPS.remove(upper_name)
    else:
        raise ValueError(f"Debug group {group} does not exist")

@enforce_types
def _DEBUG_UPDATE_GROUPS(group: str) -> None:
    global _DEBUG_NEW_GROUPS_ENABLE
    if (group in _DEBUG_ENABLED_GROUPS) or (group in _DEBUG_DISABLED_GROUPS):
        return
    elif _DEBUG_NEW_GROUPS_ENABLE:
        _DEBUG_ENABLED_GROUPS.add(group.upper())
    else:
        _DEBUG_DISABLED_GROUPS.add(group.upper())

@enforce_types
def _DEBUG_GROUP_ENABLED(group: str) -> bool:
    return group.upper() in _DEBUG_ENABLED_GROUPS
# --- END DEBUG GROUP SETTINGS

# --- MAIN DEBUG
@enforce_types
def DEBUG_PRINT(lazy_var_eval: Callable, group: str="DEFAULT") -> None:
    _DEBUG_UPDATE_GROUPS(group)
    if _DEBUG_GROUP_ENABLED(group):
        print(lazy_var_eval())

@enforce_types
def DEBUG_PUSH(lazy_var_eval: Callable, group: str="DEFAULT") -> None:
    _DEBUG_UPDATE_GROUPS(group)
    if _DEBUG_GROUP_ENABLED(group):
        _DEBUG_QUEUE.append(lazy_var_eval())

@enforce_types
def _DEBUG_QUEUE_CLEAR() -> None:
	_DEBUG_QUEUE.clear()

@enforce_types
def DEBUG_CLEAR() -> None:
    _DEBUG_QUEUE_CLEAR()

@enforce_types
def DEBUG_POP() -> None:
    # print("-----DEBUG LOGGING OUTPUT-----")
    for msg in _DEBUG_QUEUE:
        print(msg)
    # print("-----END DEBUG LOGGING OUTPUT-----")
    _DEBUG_QUEUE_CLEAR()
# --- END MAIN DEBUG

# --- PYTHON INFO FUNCTIONS
@enforce_types
def GET_CLASS_FUNCTIONS(obj: Any) -> List[str]:
    if obj is None:
        raise ValueError("object cannot be None")
    return [method for method in dir(obj)
            if callable(getattr(obj, method)) and not (method.startswith('_') or method.endswith('_'))]

@enforce_types
def GET_CLASS_VARIABLES(obj):
    if obj is None:
        raise ValueError("object cannot be None")
    return [(attr, type(getattr(obj, attr)).__name__) for attr in dir(obj)
            if not callable(getattr(obj, attr)) and not (attr.startswith('_') or attr.endswith('_'))]

@enforce_types
def GET_CLASS_DUNDERS(obj):
    if obj is None:
        raise ValueError("object cannot be None")
    return [method for method in dir(obj)
            if method.startswith('__') and method.endswith('__')]
# --- END PYTHON INFO FUNCTIONS

# --- PLOTTING FUNCTIONS
def easy_plot(title, x_label, xs, y_label, ys, x_scale=None, y_scale=None, x_range=None, y_range=None, grid_lines=None):
    plt.clf()
    plt.plot(xs, ys)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_scale:
        plt.xscale(x_scale)
    if y_scale:
        plt.yscale(y_scale)
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)
    if grid_lines:
        plt.grid(True)

    plt.show()

def easy_lines(title, x_label, line_xs, y_label, line_ys, x_scale=None, y_scale=None, line_names=None, x_range=None, y_range=None, grid_lines=None):
    plt.clf()

    num_lines = len(line_xs)
    if line_names is None:
        line_names = [f'Line {i+1}' for i in range(num_lines)]
    else:
        if len(line_names) != num_lines:
            raise ValueError("line_names must have the same length as line_xs and line_ys")

    for i in range(num_lines):
        plt.plot(line_xs[i], line_ys[i], label=line_names[i])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_scale:
        plt.xscale(x_scale)
    if y_scale:
        plt.yscale(y_scale)
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)
    if grid_lines:
        plt.grid(True)

    plt.legend()
    plt.show()
# --- END PLOTTING FUNCTIONS