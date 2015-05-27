import types

import sys
if sys.version_info[0] >= 3:
    from ..extern import byteplay3 as byteplay
else:
    from ..extern import byteplay
from ..extern.six.moves import range


def new_closure(vals):
    """
    Build a new closure
    """
    args = ','.join('x%i' % i for i in range(len(vals)))
    f = eval("lambda %s:lambda:(%s)" % (args, args))
    if sys.version_info[0] >= 3:
        return f(*vals).__closure__
    return f(*vals).func_closure


def _inject_closure_values_fix_closures(c, injected, **kwargs):
    """
    Recursively fix closures

    Python bytecode for a closure looks like:

        LOAD_CLOSURE var1
        BUILD_TUPLE <n_of_vars_closed_over>
        LOAD_CONST <code_object_containing_closure>
        MAKE_CLOSURE

    This function finds closures and adds the injected closed variables in the
    right place.
    """
    code = c.code
    orig_len = len(code)
    for iback, (opcode, value) in enumerate(reversed(code)):
        i = orig_len - iback - 1

        if opcode != byteplay.MAKE_CLOSURE:
            continue

        codeobj = code[i-1]
        assert codeobj[0] == byteplay.LOAD_CONST

        build_tuple = code[i-2]
        assert build_tuple[0] == byteplay.BUILD_TUPLE
        n_closed = build_tuple[1]

        load_closures = code[i-2-n_closed:i-2]
        assert all(o == byteplay.LOAD_CLOSURE for o, _ in load_closures)

        newlcs = [(byteplay.LOAD_CLOSURE, inj) for inj in injected]

        code[i-2] = byteplay.BUILD_TUPLE, n_closed + len(injected)
        code[i-2:i-2] = newlcs

        _inject_closure_values_fix_code(codeobj[1], injected, **kwargs)


def _inject_closure_values_fix_code(c, injected, **kwargs):
    """
    Fix code objects, recursively fixing any closures
    """
    # Add more closure variables
    c.freevars += injected

    # Replace LOAD_GLOBAL with LOAD_DEREF (fetch from closure cells)
    # for named variables
    for i, (opcode, value) in enumerate(c.code):
        if opcode == byteplay.LOAD_GLOBAL and value in kwargs:
            c.code[i] = byteplay.LOAD_DEREF, value

    _inject_closure_values_fix_closures(c, injected, **kwargs)

    return c


def _inject_closure_values(func, **kwargs):
    for name in kwargs:
        if sys.version_info[0] >= 3:
            co_freevars = func.__code__.co_freevars
        else:
            co_freevars = func.func_code.co_freevars
        assert name not in co_freevars, ("BUG! Tried to inject "
            "closure variable where there is already a closure variable of the "
            "same name: {0}".format(name))

    cellvalues = []
    if sys.version_info[0] >= 3:
        func_closure = func.__closure__
        func_code = func.__code__
        func_globals = func.__globals__
        func_name = func.__name__
        func_defaults = func.__defaults__
    else:
        func_closure = func.func_closure
        func_code = func.func_code
        func_globals = func.func_globals
        func_name = func.func_name
        func_defaults = func.func_defaults
    if func_closure:
        cellvalues = [c.cell_contents for c in func_closure]

    injected = tuple(sorted(kwargs))
    # Insert the closure values into the new cells
    cellvalues.extend(kwargs[key] for key in injected)

    c = byteplay.Code.from_code(func_code)

    _inject_closure_values_fix_code(c, injected, **kwargs)

    code = c.to_code()
    closure = new_closure(cellvalues)

    args = code, func_globals, func_name, func_defaults, closure
    return types.FunctionType(*args)


def inject_closure_values(func, **kwargs):
    """
    Returns a new function identical to the previous one except that it acts as
    though global variables named in `kwargs` have been closed over with the
    values specified in the `kwargs` dictionary.

    Works on properties, class/static methods and functions.

    This can be useful for mocking and other nefarious activities.
    """
    wrapped_by = None

    if isinstance(func, property):
        fget, fset, fdel = func.fget, func.fset, func.fdel
        if fget: fget = fix_func(fget, **kwargs)
        if fset: fset = fix_func(fset, **kwargs)
        if fdel: fdel = fix_func(fdel, **kwargs)
        wrapped_by = type(func)
        return wrapped_by(fget, fset, fdel)

    elif isinstance(func, (staticmethod, classmethod)):
        func = func.__func__
        wrapped_by = type(func)

    newfunc = _inject_closure_values(func, **kwargs)

    if wrapped_by:
        newfunc = wrapped_by(newfunc)
    return newfunc
