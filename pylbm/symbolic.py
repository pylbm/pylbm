import sympy as sp
import inspect
import sys

nx, ny, nz, nv = sp.symbols("nx, ny, nz, nv", integer=True)
ix, iy, iz, iv = sp.symbols("ix, iy, iz, iv", integer=True)

def set_order(array, sorder, remove_ind=None):
    out = [-1]*len(sorder)
    for i, s in enumerate(sorder):
        out[s] = array[i]
    if remove_ind:
        for i in remove_ind:
            out.pop(sorder[i])
    return out

def indexed(name, shape, index=[iv, ix, iy, iz], list_ind=None, ranges=None, permutation=None, remove_ind=None):
    if not permutation:
        permutation = range(len(index))

    output =  sp.IndexedBase(name, set_order(shape, permutation, remove_ind))

    if ranges:
        ind = [set_order([k] + index[1:], permutation, remove_ind) for k in ranges]
        return sp.Matrix([output[i] for i in ind])
    elif list_ind is not None:
        ind = []
        indices = index[1:]
        for il, l in enumerate(list_ind):
            tmp_ind = []
            for ik, k in enumerate(l):
                tmp_ind.append(indices[ik] - int(k))
            ind.append(set_order([il] + tmp_ind, permutation, remove_ind))
        return sp.Matrix([output[i] for i in ind])
    else:
        return output[set_order(index, permutation, remove_ind)]

def space_loop(ranges, permutation=None):
    if not permutation:
        permutation = range(len(ranges) + 1)

    indices = [ix, iy, iz]
    idx = []
    for ir, r in enumerate(ranges):
        idx.append(sp.Idx(indices[ir], r))
    return set_order([0] + idx, permutation, remove_ind=[0])

def getargspec_permissive(func):
    """
    find in https://github.com/neithere/argh/blob/master/argh/compat.py

    An `inspect.getargspec` with a relaxed sanity check to support Cython.
    Motivation:
        A Cython-compiled function is *not* an instance of Python's
        types.FunctionType.  That is the sanity check the standard Py2
        library uses in `inspect.getargspec()`.  So, an exception is raised
        when calling `argh.dispatch_command(cythonCompiledFunc)`.  However,
        the CyFunctions do have perfectly usable `.func_code` and
        `.func_defaults` which is all `inspect.getargspec` needs.
        This function just copies `inspect.getargspec()` from the standard
        library but relaxes the test to a more duck-typing one of having
        both `.func_code` and `.func_defaults` attributes.
    """
    if inspect.ismethod(func):
        func = func.im_func

    # Py2 Stdlib uses isfunction(func) which is too strict for Cython-compiled
    # functions though such have perfectly usable func_code, func_defaults.
    if not (hasattr(func, "func_code") and hasattr(func, "func_defaults")):
        raise TypeError('{!r} missing func_code or func_defaults'.format(func))

    args, varargs, varkw = inspect.getargs(func.func_code)
    return inspect.ArgSpec(args, varargs, varkw, func.func_defaults)

PY3 = sys.version_info >= (3,)

if PY3:
    from inspect import getfullargspec as getargspec
else:
    getargspec = getargspec_permissive

def call_genfunction(function, args):
    from .context import queue
    try:
        func_args = function.arg_dict.keys()
        d = {k:args[k] for k in func_args}
        d['queue'] = queue
    except:
        func_args = getargspec(function).args
        d = {k:args[k] for k in func_args}
    function(**d)
