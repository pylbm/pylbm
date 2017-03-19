import sympy as sp
import sys

PY3 = sys.version_info >= (3,)

if PY3:
    from inspect import getfullargspec as getargspec
else:
    from inspect import getargspec
    
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
        ind = [set_order([k, *index[1:]], permutation, remove_ind) for k in ranges]
        return sp.Matrix([output[i] for i in ind])
    elif list_ind is not None:
        ind = []
        indices = index[1:]
        for il, l in enumerate(list_ind):
            tmp_ind = []
            for ik, k in enumerate(l):
                tmp_ind.append(indices[ik] - int(k))
            ind.append(set_order([il, *tmp_ind], permutation, remove_ind))
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
    return set_order([0, *idx], permutation, remove_ind=[0])

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
    