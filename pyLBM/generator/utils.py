from __future__ import print_function
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
from string import Template
import numpy as np
from six.moves import range

def matMult(A, x, y, sorder=None, vectorized=True, indent=''):
    """
    return a string representing the unroll matrix vector operation y = Ax

    Parameters
    ----------
    A : numpy array
      the matrix coefficients
    x : string
      input vector in a string format
    y : string
      output vector in a string format where the matrix vector product is stored
    indent : string
      representing spaces or tabs which are set at the beginning of each line
    prefix : string
      some string to add before indices
    suffix : string
      some string to add after indices

    Returns
    -------
    code : string
      the string representing the unroll matrix vector operation y = Ax

    Examples
    --------

    >>> import numpy as np
    >>> A = np.arange(12).reshape(4,3)
    >>> A[2, :] *= -1

    >>> print matMult(A, 'm', 'f')
    f[0] =  + m[1] + 2.0000000000000000*m[2]
    f[1] =  + 3.0000000000000000*m[0] + 4.0000000000000000*m[1] + 5.0000000000000000*m[2]
    f[2] =  - 6.0000000000000000*m[0] - 7.0000000000000000*m[1] - 8.0000000000000000*m[2]
    f[3] =  + 9.0000000000000000*m[0] + 10.0000000000000000*m[1] + 11.0000000000000000*m[2]

    >>> print matMult(A, 'm', 'f', '\t', prefix = 'i, j, ')
        f[i, j, 0] =  + m[i, j, 1] + 2.0000000000000000*m[i, j, 2]
        f[i, j, 1] =  + 3.0000000000000000*m[i, j, 0] + 4.0000000000000000*m[i, j, 1] + 5.0000000000000000*m[i, j, 2]
        f[i, j, 2] =  - 6.0000000000000000*m[i, j, 0] - 7.0000000000000000*m[i, j, 1] - 8.0000000000000000*m[i, j, 2]
        f[i, j, 3] =  + 9.0000000000000000*m[i, j, 0] + 10.0000000000000000*m[i, j, 1] + 11.0000000000000000*m[i, j, 2]

    >>> print matMult(A, 'm', 'f', '\t', suffix = ', i, j')
        f[0, i, j] =  + m[1, i, j] + 2.0000000000000000*m[2, i, j]
        f[1, i, j] =  + 3.0000000000000000*m[0, i, j] + 4.0000000000000000*m[1, i, j] + 5.0000000000000000*m[2, i, j]
        f[2, i, j] =  - 6.0000000000000000*m[0, i, j] - 7.0000000000000000*m[1, i, j] - 8.0000000000000000*m[2, i, j]
        f[3, i, j] =  + 9.0000000000000000*m[0, i, j] + 10.0000000000000000*m[1, i, j] + 11.0000000000000000*m[2, i, j]

    """
    nvk1, nvk2 = A.shape
    code = ''

    if sorder is not None:
        tmp = ['']*len(sorder)
        for i, v in enumerate(sorder[1:]):
            tmp[v] = ':' if vectorized else 'i{0}'.format(i)

    for i in range(nvk1):
        if sorder is None:
            tmp = [str(i)]
        else:
            tmp[sorder[0]] = str(i)

        code += indent + "{0}[{1}] = ".format(y, ', '.join(tmp))

        for j in range(nvk2):
            coef = A[i, j]
            scoef = '' if  abs(coef) == 1 else '{0:.16f}*'.format(abs(coef))
            sign = ' + ' if coef > 0 else ' - '

            if sorder is None:
                tmp = [str(j)]
            else:
                tmp[sorder[0]] = str(j)

            if coef != 0:
                code += "{0}{1}{2}[{3}]".format(sign, scoef, x, ', '.join(tmp))

        code += "\n"

    return code

def give_slice(X):
    res = []
    for x in X:
        res.append([])
        for iv, v in enumerate(x):
            if v > 0:
                res[-1].append('{0:d}:'.format(v))
            elif v < 0:
                res[-1].append(':{0:d}'.format(v))
            else:
                res[-1].append(':')
    return res

def give_indices(X):
    res = []
    for x in X:
        res.append([])
        for iv, v in enumerate(x):
            if v > 0:
                res[-1].append('i{0:d} - {1:d}'.format(iv, v))
            elif v < 0:
                res[-1].append('i{0:d} + {1:d}'.format(iv, -v))
            else:
                res[-1].append('i{0:d}'.format(iv))
    return res

def get_indices(s, ind, sorder):
    if s is '':
        res = [str(ind)]
    else:
        res = ['']*len(sorder)
        for i, v in enumerate(sorder[1:]):
            res[v] = s[i]
        res[sorder[0]] = str(ind)
    return res

def load_or_store(x, y, load_list, store_list, sorder, indent='', vectorized = True, avoid_copy=True):

    if load_list is not None:
        s1 = give_slice(load_list) if vectorized else give_indices(load_list)
    if store_list is not None:
        s2 = give_slice(store_list) if vectorized else give_indices(store_list)

    if load_list is None:
        s1 = ['']*len(s2)
    if store_list is None:
        s2 = ['']*len(s1)

    t = Template("$indent$x[$indx] = $y[$indy]")
    code = ''
    ind = 0
    for ss1, ss2 in zip(s1, s2):
        tmp1 = get_indices(ss1, ind, sorder)
        tmp2 = get_indices(ss2, ind, sorder)
        if not (tmp1 == tmp2 and avoid_copy):
            code += "{0}{1}[{2}] = {3}[{4}]\n".format(indent, x, ', '.join(tmp2),
                        y, ', '.join(tmp1))
        ind += 1

    return code

if __name__ == '__main__':
    import numpy as np
    v = np.asarray([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

    nv = 2
    nspace = [0, 1]
    print(load_or_store('f', 'f', -v, v, nv, nspace))
    print(load_or_store('floc', 'f', v, None, nv, nspace, vectorized=False))
    print(load_or_store('f', 'floc', None, np.zeros(v.shape), nv, nspace, vectorized=False))

    nv = 1
    nspace = [0, 2]
    print(load_or_store('f', 'f', -v, v, nv, nspace))
    print(load_or_store('floc', 'f', v, None, nv, nspace, vectorized=False))
    print(load_or_store('f', 'floc', None, np.zeros(v.shape), nv, nspace, vectorized=False))

    A = np.arange(12).reshape(4,3)
    A[2, :] *= -1
    print(matMult(A, 'm', 'f'))
    print(matMult(A, 'm', 'f', nv, nspace, '\t'))
