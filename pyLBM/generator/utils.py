# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

def matMult(A, x, y, indent='', prefix='', suffix=''):
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

    for i in xrange(nvk1):
        code += indent + "{1}[{2}{0:d}{3}] = ".format(i, y, prefix, suffix)
        for j in xrange(nvk2):
            coef = A[i, j]
            scoef = '' if  abs(coef) == 1 else '{0:.16f}*'.format(abs(coef))
            sign = ' + ' if coef > 0 else ' - '

            if coef != 0:
                code += "{1}{2}{3}[{4}{0:d}{5}]".format(j, sign, scoef, x, prefix, suffix)

        code += "\n"

    return code

def give_slice(X):
    res = []
    for x in X:
        s = ''
        lx = len(x)
        for iv, v in enumerate(x):
            if v > 0:
                s += '{0:d}:'.format(v)
            elif v < 0:
                s += ':{0:d}'.format(v)
            else:
                s += ':'
            comma = (', ' if iv < lx - 1 else '')
            s += comma
        res.append(s)
    return res

def give_indices(X):
    res = []
    for x in X:
        s = ''
        lx = len(x)
        for iv, v in enumerate(x):
            if v > 0:
                s += 'i{0:d} - {1:d}'.format(iv, v)
            elif v < 0:
                s += 'i{0:d} + {1:d}'.format(iv, -v)
            else:
                s += 'i{0:d}'.format(iv)
            comma = (', ' if iv < lx - 1 else '')
            s += comma
        res.append(s)
    return res

def load_or_store(x, y, load_list, store_list, indent='', vec_form = True, avoid_copy=True, nv_on_beg=True):
    if load_list is not None:
        s1 = give_slice(load_list) if vec_form else give_indices(load_list)
    if store_list is not None:
        s2 = give_slice(store_list) if vec_form else give_indices(store_list)

    if load_list is None:
        s1 = ['']*len(s2)
    if store_list is None:
        s2 = ['']*len(s1)

    code = ''
    ind = 0
    for ss1, ss2 in zip(s1, s2):
        if not (ss1 == ss2 and avoid_copy):
            comma1 = '' if ss1 == '' else ', '
            comma2 = '' if ss2 == '' else ', '
            if nv_on_beg:
                code += "{6}{0}[{1:d}{4}{2}] = {7}[{1:d}{5}{3}]\n".format(x, ind,
                        ss2, ss1, comma2, comma1, indent, y)
            else:
                code += "{6}{0}[{2}{4}{1:d}] = {7}[{3}{5}{1:d}]\n".format(x, ind,
                        ss2, ss1, comma2, comma1, indent, y)
        ind += 1

    return code

if __name__ == '__main__':
    import numpy as np
    v = np.asarray([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

    print load_or_store('f', 'f', -v, v)
    print load_or_store('floc', 'f', v, None, vec_form=False, nv_on_beg=False)
    print load_or_store('f', 'floc', None, np.zeros(v.shape), vec_form=False, nv_on_beg=False)
