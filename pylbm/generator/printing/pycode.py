# FIXME: make pylint happy !
#pylint: disable=all
"""
Python code printers

This module contains python code printers for plain python as well as NumPy & SciPy enabled code.
"""
from collections import defaultdict
from itertools import chain
from sympy.core import S, Add, Eq, Symbol
from sympy.printing.precedence import precedence
from sympy.printing.codeprinter import CodePrinter
from sympy.core.sympify import _sympify, sympify
from sympy.core.basic import Basic

_kw_py2and3 = {
    'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif',
    'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in',
    'is', 'lambda', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while',
    'with', 'yield', 'None'  # 'None' is actually not in Python 2's keyword.kwlist
}
_kw_only_py2 = {'exec', 'print'}
_kw_only_py3 = {'False', 'nonlocal', 'True'}

_known_functions = {
    'Abs': 'abs',
}
_known_functions_math = {
    'acos': 'acos',
    'acosh': 'acosh',
    'asin': 'asin',
    'asinh': 'asinh',
    'atan': 'atan',
    'atan2': 'atan2',
    'atanh': 'atanh',
    'ceiling': 'ceil',
    'cos': 'cos',
    'cosh': 'cosh',
    'erf': 'erf',
    'erfc': 'erfc',
    'exp': 'exp',
    'expm1': 'expm1',
    'factorial': 'factorial',
    'floor': 'floor',
    'gamma': 'gamma',
    'hypot': 'hypot',
    'loggamma': 'lgamma',
    'log': 'log',
    'ln': 'log',
    'log10': 'log10',
    'log1p': 'log1p',
    'log2': 'log2',
    'sin': 'sin',
    'sinh': 'sinh',
    'Sqrt': 'sqrt',
    'tan': 'tan',
    'tanh': 'tanh'
}  # Not used from ``math``: [copysign isclose isfinite isinf isnan ldexp frexp pow modf
# radians trunc fmod fsum gcd degrees fabs]
_known_constants_math = {
    'Exp1': 'e',
    'Pi': 'pi',
    'E': 'e'
    # Only in python >= 3.5:
    # 'Infinity': 'inf',
    # 'NaN': 'nan'
}

def _print_known_func(self, expr):
    known = self.known_functions[expr.__class__.__name__]
    return '{name}({args})'.format(name=self._module_format(known),
                                   args=', '.join(map(lambda arg: self._print(arg), expr.args)))


def _print_known_const(self, expr):
    known = self.known_constants[expr.__class__.__name__]
    return self._module_format(known)


class AbstractPythonCodePrinter(CodePrinter):
    printmethod = "_pythoncode"
    language = "Python"
    reserved_words = _kw_py2and3.union(_kw_only_py3)
    modules = None  # initialized to a set in __init__
    tab = '    '
    _kf = dict(chain(
        _known_functions.items(),
        [(k, 'math.' + v) for k, v in _known_functions_math.items()]
    ))
    _kc = {k: 'math.'+v for k, v in _known_constants_math.items()}
    _operators = {'and': 'and', 'or': 'or', 'not': 'not'}
    _default_settings = dict(
        CodePrinter._default_settings,
        user_functions={},
        precision=17,
        inline=True,
        fully_qualified_modules=True,
        contract=False,
        human=True,
        standard='python3',
    )

    def __init__(self, settings=None):
        super(AbstractPythonCodePrinter, self).__init__(settings)

        # Python standard handler
        std = self._settings['standard']
        if std is None:
            import sys
            std = 'python{}'.format(sys.version_info.major)
        if std not in ('python2', 'python3'):
            raise ValueError('Unrecognized python standard : {}'.format(std))
        self.standard = std

        self.module_imports = defaultdict(set)

        # Known functions and constants handler
        self.known_functions = dict(self._kf, **(settings or {}).get(
            'user_functions', {}))
        self.known_constants = dict(self._kc, **(settings or {}).get(
            'user_constants', {}))

    def _declare_number_const(self, name, value):
        return "%s = %s" % (name, value)

    def _module_format(self, fqn, register=True):
        parts = fqn.split('.')
        if register and len(parts) > 1:
            self.module_imports['.'.join(parts[:-1])].add(parts[-1])

        if self._settings['fully_qualified_modules']:
            return fqn
        else:
            return fqn.split('(')[0].split('[')[0].split('.')[-1]

    def _format_code(self, lines):
        return lines

    def _get_statement(self, codestring):
        return "{}".format(codestring)

    def _get_comment(self, text):
        return "  # {0}".format(text)

    def _expand_fold_binary_op(self, op, args):
        """
        This method expands a fold on binary operations.

        ``functools.reduce`` is an example of a folded operation.

        For example, the expression

        `A + B + C + D`

        is folded into

        `((A + B) + C) + D`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            return "%s(%s, %s)" % (
                self._module_format(op),
                self._expand_fold_binary_op(op, args[:-1]),
                self._print(args[-1]),
            )

    def _expand_reduce_binary_op(self, op, args):
        """
        This method expands a reductin on binary operations.

        Notice: this is NOT the same as ``functools.reduce``.

        For example, the expression

        `A + B + C + D`

        is reduced into:

        `(A + B) + (C + D)`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            N = len(args)
            Nhalf = N // 2
            return "%s(%s, %s)" % (
                self._module_format(op),
                self._expand_reduce_binary_op(args[:Nhalf]),
                self._expand_reduce_binary_op(args[Nhalf:]),
            )

    def _get_einsum_string(self, subranks, contraction_indices):
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ""
        counter = 0
        d = {j: min(i) for i in contraction_indices for j in i}
        indices = []
        for rank_arg in subranks:
            lindices = []
            for i in range(rank_arg):
                if counter in d:
                    lindices.append(d[counter])
                else:
                    lindices.append(counter)
                counter += 1
            indices.append(lindices)
        mapping = {}
        letters_free = []
        letters_dum = []
        for i in indices:
            for j in i:
                if j not in mapping:
                    l = next(letters)
                    mapping[j] = l
                else:
                    l = mapping[j]
                contraction_string += l
                if j in d:
                    if l not in letters_dum:
                        letters_dum.append(l)
                else:
                    letters_free.append(l)
            contraction_string += ","
        contraction_string = contraction_string[:-1]
        return contraction_string, letters_free, letters_dum

    def _print_NaN(self, expr):
        return "float('nan')"

    def _print_Infinity(self, expr):
        return "float('inf')"

    def _print_NegativeInfinity(self, expr):
        return "float('-inf')"

    def _print_ComplexInfinity(self, expr):
        return self._print_NaN(expr)

    def _print_Mod(self, expr):
        PREC = precedence(expr)
        return ('{0} % {1}'.format(*map(lambda x: self.parenthesize(x, PREC), expr.args)))

    def _print_Piecewise(self, expr):
        result = []
        i = 0
        for arg in expr.args:
            e = arg.expr
            c = arg.cond
            if i == 0:
                result.append('(')
            result.append('(')
            result.append(self._print(e))
            result.append(')')
            result.append(' if ')
            result.append(self._print(c))
            result.append(' else ')
            i += 1
        result = result[:-1]
        if result[-1] == 'True':
            result = result[:-2]
            result.append(')')
        else:
            result.append(' else None)')
        return ''.join(result)

    def _print_Relational(self, expr):
        "Relational printer for Equality and Unequality"
        op = {
            '==' :'equal',
            '!=' :'not_equal',
            '<'  :'less',
            '<=' :'less_equal',
            '>'  :'greater',
            '>=' :'greater_equal',
        }
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return '({lhs} {op} {rhs})'.format(op=expr.rel_op, lhs=lhs, rhs=rhs)
        return super(AbstractPythonCodePrinter, self)._print_Relational(expr)

    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_Sum(self, expr):
        loops = (
            'for {i} in range({a}, {b}+1)'.format(
                i=self._print(i),
                a=self._print(a),
                b=self._print(b))
            for i, a, b in expr.limits)
        return '(builtins.sum({function} {loops}))'.format(
            function=self._print(expr.function),
            loops=' '.join(loops))

    def _print_ImaginaryUnit(self, expr):
        return '1j'

    def _print_KroneckerDelta(self, expr):
        a, b = expr.args

        return '(1 if {a} == {b} else 0)'.format(
            a = self._print(a),
            b = self._print(b)
        )

    def _print_MatrixBase(self, expr):
        name = expr.__class__.__name__
        func = self.known_functions.get(name, name)
        return "%s(%s)" % (func, self._print(expr.tolist()))

    _print_SparseMatrix = \
        _print_MutableSparseMatrix = \
        _print_ImmutableSparseMatrix = \
        _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        lambda self, expr: self._print_MatrixBase(expr)

    def _indent_codestring(self, codestring):
        return '\n'.join([self.tab + line for line in codestring.split('\n')])

    def _print_FunctionDefinition(self, fd):
        body = '\n'.join(map(lambda arg: self._print(arg), fd.body))
        return "def {name}({parameters}):\n{body}".format(
            name=self._print(fd.name),
            parameters=', '.join([self._print(var.symbol) for var in fd.parameters]),
            body=self._indent_codestring(body)
        )

    def _print_While(self, whl):
        body = '\n'.join(map(lambda arg: self._print(arg), whl.body))
        return "while {cond}:\n{body}".format(
            cond=self._print(whl.condition),
            body=self._indent_codestring(body)
        )

    def _print_Declaration(self, decl):
        return '%s = %s' % (
            self._print(decl.variable.symbol),
            self._print(decl.variable.value)
        )

    def _print_Return(self, ret):
        arg, = ret.args
        return 'return %s' % self._print(arg)

    def _print_Print(self, prnt):
        print_args = ', '.join(map(lambda arg: self._print(arg), prnt.print_args))
        if prnt.format_string != None: # Must be '!= None', cannot be 'is not None'
            print_args = '{0} % ({1})'.format(
                self._print(prnt.format_string), print_args)
        if prnt.file != None: # Must be '!= None', cannot be 'is not None'
            print_args += ', file=%s' % self._print(prnt.file)

        if self.standard == 'python2':
            return 'print %s' % print_args
        return 'print(%s)' % print_args

    def _print_Stream(self, strm):
        if str(strm.name) == 'stdout':
            return self._module_format('sys.stdout')
        elif str(strm.name) == 'stderr':
            return self._module_format('sys.stderr')
        else:
            return self._print(strm.name)

    def _print_NoneToken(self, arg):
        return 'None'


class PythonCodePrinter(AbstractPythonCodePrinter):

    def _print_sign(self, e):
        return '(0.0 if {e} == 0 else {f}(1, {e}))'.format(
            f=self._module_format('math.copysign'), e=self._print(e.args[0]))

    def _print_Not(self, expr):
        PREC = precedence(expr)
        return self._operators['not'] + self.parenthesize(expr.args[0], PREC)

    def _print_Indexed(self, expr):
        base = expr.args[0]
        index = expr.args[1:]
        return "{}[{}]".format(str(base), ", ".join([self._print(ind) for ind in index]))

    def _hprint_Pow(self, expr, rational=False, sqrt='math.sqrt'):
        """Printing helper function for ``Pow``

        Notes
        =====

        This only preprocesses the ``sqrt`` as math formatter

        Examples
        ========

        >>> from sympy.functions import sqrt
        >>> from sympy.printing.pycode import PythonCodePrinter
        >>> from sympy.abc import x

        Python code printer automatically looks up ``math.sqrt``.

        >>> printer = PythonCodePrinter({'standard':'python3'})
        >>> printer._hprint_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._hprint_Pow(sqrt(x), rational=False)
        'math.sqrt(x)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=False)
        '1/math.sqrt(x)'

        Using sqrt from numpy or mpmath

        >>> printer._hprint_Pow(sqrt(x), sqrt='numpy.sqrt')
        'numpy.sqrt(x)'
        >>> printer._hprint_Pow(sqrt(x), sqrt='mpmath.sqrt')
        'mpmath.sqrt(x)'

        See Also
        ========

        sympy.printing.str.StrPrinter._print_Pow
        """
        PREC = precedence(expr)

        if expr.exp == S.Half and not rational:
            func = self._module_format(sqrt)
            arg = self._print(expr.base)
            return '{func}({arg})'.format(func=func, arg=arg)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                func = self._module_format(sqrt)
                num = self._print(S.One)
                arg = self._print(expr.base)
                return "{num}/{func}({arg})".format(
                    num=num, func=func, arg=arg)

        base_str = self.parenthesize(expr.base, PREC, strict=False)
        exp_str = self.parenthesize(expr.exp, PREC, strict=False)
        return "{}**{}".format(base_str, exp_str)

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational)

    def _print_Rational(self, expr):
        if self.standard == 'python2':
            return '{}./{}.'.format(expr.p, expr.q)
        return '{}/{}'.format(expr.p, expr.q)

    def _print_Half(self, expr):
        return self._print_Rational(expr)

    _print_lowergamma = CodePrinter._print_not_supported
    _print_uppergamma = CodePrinter._print_not_supported
    _print_fresnelc = CodePrinter._print_not_supported
    _print_fresnels = CodePrinter._print_not_supported


for k in PythonCodePrinter._kf:
    setattr(PythonCodePrinter, '_print_%s' % k, _print_known_func)

for k in _known_constants_math:
    setattr(PythonCodePrinter, '_print_%s' % k, _print_known_const)


def pycode(expr, **settings):
    """ Converts an expr to a string of Python code

    Parameters
    ==========

    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    standard : str or None, optional
        If 'python2', Python 2 sematics will be used.
        If 'python3', Python 3 sematics will be used.
        If None, the standard will be automatically detected.
        Default is 'python3'. And this parameter may be removed in the
        future.

    Examples
    ========

    >>> from sympy import tan, Symbol
    >>> from sympy.printing.pycode import pycode
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'

    """
    return PythonCodePrinter(settings).doprint(expr)


_not_in_mpmath = 'log1p log2'.split()
_in_mpmath = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_mpmath]
_known_functions_mpmath = dict(_in_mpmath, **{
    'beta': 'beta',
    'fresnelc': 'fresnelc',
    'fresnels': 'fresnels',
    'sign': 'sign',
})
_known_constants_mpmath = {
    'Exp1': 'e',
    'Pi': 'pi',
    'GoldenRatio': 'phi',
    'EulerGamma': 'euler',
    'Catalan': 'catalan',
    'NaN': 'nan',
    'Infinity': 'inf',
    'NegativeInfinity': 'ninf'
}


class MpmathPrinter(PythonCodePrinter):
    """
    Lambda printer for mpmath which maintains precision for floats
    """
    printmethod = "_mpmathcode"

    language = "Python with mpmath"

    _kf = dict(chain(
        _known_functions.items(),
        [(k, 'mpmath.' + v) for k, v in _known_functions_mpmath.items()]
    ))
    _kc = {k: 'mpmath.'+v for k, v in _known_constants_mpmath.items()}

    def _print_Float(self, e):
        # XXX: This does not handle setting mpmath.mp.dps. It is assumed that
        # the caller of the lambdified function will have set it to sufficient
        # precision to match the Floats in the expression.

        # Remove 'mpz' if gmpy is installed.
        args = str(tuple(map(int, e._mpf_)))
        return '{func}({args})'.format(func=self._module_format('mpmath.mpf'), args=args)


    def _print_Rational(self, e):
        return "{func}({p})/{func}({q})".format(
            func=self._module_format('mpmath.mpf'),
            q=self._print(e.q),
            p=self._print(e.p)
        )

    def _print_Half(self, e):
        return self._print_Rational(e)

    def _print_uppergamma(self, e):
        return "{0}({1}, {2}, {3})".format(
            self._module_format('mpmath.gammainc'),
            self._print(e.args[0]),
            self._print(e.args[1]),
            self._module_format('mpmath.inf'))

    def _print_lowergamma(self, e):
        return "{0}({1}, 0, {2})".format(
            self._module_format('mpmath.gammainc'),
            self._print(e.args[0]),
            self._print(e.args[1]))

    def _print_log2(self, e):
        return '{0}({1})/{0}(2)'.format(
            self._module_format('mpmath.log'), self._print(e.args[0]))

    def _print_log1p(self, e):
        return '{0}({1}+1)'.format(
            self._module_format('mpmath.log'), self._print(e.args[0]))

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='mpmath.sqrt')


for k in MpmathPrinter._kf:
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_func)

for k in _known_constants_mpmath:
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_const)


_not_in_numpy = 'erf erfc factorial gamma loggamma'.split()
_in_numpy = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_numpy]
_known_functions_numpy = dict(_in_numpy, **{
    'acos': 'arccos',
    'acosh': 'arccosh',
    'asin': 'arcsin',
    'asinh': 'arcsinh',
    'atan': 'arctan',
    'atan2': 'arctan2',
    'atanh': 'arctanh',
    'exp2': 'exp2',
    'sign': 'sign',
})
_known_constants_numpy = {
    'Exp1': 'e',
    'Pi': 'pi',
    'EulerGamma': 'euler_gamma',
    'NaN': 'nan',
    'Infinity': 'PINF',
    'NegativeInfinity': 'NINF'
}


class NumPyPrinter(PythonCodePrinter):
    """
    Numpy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """
    printmethod = "_numpycode"
    language = "Python with NumPy"

    _kf = dict(chain(
        PythonCodePrinter._kf.items(),
        [(k, 'numpy.' + v) for k, v in _known_functions_numpy.items()]
    ))
    _kc = {k: 'numpy.'+v for k, v in _known_constants_numpy.items()}

    def doprint(self, expr, assign_to=None):
        """
        Print the expression as code.

        Parameters
        ----------
        expr : Expression
            The expression to be printed.

        assign_to : Symbol, string, MatrixSymbol, list of strings or Symbols (optional)
            If provided, the printed code will set the expression to a variable or multiple variables
            with the name or names given in ``assign_to``.
        """
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from ..ast import CodeBlock, Assignment

        def _handle_assign_to(expr, assign_to):
            if isinstance(expr, Eq):
                expr = Assignment(expr.lhs, expr.rhs)
            if assign_to is None:
                return sympify(expr)
            if isinstance(assign_to, (list, tuple)):
                if len(expr) != len(assign_to):
                    raise ValueError('Failed to assign an expression of length {} to {} variables'.format(len(expr), len(assign_to)))
                return CodeBlock(*[_handle_assign_to(lhs, rhs) for lhs, rhs in zip(expr, assign_to)])
            if isinstance(assign_to, str):
                if expr.is_Matrix:
                    assign_to = MatrixSymbol(assign_to, *expr.shape)
                else:
                    assign_to = Symbol(assign_to)
            elif not isinstance(assign_to, Basic):
                raise TypeError("{} cannot assign to object of type {}".format(
                        type(self).__name__, type(assign_to)))
            return Assignment(assign_to, expr)

        expr = _handle_assign_to(expr, assign_to)

        # keep a set of expressions that are not strictly translatable to Code
        # and number constants that must be declared and initialized
        self._not_supported = set()
        self._number_symbols = set()

        lines = self._print(expr).splitlines()

        # format the output
        if self._settings["human"]:
            frontlines = []
            if self._not_supported:
                frontlines.append(self._get_comment(
                        "Not supported in {}:".format(self.language)))
                for expr in sorted(self._not_supported, key=str):
                    frontlines.append(self._get_comment(type(expr).__name__))
            for name, value in sorted(self._number_symbols, key=str):
                frontlines.append(self._declare_number_const(name, value))
            lines = frontlines + lines
            lines = self._format_code(lines)
            result = "\n".join(lines)
        else:
            lines = self._format_code(lines)
            num_syms = {(k, self._print(v)) for k, v in self._number_symbols}
            result = (num_syms, self._not_supported, "\n".join(lines))
        self._not_supported = set()
        self._number_symbols = set()
        return result

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _print_seq(self, seq):
        "General sequence printer: converts to tuple"
        # Print tuples here instead of lists because numba supports
        #     tuples in nopython mode.
        delimiter=', '
        return '({},)'.format(delimiter.join(self._print(item) for item in seq))

    def _print_MatMul(self, expr):
        "Matrix multiplication printer"
        if expr.as_coeff_matrices()[0] is not S.One:
            expr_list = expr.as_coeff_matrices()[1]+[(expr.as_coeff_matrices()[0])]
            return '({0})'.format(').dot('.join(self._print(i) for i in expr_list))
        return '({0})'.format(').dot('.join(self._print(i) for i in expr.args))

    def _print_MatPow(self, expr):
        "Matrix power printer"
        return '{0}({1}, {2})'.format(self._module_format('numpy.linalg.matrix_power'),
            self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_Inverse(self, expr):
        "Matrix inverse printer"
        return '{0}({1})'.format(self._module_format('numpy.linalg.inv'),
            self._print(expr.args[0]))

    def _print_DotProduct(self, expr):
        # DotProduct allows any shape order, but numpy.dot does matrix
        # multiplication, so we have to make sure it gets 1 x n by n x 1.
        arg1, arg2 = expr.args
        if arg1.shape[0] != 1:
            arg1 = arg1.T
        if arg2.shape[1] != 1:
            arg2 = arg2.T

        return "%s(%s, %s)" % (self._module_format('numpy.dot'),
                               self._print(arg1),
                               self._print(arg2))

    def _print_MatrixSolve(self, expr):
        return "%s(%s, %s)" % (self._module_format('numpy.linalg.solve'),
                               self._print(expr.matrix),
                               self._print(expr.vector))

    def _print_ZeroMatrix(self, expr):
        return '{}({})'.format(self._module_format('numpy.zeros'),
            self._print(expr.shape))

    def _print_OneMatrix(self, expr):
        return '{}({})'.format(self._module_format('numpy.ones'),
            self._print(expr.shape))

    def _print_FunctionMatrix(self, expr):
        from sympy.core.function import Lambda
        from sympy.abc import i, j
        lamda = expr.lamda
        if not isinstance(lamda, Lambda):
            lamda = Lambda((i, j), lamda(i, j))
        return '{}(lambda {}: {}, {})'.format(self._module_format('numpy.fromfunction'),
            ', '.join(self._print(arg) for arg in lamda.args[0]),
            self._print(lamda.args[1]), self._print(expr.shape))

    def _print_HadamardProduct(self, expr):
        func = self._module_format('numpy.multiply')
        return ''.join('{}({}, '.format(func, self._print(arg)) \
            for arg in expr.args[:-1]) + "{}{}".format(self._print(expr.args[-1]),
            ')' * (len(expr.args) - 1))

    def _print_KroneckerProduct(self, expr):
        func = self._module_format('numpy.kron')
        return ''.join('{}({}, '.format(func, self._print(arg)) \
            for arg in expr.args[:-1]) + "{}{}".format(self._print(expr.args[-1]),
            ')' * (len(expr.args) - 1))

    def _print_Adjoint(self, expr):
        return '{}({}({}))'.format(
            self._module_format('numpy.conjugate'),
            self._module_format('numpy.transpose'),
            self._print(expr.args[0]))

    def _print_DiagonalOf(self, expr):
        vect = '{}({})'.format(
            self._module_format('numpy.diag'),
            self._print(expr.arg))
        return '{}({}, (-1, 1))'.format(
            self._module_format('numpy.reshape'), vect)

    def _print_DiagMatrix(self, expr):
        return '{}({})'.format(self._module_format('numpy.diagflat'),
            self._print(expr.args[0]))

    def _print_DiagonalMatrix(self, expr):
        return '{}({}, {}({}, {}))'.format(self._module_format('numpy.multiply'),
            self._print(expr.arg), self._module_format('numpy.eye'),
            self._print(expr.shape[0]), self._print(expr.shape[1]))

    def _print_Piecewise(self, expr):
        "Piecewise function printer"
        exprs = '[{0}]'.format(','.join(self._print(arg.expr) for arg in expr.args))
        conds = '[{0}]'.format(','.join(self._print(arg.cond) for arg in expr.args))
        # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
        #     it will behave the same as passing the 'default' kwarg to select()
        #     *as long as* it is the last element in expr.args.
        # If this is not the case, it may be triggered prematurely.
        return '{0}({1}, {2}, default={3})'.format(
            self._module_format('numpy.select'), conds, exprs,
            self._print(S.NaN))

    def _print_Relational(self, expr):
        "Relational printer for Equality and Unequality"
        op = {
            '==' :'equal',
            '!=' :'not_equal',
            '<'  :'less',
            '<=' :'less_equal',
            '>'  :'greater',
            '>=' :'greater_equal',
        }
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return '{op}({lhs}, {rhs})'.format(op=self._module_format('numpy.'+op[expr.rel_op]),
                                               lhs=lhs, rhs=rhs)
        return super(NumPyPrinter, self)._print_Relational(expr)

    def _print_And(self, expr):
        "Logical And printer"
        # We have to override LambdaPrinter because it uses Python 'and' keyword.
        # If LambdaPrinter didn't define it, we could use StrPrinter's
        # version of the function and add 'logical_and' to NUMPY_TRANSLATIONS.
        return '{0}.reduce(({1}))'.format(self._module_format('numpy.logical_and'), ','.join(self._print(i) for i in expr.args))

    def _print_Or(self, expr):
        "Logical Or printer"
        # We have to override LambdaPrinter because it uses Python 'or' keyword.
        # If LambdaPrinter didn't define it, we could use StrPrinter's
        # version of the function and add 'logical_or' to NUMPY_TRANSLATIONS.
        return '{0}.reduce(({1}))'.format(self._module_format('numpy.logical_or'), ','.join(self._print(i) for i in expr.args))

    def _print_Not(self, expr):
        "Logical Not printer"
        # We have to override LambdaPrinter because it uses Python 'not' keyword.
        # If LambdaPrinter didn't define it, we would still have to define our
        #     own because StrPrinter doesn't define it.
        return '{0}({1})'.format(self._module_format('numpy.logical_not'), ','.join(self._print(i) for i in expr.args))

    def _print_Pow(self, expr, rational=False):
        # XXX Workaround for negative integer power error
        from sympy.core.power import Pow
        if expr.exp.is_integer and expr.exp.is_negative:
            expr = Pow(expr.base, expr.exp.evalf(), evaluate=False)
        if expr.exp.is_integer and not expr.exp.is_negative:
            line = self._print(expr.base)
            for i in range(1, expr.exp):
                line += '*' + self._print(expr.base)
            return f'({line})'
        return self._hprint_Pow(expr, rational=rational, sqrt='numpy.sqrt')

    def _print_Min(self, expr):
        return '{0}(({1}), axis=0)'.format(self._module_format('numpy.amin'), ','.join(self._print(i) for i in expr.args))

    def _print_Max(self, expr):
        return '{0}(({1}), axis=0)'.format(self._module_format('numpy.amax'), ','.join(self._print(i) for i in expr.args))

    def _print_arg(self, expr):
        return "%s(%s)" % (self._module_format('numpy.angle'), self._print(expr.args[0]))

    def _print_im(self, expr):
        return "%s(%s)" % (self._module_format('numpy.imag'), self._print(expr.args[0]))

    def _print_Mod(self, expr):
        return "%s(%s)" % (self._module_format('numpy.mod'), ', '.join(
            map(lambda arg: self._print(arg), expr.args)))

    def _print_re(self, expr):
        return "%s(%s)" % (self._module_format('numpy.real'), self._print(expr.args[0]))

    def _print_sinc(self, expr):
        return "%s(%s)" % (self._module_format('numpy.sinc'), self._print(expr.args[0]/S.Pi))

    def _print_MatrixBase(self, expr):
        func = self.known_functions.get(expr.__class__.__name__, None)
        if func is None:
            func = self._module_format('numpy.array')
        return "%s(%s)" % (func, self._print(expr.tolist()))

    def _print_Identity(self, expr):
        shape = expr.shape
        if all([dim.is_Integer for dim in shape]):
            return "%s(%s)" % (self._module_format('numpy.eye'), self._print(expr.shape[0]))
        else:
            raise NotImplementedError("Symbolic matrix dimensions are not yet supported for identity matrices")

    def _print_BlockMatrix(self, expr):
        return '{0}({1})'.format(self._module_format('numpy.block'),
                                 self._print(expr.args[0].tolist()))

    def _print_CodegenArrayTensorProduct(self, expr):
        array_list = [j for i, arg in enumerate(expr.args) for j in
                (self._print(arg), "[%i, %i]" % (2*i, 2*i+1))]
        return "%s(%s)" % (self._module_format('numpy.einsum'), ", ".join(array_list))

    def _print_CodegenArrayContraction(self, expr):
        from sympy.codegen.array_utils import CodegenArrayTensorProduct
        base = expr.expr
        contraction_indices = expr.contraction_indices
        if not contraction_indices:
            return self._print(base)
        if isinstance(base, CodegenArrayTensorProduct):
            counter = 0
            d = {j: min(i) for i in contraction_indices for j in i}
            indices = []
            for rank_arg in base.subranks:
                lindices = []
                for i in range(rank_arg):
                    if counter in d:
                        lindices.append(d[counter])
                    else:
                        lindices.append(counter)
                    counter += 1
                indices.append(lindices)
            elems = ["%s, %s" % (self._print(arg), ind) for arg, ind in zip(base.args, indices)]
            return "%s(%s)" % (
                self._module_format('numpy.einsum'),
                ", ".join(elems)
            )
        raise NotImplementedError()

    def _print_CodegenArrayDiagonal(self, expr):
        diagonal_indices = list(expr.diagonal_indices)
        if len(diagonal_indices) > 1:
            # TODO: this should be handled in sympy.codegen.array_utils,
            # possibly by creating the possibility of unfolding the
            # CodegenArrayDiagonal object into nested ones. Same reasoning for
            # the array contraction.
            raise NotImplementedError
        if len(diagonal_indices[0]) != 2:
            raise NotImplementedError
        return "%s(%s, 0, axis1=%s, axis2=%s)" % (
            self._module_format("numpy.diagonal"),
            self._print(expr.expr),
            diagonal_indices[0][0],
            diagonal_indices[0][1],
        )

    def _print_CodegenArrayPermuteDims(self, expr):
        return "%s(%s, %s)" % (
            self._module_format("numpy.transpose"),
            self._print(expr.expr),
            self._print(expr.permutation.array_form),
        )

    def _print_CodegenArrayElementwiseAdd(self, expr):
        return self._expand_fold_binary_op('numpy.add', expr.args)

    def _print_Indexed(self, expr):
        from sympy.tensor.indexed import Idx

        elem = []
        for i in range(expr.rank):
            e = expr.indices[i]
            if isinstance(e, Add) and e.has(Idx):
                lower = 0
                upper = 0
                for a in e.args:
                    if isinstance(a, Idx):
                        lower += a.lower
                        upper += a.upper
                    else:
                        lower += a
                        upper += a
                elem.append("%s:%s"%(lower, upper))
            else:
                elem.append(self._print(expr.indices[i]))
        return "%s[%s]" % (self._print(expr.base.label), ', '.join(elem))

    def _print_Idx(self, expr):
        return "%s:%s"%(expr.lower, expr.upper)

    def _print_For(self, expr):
        lines = []
        for e in expr.body:
            temp1, temp2, addlines = self.doprint(e)
            if isinstance(addlines, str):
                lines.append(addlines)
            else:
                lines += addlines
        return "\n".join(lines)

    def _print_Assignment(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.matrices import MatrixBase, MatrixSlice
        from sympy.codegen.ast import Assignment

        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        if isinstance(expr.rhs, Piecewise):
            # Here we modify Piecewise so each expression is now
            # an Assignment, and then continue on the print.
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        #elif isinstance(lhs, MatrixSymbol):
        elif isinstance(lhs, (MatrixBase, MatrixSymbol, MatrixSlice)):
            # Here we form an Assignment for each element in the array,
            # printing each one.
            lines = []
            for (i, j) in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return "\n".join(lines)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            # hack to avoid the printing of m[i] = m[i]
            if lhs_code == rhs_code:
                return ""
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "    "
        inc_token = (':', ":\n")
        dec_token = ('end\n',)

        code = [ line.lstrip(' \t') for line in code ]


        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.endswith, dec_token)))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line == '' or line == '\n':
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty

    _print_lowergamma = CodePrinter._print_not_supported
    _print_uppergamma = CodePrinter._print_not_supported
    _print_fresnelc = CodePrinter._print_not_supported
    _print_fresnels = CodePrinter._print_not_supported

    _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        _print_MatrixBase

for k in NumPyPrinter._kf:
    setattr(NumPyPrinter, '_print_%s' % k, _print_known_func)

for k in NumPyPrinter._kc:
    setattr(NumPyPrinter, '_print_%s' % k, _print_known_const)


_known_functions_scipy_special = {
    'erf': 'erf',
    'erfc': 'erfc',
    'besselj': 'jv',
    'bessely': 'yv',
    'besseli': 'iv',
    'besselk': 'kv',
    'factorial': 'factorial',
    'gamma': 'gamma',
    'loggamma': 'gammaln',
    'digamma': 'psi',
    'RisingFactorial': 'poch',
    'jacobi': 'eval_jacobi',
    'gegenbauer': 'eval_gegenbauer',
    'chebyshevt': 'eval_chebyt',
    'chebyshevu': 'eval_chebyu',
    'legendre': 'eval_legendre',
    'hermite': 'eval_hermite',
    'laguerre': 'eval_laguerre',
    'assoc_laguerre': 'eval_genlaguerre',
    'beta': 'beta',
    'LambertW' : 'lambertw',
}

_known_constants_scipy_constants = {
    'GoldenRatio': 'golden_ratio',
    'Pi': 'pi',
}

class SciPyPrinter(NumPyPrinter):

    language = "Python with SciPy"

    _kf = dict(chain(
        NumPyPrinter._kf.items(),
        [(k, 'scipy.special.' + v) for k, v in _known_functions_scipy_special.items()]
    ))
    _kc =dict(chain(
        NumPyPrinter._kc.items(),
        [(k, 'scipy.constants.' + v) for k, v in _known_constants_scipy_constants.items()]
    ))

    def _print_SparseMatrix(self, expr):
        i, j, data = [], [], []
        for (r, c), v in expr._smat.items():
            i.append(r)
            j.append(c)
            data.append(v)

        return "{name}({data}, ({i}, {j}), shape={shape})".format(
            name=self._module_format('scipy.sparse.coo_matrix'),
            data=data, i=i, j=j, shape=expr.shape
        )

    _print_ImmutableSparseMatrix = _print_SparseMatrix

    # SciPy's lpmv has a different order of arguments from assoc_legendre
    def _print_assoc_legendre(self, expr):
        return "{0}({2}, {1}, {3})".format(
            self._module_format('scipy.special.lpmv'),
            self._print(expr.args[0]),
            self._print(expr.args[1]),
            self._print(expr.args[2]))

    def _print_lowergamma(self, expr):
        return "{0}({2})*{1}({2}, {3})".format(
            self._module_format('scipy.special.gamma'),
            self._module_format('scipy.special.gammainc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]))

    def _print_uppergamma(self, expr):
        return "{0}({2})*{1}({2}, {3})".format(
            self._module_format('scipy.special.gamma'),
            self._module_format('scipy.special.gammaincc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]))

    def _print_fresnels(self, expr):
        return "{0}({1})[0]".format(
                self._module_format("scipy.special.fresnel"),
                self._print(expr.args[0]))

    def _print_fresnelc(self, expr):
        return "{0}({1})[1]".format(
                self._module_format("scipy.special.fresnel"),
                self._print(expr.args[0]))

    def _print_airyai(self, expr):
        return "{0}({1})[0]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_airyaiprime(self, expr):
        return "{0}({1})[1]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_airybi(self, expr):
        return "{0}({1})[2]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_airybiprime(self, expr):
        return "{0}({1})[3]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))


for k in SciPyPrinter._kf:
    setattr(SciPyPrinter, '_print_%s' % k, _print_known_func)

for k in SciPyPrinter._kc:
    setattr(SciPyPrinter, '_print_%s' % k, _print_known_const)


class SymPyPrinter(PythonCodePrinter):

    language = "Python with SymPy"

    _kf = {k: 'sympy.' + v for k, v in chain(
        _known_functions.items(),
        _known_functions_math.items()
    )}

    def _print_Function(self, expr):
        mod = expr.func.__module__ or ''
        return '%s(%s)' % (self._module_format(mod + ('.' if mod else '') + expr.func.__name__),
                           ', '.join(map(lambda arg: self._print(arg), expr.args)))

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='sympy.sqrt')
