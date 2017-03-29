from sympy.core import Symbol, Expr, Tuple
from sympy.core.sympify import _sympify, sympify
from sympy.tensor import Idx, IndexedBase, Indexed
from sympy.core.basic import Basic
from sympy.core.relational import Relational
from sympy.core.compatibility import is_sequence, string_types


from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.core import Expr, Tuple, Symbol, sympify, S
from sympy.core.compatibility import is_sequence, string_types, NotIterable, range

class Assignment(Relational):
    """
    Represents variable assignment for code generation.

    Parameters
    ----------
    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Expr
        Sympy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    Examples
    ========

    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from sympy.printing.codeprinter import Assignment
    >>> x, y, z = symbols('x, y, z')
    >>> Assignment(x, y)
    Assignment(x, y)
    >>> Assignment(x, 0)
    Assignment(x, 0)
    >>> A = MatrixSymbol('A', 1, 3)
    >>> mat = Matrix([x, y, z]).T
    >>> Assignment(A, mat)
    Assignment(A, Matrix([[x, y, z]]))
    >>> Assignment(A[0, 1], x)
    Assignment(A[0, 1], x)
    """

    rel_op = ':='
    __slots__ = []

    def __new__(cls, lhs, rhs=0, **assumptions):
        from sympy.matrices.expressions.matexpr import (
            MatrixElement, MatrixSymbol)
        from sympy.tensor.indexed import Indexed
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed)
        #if not isinstance(lhs, assignable):
        #    raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
        # Indexed types implement shape, but don't define it until later. This
        # causes issues in assignment validation. For now, matrices are defined
        # as anything with a shape that is not an Indexed
        lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs, Indexed)
        rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs, Indexed)
        # If lhs and rhs have same structure, then this assignment is ok
        if lhs_is_mat:
            if not rhs_is_mat:
                raise ValueError("Cannot assign a scalar to a matrix.")
            elif lhs.shape != rhs.shape:
                raise ValueError("Dimensions of lhs and rhs don't align.")
        elif rhs_is_mat and not lhs_is_mat:
            raise ValueError("Cannot assign a matrix to a scalar.")
        return Relational.__new__(cls, lhs, rhs, **assumptions)

class AssignmentIf(Assignment):
    rel_op = '=='
    __slots__ = []

class For(Basic):
    def __new__(cls, idx, expr, **kw_args):

        if isinstance(idx, Idx):
            index = Tuple(idx)
        elif is_sequence(idx):
            index = Tuple(*idx)
        else:
            raise TypeError("Loop object requires an Idx or a list of Idx.")

        if is_sequence(expr):
            expr = Tuple(*expr)
        else:
            expr = Tuple(expr)

        args = index, expr
        obj = Basic.__new__(cls, *args, **kw_args)
        return obj

    @property
    def index(self):
        return self.args[0]        

    @property
    def expr(self):
        return self.args[1]        

    def _sympystr(self, p):
        return p.doprint(p)

class If(Basic):
    def __new__(cls, *args, **options):
        newargs = []
        for ec in args:
            cond = ec[0]
            expr = ec[1]
            if is_sequence(expr):
                expr = Tuple(*expr)
            else:
                expr = Tuple(expr)
            newargs.append(Tuple(cond, expr))

        obj = Basic.__new__(cls, *newargs, **options)
        return obj
    
    @property
    def statement(self):
        return self.args      

class IdxRange(Basic):
    def __new__(cls, *args):
        if len(args) > 4:
            raise ValueError("Range is defined by (top), (start, stop) or (start, stop, step)")

        # for a in args:
        #     print(type(a))
        #     if not isinstance(a, (int, Symbol, Add)):
        #         raise TypeError("args in Range must be integer or SymPy Idx")

        # expand range
        label = args[0]

        slc = slice(*args[1:])

        if slc.step == 0:
            raise ValueError("step cannot be 0")

        start, stop, step = slc.start or 0, slc.stop, slc.step or 1

        start = sympify(start)
        stop = sympify(stop)
        step = sympify(step)

        return Basic.__new__(cls, label, start, stop, step)

    @property
    def label(self):
        return self.args[0]

    @property
    def start(self):
        return self.args[1]

    @property
    def stop(self):
        return self.args[2]

    @property
    def step(self):
        return self.args[3]                


class IndexedIntBase(IndexedBase):
    is_integer = True
    is_Integer = True
