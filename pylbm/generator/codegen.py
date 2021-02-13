# FIXME: make pylint happy !
#pylint: disable=all

# from sympy.utilities.codegen import CodeGen, CodeGenError, CodeGenArgumentListError, ResultBase, Result, InputArgument, InOutArgument, OutputArgument
# from sympy.core import Symbol, S, Expr, Tuple, Equality, Function, sympify
# from sympy.core.compatibility import is_sequence, StringIO, string_types
# from sympy.printing.codeprinter import AssignmentError
# from sympy.core.sympify import _sympify, sympify

# from sympy.tensor import Idx, Indexed, IndexedBase
# from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
#                             MatrixExpr, MatrixSlice)
# from sympy.core.basic import Basic

# default_settings = {'export': True}

# from .ast import For, If
# from .printing.cython import cython_code, CythonCodePrinter
# from .printing.numpy import numpy_code, NumpyCodePrinter
# from .printing.loopy import loopy_code, LoopyCodePrinter

"""
module for generating C, C++, Fortran77, Fortran90, Julia, Rust
and Octave/Matlab routines that evaluate sympy expressions.
This module is work in progress.
Only the milestones with a '+' character in the list below have been completed.

--- How is sympy.utilities.codegen different from sympy.printing.ccode? ---

We considered the idea to extend the printing routines for sympy functions in
such a way that it prints complete compilable code, but this leads to a few
unsurmountable issues that can only be tackled with dedicated code generator:

- For C, one needs both a code and a header file, while the printing routines
  generate just one string. This code generator can be extended to support
  .pyf files for f2py.

- SymPy functions are not concerned with programming-technical issues, such
  as input, output and input-output arguments. Other examples are contiguous
  or non-contiguous arrays, including headers of other libraries such as gsl
  or others.

- It is highly interesting to evaluate several sympy functions in one C
  routine, eventually sharing common intermediate results with the help
  of the cse routine. This is more than just printing.

- From the programming perspective, expressions with constants should be
  evaluated in the code generator as much as possible. This is different
  for printing.

--- Basic assumptions ---

* A generic Routine data structure describes the routine that must be
  translated into C/Fortran/... code. This data structure covers all
  features present in one or more of the supported languages.

* Descendants from the CodeGen class transform multiple Routine instances
  into compilable code. Each derived class translates into a specific
  language.

* In many cases, one wants a simple workflow. The friendly functions in the
  last part are a simple api on top of the Routine/CodeGen stuff. They are
  easier to use, but are less powerful.

--- Milestones ---

+ First working version with scalar input arguments, generating C code,
  tests
+ Friendly functions that are easier to use than the rigorous
  Routine/CodeGen workflow.
+ Integer and Real numbers as input and output
+ Output arguments
+ InputOutput arguments
+ Sort input/output arguments properly
+ Contiguous array arguments (numpy matrices)
+ Also generate .pyf code for f2py (in autowrap module)
+ Isolate constants and evaluate them beforehand in double precision
+ Fortran 90
+ Octave/Matlab

- Common Subexpression Elimination
- User defined comments in the generated code
- Optional extra include lines for libraries/objects that can eval special
  functions
- Test other C compilers and libraries: gcc, tcc, libtcc, gcc+gsl, ...
- Contiguous array arguments (sympy matrices)
- Non-contiguous array arguments (sympy matrices)
- ccode must raise an error when it encounters something that can not be
  translated into c. ccode(integrate(sin(x)/x, x)) does not make sense.
- Complex numbers as input and output
- A default complex datatype
- Include extra information in the header: date, user, hostname, sha1
  hash, ...
- Fortran 77
- C++
- Python
- Julia
- Rust
- ...

"""

import os
import textwrap
from io import StringIO

from sympy import __version__ as sympy_version
from .ast import Assignment, For, If, WithBody
# from sympy.codegen import Assignment
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic
from sympy.core.compatibility import is_sequence
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
                            MatrixExpr, MatrixSlice)

from .printing.pycode import NumPyPrinter
from .printing.cython import CythonCodePrinter
from .printing.loopy import LoopyCodePrinter

__all__ = [
    # description of routines
    "Routine",
    "Argument", "InputArgument", "OutputArgument", "Result",
    # routines -> code
    "CodeGen", "CythonCodeGen", "NumPyCodeGen", "LoopyCodeGen",
    # friendly functions
    "codegen", "make_routine",
]


#
# Description of routines
#


class Routine:
    """Generic description of evaluation routine for set of expressions.

    A CodeGen class can translate instances of this class into code in a
    particular language.  The routine specification covers all the features
    present in these languages.  The CodeGen part must raise an exception
    when certain features are not present in the target language.  For
    example, multiple return values are possible in Python, but not in C or
    Fortran.  Another example: Fortran and Python support complex numbers,
    while C does not.

    """

    def __init__(self, name, arguments, results, statements, idx_vars, local_vars, global_vars, settings):
        """Initialize a Routine instance.

        Parameters
        ==========

        name : string
            Name of the routine.

        arguments : list of Arguments
            These are things that appear in arguments of a routine, often
            appearing on the right-hand side of a function call.  These are
            commonly InputArguments but in some languages, they can also be
            OutputArguments or InOutArguments (e.g., pass-by-reference in C
            code).

        results : list of Results
            These are the return values of the routine, often appearing on
            the left-hand side of a function call.  The difference between
            Results and OutputArguments and when you should use each is
            language-specific.

        statements : list of Expressions
            These are the different lines corresponding to the body of a
            routine.

        local_vars : list of Results
            These are variables that will be defined at the beginning of the
            function.

        global_vars : list of Symbols
            Variables which will not be passed into the function.

        """

        # extract all input symbols and all symbols appearing in an expression
        input_symbols = set()
        output_symbols = set()
        symbols = set()
        for arg in arguments:
            if isinstance(arg, OutputArgument):
                output_symbols.add(arg.name)
                symbols.update(arg.expr.free_symbols - arg.expr.atoms(Indexed))
            elif isinstance(arg, InputArgument):
                input_symbols.add(arg.name)
            elif isinstance(arg, InOutArgument):
                input_symbols.add(arg.name)
                symbols.update(arg.expr.free_symbols - arg.expr.atoms(Indexed))
            else:
                raise ValueError("Unknown Routine argument: %s" % arg)

        for r in results:
            if not isinstance(r, Result):
                raise ValueError("Unknown Routine result: %s" % r)
            symbols.update(r.expr.free_symbols - r.expr.atoms(Indexed))

        local_symbols = set()
        for r in local_vars:
            if isinstance(r, Result):
                symbols.update(r.expr.free_symbols - r.expr.atoms(Indexed))
                local_symbols.add(r.name)
            else:
                local_symbols.add(r)

        symbols = {s.label if isinstance(s, Idx) else s for s in symbols}

        # Check that all symbols in the expressions are covered by
        # InputArguments/InOutArguments---subset because user could
        # specify additional (unused) InputArguments or local_vars.
        notcovered = symbols.difference(
            input_symbols.union(output_symbols).union(idx_vars).union(local_symbols).union(global_vars))
        if notcovered != set():
            raise ValueError("Symbols needed for output are not in input " +
                             ", ".join([str(x) for x in notcovered]))

        self.name = name
        self.arguments = arguments
        self.results = results
        self.statements = statements
        self.idx_vars = idx_vars
        self.local_vars = local_vars
        self.global_vars = global_vars
        self.settings = settings

    def __str__(self):
        return self.__class__.__name__ + "({name!r}, {arguments}, {results}, {statements}, {local_vars}, {global_vars})".format(**self.__dict__)

    __repr__ = __str__

    @property
    def variables(self):
        """Returns a set of all variables possibly used in the routine.

        For routines with unnamed return values, the dummies that may or
        may not be used will be included in the set.

        """
        v = set(self.local_vars)
        for arg in self.arguments:
            v.add(arg.name)
        for res in self.results:
            v.add(res.result_var)
        return v

    @property
    def result_variables(self):
        """Returns a list of OutputArgument, InOutArgument and Result.

        If return values are present, they are at the end ot the list.
        """
        args = [arg for arg in self.arguments if isinstance(
            arg, (OutputArgument, InOutArgument))]
        args.extend(self.results)
        return args

COMPLEX_ALLOWED = False
def get_default_datatype(expr, complex_allowed=None):
    """Derives an appropriate datatype based on the expression."""
    if complex_allowed is None:
        complex_allowed = COMPLEX_ALLOWED
    if complex_allowed:
        final_dtype = "complex"
    else:
        final_dtype = "float"
    if expr.is_integer:
        return "int"
    elif expr.is_real:
        return "float"
    elif isinstance(expr, MatrixBase):
        #check all entries
        dt = "int"
        for element in expr:
            if dt == "int" and not element.is_integer:
                dt = "float"
            if dt == "float" and not element.is_real:
                return final_dtype
        return dt
    else:
        return final_dtype


class Variable:
    """Represents a typed variable."""

    def __init__(self, name, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol or MatrixSymbol

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimension : sequence containing tupes, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        if not isinstance(name, (Symbol, MatrixSymbol)):
            raise TypeError("The first argument must be a sympy symbol.")
        if datatype is None:
            datatype = get_default_datatype(name)
        if dimensions and not isinstance(dimensions, (tuple, list)):
            raise TypeError(
                "The dimension argument must be a sequence of tuples")

        self._name = name
        self.datatype = datatype
        self.dimensions = dimensions
        self.precision = precision

    def __str__(self):
        return "%s(%r)" % (self.__class__.__name__, self.name)

    __repr__ = __str__

    @property
    def name(self):
        return self._name

class Argument(Variable):
    """An abstract Argument data structure: a name and a data type.

    This structure is refined in the descendants below.

    """
    pass


class InputArgument(Argument):
    pass


class ResultBase:
    """Base class for all "outgoing" information from a routine.

    Objects of this class stores a sympy expression, and a sympy object
    representing a result variable that will be used in the generated code
    only if necessary.

    """
    def __init__(self, expr, result_var):
        self.expr = expr
        self.result_var = result_var

    def __str__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.expr,
            self.result_var)

    __repr__ = __str__


class OutputArgument(Argument, ResultBase):
    """OutputArgument are always initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol, MatrixSymbol
            The name of this variable.  When used for code generation, this
            might appear, for example, in the prototype of function in the
            argument list.

        result_var : Symbol, Indexed
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".

        expr : object
            The expression that should be output, typically a SymPy
            expression.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimension : sequence containing tupes, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """

        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.result_var, self.expr)

    __repr__ = __str__


class InOutArgument(Argument, ResultBase):
    """InOutArgument are never initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        if not datatype:
            datatype = get_default_datatype(expr)
        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)
    __init__.__doc__ = OutputArgument.__init__.__doc__


    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.expr,
            self.result_var)

    __repr__ = __str__


class Result(Variable, ResultBase):
    """An expression for a return value.

    The name result is used to avoid conflicts with the reserved word
    "return" in the python language.  It is also shorter than ReturnValue.

    These may or may not need a name in the destination (e.g., "return(x*y)"
    might return a value without ever naming it).

    """

    def __init__(self, expr, name=None, result_var=None, datatype=None,
                 dimensions=None, precision=None):
        """Initialize a return value.

        Parameters
        ==========

        expr : SymPy expression

        name : Symbol, MatrixSymbol, optional
            The name of this return variable.  When used for code generation,
            this might appear, for example, in the prototype of function in a
            list of return values.  A dummy name is generated if omitted.

        result_var : Symbol, Indexed, optional
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".  Defaults to
            `name` if omitted.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the expr argument.

        dimension : sequence containing tupes, optional
            If present, this variable is interpreted as an array,
            where this sequence of tuples specifies (lower, upper)
            bounds for each index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        # Basic because it is the base class for all types of expressions
        if not isinstance(expr, (Basic, MatrixBase)):
            raise TypeError("The first argument must be a sympy expression.")

        if name is None:
            name = 'result_%d' % abs(hash(expr))

        if datatype is None:
            #try to infer data type from the expression
            datatype = get_default_datatype(expr)

        if isinstance(name, str):
            if isinstance(expr, (MatrixBase, MatrixExpr)):
                name = MatrixSymbol(name, *expr.shape)
            else:
                name = Symbol(name)

        if result_var is None:
            result_var = name

        Variable.__init__(self, name, datatype=datatype,
                          dimensions=dimensions, precision=precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.expr, self.name,
            self.result_var)

    __repr__ = __str__


#
# Transformation of routine objects into code
#

class CodeGen:
    """Abstract class for the code generators."""

    printer = None  # will be set to an instance of a CodePrinter subclass
    default_datatypes = None
    has_output = True

    def _indent_code(self, codelines):
        return self.printer.indent_code(codelines)

    def _printer_method_with_settings(self, method, settings=None, *args, **kwargs):
        settings = settings or {}
        ori = {k: self.printer._settings[k] for k in settings if self.printer._settings.get(k, None)}
        for k, v in settings.items():
            self.printer._settings[k] = v
        result = getattr(self.printer, method)(*args, **kwargs)
        for k, v in ori.items():
            self.printer._settings[k] = v
        return result

    def _get_type(self, stype):
        return self.default_datatypes[stype]

    def _get_symbol(self, s):
        """Returns the symbol as fcode prints it."""
        if self.printer._settings['human']:
            expr_str = self.printer.doprint(s)
        else:
            constants, not_supported, expr_str = self.printer.doprint(s)
            if constants or not_supported:
                raise ValueError("Failed to print %s" % str(s))
        return expr_str.strip()

    def _cse_process(self, expr):
        from sympy.simplify.cse_main import cse

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            for e in expr:
                if not e.is_Equality:
                    raise CodeGenError("Lists of expressions must all be Equalities. {} is not.".format(e))

            # create a list of right hand sides and simplify them
            rhs = [e.rhs for e in expr]
            common, simplified = cse(rhs)

            # pack the simplified expressions back up with their left hand sides
            expr = [Equality(e.lhs, rhs) for e, rhs in zip(expr, simplified)]
        else:
            rhs = [expr]

            if isinstance(expr, Equality):
                common, simplified = cse(expr.rhs) #, ignore=in_out_args)
                expr = Equality(expr.lhs, simplified[0])
            else:
                common, simplified = cse(expr)
                expr = simplified

        local_vars = {Result(b, a): None for a, b in common}

        local_symbols = {a for a,_ in common}
        local_expressions = Tuple(*[b for _,b in common])
        return local_vars, local_symbols, local_expressions, expr

    def _update_symbols(self, symbols):
        new_symbols = set()
        new_symbols.update(symbols)

        for symbol in symbols:
            if isinstance(symbol, Idx):
                new_symbols.remove(symbol)
                new_symbols.update(symbol.args[1].free_symbols)
            if isinstance(symbol, Indexed):
                new_symbols.remove(symbol)
        return new_symbols

    def _get_statements_and_outputs(self, name, expressions, symbols, local_vars):
        # Decide whether to use output argument or return value
        return_val = []
        output_args = []

        def extract(expressions, return_index=1):
            new_expr = []
            for iexpr, expr in enumerate(expressions):
                output_name = 'out%s' % (return_index)
                if isinstance(expr, (Equality, Assignment)):
                    new_expr.append(expr)
                    out_arg = expr.lhs
                    expr = expr.rhs
                    if isinstance(out_arg, Indexed):
                        dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                        symbol = out_arg.base.label
                    elif isinstance(out_arg, Symbol):
                        dims = []
                        symbol = out_arg
                    elif isinstance(out_arg, MatrixSymbol):
                        dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                        symbol = out_arg
                    else:
                        raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                        "can define output arguments.")

                    if symbol in symbols:
                        if self.has_output:
                            if expr.has(symbol):
                                output_args.append(
                                    InOutArgument(symbol, out_arg, expr, dimensions=dims))
                            else:
                                output_args.append(
                                    OutputArgument(symbol, out_arg, expr, dimensions=dims))
                        else:
                            if expr.has(symbol):
                                output_args.append(
                                    InOutArgument(symbol, out_arg, expr, dimensions=dims))
                            if isinstance(out_arg, Indexed):
                                output_args.append(
                                    InOutArgument(symbol, out_arg, expr, dimensions=dims))
                            return_val.append(Result(expr, name=symbol, result_var=out_arg))

                    # remove duplicate arguments when they are not local variables
                    if symbol not in local_vars:
                        # avoid duplicate arguments
                        symbols.remove(symbol)
                elif isinstance(expr, WithBody): # we should add all the classes which have a CodeBlock
                    new_expr.append(expr)
                elif isinstance(expr, (ImmutableMatrix, MatrixSlice)):
                    # Create a "dummy" MatrixSymbol to use as the Output arg
                    out_arg = MatrixSymbol(output_name, *expr.shape)
                    return_index += 1
                    new_expr.append(Assignment(out_arg, expr))
                    dims = tuple([(S.Zero, dim - 1) for dim in out_arg.shape])
                    if self.has_output:
                        output_args.append(
                            OutputArgument(out_arg, out_arg, expr, dimensions=dims))
                    else:
                        return_val.append(Result(expr, name=out_arg.name, result_var=out_arg))

                else:
                    r = Result(expr, output_name)
                    return_index += 1
                    new_expr.append(Assignment(r.result_var, r.expr))
                    return_val.append(r)
            return new_expr

        statements = extract(expressions)
        return statements, return_val, output_args

    def __init__(self, project="project", cse=False):
        """Initialize a code generator.

        Derived classes will offer more options that affect the generated
        code.

        """
        self.project = project
        self.cse = cse

    def routine(self, name, expr, argument_sequence=None, user_local_vars=None, global_vars=None, settings=None):
        """Creates an Routine object that is appropriate for this language.

        This implementation is appropriate for at least C/Fortran.  Subclasses
        can override this if necessary.

        Here, we assume at most one return value (the l-value) which must be
        scalar.  Additional outputs are OutputArguments (e.g., pointers on
        right-hand-side or pass-by-reference).  Matrices are always returned
        via OutputArguments.  If ``argument_sequence`` is None, arguments will
        be ordered alphabetically, but with all InputArguments first, and then
        OutputArgument and InOutArguments.

        """

        if self.cse:
            local_vars, local_symbols, local_expressions, expr = self._cse_process(expr)
        else:
            local_expressions = Tuple()

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        idx_vars = {i for i in expressions.atoms(Idx)}

        if self.cse:
            if {i.label for i in expressions.atoms(Idx)} != set():
                raise CodeGenError("CSE and Indexed expressions do not play well together yet")
        else:
            # local variables for indexed expressions
            local_vars = set()
            local_symbols = idx_vars.copy()

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)
        user_local_vars = set() if user_local_vars is None else set(user_local_vars)

        local_vars.update(user_local_vars)
        local_symbols.update(local_vars)

        # symbols that should be arguments
        symbols = (expressions.free_symbols | local_expressions.free_symbols) - local_symbols - global_vars
        symbols = self._update_symbols(symbols)

        statements, return_val, output_args = self._get_statements_and_outputs(name, expressions, symbols, local_vars)
        expressions = Tuple(*statements)

        arg_list = []

        # setup input argument list

        # helper to get dimensions for data for array-like args
        def dimensions(s):
            return [(S.Zero, dim - 1) for dim in s.shape]

        array_symbols = {}
        for array in expressions.atoms(Indexed) | local_expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol) | local_expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            if symbol in array_symbols:
                array = array_symbols[symbol]
                metadata = {'dimensions': dimensions(array)}
            else:
                metadata = {}

            arg_list.append(InputArgument(symbol, **metadata))

        output_args.sort(key=lambda x: str(x.name))
        arg_list.extend(output_args)

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    if isinstance(symbol, (IndexedBase, MatrixSymbol)):
                        metadata = {'dimensions': dimensions(symbol)}
                    else:
                        metadata = {}
                    new_args.append(InputArgument(symbol, **metadata))
            arg_list = new_args

        return Routine(name, arg_list, return_val, statements, idx_vars, local_vars, global_vars, settings)

    def write(self, routines, prefix, to_files=False, header=True, empty=True):
        """Writes all the source code files for the given routines.

        The generated source is returned as a list of (filename, contents)
        tuples, or is written to files (see below).  Each filename consists
        of the given prefix, appended with an appropriate extension.

        Parameters
        ==========

        routines : list
            A list of Routine instances to be written

        prefix : string
            The prefix for the output files

        to_files : bool, optional
            When True, the output is written to files.  Otherwise, a list
            of (filename, contents) tuples is returned.  [default: False]

        header : bool, optional
            When True, a header comment is included on top of each source
            file. [default: True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files. [default: True]

        """
        if to_files:
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                with open(filename, "w") as f:
                    dump_fn(self, routines, f, prefix, header, empty)
        else:
            result = []
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                contents = StringIO()
                dump_fn(self, routines, contents, prefix, header, empty)
                result.append((filename, contents.getvalue()))
            return result

    def dump_code(self, routines, f, prefix, header=True, empty=True):
        """Write the code by calling language specific methods.

        The generated file contains all the definitions of the routines in
        low-level code and refers to the header file if appropriate.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """

        code_lines = self._preprocessor_statements(prefix)

        for routine in routines:
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_opening(routine))
            code_lines.extend(self._declare_arguments(routine))
            code_lines.extend(self._declare_globals(routine))
            code_lines.extend(self._declare_locals(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._call_printer(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_ending(routine))

        code_lines = self._indent_code(''.join(code_lines))

        if header:
            code_lines = ''.join(self._get_header() + [code_lines])

        if code_lines:
            f.write(code_lines)


class CodeGenError(Exception):
    pass


class CodeGenArgumentListError(Exception):
    @property
    def missing_args(self):
        return self.args[1]


header_comment = """Code generated with sympy %(version)s

See http://www.sympy.org/ for more information.

This file is part of '%(project)s'
"""

class CythonCodeGen(CodeGen):
    """Generator for Cython code.

    The .write() method inherited from CodeGen will output a code file <prefix>.pyx.

    """

    code_extension = "pyx"
    has_output = False

    default_datatypes = {'int': 'int',
                         'float': 'double',
                         'complex': 'double'}

    def __init__(self, project='project', printer=None, settings={}):
        super(CythonCodeGen, self).__init__(project)
        self.printer = printer or CythonCodePrinter(settings)

    def _get_header(self):
        code_lines = []
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            if line == '':
                code_lines.append("#\n")
            else:
                code_lines.append("#   %s\n" % line)
        code_lines += ["#!python\n",
                      "#cython: language_level=3\n",
                      "#cython: boundscheck=False\n",
                      "#cython: wraparound=False\n",
                      "#cython: cdivision=True\n",
                      "#cython: binding=True\n",
                      "#import cython\n",
                      "from libc.math cimport *\n",
                     ]
        return code_lines + ["\n\n"]

    def _preprocessor_statements(self, prefix):

        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        code_list = []

        export = True
        # export = self.settings.pop('export', True)
        if export:
            code_list.append("def ")
        else:
            code_list.append("cdef void ")

        # Inputs
        args = []

        for arg in routine.arguments:
            if isinstance(arg, OutputArgument):
                raise CodeGenError("Cython: invalid argument of type %s" %
                                   str(type(arg)))

            if isinstance(arg, (InputArgument, InOutArgument)):

                name = self._get_symbol(arg.name)

                if not arg.dimensions:
                    # If it is a scalar
                    if isinstance(arg, ResultBase):
                        # if it is an output
                        args.append("*%s %s" % (self._get_type(arg.datatype), name))
                    else:
                        # if it is an output
                        args.append("%s %s" % (self._get_type(arg.datatype), name))
                else:
                    if not export and len(arg.dimensions) == 1:
                        # if the dimension is 1
                        args.append("*%s %s" % (self._get_type(arg.datatype), name))
                    else:
                        array_type = self._get_type(arg.datatype) + '[' + ', '.join([':']*len(arg.dimensions)) + ':1]'
                        args.append("%s %s" % (array_type, name))
            else:
                raise CodeGenError("Unknown Argument type: %s" % type(arg))

        args = ", ".join(args)
        code_list.append("%s(%s)%s\n" % (routine.name, args, ":" if export else " nogil:"))
        code_list = [ "".join(code_list) ]

        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        return []

    def _declare_locals(self, routine):
        args = []
        for l in routine.idx_vars:
            args.append("cdef int %s\n" % self._get_symbol(l))

        for g in routine.local_vars:
            if isinstance(g, Symbol):
                args.append("cdef double %s\n"%(self._get_symbol(g)))
            else:
                shape = [d for d in g.shape if d!=1]
                args.append("cdef double %s[%s]\n"%(self._get_symbol(g), ','.join("%s"%s for s in shape)))
        return ["".join(args)]

    def _call_printer(self, routine):
        code_lines = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = result.name
                t = self._get_type(result.datatype)
                code_lines.append("{0} {1};\n".format(t, str(assign_to)))

        for statement in routine.statements:
            expr = statement
            if isinstance(statement, Equality):
                expr = Assignment(statement.lhs, statement.rhs)

            settings = routine.settings
            settings.update(dict(human=False, dereference=dereference))
            constants, not_c, c_expr = self._printer_method_with_settings(
                'doprint', settings,
                expr)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))
            code_lines.append("%s\n" % c_expr)

        return code_lines

    def _get_routine_ending(self, routine):
        return ["#end\n"]

    def _indent_code(self, codelines):
        p = CythonCodePrinter()
        return p.indent_code(codelines)

    def dump_pyx(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_pyx.extension = code_extension
    dump_pyx.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_pyx]

class NumPyCodeGen(CodeGen):
    """Generator for numpy code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.py.

    """

    code_extension = "py"
    has_output = False

    def __init__(self, project='project', printer=None, settings={}):
        super(NumPyCodeGen, self).__init__(project)
        self.printer = printer or NumPyPrinter(settings)

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            if line == '':
                code_lines.append("#\n")
            else:
                code_lines.append("#   %s\n" % line)
        code_lines.append("import numpy\n")
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        code_list = []
        code_list.append("def ")

        # Inputs
        args = []
        for i, arg in enumerate(routine.arguments):
            if isinstance(arg, OutputArgument):
                raise CodeGenError("NumPy: invalid argument of type %s" %
                                   str(type(arg)))
            if isinstance(arg, (InputArgument, InOutArgument)):
                args.append("%s" % self._get_symbol(arg.name))
        args = ", ".join(args)
        code_list.append("%s(%s):\n" % (routine.name, args))
        code_list = [ "".join(code_list) ]

        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        return []

    def _declare_locals(self, routine):
        return []

    def _get_routine_ending(self, routine):
        outs = []
        for result in routine.results:
            if isinstance(result, Result):
                # Note: name not result_var; want `y` not `y[i]` for Indexed
                s = self._get_symbol(result.name)
            else:
                raise CodeGenError("unexpected object in Routine results")
            outs.append(s)
        if outs:
            return ["return " + ", ".join(outs) + "\n#end\n"]
        else:
            return ["#end\n"]

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        # for i, result in enumerate(routine.results):

        for statement in routine.statements:
            expr = statement
            if isinstance(statement, Equality):
                expr = Assignment(statement.lhs, statement.rhs)

            constants, not_supported, py_expr = self._printer_method_with_settings(
                'doprint', dict(human=False), expr)

            for obj, v in sorted(constants, key=str):
                declarations.append(
                    "%s = %s\n" % (obj, v))
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append(
                    "# unsupported: %s\n" % (name))
            code_lines.append("%s\n" % (py_expr))
        return declarations + code_lines

    def _indent_code(self, codelines):
        p = NumPyPrinter()
        return p.indent_code(codelines)

    def dump_py(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_py.extension = code_extension
    dump_py.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_py]

class LoopyCodeGen(CodeGen):
    """Generator for Cython code.

    The .write() method inherited from CodeGen will output a code file <prefix>.pyx.

    """

    code_extension = "py"
    has_output = False

    _default_settings = {"prefetch": None}

    default_datatypes = {'int': 'int',
                         'float': 'float',
                         'complex': 'complex'}

    def __init__(self, project='project', printer=None, settings={}):
        super(LoopyCodeGen, self).__init__(project)
        self.printer = printer or LoopyCodePrinter(settings)

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        # for i, result in enumerate(routine.results):

        for statement in routine.statements:
            expr = statement
            if isinstance(statement, Equality):
                expr = Assignment(statement.lhs, statement.rhs)

            constants, not_supported, py_expr = self._printer_method_with_settings(
                'doprint', dict(human=False), expr)

            for obj, v in sorted(constants, key=str):
                declarations.append(
                    "%s = %s\n" % (obj, v))
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append(
                    "# unsupported: %s\n" % (name))
            code_lines.append("%s\n" % (py_expr))
        return declarations + code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_header(self):
        code_lines = ["import loopy as lp\n",
                      "import numpy as np\n"
                     ]
        return code_lines + ["\n\n"]

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        self.printer.instr = 0
        code_list = []
        code_list.append("%s = lp.make_kernel("%routine.name)
        name = []
        bounds = []
        for i in routine.idx_vars:
            if isinstance(i, Idx):
                name.append("%s_"%i.label)
                bounds.append("0<={ilabel}_<{upper}".format(ilabel=i.label, upper=i.upper-i.lower))

        if len(name) > 0:
            code_list.append('"{[%s]:%s}",'%(",".join(name), " and ".join(bounds)))
        code_list.append('"""  # noqa (silences flake8 line length warning)\n')
        code_list = [ "\n".join(code_list) ]
        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        return []

    def _declare_locals(self, routine):
        return []

    def _get_routine_ending(self, routine):
        code_list = []
        code_list.append('""",')

        # Inputs
        args = []
        dtypes = []
        for i, arg in enumerate(routine.arguments):
            if isinstance(arg, OutputArgument):
                raise CodeGenError("Loopy: invalid argument of type %s" %
                                   str(type(arg)))

            if isinstance(arg, (InputArgument, InOutArgument)):
                name = self._get_symbol(arg.name)
                if arg.dimensions:
                    dims = ["{}".format(d[1]-d[0]+1) for d in arg.dimensions]
                    dtype = self._get_type(arg.datatype)
                    if dtype == 'int':
                        dtype = 'np.int32'
                    args.append('lp.GlobalArg("{name}", dtype={dtype}, shape="{shape}")'.format(name=name, dtype=dtype, shape=", ".join(dims)))
                else:
                    args.append('lp.ValueArg("{name}", dtype={dtype})'.format(name=name, dtype=self._get_type(arg.datatype)))
        for i, arg in enumerate(routine.local_vars):
            if isinstance(arg, Symbol):
                args.append('lp.TemporaryVariable("{name}", dtype=float)'.format(name=self._get_symbol(arg)))
            else:
                dims = [d for d in arg.shape if d!=1]
                args.append('lp.TemporaryVariable("{name}", dtype=float, shape="{shape}")'.format(name=self._get_symbol(arg), shape=','.join("%s"%s for s in dims)))

        code_list.append('[')
        args = ",\n".join(args)
        code_list.append(args)
        code_list.append('])#endArg\n')

        # add type
        dim = len(routine.idx_vars)
        if dim == 1:
            block_size = [256]
        if dim == 2:
            block_size = [16, 16]
        if dim == 3:
            block_size = [4, 4, 4]

        # for i, idx in enumerate(routine.idx_vars[-1::-1]):
        i = 0
        for idx in routine.idx_vars:
            code_list.append('{name} = lp.split_iname({name}, "{label}", {block}, outer_tag="g.{ilabel}", inner_tag="l.{ilabel}")'.format(name = routine.name, label = "%s_"%idx.label, ilabel=i, block=block_size[i]))
            i += 1
        code_list.append('{name} = lp.expand_subst({name})\n'.format(name=routine.name))
        code_list.append('{name} = lp.set_options({name}, no_numpy = True)\n'.format(name=routine.name))

        prefetch = routine.settings.get("prefetch", None)
        if prefetch:
            for var in prefetch:
                indices = []
                for idx in routine.idx_vars:
                    indices.append("%s__inner"%idx.label)
                code_list.append('{name} = lp.add_prefetch({name}, "{var}", "{label}", fetch_bounding_box=True)\n'.format(name=routine.name, var=var.base.label, label=",".join(indices)))

        code_list = [ "\n".join(code_list) ]
        return code_list

    def _indent_code(self, codelines):
        p = LoopyCodePrinter()
        return p.indent_code(codelines)

    def dump_py(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_py.extension = code_extension
    dump_py.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_py]


def get_code_generator(language, project=None, standard=None, printer=None, settings=None):
    CodeGenClass = {"NUMPY": NumPyCodeGen,
                    "CYTHON": CythonCodeGen,
                    "LOOPY": LoopyCodeGen}.get(language.upper())
    if CodeGenClass is None:
        raise ValueError("Language '%s' is not supported." % language)
    return CodeGenClass(project, printer, settings=settings)

#
# Friendly functions
#

def codegen(name_expr, language=None, prefix=None, project="project",
            to_files=False, header=True, empty=True, argument_sequence=None,
            global_vars=None, standard=None, code_gen=None, printer = None, settings=None):
    """Generate source code for expressions in a given language.

    Parameters
    ==========

    name_expr : tuple, or list of tuples
        A single (name, expression) tuple or a list of (name, expression)
        tuples.  Each tuple corresponds to a routine.  If the expression is
        an equality (an instance of class Equality) the left hand side is
        considered an output argument.  If expression is an iterable, then
        the routine will have multiple outputs.

    language : string,
        A string that indicates the source code language.  This is case
        insensitive.  Currently, 'C', 'F95' and 'Octave' are supported.
        'Octave' generates code compatible with both Octave and Matlab.

    prefix : string, optional
        A prefix for the names of the files that contain the source code.
        Language-dependent suffixes will be appended.  If omitted, the name
        of the first name_expr tuple is used.

    project : string, optional
        A project name, used for making unique preprocessor instructions.
        [default: "project"]

    to_files : bool, optional
        When True, the code will be written to one or more files with the
        given prefix, otherwise strings with the names and contents of
        these files are returned. [default: False]

    header : bool, optional
        When True, a header is written on top of each source file.
        [default: True]

    empty : bool, optional
        When True, empty lines are used to structure the code.
        [default: True]

    argument_sequence : iterable, optional
        Sequence of arguments for the routine in a preferred order.  A
        CodeGenError is raised if required arguments are missing.
        Redundant arguments are used without warning.  If omitted,
        arguments will be ordered alphabetically, but with all input
        arguments first, and then output or in-out arguments.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    standard : string

    code_gen : CodeGen instance
        An instance of a CodeGen subclass. Overrides ``language``.

    Examples
    ========

    >>> from sympy.utilities.codegen import codegen
    >>> from sympy.abc import x, y, z
    >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    ...     ("f", x+y*z), "C89", "test", header=False, empty=False)
    >>> print(c_name)
    test.c
    >>> print(c_code)
    #include "test.h"
    #include <math.h>
    double f(double x, double y, double z) {
       double out1;
       out1 = x + y*z;
       return out1;
    }
    <BLANKLINE>
    >>> print(h_name)
    test.h
    >>> print(c_header)
    #ifndef PROJECT__TEST__H
    #define PROJECT__TEST__H
    double f(double x, double y, double z);
    #endif
    <BLANKLINE>

    Another example using Equality objects to give named outputs.  Here the
    filename (prefix) is taken from the first (name, expr) pair.

    >>> from sympy.abc import f, g
    >>> from sympy import Eq
    >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    ...      [("myfcn", x + y), ("fcn2", [Eq(f, 2*x), Eq(g, y)])],
    ...      "C99", header=False, empty=False)
    >>> print(c_name)
    myfcn.c
    >>> print(c_code)
    #include "myfcn.h"
    #include <math.h>
    double myfcn(double x, double y) {
       double out1;
       out1 = x + y;
       return out1;
    }
    void fcn2(double x, double y, double *f, double *g) {
       (*f) = 2*x;
       (*g) = y;
    }
    <BLANKLINE>

    If the generated function(s) will be part of a larger project where various
    global variables have been defined, the 'global_vars' option can be used
    to remove the specified variables from the function signature

    >>> from sympy.utilities.codegen import codegen
    >>> from sympy.abc import x, y, z
    >>> [(f_name, f_code), header] = codegen(
    ...     ("f", x+y*z), "F95", header=False, empty=False,
    ...     argument_sequence=(x, y), global_vars=(z,))
    >>> print(f_code)
    REAL*8 function f(x, y)
    implicit none
    REAL*8, intent(in) :: x
    REAL*8, intent(in) :: y
    f = x + y*z
    end function
    <BLANKLINE>

    """

    # Initialize the code generator.
    if language is None:
        if code_gen is None:
            raise ValueError("Need either language or code_gen")
    else:
        if code_gen is not None:
            raise ValueError("You cannot specify both language and code_gen.")
        code_gen = get_code_generator(language, project, standard, printer)

    if isinstance(name_expr[0], str):
        # single tuple is given, turn it into a singleton list with a tuple.
        name_expr = [name_expr]

    if prefix is None:
        prefix = name_expr[0][0]

    # Construct Routines appropriate for this code_gen from (name, expr) pairs.
    routines = []
    for name, expr in name_expr:
        routines.append(code_gen.routine(name, expr, argument_sequence,
                                         global_vars=global_vars, settings=settings))

    # Write the code.
    return code_gen.write(routines, prefix, to_files, header, empty)


def make_routine(name, expr, argument_sequence=None,
                 user_local_vars=None, global_vars=None, language="F95", settings=None):
    """A factory that makes an appropriate Routine from an expression.

    Parameters
    ==========

    name : string
        The name of this routine in the generated code.

    expr : expression or list/tuple of expressions
        A SymPy expression that the Routine instance will represent.  If
        given a list or tuple of expressions, the routine will be
        considered to have multiple return values and/or output arguments.

    argument_sequence : list or tuple, optional
        List arguments for the routine in a preferred order.  If omitted,
        the results are language dependent, for example, alphabetical order
        or in the same order as the given expressions.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    language : string, optional
        Specify a target language.  The Routine itself should be
        language-agnostic but the precise way one is created, error
        checking, etc depend on the language.  [default: "F95"].

    A decision about whether to use output arguments or return values is made
    depending on both the language and the particular mathematical expressions.
    For an expression of type Equality, the left hand side is typically made
    into an OutputArgument (or perhaps an InOutArgument if appropriate).
    Otherwise, typically, the calculated expression is made a return values of
    the routine.

    Examples
    ========

    >>> from sympy.utilities.codegen import make_routine
    >>> from sympy.abc import x, y, f, g
    >>> from sympy import Eq
    >>> r = make_routine('test', [Eq(f, 2*x), Eq(g, x + y)])
    >>> [arg.result_var for arg in r.results]
    []
    >>> [arg.name for arg in r.arguments]
    [x, y, f, g]
    >>> [arg.name for arg in r.result_variables]
    [f, g]
    >>> r.local_vars
    set()

    Another more complicated example with a mixture of specified and
    automatically-assigned names.  Also has Matrix output.

    >>> from sympy import Matrix
    >>> r = make_routine('fcn', [x*y, Eq(f, 1), Eq(g, x + g), Matrix([[x, 2]])])
    >>> [arg.result_var for arg in r.results]  # doctest: +SKIP
    [result_5397460570204848505]
    >>> [arg.expr for arg in r.results]
    [x*y]
    >>> [arg.name for arg in r.arguments]  # doctest: +SKIP
    [x, y, f, g, out_8598435338387848786]

    We can examine the various arguments more closely:

    >>> from sympy.utilities.codegen import (InputArgument, OutputArgument,
    ...                                      InOutArgument)
    >>> [a.name for a in r.arguments if isinstance(a, InputArgument)]
    [x, y]

    >>> [a.name for a in r.arguments if isinstance(a, OutputArgument)]  # doctest: +SKIP
    [f, out_8598435338387848786]
    >>> [a.expr for a in r.arguments if isinstance(a, OutputArgument)]
    [1, Matrix([[x, 2]])]

    >>> [a.name for a in r.arguments if isinstance(a, InOutArgument)]
    [g]
    >>> [a.expr for a in r.arguments if isinstance(a, InOutArgument)]
    [g + x]

    """

    # initialize a new code generator
    code_gen = get_code_generator(language)

    return code_gen.routine(name, expr, argument_sequence, user_local_vars, global_vars, settings)
