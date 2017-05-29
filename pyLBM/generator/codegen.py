from sympy.utilities.codegen import CodeGen, CodeGenError, ResultBase, Result, InputArgument, InOutArgument, OutputArgument
from sympy.core import Symbol, S, Expr, Tuple, Equality, Function, sympify
from sympy.core.compatibility import is_sequence, StringIO, string_types
from sympy.printing.codeprinter import AssignmentError
from sympy.core.sympify import _sympify, sympify

from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
                            MatrixExpr, MatrixSlice)
from sympy.core.basic import Basic

default_settings = {'export': True}

from .ast import For, If
from .printing.cython import cython_code, CythonCodePrinter
from .printing.numpy import numpy_code, NumpyCodePrinter
from .printing.loopy import loopy_code, LoopyCodePrinter

class Routine(object):
    """Generic description of evaluation routine for set of expressions.

    A CodeGen class can translate instances of this class into code in a
    particular language.  The routine specification covers all the features
    present in these languages.  The CodeGen part must raise an exception
    when certain features are not present in the target language.  For
    example, multiple return values are possible in Python, but not in C or
    Fortran.  Another example: Fortran and Python support complex numbers,
    while C does not.

    """

    def __init__(self, name, arguments, instructions, idx_vars, local_vars, settings={}):
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

        instructions : list
            Instructions of the routine.

        local_vars : list of Symbols
            These are used internally by the routine.

        global_vars : list of Symbols
            Variables which will not be passed into the function.

        """

        # extract all input symbols and all symbols appearing in an expression
        input_symbols = set([])
        symbols = set([])
        for arg in arguments:
            if isinstance(arg, OutputArgument):
                symbols.update(arg.expr.free_symbols)
            elif isinstance(arg, InputArgument):
                input_symbols.add(arg.name)
            elif isinstance(arg, InOutArgument):
                input_symbols.add(arg.name)
                symbols.update(arg.expr.free_symbols)
            else:
                raise ValueError("Unknown Routine argument: %s" % arg)

        for i in instructions:
            symbols.update(i.free_symbols)

        symbols = set([s.label if isinstance(s, Idx) else s for s in symbols])

        # Check that all symbols in the expressions are covered by
        # InputArguments/InOutArguments---subset because user could
        # specify additional (unused) InputArguments or local_vars.
        dummy = [i.label for i in idx_vars]
        notcovered = symbols.difference(
            input_symbols.union(dummy).union(local_vars))
        if notcovered != set([]):
            raise ValueError("Symbols needed for output are not in input " +
                             ", ".join([str(x) for x in notcovered]))

        self.name = name
        self.arguments = arguments
        self.instructions = instructions
        self.idx_vars = idx_vars
        self.local_vars = local_vars
        self.settings = settings

    def __str__(self):
        return self.__class__.__name__ + "({name!r}, {arguments}, {instructions}, {idx_vars}, {local_vars})".format(**self.__dict__)

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


def get_dims_and_symbol(expr):
    if isinstance(expr, Indexed):
        dims = tuple([ (S.Zero, dim - 1) for dim in expr.shape])
        symbol = expr.base.label
    elif isinstance(expr, Symbol):
        dims = []
        symbol = expr
    elif isinstance(expr, MatrixSymbol):
        dims = tuple([ (S.Zero, dim - 1) for dim in expr.shape if dim != 1])
        symbol = expr
    elif isinstance(expr, MatrixBase):
        # if we have a Matrix, we set line by line the code
        # useful when you have indexes like
        # A[i, j, 0] = B[i, j, 4]

        # todo: regarder que les matrices pour le in et out ont la meme taille
        # todo: regarder si le symbole est toujours le meme dans le terme de gauche (expr)
        for i in range(expr.shape[0]):
            symbol = expr[i].base.label
            dims = tuple([ (S.Zero, dim - 1) for dim in expr[i].base.shape if dim != 1])
    elif isinstance(expr, MatrixSlice):
        symbol = expr.parent
        dims = tuple([ (S.Zero, dim - 1) for dim in symbol.shape if dim != 1])
    else:
        raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                            "can define output arguments.")
    return dims, symbol

def extract(expressions, symbols):
    # extract arguments and instructions of the routine
    output_args = []
    instructions = []
    for expr in expressions:
        instructions.append(expr)
        if isinstance(expr, Equality):
            out_arg = expr.lhs
            expr = expr.rhs
            dims, symbol = get_dims_and_symbol(out_arg)

            if symbol in symbols:
                output_args.append(
                    InOutArgument(symbol, out_arg, expr, dimensions=dims))

                # avoid duplicate arguments
                symbols.remove(symbol)
        elif isinstance(expr, For):
            args, vals = extract(expr.expr, symbols)
            output_args += args
        elif isinstance(expr, If):
            for c, e in expr.statement:
                args, vals = extract(e, symbols)
                output_args += args
        else:
            raise TypeError("The expression must be a For or an equality (Eq).")
    return output_args, instructions


class LBMCodeGen(CodeGen):
    """Generator for Cython code.

    The .write() method inherited from CodeGen will output a code file <prefix>.pyx.

    """

    code_extension = None

    def routine(self, name, expr, argument_sequence, local_vars, settings):
        """Specialized Routine creation for Cython."""

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        self.settings = settings

        # local variables
        idx_vars = set()
        symbol_idx_vars = set()
        for l in expressions.atoms(For):
            idx_vars.update({i for i in l.atoms(Idx)})
            # remove symbols that have the same name of Idx in loop
            name_idx = [i.label.name for i in l.atoms(Idx)]
            symbol_idx_vars.update({i for i in l.atoms(Symbol) if i.name in name_idx})

        score_table = {}
        for i in idx_vars:
            score_table[i] = 0

        def rate_index_position(p):
            return p*5

        arrays = expressions.atoms(Indexed)
        for arr in arrays:
            for p, ind in enumerate(arr.indices):
                try:
                    score_table[ind] += rate_index_position(p)
                except KeyError:
                    pass

        idx_order = sorted(idx_vars, key=lambda x: score_table[x])

        # local variables
        local_vars = set() if local_vars is None else set(local_vars)

        # symbols that should be arguments
        symbols = expressions.free_symbols - idx_vars - local_vars - symbol_idx_vars

        new_symbols = set([])
        new_symbols.update(symbols)

        for symbol in symbols:
            if isinstance(symbol, Idx):
                new_symbols.remove(symbol)
                if symbol.label in idx_vars:
                    new_symbols.update(symbol.args[1].free_symbols)
                else:
                    new_symbols.update([symbol.label])
        symbols = new_symbols

        output_args, instructions = extract(expressions, symbols)

        arg_list = []

        # setup input argument list
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            if symbol in array_symbols:
                dims = []
                array = array_symbols[symbol]
                for dim in array.shape:
                    if dim != 1:
                        dims.append((S.Zero, dim - 1))
                metadata = {'dimensions': dims}
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
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        return Routine(name, arg_list, instructions, idx_order, local_vars, settings)

    def code_generator(self, expr, assign_to=None):
        pass

    def _get_symbol(self, s):
        """Print the symbol appropriately."""
        return self.code_generator(s).strip()

    def _call_printer(self, routine):
        code_lines = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        for instruction in routine.instructions:
            constants, not_supported, expr = self.code_generator(instruction, human=False)
            code_lines.append("%s\n" % (expr))
        return code_lines

        # declarations = []
        # code_lines = []
        # for i, result in enumerate(routine.results):
        #     print(self.settings)
        #     constants, not_supported, jl_expr = self.code_generator(result, human=False, **self.settings)

        #     for obj, v in sorted(constants, key=str):
        #         declarations.append(
        #             "%s = %s\n" % (obj, v))
        #     for obj in sorted(not_supported, key=str):
        #         if isinstance(obj, Function):
        #             name = obj.func
        #         else:
        #             name = obj
        #         declarations.append(
        #             "# unsupported: %s\n" % (name))
        #     code_lines.append("%s\n" % (jl_expr))
        # return declarations + code_lines

    def _get_routine_opening(self, routine):
        return []

    def _get_routine_ending(self, routine):
        return []


class CythonCodeGen(LBMCodeGen):
    """Generator for Cython code.

    The .write() method inherited from CodeGen will output a code file <prefix>.pyx.

    """

    code_extension = "pyx"

    def code_generator(self, expr, assign_to=None, **settings):
        return cython_code(expr, assign_to, **settings)

    def _get_header(self):
        code_lines = ["#!python\n",
                      "#cython: boundscheck=False\n",
                      "#cython: wraparound=False\n",
                      "#cython: cdivision=True\n",
                      "#cython: binding=True\n",
                      "#import cython\n",
                      "from libc.math cimport *\n",
                     ]
        return code_lines + ["\n\n"]

    def _preprocessor_statements(self, prefix):
        # code_lines = ["#!python\n",
        #               "#cython: boundscheck=False\n",
        #               "#cython: wraparound=False\n",
        #               "#cython: cdivision=True\n",
        #               "#cython: binding=True\n",
        #               "#import cython\n",
        #               "from libc.math cimport *\n",
        #              ]
        # return code_lines + ["\n\n"]
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
        for i, arg in enumerate(routine.arguments):
            if isinstance(arg, OutputArgument):
                raise CodeGenError("Cython: invalid argument of type %s" %
                                   str(type(arg)))

            if isinstance(arg, (InputArgument, InOutArgument)):
                name = self._get_symbol(arg.name)
                if not arg.dimensions:
                    # If it is a scalar
                    if isinstance(arg, ResultBase):
                        # if it is an output
                        args.append((arg.get_datatype('C'), "*%s" % name))
                    else:
                        # if it is an output
                        args.append((arg.get_datatype('C'), name))
                else:
                    if not export and len(arg.dimensions) == 1:
                        # if the dimension is 1
                        args.append((arg.get_datatype('C'), "*%s" % name))
                    else:
                        args.append((arg.get_datatype('C') + '[' + ', '.join([':']*len(arg.dimensions)) + ':1]', "%s" % name))

        args = ", ".join([ "%s %s" % t for t in args])
        code_list.append("%s(%s)%s\n" % (routine.name, args, ":" if export else " nogil:"))
        code_list = [ "".join(code_list) ]

        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        args = []
        for g in routine.local_vars:
            if isinstance(g, Symbol):
                args.append("cdef double %s\n"%(self._get_symbol(g)))
            else:
                shape = [d for d in g.shape if d!=1]
                args.append("cdef double %s[%s]\n"%(self._get_symbol(g), ','.join("%s"%s for s in shape)))
        return ["".join(args)]

    def _declare_locals(self, routine):
        s = []
        for l in routine.idx_vars:
            s.append("cdef int %s\n" % l.label)
        return s + ['\n']

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


class NumpyCodeGen(LBMCodeGen):
    """Generator for Cython code.

    The .write() method inherited from CodeGen will output a code file <prefix>.pyx.

    """

    code_extension = "py"

    def code_generator(self, expr, assign_to=None, **settings):
        return numpy_code(expr, assign_to, **settings)

    def _get_header(self):
        code_lines = ["import numpy as np\n",
                     ]
        return code_lines + ["\n\n"]

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
                raise CodeGenError("Numpy: invalid argument of type %s" %
                                   str(type(arg)))

            if isinstance(arg, (InputArgument, InOutArgument)):
                name = self._get_symbol(arg.name)
                args.append(name)
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
        return ["#end\n"]

    def _indent_code(self, codelines):
        p = NumpyCodePrinter()
        return p.indent_code(codelines)

    def dump_py(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_py.extension = code_extension
    dump_py.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_py]

class LoopyCodeGen(LBMCodeGen):
    """Generator for Cython code.

    The .write() method inherited from CodeGen will output a code file <prefix>.pyx.

    """

    code_extension = "py"

    _default_settings = {"prefetch": None}

    def code_generator(self, expr, assign_to=None, **settings):
        return loopy_code(expr, assign_to, **settings)

    def _preprocessor_statements(self, prefix):
        return []

    def _get_header(self):
        code_lines = ["import loopy as lp\n",
                      "import numpy as np\n"
                     ]
        return code_lines + ["\n\n"]

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
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
                    dtype = arg.get_datatype('PYTHON')
                    if dtype == 'int':
                        dtype = 'np.int32'
                    args.append('lp.GlobalArg("{name}", dtype={dtype}, shape="{shape}")'.format(name=name, dtype=dtype, shape=", ".join(dims)))
                else:
                    args.append('lp.ValueArg("{name}", dtype={dtype})'.format(name=name, dtype=arg.get_datatype('PYTHON')))
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

        for i, idx in enumerate(routine.idx_vars[-1::-1]):
            code_list.append('{name} = lp.split_iname({name}, "{label}", {block}, outer_tag="g.{ilabel}", inner_tag="l.{ilabel}")'.format(name = routine.name,
            label = "%s_"%idx.label, ilabel=i, block=block_size[i]))
        code_list.append('{name} = lp.expand_subst({name})\n'.format(name=routine.name))
        code_list.append('{name} = lp.set_options({name}, no_numpy = True)\n'.format(name=routine.name))

        prefetch = routine.settings.get("prefetch", None)

        if prefetch:
            for var in prefetch:
                indices = []
                for idx in routine.idx_vars:
                    indices.append("%s__inner"%idx.label)
                # for i in range(var.rank):
                #     if isinstance(var.indices[i], Idx):
                #         indices.append("%s__inner"%var.indices[i].label)
                code_list.append('{name} = lp.add_prefetch({name}, "{var}", "{label}", fetch_bounding_box=True)\n'.format(name=routine.name, var=var.base.label, label=",".join(indices)))
#            print("PREFETCH")
        #     label = []
        #     for var in self._settings["prefetch"]:
        #         indices = []
        #         for i in var.indices:
        #             if isinstance(i, Idx):
        #                 indices.append("%s__inner"%i.label)
        #         code_list.append('{name} = lp.add_prefetch({name}, "{var}", "{label}", fetch_bounding_box=True)'.format(name=routine.name, var=var.label, label=",".join(indices)))
        #print(LoopyCodePrinter()._sort_optimized(routine.local_vars, routine.instructions))
# one_time_step = lp.split_iname(one_time_step, "ii", 16, outer_tag="g.1", inner_tag="l.1")
# one_time_step = lp.split_iname(one_time_step, "jj", 16, outer_tag="g.0", inner_tag="l.0")
# one_time_step = lp.expand_subst(one_time_step)
# one_time_step = lp.add_prefetch(one_time_step, "f", "ii_inner,jj_inner", fetch_bounding_box=True)
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

def get_code_generator(language, project):
    CodeGenClass = {"NUMPY" : NumpyCodeGen,
                    "CYTHON": CythonCodeGen,
                    "LOOPY": LoopyCodeGen}.get(language.upper())
    if CodeGenClass is None:
        raise ValueError("Language '%s' is not supported." % language)
    return CodeGenClass(project)

def codegen(name_expr, language, prefix=None, project="project",
            to_files=False, header=True, empty=True, argument_sequence=None,
            global_vars=None, settings={}):
    # Initialize the code generator.
    code_gen = get_code_generator(language, project)

    if isinstance(name_expr[0], string_types):
        # single tuple is given, turn it into a singleton list with a tuple.
        name_expr = [name_expr]

    if prefix is None:
        prefix = name_expr[0][0]

    # Construct Routines appropriate for this code_gen from (name, expr) pairs.
    routines = []
    for name, expr in name_expr:
        routines.append(code_gen.routine(name, expr, argument_sequence,
                                         global_vars, settings))

    # Write the code.
    return code_gen.write(routines, prefix, to_files, header, empty)

def make_routine(name_expr, argument_sequence=None, local_vars=None, settings={}):
    if isinstance(name_expr[0], string_types):
        # single tuple is given, turn it into a singleton list with a tuple.
        name_expr = [name_expr]

    routines = []
    for name, expr in name_expr:
        routines.append(LBMCodeGen().routine(name, expr, argument_sequence,
                                             local_vars, settings))

    return routines
