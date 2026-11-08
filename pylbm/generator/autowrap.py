# FIXME: make pylint happy !
# pylint: disable=all

import sys
import os
import shutil
import tempfile
from subprocess import STDOUT, CalledProcessError, check_output
import importlib

from .codegen import get_code_generator


class CodeWrapError(Exception):
    pass


class CodeWrapper(object):
    """Base Class for code wrappers"""

    _module_name = "wrapped_module"

    @property
    def filename(self):
        return "%s_%s" % (self._module_name, self._module_counter)

    @property
    def full_path(self):
        return "%s/%s_%s" % (self.workdir, self._module_name, self._module_counter)

    @property
    def module_name(self):
        return "%s_%s" % (self._module_name, self._module_counter)

    def __init__(
        self, generator, filepath=None, flags=[], generate=True, verbose=False
    ):
        """
        generator -- the code generator to use
        """
        self.generator = generator
        self.filepath = filepath
        self.workdir = self.filepath or tempfile.mkdtemp("_sympy_compile")
        self.flags = flags
        self.generate = generate
        self.verbose = verbose
        self._module_counter = 0

    def _generate_code(self, routines):
        self.generator.write(routines, self.full_path, True, True, False)

    def wrap_code(self, routines):
        if not os.access(self.workdir, os.F_OK):
            os.mkdir(self.workdir)

        try:
            sys.path.append(self.workdir)
            if self.generate:
                self._prepare_files(routines)
                self._generate_code(routines)
                self._process_files(routines)
            if self.module_name in sys.modules:
                sys.modules.pop(self.module_name)
            importlib.invalidate_caches()
            mod = importlib.import_module(self.module_name)
        finally:
            sys.path.remove(self.workdir)
            self._module_counter += 1
            if not self.filepath:
                try:
                    shutil.rmtree(self.workdir)
                except OSError:
                    # Could be some issues on Windows
                    pass

        return mod

    def _process_files(self, routine):
        command = self.command
        command.extend(self.flags)

        try:
            retoutput = check_output(command, stderr=STDOUT)
        except CalledProcessError as e:
            raise CodeWrapError(
                "Error while executing command: %s. Command output is:\n%s"
                % (" ".join(command), e.output.decode())
            )
        if self.verbose:
            print(retoutput)


class CythonCodeWrapper(CodeWrapper):
    setup_template = """\
import os
import subprocess
from setuptools import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np


def gcc_major(compiler):
    # Return the major version of GCC, or None for any other compiler
    # (clang and Intel compilers also define __GNUC__).
    try:
        out = subprocess.run(
            compiler + ["-dM", "-E", "-x", "c", os.devnull],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    macros = dict(
        line.split()[1:3] for line in out.splitlines()
        if line.startswith("#define ") and len(line.split()) >= 3
    )
    if "__clang__" in macros or "__INTEL_COMPILER" in macros:
        return None
    if "__GNUC__" not in macros:
        return None
    return int(macros["__GNUC__"])


class BuildExt(build_ext):
    def build_extensions(self):
        # GCC 15 SLP vectorizer miscompiles the fused one_time_step kernel
        # on x86_64 (wrong values, then NaN), even at -O2.
        if self.compiler.compiler_type == "unix":
            major = gcc_major(self.compiler.compiler_so[:1])
            if major is not None and major >= 15:
                for ext in self.extensions:
                    ext.extra_compile_args.append("-fno-tree-slp-vectorize")
        super().build_extensions()


ext_mods = [Extension(
    "{modname}",
    {pyxfilename},
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3', '-w'],
)]
setup(ext_modules=cythonize(ext_mods), cmdclass={{"build_ext": BuildExt}})
"""

    pyproject_template = """\
[build-system]
requires = ["setuptools", "wheel", "Cython"]
"""

    @property
    def command(self):
        setup_file = os.path.join(self.workdir, "setup.py")
        with open(setup_file, "w") as f:
            f.write(
                self.setup_template.format(
                    modname=self.module_name, pyxfilename=[self.full_path + ".pyx"]
                )
            )

        pyproject_file = os.path.join(self.workdir, "pyproject.toml")
        with open(pyproject_file, "w") as f:
            f.write(self.pyproject_template)

        if self.verbose:
            print(open(self.full_path + ".pyx").read())

        command = [
            sys.executable,
            setup_file,
            "build_ext",
            f"--build-lib={self.workdir}",
        ]
        return command

    def _prepare_files(self, routines):
        pass


class PythonCodeWrapper(CodeWrapper):
    @property
    def command(self):
        return []

    def _prepare_files(self, routines):
        pass

    def _process_files(self, routines):
        if self.verbose:
            print(open(self.full_path + ".py").read())


def get_code_wrapper(backend):
    CodeWrapClass = {
        "NUMPY": PythonCodeWrapper,
        "CYTHON": CythonCodeWrapper,
        "LOOPY": PythonCodeWrapper,
    }.get(backend.upper())
    if CodeWrapClass is None:
        raise ValueError("Language '%s' is not supported." % backend)
    return CodeWrapClass


def autowrap(
    routines,
    backend="cython",
    tempdir=None,
    generate=True,
    args=None,
    flags=[],
    verbose=False,
):
    code_generator = get_code_generator(backend, "project")
    CodeWrapperClass = get_code_wrapper(backend)
    code_wrapper = CodeWrapperClass(code_generator, tempdir, flags, generate, verbose)

    return code_wrapper.wrap_code(routines)
