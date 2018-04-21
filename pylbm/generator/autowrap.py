from __future__ import print_function, division

import sys
import os
import shutil
import tempfile
from subprocess import STDOUT, CalledProcessError, check_output

from .codegen import get_code_generator

class CodeWrapError(Exception):
    pass


class CodeWrapper(object):
    """Base Class for code wrappers"""
    _module_name = "wrapped_module"
    _module_counter = 0

    @property
    def filename(self):
        return "%s_%s" % (self._module_name, CodeWrapper._module_counter)

    @property
    def module_name(self):
        return "%s_%s" % (self._module_name, CodeWrapper._module_counter)

    def __init__(self, generator, filepath=None, flags=[], verbose=False):
        """
        generator -- the code generator to use
        """
        self.generator = generator
        self.filepath = filepath
        self.flags = flags
        self.verbose = verbose

    def _generate_code(self, routines):
        self.generator.write(
            routines, self.filename, True, True, False)

    def wrap_code(self, routines):
        workdir = self.filepath or tempfile.mkdtemp("_sympy_compile")
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        oldwork = os.getcwd()
        os.chdir(workdir)

        try:
            sys.path.append(workdir)
            self._prepare_files(routines)
            self._generate_code(routines)
            self._process_files(routines)
            mod = __import__(self.module_name)
        finally:
            sys.path.remove(workdir)
            CodeWrapper._module_counter += 1
            os.chdir(oldwork)
            if not self.filepath:
                try:
                    shutil.rmtree(workdir)
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
                "Error while executing command: %s. Command output is:\n%s" % (
                    " ".join(command), e.output.decode()))
        if self.verbose:
            print(retoutput)


class CythonCodeWrapper(CodeWrapper):
    @property
    def command(self):
        bld = open(self.filename + '.pyxbld', "w")
        code = """
def make_ext(modname, pyxfilename):
    from distutils.extension import Extension

    return Extension(name = modname,
                     sources=[pyxfilename],
                     #extra_compile_args = ['-O3', '-fopenmp, '-w'],
                     #extra_link_args= ['-fopenmp'])
                     extra_compile_args = ['-O3', '-w']
                     #extra_compile_args = ['-O3', '-fopenmp', '-w'],
                     #extra_link_args= ['-fopenmp'])
                    )
                    """
        bld.write(code)
        bld.close()
        bld = open('build.py', 'w')
        code = """
import pyximport
pyximport.install(build_dir=\'.\', inplace=True)
import {module_name}
        """.format(module_name=self.filename)
        bld.write(code)
        bld.close()

        if self.verbose:
            print(open(self.filename + '.pyx').read())

        command = [sys.executable, 'build.py']

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
            print(open(self.filename + '.py').read())

def get_code_wrapper(backend):
    CodeWrapClass = {"NUMPY" : PythonCodeWrapper,
                     "CYTHON": CythonCodeWrapper,
                     "LOOPY": PythonCodeWrapper}.get(backend.upper())
    if CodeWrapClass is None:
        raise ValueError("Language '%s' is not supported." % backend)
    return CodeWrapClass

def autowrap(routines, backend='cython', tempdir=None, args=None, flags=[],
    verbose=False):

    code_generator = get_code_generator(backend, "project")
    CodeWrapperClass = get_code_wrapper(backend)
    code_wrapper = CodeWrapperClass(code_generator, tempdir, flags, verbose)

    return code_wrapper.wrap_code(routines)
