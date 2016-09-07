from __future__ import print_function, division

import os
import sys
import shutil
import copy

import IPython.nbformat as nbformat
from IPython.nbconvert import RSTExporter
from IPython.utils.process import get_output_error_code
from IPython.testing.tools import get_ipython_cmd

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen  # Py3k

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
TUTORIAL_DIR = os.path.abspath(os.path.join(DOC_DIR, '..', 'notebooks'))
SRC_DIR = os.path.join(DOC_DIR, '_source/')
OUTPUT_TUTO_DIR = os.path.join(SRC_DIR, 'notebooks')
NOTEBOOKS = []

ipy_cmd = get_ipython_cmd(as_string=True) + " "

def clean():
    # Clean tutorial file
    fname = os.path.join(SRC_DIR, './tutorial.rst')
    if os.path.isfile(fname):
        os.remove(fname)

    for fname in NOTEBOOKS:
        os.remove(SRC_DIR + '/' + fname[1] + '.rst')

    # Clear tutorial notebooks dir
    if os.path.isdir(OUTPUT_TUTO_DIR):
        shutil.rmtree(OUTPUT_TUTO_DIR)

def main():
    global NOTEBOOKS
    clean()

    shutil.copytree(TUTORIAL_DIR, OUTPUT_TUTO_DIR)

    # Get notebooks
    notebooks = list(get_notebook_filenames(SRC_DIR))
    notebooks.sort(key=lambda x: x[1])
    NOTEBOOKS = copy.copy(notebooks)

    create_notebooks(notebooks)

    # Get tutorial notebooks
    notebooks = list(get_notebook_filenames(OUTPUT_TUTO_DIR))
    notebooks.sort(key=lambda x: x[1])
    create_tutorial_rst(notebooks)

def get_notebook_filenames(notebooks_dir):
    """ Yield (filename, name) elements for all examples. The examples
    are organized in directories, therefore the name can contain a
    forward slash.
    """
    for (dirpath, dirnames, filenames) in os.walk(notebooks_dir):
        for fname in filenames:
            if not fname.endswith('.ipynb') or dirpath.count('.ipynb_checkpoints') > 0:
                continue
            filename = os.path.join(dirpath, fname)
            name = filename[len(notebooks_dir):].lstrip('/\\')[:-6]
            name = name.replace('\\', '/')
            yield filename, name

def create_notebooks(notebooks):

    for filename, name in notebooks:
        head, tail = os.path.split(filename)
        print("\tgenerate {0}.rst".format(name))
        #command = ipy_cmd +'nbconvert --to rst --execute {0} --output {1}.rst --template myrst.tpl'.format(filename, SRC_DIR + name)
        command = ipy_cmd +'nbconvert --to rst {0} --output-dir {1} --template myrst.tpl'.format(filename, head)
        out, err, return_code = get_output_error_code(command)
        if return_code != 0:
            print(err)
            clean()
            sys.exit()

def create_tutorial_rst(notebooks):
    export = os.path.join(SRC_DIR, './tutorial.rst')
    with open(export, 'w') as f:
        f.write('Tutorial\n')
        f.write('========\n')
        f.write('\n')
        f.write(".. toctree::\n")
        f.write("   :hidden:\n")
        f.write("   :titlesonly:\n\n")
        for filename, name in notebooks:
            f.write("   {0}<{1}>\n".format(name, './notebooks/' + name))
        f.write('\n')
        #f.write('    :maxdepth: 1\n\n')

        for filename, name in notebooks:
            f.write(":download:`get the notebook<{0}>`\n\n".format('./notebooks/' + name + '.ipynb'))
            f.write("\n:doc:`{0}`\n\n".format('./notebooks/' + name))
            meta = filename[:-6] + '.meta'
            if os.path.isfile(meta):
                lines = open(meta).readlines()
                f.writelines(lines)
                f.write('\n')


if __name__ == '__main__':
    main()
