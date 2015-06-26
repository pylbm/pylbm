from __future__ import print_function, division

import os
import sys
import shutil

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
NOTEBOOKS_DIR = os.path.abspath(os.path.join(DOC_DIR, '..', 'notebooks'))
OUTPUT_DIR = os.path.join(DOC_DIR, '_source/notebooks')

ipy_cmd = get_ipython_cmd(as_string=True) + " "

def clean():
    # Clear notebooks dir
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    # Clean tutorial file
    fname = os.path.join(DOC_DIR, './_source/tutorial.rst')
    if os.path.isfile(fname):
        os.remove(fname)

def main():

    clean()

    shutil.copytree(NOTEBOOKS_DIR, OUTPUT_DIR)

    # Get notebooks
    notebooks = list(get_notebook_filenames(OUTPUT_DIR))
    notebooks.sort(key=lambda x: x[1])

    create_notebooks(notebooks)
    # create_examples_list(examples)


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
        command = 'cd {0}; '.format(head) + ipy_cmd +'nbconvert --to rst {0}'.format(tail)
        get_output_error_code(command)

    export = os.path.join(OUTPUT_DIR, '..', 'tutorial.rst')
    with open(export, 'w') as f:
        f.write('Tutorial\n')
        f.write('========\n')
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
