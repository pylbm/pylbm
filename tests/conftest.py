import re
import os
import pytest
import h5py
from functools import wraps
import shutil
import tempfile
import pylbm
import mpi4py.MPI as mpi
import numpy as np
import warnings

def setup_function(function):
    if mpi.COMM_WORLD.Get_rank() > 0:
        import sys
        sys.stdout = open(os.devnull, 'w')

_schemes = {
    'D1Q2': {'num': [1, 2],
             'vx': [1, -1],
    },
    'D1Q3': {'num': list(range(3)),
             'vx': [0, 1, -1],
    },
    'D1Q5': {'num': list(range(5)),
             'vx': [0, 1, -1, 2, -2],
    },
    'D1Q3Q5': [{'num': list(range(3)),
                'vx': [0, 1, -1],
               },
               {'num': list(range(5)),
               'vx': [0, 1, -1, 2, -2],
               }],
    'D2Q4': { 'num': [1, 2, 3, 4],
              'vx': [1, 0, -1, 0],
              'vy': [0, 1, 0, -1],
    },
    'D2Q5': { 'num': list(range(5)),
              'vx': [0, 1, 0, -1, 0],
              'vy': [0, 0, 1, 0, -1],
    },
    'D2Q9': { 'num': list(range(9)),
              'vx': [0, 1, 0, -1, 0, 1, -1, -1, 1 ],
              'vy': [0, 0, 1, 0, -1, 1, 1, -1, -1],
    },
    'D2Q5Q9': [{ 'num': list(range(5)),
                 'vx': [0, 1, 0, -1, 0],
                 'vy': [0, 0, 1, 0, -1],
               },
               { 'num': list(range(9)),
                 'vx': [0, 1, 0, -1, 0, 1, -1, -1, 1 ],
                 'vy': [0, 0, 1, 0, -1, 1, 1, -1, -1],
               }],
    'D3Q19': {'num': list(range(19)),
              'vx': [ 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1],
              'vy': [ 0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1],
              'vz': [ 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
    },
    'D3Q27': {'num': list(range(27)),
              'vx': [0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1],
              'vy': [0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1],
              'vz': [0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1],
    }
}   

@pytest.fixture(params=_schemes.items(), ids=_schemes.keys())
def schemes(request):
    request.dim = int(re.search(r'D(\d)', request.param[0]).group(1))

    if isinstance(request.param[1], list):
        velocities = [{'velocities': l['num']} for l in request.param[1]]
    else:
        velocities = [{'velocities': request.param[1]['num']}]

    request.dico = {'dim': request.dim,
                    'schemes': velocities,
                   }
    request.scheme = request.param[1]
    return request

def pytest_addoption(parser):
    group = parser.getgroup("h5 file comparison")
    group.addoption('--h5diff', action='store_true',
                    help="Enable comparison of h5 files to reference files")
    group.addoption('--h5diff-generate-path',
                    help="directory to generate reference h5 files in, relative "
                    "to location where py.test is run", action='store')
    group.addoption('--h5diff-reference-path',
                    help="directory containing baseline h5 files, relative to "
                    "location where py.test is run. This can also be a URL or a "
                    "set of comma-separated URLs (in case mirrors are "
                    "specified)", action='store')

def pytest_configure(config):

    if config.getoption("--h5diff") or config.getoption("--h5diff-generate-path") is not None:

        reference_dir = config.getoption("--h5diff-reference-path")
        generate_dir = config.getoption("--h5diff-generate-path")

        if reference_dir is not None and generate_dir is not None:
            warnings.warn("Ignoring --h5diff-reference-path since --h5diff-generate-path is set")

        if reference_dir is not None:
            reference_dir = os.path.abspath(reference_dir)
        if generate_dir is not None:
            reference_dir = os.path.abspath(generate_dir)

        config.pluginmanager.register(H5Comparison(config,
                                                   reference_dir=reference_dir,
                                                   generate_dir=generate_dir))

class H5File:
    @staticmethod
    def read(filename):
        if mpi.COMM_WORLD.Get_rank() == 0:
            return h5py.File(filename, 'r')

    @staticmethod
    def write(filename, data):
        grid = [data.domain.x]
        if data.dim > 1:
            grid.append(data.domain.y)
        if data.dim == 3:
            grid.append(data.domain.z)
        h5 = pylbm.H5File(data.domain.mpi_topo, os.path.basename(filename), os.path.dirname(filename))
        h5.set_grid(*grid)

        slices = []
        for i in data.domain.in_or_out.shape:
            slices.append(slice(1, i-1))
        slices = tuple(slices)
        
        for consm in data.scheme.consm.keys():
            clean_data = data.m[consm].copy()
            clean_data[data.domain.in_or_out[slices] != data.domain.valin] = 0
            h5.add_scalar(consm.name, clean_data)
        
        h5.save()

    @classmethod
    def compare(cls, reference_file, test_file, atol=None, rtol=None):
        if mpi.COMM_WORLD.Get_rank() == 0:
            f1 = cls.read(reference_file)
            f2 = cls.read(test_file)

            try:
                for k in f1.keys():
                    assert f1[k][...] == pytest.approx(f2[k][...], rel=rtol, abs=atol)
            except AssertionError as exc:
                message = "\n\na: {0}".format(test_file) + '\n'
                message += "b: {0}".format(reference_file) + '\n'
                message += exc.args[0]
                return False, message
            else:
                return True, ""
        return True, ""

class H5Comparison:
    
    def __init__(self, config, reference_dir=None, generate_dir=None):
        self.config = config
        self.reference_dir = reference_dir
        self.generate_dir = generate_dir

    def pytest_runtest_setup(self, item):

        compare = item.get_closest_marker('h5diff')

        if compare is None:
            return

        extension = 'h5'
        atol = compare.kwargs.get('atol', 1e-7)
        rtol = compare.kwargs.get('rtol', 1e-14)

        single_reference = compare.kwargs.get('single_reference', False)

        original = item.function

        @wraps(item.function)
        def item_function_wrapper(*args, **kwargs):

            reference_dir = compare.kwargs.get('reference_dir', None)
            if reference_dir is None:
                if self.reference_dir is None:
                    reference_dir = os.path.join(os.path.dirname(item.fspath.strpath), 'reference')
                else:
                    reference_dir = self.reference_dir
            else:
                if not reference_dir.startswith(('http://', 'https://')):
                    reference_dir = os.path.join(os.path.dirname(item.fspath.strpath), reference_dir)

            baseline_remote = reference_dir.startswith('http')

            # Run test and get figure object
            import inspect
            if inspect.ismethod(original):  # method
                data = original(*args[1:], **kwargs)
            else:  # function
                data = original(*args, **kwargs)

            # Find test name to use as plot name
            filename = compare.kwargs.get('filename', None)
            if filename is None:
                if single_reference:
                    filename = original.__name__ + '.' + extension
                else:
                    filename = item.name + '.' + extension
                    filename = filename.replace('[', '_').replace(']', '_')
                    filename = filename.replace('_.' + extension, '.' + extension)

            # What we do now depends on whether we are generating the reference
            # files or simply running the test.
            if self.generate_dir is None:

                # Save the figure
                result_dir = tempfile.mkdtemp()
                test_h5file = os.path.abspath(os.path.join(result_dir, filename))

                H5File.write(test_h5file, data)

                # Find path to baseline array
                baseline_file_ref = os.path.abspath(os.path.join(os.path.dirname(item.fspath.strpath), reference_dir, filename))

                if not os.path.exists(baseline_file_ref):
                    raise Exception("""File not found for comparison test
                                    Generated file:
                                    \t{test}
                                    This is expected for new tests.""".format(
                        test=test_h5file))

                # distutils may put the baseline arrays in non-accessible places,
                # copy to our tmpdir to be sure to keep them in case of failure
                baseline_file = os.path.abspath(os.path.join(result_dir, 'reference-' + filename))
                shutil.copyfile(baseline_file_ref, baseline_file)

                identical, msg = H5File.compare(baseline_file, test_h5file, atol=atol, rtol=rtol)

                if identical:
                    shutil.rmtree(result_dir)
                else:
                    raise Exception(msg)

            else:

                if not os.path.exists(self.generate_dir):
                    os.makedirs(self.generate_dir)

                H5File.write(os.path.abspath(os.path.join(self.generate_dir, filename)), data)

                pytest.skip("Skipping test, since generating data")

        if item.cls is not None:
            setattr(item.cls, item.function.__name__, item_function_wrapper)
        else:
            item.obj = item_function_wrapper