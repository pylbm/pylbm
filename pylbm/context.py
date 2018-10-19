# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
queue = None #pylint: disable=invalid-name

def set_queue(backend):
    global queue #pylint: disable=invalid-name
    if backend.upper() == "LOOPY":
        try:
            import pyopencl as cl
        except ImportError:
            raise ImportError('Please install loo.py')
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
