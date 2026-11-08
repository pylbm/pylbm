# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
queue = None  # pylint: disable=invalid-name


def set_queue(backend):
    """
    Create the OpenCL queue of the Loopy backend. It is created once and
    shared by all the simulations: the arrays of a simulation belong to the
    context of the queue they were created with, and a new context would
    make them unusable.
    """
    global queue  # pylint: disable=invalid-name
    if backend.upper() == "LOOPY" and queue is None:
        try:
            import pyopencl as cl
        except ImportError:
            raise ImportError("Please install pylbm[gpu]")
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
