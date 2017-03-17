import pyopencl as cl
import pyopencl.array

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
