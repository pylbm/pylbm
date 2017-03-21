
queue = None

def set_queue(backend):
    global queue
    if not init_queue:
        init_queue = True
        if backend.upper() == "LOOPY":
            import pyopencl as cl
            import pyopencl.array

            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
