import time
import inspect
from tqdm import tqdm
from mpi4py import MPI
from datetime import timedelta
from functools import wraps
from typing import Tuple, Callable, Any

mpi_world = MPI.COMM_WORLD # get your comms on all workers
mpi_world.barrier() # synchronization

def user_defined(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(**(dict(zip(list(inspect.signature(func).parameters.keys())[:len(args)], args)) 
                       | {key:kwargs[key] for key in list(inspect.signature(func).parameters.keys()) if key in kwargs}))
    return wrapper

def run_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        """
        Calculate the run time of a function and return that (in seconds) with the function result
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        stop_time = time.perf_counter()
        return result, stop_time - start_time
    return wrapper

def chunk_gen(pieces):
    for i in range(len(pieces)):
        yield pieces[i]

class MPI_PROCESS:
    def __init__(self, chunkfunc, workfunc, resultfunc, initfunc=None, finalfunc=None, **kwargs):
        self.cfunc = chunkfunc # returns number of chunks and a chunk generator
        self.wfunc = workfunc # function the worker will run on the data
        self.rfunc = resultfunc # write results out or add to a list here
        self.ifunc = initfunc # create output rasters here, if needed
        self.ffunc = finalfunc # return list of results here, if needed
        self.kwargs = kwargs # all arguments needed for downstream, user specified functions

    def __call__(self):
        self.master(mpi_world, **self.kwargs) if (mpi_world.rank == 0) else self.worker(mpi_world, **self.kwargs) # run your process depending on master/worker
        mpi_world.barrier() # synchronization
        results = self.final(**self.kwargs) if (mpi_world.rank == 0) else None # any wrap up functionality needed
        mpi_world.barrier()
        return results

    def master(self, mpi_world, **kwargs):
        mpi_status = MPI.Status()
        kwargs.update(self.ifunc(**kwargs)) if self.ifunc is not None else print('no master initialization required.')
        total_chunks, chunks = self.cfunc(**kwargs) #user specifies how they want to chunk things

        # Give starting chunks to workers
        for worker in range(1, mpi_world.size):
            mpi_world.send(next(chunks, None), dest=worker)

        # Listen for workers and handle results
        total_model_time, total_write_time = 0.0, 0.0
        for _ in tqdm(range(total_chunks)): # still need to have worker return if it is free or not even if worker writes out
            result, chunk, model_time = mpi_world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi_status)
            _, write_time = run_time(self.rfunc)(result, **({'chunk': chunk} | kwargs)) # user specifies what they want to do with results

            # Send a new chunk if there are chunks remaining to be sent
            mpi_world.send(next(chunks, None), dest=mpi_status.Get_source())
            # Update the performance timers
            total_model_time += model_time
            total_write_time += write_time

        # Print performance timers
        print('Model time:', timedelta(seconds=total_model_time), 'hh:mm:ss')
        print('Outputs write time:', timedelta(seconds=total_write_time), 'hh:mm:ss')

    def worker(self, mpi_world, **kwargs):
        chunk = mpi_world.recv(source=0)
        while chunk is not None:
            results, model_time = run_time(self.wfunc)(chunk, **kwargs)
            mpi_world.send((results, chunk, model_time), dest=0)
            chunk = mpi_world.recv(source=0)

    def final(self, **kwargs):
        return self.ffunc(**kwargs) if self.ffunc is not None else None # user specifies wrap up function could be none / True

