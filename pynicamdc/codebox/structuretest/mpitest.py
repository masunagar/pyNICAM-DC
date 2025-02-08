from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name() 

print(f"Hello, world! from rank {rank} out of {size}")
print(f"Hello, World! I am process {rank} of {size} on {name}\n")
