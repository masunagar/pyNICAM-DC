from mpi4py import MPI
 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = {'x': 1, 'y': 2.0}
    for i in range(1, size):
        req = comm.isend(data, dest=i, tag=0)  # Non-blocking send
        req.wait()  # Ensure completion
        print(f'Process {rank} sent data to {i}: {data}')
 
else:
    req = comm.irecv(source=0, tag=0)  # Non-blocking receive
    data = req.wait()
    print(f'Process {rank} received data: {data}')
