from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 3:
    raise ValueError("This program requires at least 3 processes.")

# Master process (rank 0)
if rank == 0:
    data_list = [{'value': 42}, {'value': 3.14}]  # Different data for workers
    workers = [1, 2]  # Sending to ranks 1 and 2
    requests = []

    for i, worker in enumerate(workers):
        req = comm.isend(data_list[i], dest=worker, tag=i)  # Non-blocking send
        requests.append(req)  # Store requests

    # Wait for all sends to complete
    MPI.Request.Waitall(requests)
    print(f"Master (rank {rank}) sent data to workers.")

# Worker processes (ranks 1 and 2)
elif rank in [1, 2]:
    req = comm.irecv(source=0, tag=rank-1)  # Non-blocking receive
    data = req.wait()  # Wait for data to be received
    print(f"Worker (rank {rank}) received data: {data}")
