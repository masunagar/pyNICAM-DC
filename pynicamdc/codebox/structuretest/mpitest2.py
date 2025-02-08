from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name() 

print(f"Hello, world! from rank {rank} out of {size}")
print(f"Hello, World! I am process {rank} of {size} on {name}\n")

# マスタープロセス
if rank == 0:
    data = {'x': 1, 'y': 2.0}
    # マスタープロセスは、すべてのワーカープロセスのランクを通して、
    # ワーカープロセスにデータを送信する
    for i in range(1, size):
        comm.send(data, dest=i, tag=i)
        print('Process {} sent data:'.format(rank), data)
 
# ワーカープロセス
else:
    # 各ワーカープロセスはマスタープロセスからデータを受け取る
    data = comm.recv(source=0, tag=rank)
    print(f'Process {rank} received data: {data}')

print('hahaha')



