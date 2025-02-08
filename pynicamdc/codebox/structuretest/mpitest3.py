from mpi4py import MPI
 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# マスタープロセス
if rank == 0:
    data = {'x': 1, 'y': 2.0}
    # マスタープロセスは、すべてのワーカープロセスのランクを通して、
    # ワーカープロセスにデータを送信する
    for i in range(1, size):
        req = comm.send(data,dest=i,tag=i)
        req.wait()
        print('Process {} sent data:'.format(rank), data)
 
# ワーカープロセス
else:
    # 各ワーカープロセスはマスタープロセスからデータを受け取る
    req = comm.recv(source=0,tag=rank)
    data = req.wait()
    print(f'Process {rank} received data: {data}')
