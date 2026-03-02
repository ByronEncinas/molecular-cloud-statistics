from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

def tarea(x):
    return x*x

with MPIPoolExecutor() as pool:
    resultados = pool.map(tarea, range(10))
    print(list(resultados))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for i in range(rank, 1000, size):
    print(i)

# srun -n 20 python3 mpi_test.py
