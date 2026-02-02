# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI
from time import time

globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank

# Dimension du problème (peut-être changé)
dim = 120

nloc = dim//nbp

# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(nloc*rank,nloc*(rank+1))])
print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
print(f"u = {u}")

globCom.Barrier()
local_time = 0.0
deb = time()
vi = A.dot(u)
#print(f"vi = {vi}")

v = np.empty(dim)

globCom.Allgather(vi, v)

fin = time()
local_time = fin - deb
print(f"Process {rank}: Temps total du calcul: {local_time:.6f}s")
# Produit matrice-vecteur
# v = A.dot(u)
if(rank == 0): print(f"v = {v}")