import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1) + c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

# Initialisation MPI - Récupération du rang et du nombre de processus
comm = MPI.COMM_WORLD 
rank = comm.Get_rank() 
nbp = comm.Get_size() 

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

block_size = height // nbp
start_y = rank * block_size
end_y = (rank + 1) * block_size if rank != nbp - 1 else height
local_height = end_y - start_y

# CORRECTION: (height, width) pour convergence global et (local_height, width) pour local
local_convergence = np.empty((local_height, width), dtype=np.double)

# Calcul local du temps et de l'ensemble de Mandelbrot
local_time = 0.0
deb = time()
for y in range(start_y, end_y):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        local_convergence[y-start_y, x] = mandelbrot_set.convergence(c, smooth=True)
fin = time()
local_time = fin - deb
print(f"Process {rank}: Temps local du calcul: {local_time:.3f}s")

if rank == 0:
    # CORRECTION: convergence global en (height, width)
    convergence = np.empty((height, width), dtype=np.double)
    convergence[start_y:end_y, :] = local_convergence
    
    # Réception des autres processus
    for src in range(1, nbp):
        src_start = src * block_size
        src_end = src_start + block_size if src < nbp - 1 else height
        src_size = src_end - src_start
        recv_buf = np.empty((src_size, width), dtype=np.double)
        comm.Recv(recv_buf, source=src, tag=src)
        convergence[src_start:src_end, :] = recv_buf
    
    # Constitution de l'image résultante
    image_time_deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence)*255))
    image_time_fin = time()
    print(f"Temps de constitution de l'image : {image_time_fin-image_time_deb:.3f}s")
    image.show()
    
else:
    # Les autres processus envoient leurs résultats et leur temps
    comm.Send(local_convergence, dest=0, tag=rank)
    comm.send(local_time, dest=0, tag=rank+nbp)
