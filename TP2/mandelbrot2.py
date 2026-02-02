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
        if c.real*c.real + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1) + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

# Initialisation MPI
comm = MPI.COMM_WORLD 
rank = comm.Get_rank() 
nbp = comm.Get_size()

# Paramètres
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

# NOUVELLE DISTRIBUTION : répartition cyclique des lignes
# Chaque processus reçoit une ligne sur nbp
local_rows = list(range(rank, height, nbp))
local_height = len(local_rows)

print(f"Process {rank}: traite {local_height} lignes (répartition cyclique)")

# Calcul local
local_convergence = np.empty((local_height, width), dtype=np.double)
local_time = 0.0

deb = time()
for i, y in enumerate(local_rows):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        local_convergence[i, x] = mandelbrot_set.convergence(c, smooth=True)
fin = time()
local_time = fin - deb

print(f"Process {rank}: Temps local du calcul: {local_time:.3f}s")

# Collecte des résultats
if rank == 0:
    convergence = np.empty((height, width), dtype=np.double)
    all_times = [local_time]
    
    # Place ses propres données
    for i, y in enumerate(local_rows):
        convergence[y, :] = local_convergence[i, :]
    
    # Réception des autres processus
    for src in range(1, nbp):
        # Reçoit la taille des données
        src_height = comm.recv(source=src, tag=src)
        src_data = np.empty((src_height, width), dtype=np.double)
        
        comm.Recv(src_data, source=src, tag=src+1000)
        
        # Reçoit les indices des lignes
        src_rows = np.empty(src_height, dtype=int)
        comm.Recv(src_rows, source=src, tag=src+2000)
        
        # Place les données
        for i, y in enumerate(src_rows):
            convergence[y, :] = src_data[i, :]
        
        # Reçoit le temps
        src_time = comm.recv(source=src, tag=src+3000)
        all_times.append(src_time)

    # Constitution de l'image
    image_time_deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence)*255))
    image_time_fin = time()
    print(f"\nTemps de constitution de l'image: {image_time_fin-image_time_deb:.3f}s")
    image.save(f"mandelbrot_cyclique_{nbp}procs.png")
    
else:
    # Envoi de la taille
    comm.send(local_height, dest=0, tag=rank)
    # Envoi des données
    comm.Send(local_convergence, dest=0, tag=rank+1000)
    # Envoi des indices des lignes
    comm.Send(np.array(local_rows, dtype=int), dest=0, tag=rank+2000)
    # Envoi du temps
    comm.send(local_time, dest=0, tag=rank+3000)