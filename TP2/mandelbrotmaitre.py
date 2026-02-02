import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

comm = MPI.COMM_WORLD
nbp = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z: complex
        iter: int

        # Vérifier d'abord si le complexe n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1) * (c.real + 1) + c.imag * c.imag < 0.0625:
            return self.max_iterations
        #  2. Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z))) / log(2)
                return iter
        return self.max_iterations

# Paramètres pour l'ensemble de Mandelbrot
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

# Stratégie maître-esclave
if rank == 0:
    deb = time()
    global_convergence = np.empty((height, width), dtype=np.double)

    for i in range(1,nbp):
        comm.send(i-1, dest=i)
    tasks = nbp
    stat = MPI.Status()

    while(tasks < height):
        new_line = comm.recv(status=stat)
        n_line = stat.Get_tag()

        source = stat.source
        comm.send(tasks, dest=source)
        tasks += 1
        global_convergence[n_line,:] = new_line

    for i in range(1,nbp):
        new_line = comm.recv(status=stat)
        n_line = stat.Get_tag()

        source = stat.source
        comm.send(-1, dest=source)
        global_convergence[n_line,:] = new_line

    fin = time()
    print(f"Temps du calcul total: {fin-deb}")

    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(global_convergence) * 255))
    image.show()


else:
    deb = time()
    n_line = comm.recv(source=0)
    while(n_line != -1):
        local_convergence = np.empty(width, dtype=np.double)

        for x in range(width):
            c = complex(-2. + (3. / width) * x, -1.125 + (2.25 / height) * n_line)
            local_convergence[x] = mandelbrot_set.convergence(c, smooth=True)

        # Résultats locaux sont renvoyés au maître
        comm.send(local_convergence, dest=0, tag=n_line)
        n_line = comm.recv(source=0)

    fin = time()
    print(f"Temps du calcul de l'ensemble de Mandelbrot Rank {rank}: {fin-deb}")
