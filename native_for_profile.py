import numpy as np

#from solver.native import gauss_seidel_native

def gauss_seidel_native(f):
    newf = f.copy()
    for i in range(1, newf.shape[0]-1):
        for j in range(1, newf.shape[1]-1):
            newf[i, j] = 0.25 * np.sum([newf[i, j+1], newf[i, j-1], newf[i+1, j], newf[i-1, j]])
    return newf

def run():
    # Grid size
    grid_size = 256
    f = np.random.rand(grid_size, grid_size)
    f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0  # Boundary conditions

    # Number of iterations
    iterations = 1000

    # Perform Gauss-Seidel solver iterations
    for _ in range(iterations):
        f = gauss_seidel_native(f)

run()