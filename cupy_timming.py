import numpy as np
import matplotlib.pyplot as plt
import time

from solver.gpu import jacobi_cupy

repeat = 50
iterations = 1000

# Setup and performance testing with varying grid sizes
def run_solver_with_grids():
    grid_sizes = [32, 64, 128, 256, 512]
    times = []

    # Warmup
    jacobi_cupy(np.random.rand(32, 32), 1000)

    for size in grid_sizes:
        print(f"Running for grid size {size} x {size}")
        f = np.random.rand(size, size)
        f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0  # Boundary conditions
       
        start_time = time.time()
        for _ in range(repeat):
            _ = jacobi_cupy(f, iterations)
        times.append((time.time() - start_time)/3)

    # Plot the time taken based on different grid sizes
    plt.plot(grid_sizes, times, marker='o', label="CuPy")

    plt.title("Grid Size vs Time: Gauss Seidel CuPy")
    plt.xlabel("Grid Size (N x N)")
    plt.ylabel("Time (seconds)")
    plt.grid()
    plt.savefig("task6.png", dpi=300)

run_solver_with_grids()