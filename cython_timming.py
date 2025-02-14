import numpy as np
import matplotlib.pyplot as plt
import time

from solver.native import gauss_seidel_native
from solver.cy import gauss_seidel_cy


# Setup and performance testing with varying grid sizes
def run_solver_with_grids():
    grid_sizes = [32, 64, 128, 256, 512]
    times = []

    for size in grid_sizes:
        print(f"Running for grid size {size} x {size}")
        f = np.random.rand(size, size)
        f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0  # Boundary conditions

        iterations = 1000

        start_time = time.time()
        for _ in range(iterations):
            f = gauss_seidel_native(f)
        times.append(time.time() - start_time)
    plt.plot(grid_sizes, times, marker='o', label="Native")

    times = []

    for size in grid_sizes:
        print(f"Running for grid size {size} x {size}")
        f = np.random.rand(size, size)
        f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0  # Boundary conditions

        iterations = 1000

        start_time = time.time()
        for _ in range(iterations):
            f = gauss_seidel_cy(f)
        times.append(time.time() - start_time)

    # Plot the time taken based on different grid sizes
    plt.plot(grid_sizes, times, marker='o', label="Cython")

    plt.title("Grid Size vs Time: Gauss Seidel Cython vs Native")
    plt.xlabel("Grid Size (N x N)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid()
    plt.savefig("task3.png", dpi=300)

run_solver_with_grids()