import numpy as np
import matplotlib.pyplot as plt
import time

from solver.gpu import jacobi_pytorch

repeat = 50
iterations = 1000

# Setup and performance testing with varying grid sizes
def run_solver_with_grids():
    grid_sizes = [32, 64, 128, 256, 512]
    times = []

    # Warmup
    jacobi_pytorch(np.random.rand(32, 32), 1000)

    for size in grid_sizes:
        print(f"Running for grid size {size} x {size}")
        f = np.random.rand(size, size)
        f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0  # Boundary conditions

        start_time = time.time()
        for _ in range(repeat):
            _ = jacobi_pytorch(f, iterations)
        times.append((time.time() - start_time)/repeat)

    # Plot the time taken based on different grid sizes
    plt.plot(grid_sizes, times, marker='o', label="PyTorch(CUDA)")

    plt.title("Grid Size vs Time: Gauss Seidel PyTorch(CUDA)")
    plt.xlabel("Grid Size (N x N)")
    plt.ylabel("Time (seconds)")
    plt.grid()
    plt.savefig("task5.png", dpi=300)

run_solver_with_grids()