import numpy as np
import matplotlib.pyplot as plt
import time

from solver.native import gauss_seidel_native
from solver.cy import gauss_seidel_cy
from solver.gpu import jacobi_cupy, jacobi_pytorch

def timming_cpu(
        solver,
        label,
        grid_sizes = [32, 64, 128, 256, 512],
        iterations = 1000
):
    times = []
    print(f"Timmings for {label}")
    for size in grid_sizes:
        print(f"Running for grid size {size} x {size}")
        f = np.random.rand(size, size)
        f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0  # Boundary conditions

        start_time = time.time()
        for _ in range(iterations):
            f = solver(f)
        times.append(time.time() - start_time)
    plt.plot(grid_sizes, times, marker='o', label=label)

def timming_gpu(
        solver,
        label,
        grid_sizes = [32, 64, 128, 256, 512],
        repeat = 1,
        iterations = 1000,
        warmup = False
):
    times = []
    print(f"Timmings for {label}")
    for size in grid_sizes:
        print(f"Running for grid size {size} x {size}")
        f = np.random.rand(size, size)
        f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0  # Boundary conditions

        start_time = time.time()
        if warmup:
            _ = solver(f, iterations)
        for _ in range(repeat):
            _ = solver(f, iterations)
        times.append((time.time() - start_time)/repeat)
    plt.plot(grid_sizes, times, marker='o', label=label)

def main():
    #timming_cpu(gauss_seidel_native, "Native")
    timming_cpu(gauss_seidel_cy, "Cython")
    timming_gpu(jacobi_cupy, "CuPy", repeat=50, warmup=True)
    timming_gpu(jacobi_pytorch, "PyTorch(CUDA)", repeat=50, warmup=True)

    plt.title("Grid Size vs Time: Cython vs Native vs CuPy vs PyTorch(CUDA)")
    plt.xlabel("Grid Size (N x N)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid()
    plt.savefig("cpu_vs_gpu.png", dpi=300)

if __name__ == "__main__":
    main()