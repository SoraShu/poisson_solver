import numpy as np

from utils.save_result import save_results

from solver.gpu import jacobi_pytorch

def main():
    # Grid sizes
    grid_sizes = [32, 64, 128, 256, 512]
    iterations = 1000

    for size in grid_sizes:
        print(f"Running for grid size {size} x {size}")
        f = np.random.rand(size, size)
        f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0

        newgird = jacobi_pytorch(f, iterations)
        save_results(newgird, f"newgrid_{size}x{size}.hdf5")

if __name__ == "__main__":
    main()