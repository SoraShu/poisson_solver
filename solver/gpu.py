import torch
import cupy


def jacobi_pytorch(f, iterations=1000):
    f = torch.tensor(f, device="cuda", dtype=torch.float32)
    f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0

    for _ in range(iterations):
        f_new = 0.25 * (torch.roll(f, shifts=1, dims=0) + torch.roll(f, shifts=-1, dims=0) +
                        torch.roll(f, shifts=1, dims=1) + torch.roll(f, shifts=-1, dims=1))
        f_new[0, :], f_new[-1, :], f_new[:, 0], f_new[:, -1] = 0, 0, 0, 0
        f = f_new

    return f.cpu().numpy()

def jacobi_cupy(f, iterations=1000):
    f = cupy.array(f)
    f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0

    for _ in range(iterations):
        f_new = 0.25 * (cupy.roll(f, shift=1, axis=0) + cupy.roll(f, shift=-1, axis=0) +
                        cupy.roll(f, shift=1, axis=1) + cupy.roll(f, shift=-1, axis=1))
        f_new[0, :], f_new[-1, :], f_new[:, 0], f_new[:, -1] = 0, 0, 0, 0
        f = f_new

    return cupy.asnumpy(f)
