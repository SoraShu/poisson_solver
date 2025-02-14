import numpy as np
cimport numpy as np

def gauss_seidel_cy(np.ndarray[np.float64_t, ndim=2] f):
    cdef int i, j
    cdef int nx = f.shape[0]
    cdef int ny = f.shape[1]

    # avoid copy
    cdef np.ndarray[np.float64_t, ndim=2] newf = np.zeros((nx, ny), dtype=np.float64)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            newf[i, j] = 0.25 * (f[i, j+1] + f[i, j-1] +
                                 f[i+1, j] + f[i-1, j])
    return newf
