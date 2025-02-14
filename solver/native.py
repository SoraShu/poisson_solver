# This file contains the native implementation of the solver

def gauss_seidel_native(f):
    
    newf = f.copy()
    for i in range(1, newf.shape[0]-1):
        for j in range(1, newf.shape[1]-1):
            newf[i, j] = 0.25 * (newf[i, j+1] + newf[i, j-1] +
                                 newf[i+1, j] + newf[i-1, j])
    return newf