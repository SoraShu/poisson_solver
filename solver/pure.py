# pure python implementation of the solver
def gauss_seidel_pure(f: list[list[float]]) -> list[list[float]]:
    newf = f.copy()
    for i in range(1, len(f)-1):
        for j in range(1, len(f[0])-1):
            newf[i][j] = 0.25 * sum((f[i][j+1], f[i][j-1], f[i+1][j], f[i-1][j]))
    return newf