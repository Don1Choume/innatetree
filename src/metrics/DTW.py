import numpy as np
from numba import jit

@jit('f8(f8[:],f8[:],i8)', nopython=True)
def DTW_distance(series1: float, series2: float, window_size: int=0) -> float:
    n = len(series1)
    m = len(series2)

    window_size = max([window_size, np.abs(n-m)])
    DTW = np.full((n, m), np.inf)
    DTW[0, 0] = 0

    if window_size > 0:
        for i in range(1, n):
            for j in range(max([1, i-window_size]), min([m, i+window_size])):
                DTW[i, j] = 0
    
        for i in range(1, n):
            for j in range(max([1, i-window_size]), min([m, i+window_size])):
                diff = np.array(series1[i]-series2[j]).reshape((1,-1))
                cost = np.linalg.norm(diff, ord=2)
                DTW[i, j] = cost + min([DTW[i-1, j  ],    # insertion
                                            DTW[i  , j-1],    # deletion
                                            DTW[i-1, j-1]])    # match
    else:
        for i in range(1, n):
            for j in range(1, m):
                diff = np.array(series1[i]-series2[j]).reshape((1,-1))
                cost = np.linalg.norm(diff, ord=2)
                DTW[i, j] = cost + min([DTW[i-1, j  ],    # insertion
                                            DTW[i  , j-1],    # deletion
                                            DTW[i-1, j-1]])    # match

    return DTW[-1, -1]