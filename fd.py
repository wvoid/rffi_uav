from numpy import zeros, arange, fix, polyfit
from math import log, sqrt

"""
Algoritmo de Higuchi
Input:  x       Una lista de valores
        kmax    Maximo valor de k   

Output: Dimension fractal que describe la lista x
"""


def hfd(x, kmax=5):
    N = len(x)

    lmk = zeros((kmax, kmax))

    for k in range(1, kmax + 1):
        for m in range(1, k + 1):

            lmki = 0

            for i in range(1, int(fix((N - m) / k)) + 1):
                lmki = lmki + abs(x[(m + i * k) - 1] - x[(m + (i - 1) * k) - 1])

            Ng = (N - 1) / (fix((N - m) / k) * k)
            lmk[m - 1][k - 1] = (lmki * Ng) / k

    Lk = [log(sum([lmk[fila][k] for fila in range(kmax)]) / (k + 1)) for k in range(kmax)]
    Lnk = [log(1 / (i + 1)) for i in range(kmax)]

    return polyfit(Lnk, Lk, 1)[0]


"""
Algoritmo de Katz (KFD)
Basado en la implementacion de Jess Monge lvarez

Input:  serie   Una lista de valores
Output:         Dimension fractal que describe la lista x
"""


def katz(serie):
    N = len(serie)
    L = sum([sqrt(1 + ((serie[i - 1] - serie[i]) ** 2)) for i in range(1, N)])

    d = max([sqrt(((0 - i) ** 2) + ((serie[0] - serie[i]) ** 2)) for i in range(1, N)])

    return log(N - 1) / (log(N - 1) + log(d / L))
    # return log(L)/log(d)