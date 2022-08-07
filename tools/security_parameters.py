from math import comb, log2


def lg_error(M, n, tau):
    """Calculate the soundness error for a given choice of parameters."""
    return max(log2(comb(k, M - tau)) - log2(comb(M, M - tau)) - (k - M + tau) * log2(n) for k in range(M - tau, M + 1))
