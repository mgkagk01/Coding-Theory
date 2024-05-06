import numpy as np
# =========== This program generates the Reliabilities of the channels using the Bhattacharyya Bounds
# and save the results to a csv file (from least reliable to most reliable); and inputs are
# K: dimension of the code
# N: length of the code
# Design SNR

def bitReversed(x, n):
    """
    Bit-reversal operation.

    Parameters
    ----------
    x: ndarray<int>, int
        a vector of indices
    n: int
        number of bits per index in ``x``

    Returns
    ----------
    ndarray<int>, int
        bit-reversed version of x

    """

    result = 0
    for i in range(n):  # for each bit number
        if (x & (1 << i)):  # if it matches that bit
            result |= (1 << (n - 1 - i))  # set the "opposite" bit in result
    return result

def getNormalizedSNR(SNRdB,k,n):
    snr = 10 **(SNRdB/10)
    return snr*k/n


# --- Inputs
K = 219 # Message bits + CRC bits
N = 256 # length of the code (power of 2)
n = int(np.log2(N)) # height of the tree

# The construction of the code is a function of K,N and the design SNR;
# For fixed K and N the frozen values can be different for different SNR
designSNRdB = 3

# Es/N0
EsN0 = getNormalizedSNR(designSNRdB,K,N)

# Initialization
z = np.zeros(N)
z[0] = np.exp(-EsN0)

# For each level of the tree
for j in range(1,n+1):
    u = 2**j
    # For each connection
    for t in range(int(u/2)):
        T = z[t]
        z[t] = 2*T - T*T
        z[int(u/2)+t] = T*T

# Find the reliabilities from least reliable to most reliable
reliabilities = np.argsort(-z, kind='mergesort')

for i in range(N):
    reliabilities[i] = bitReversed(reliabilities[i],n)

np.savetxt("Q_N"+str(N)+"_K"+str(K)+"_SNR"+str(designSNRdB)+"dB.csv",reliabilities)
print()
