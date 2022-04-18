import numpy as np
import matplotlib.pyplot as plt
from LdpcDecoder import LdpcDecoder
np.random.seed(0)
# --- Initializations
EbN0List = np.arange(0, 2.5, 0.5)
nIter = 50 # number of BP iterations
a = -1
# EbN0List = np.array([20])

# Create an LDPC object
ldpcCode = LdpcDecoder("NR_1_0_8.alist", a)
R = (ldpcCode.V - ldpcCode.C) / (ldpcCode.V - 16) # Rate of the code

# To count the number of bit errors
nErrors = np.zeros(len(EbN0List))
i = 0

for EbN0 in EbN0List:

    # --- Compute sigma^2
    sigma2 = 1.0 / (2 * R * (10**(EbN0/10)))
    print('Sigma^2 = ' + str(sigma2))
    count = 0
    nErrorCode = 0
    while nErrorCode <= 50:

        # --- Set the codeword to be the all zero
        codeword = np.zeros(ldpcCode.V)

        # --- BPSK modulation
        s = (1-2*codeword)*a

        # --- Generate noise
        n = np.random.normal(0, np.sqrt(sigma2), ldpcCode.V)

        # --- Received signal
        y = s + n

        # --- Decode
        isCorrect, codewordHat = ldpcCode.decode(y, sigma2, nIter)

        if sum(((codewordHat < 0) * 1)) > 0:
            nErrorCode += 1 # Count the number of codeword error
            nErrors[i] += sum(((codewordHat < 0) * 1))

        count += 1

    nErrors[i] = nErrors[i] / (count * ldpcCode.V)
    print(nErrors)

    i += 1




plt.semilogy(EbN0List, nErrors, marker = 'o')
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("BER")
plt.grid()
plt.show()
print()
