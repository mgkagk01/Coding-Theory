import numpy as np
from PolarCode import PolarCode


np.random.seed(0)

k = 812 # message bits
n = 1024 # length of the code
listSize = 32  # list size
divisor = np.flip(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=int))
nCRC = len(divisor)
designSNRdB = 3
frozenValues = np.zeros(n-k-nCRC)
code = ChannelCode(k, n, listSize, divisor, designSNRdB)
EbN0dB = np.arange(0,5.5,0.5)+2


# Simulation
maxNIter = 10000 # Maximum number of Monte-Carlo Simulations
nBlockErrors = 100 # Maximum Block in Error

CER = np.zeros(len(EbN0dB))
BER = np.zeros(len(EbN0dB))

e = 0
for ebn0dB in EbN0dB:
    nBitErrors = 0
    nCodeErrors = 0
    sigma2 = n / (2 * 10 ** (ebn0dB / 10) * k)


    print('# === Eb/N0 ' + str(ebn0dB) + 'dB === #')
    for Iter in range(maxNIter):

        if CER[e] == nBlockErrors:
            break

        # Generate message
        msg = np.random.randint(0,2,k)

        # Encode the message
        codeword = code.encode(msg, frozenValues)

        # BPSK modulate
        symbols = 1-2*codeword

        # Generate Noise
        z = np.sqrt(sigma2)*(np.random.normal(0,1,n))

        # Received Signal
        y = symbols + z

        # Channel decoder
        DECODED, msgHat = code.decode(y, sigma2)

        # Compute Error Rates
        CER[e] += 1*(sum((msgHat+msg) % 2) > 0)
        BER[e] += sum((msgHat+msg) % 2)/k

    CER[e] = CER[e] / Iter
    BER[e] = BER[e] / Iter

    print('--- CER ---')
    print(CER[e])
    print('--- BER ---')
    print(BER[e])
    print()
    e += 1
