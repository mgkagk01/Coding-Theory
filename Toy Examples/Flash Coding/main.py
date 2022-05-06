import numpy as np
import matplotlib.pyplot as plt
from Scheme import Scheme
from FlashChannel import FlashChannel

np.random.seed(10)
# === Initializations
k = 30 # Number of information bits
nc = 64 # length of the polar codeword
m = 2 # length of the input to RLL encoder
n = 3 # length of the output to RLL encoder
R = m/n # Rate of RLL

# --- For the Flash Channel
nCodewords = 2 # number of codewords per word line
nBitLines = int(nc*nCodewords/R) # number of bit lines
nPages = 5 # Number of pages
nWordLines = nPages # Number of word lines
alpha = 1
couplingRations = np.array([0.1, 0.008, 0.006]) * alpha
sigma2 = 0.0625



gammaXlist = np.array([0.1,0.15,0.20,0.25,0.3,0.35])

WER = np.zeros(len(gammaXlist))
q = 0
# Create an object of the scheme
scheme = Scheme(m, n, k, nc, nCodewords)

for gammaX in gammaXlist:
    couplingRations[0] = gammaX
    nErrors, count = 0, 0

    # Create a channel object
    channel = FlashChannel(nBitLines + 2 * 3, nWordLines, couplingRations, sigma2)
    while nErrors < 100 and count < 1e6:

        # To store the information data
        x = np.zeros((nPages * nCodewords, k))

        # === Encode: For all pages
        dataToStore = np.zeros((nPages, nBitLines + 2*3))
        j = 0
        for p in range(nPages):
            # --- Generate data
            x[j:j+2,:] = np.random.binomial(1, 0.5, (nCodewords,k))
            dataToStore[p,:] = scheme.encode(x[j:j+2,:])

            j += 2

        # === Channel: Add noise and Interference
        y = channel.flashChannel(dataToStore)

        # === Decode: For all pages
        xHat = np.zeros((nPages * nCodewords,k))
        # Read the pages: Hard information
        yHard = np.sign(y) * np.ones((nPages, nBitLines + 2*3))

        j = 0
        for p in range(nPages):
            xHat[j:j+2,:] = scheme.decode(yHard[p,:])
            j+=2



        for ii in range(nPages * nCodewords):
            if sum((x[ii,:] + xHat[ii,:]) % 2) > 0:
                nErrors += 1

        count = count + (nPages * nCodewords)

    WER[q] = nErrors / count
    print(WER)
    print(nErrors)
    print()
    q += 1

print(WER)
print()


plt.plot(gammaXlist, WER, linewidth=2.0)
plt.show()

