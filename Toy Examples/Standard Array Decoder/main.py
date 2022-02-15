'''
    Toy example to demonstrate the standard array decoding and syndrome decoding

'''

import numpy as np
from BinaryLinearCode import BinaryLinearCode


# === Initializations
# Generator matrix
G = np.array([[1, 0, 0, 1, 0, 1],[0, 1, 0, 1, 1, 0],[0, 0, 1, 0, 1, 1]])
H = np.array([[1, 1, 0, 1, 0, 0],[0, 1, 1, 0, 1, 0],[1, 0, 1, 0, 0, 1]])
k, n = G.shape
# Create a code object
code = BinaryLinearCode(G, H)
# probability of bit flip
p = 0.2

nIter = 1000
BER_sad, BlckER_sad = 0, 0
BER_sy, BlckER_sy = 0, 0

for Iter in range(nIter):

    # -- Create a message at random
    m = np.random.randint(low = 0, high=2, size=(1,k))

    # --- Encode
    c = code.encode(m)

    # --- Generate an error vector
    e = np.random.binomial(1, p, size=n)

    # --- Received vector
    r = (c + e) % 2

    # ====== Decode using Standard Array decoder ====== #
    cHat, msgHat = code.standardArrayDec(r)

    # --- Block error rate
    BlckER_sad += 1*(sum(sum((cHat + c) % 2)) > 0)

    # --- Bit error rate
    BER_sad += sum(sum((msgHat + m) % 2))

    # ====== Decode using Syndrome Decoder ====== #
    cHat, msgHat = code.syndromeDec(r)

    # --- Block error rate
    BlckER_sy += 1*(sum(sum((cHat + c) % 2)) > 0)

    # --- Bit error rate
    BER_sy += sum(sum((msgHat + m) % 2))


print('\n=== Performance of Standard Array Decoder ===')
print('Block Error Rate for the Standard Array Decoder ' + str(BlckER_sad/nIter))
print('Bit Error Rate for the Standard Array Decoder ' + str(BER_sad/(nIter * k)))
print('\n\n=== Performance of Syndrome Decoder ===')
print('Block Error Rate for the Syndrome Decoder ' + str(BlckER_sy/nIter))
print('Bit Error Rate for the Syndrome Decoder ' + str(BER_sy/(nIter * k)))




