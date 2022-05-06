import numpy as np
from RLL17code import RLL17code
from PolarCode import PolarCode

class Scheme():
    def __init__(self, m, n, k, nc, nCodewords):
        self.n = n
        self.m = m
        self.nCodewords = nCodewords
        self.rateRLL = m / n
        self.rll = RLL17code()
        self.polar = PolarCode(nc, k, nCodewords)


    # ========================= Encoder ========================= #
    def encode(self, x):

        # --- Step 1: Polar Code
        outputPolar = np.zeros((self.nCodewords, self.polar.n))

        for i in range(self.nCodewords):
            outputPolar[i,:], _ = self.polar.encoder(x[i,:], 0, -1)

        # --- Step 2: Interleaver
        outputIter = np.ndarray.flatten(outputPolar.T)

        # --- Step 3: RLL(1,7)
        outputRLL = self.rll.encode(outputIter)

        # --- Step 4:
        output = self.encodeNRZI(outputRLL)

        return output

    def encodeNRZI(self, rllCodeword):
        length = len(rllCodeword)
        self.nrziCodeword = np.zeros(length)
        currValue = 1
        for i in range(length):
            if rllCodeword[i]:
                self.nrziCodeword[i] = currValue * -1
                currValue = self.nrziCodeword[i]
            else:
                self.nrziCodeword[i] = currValue

        checkPEPpatterns(self.nrziCodeword)
        return self.nrziCodeword

    # ========================= Decoder ========================= #
    def decode(self, y):

        # --- Step 1: NRZI
        outputNRZI = self.decodeNRZI(y)

        # ---  Step 2: RLL(1,7)
        # output = self.rll.decode(y)
        outputRLL = self.rll.decode(outputNRZI)

        # --- Step 3: Interleaver
        outputInter = np.reshape(outputRLL, (self.polar.n, 2)).T

        # --- Step 4:
        output = np.zeros((self.nCodewords, self.polar.k))
        for i in range(self.nCodewords):
            output[i,:], _ = self.polar.decodeSC(1-2*outputInter[i,:], np.zeros(self.polar.n - self.polar.k))

        return output

    def decodeNRZI(self, y):

        length = len(y)

        msg = np.zeros(length)
        currValue = 1
        for i in range(length):
            if currValue != y[i]:
                msg[i] = 1
                currValue = y[i]
            else:
                msg[i] = 0

        return msg
