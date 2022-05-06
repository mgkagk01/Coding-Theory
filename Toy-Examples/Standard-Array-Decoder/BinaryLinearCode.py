
import numpy as np
import itertools

class BinaryLinearCode():
    def __init__(self, G, H):
        self.G = G # Generator matrix
        self.H = H # Parity Check matrix
        self.k = G.shape[0]
        self.n = G.shape[1]
        self.M = 2 ** self.k # Number of possible messages
        self.codewordsList = findCodewords(self)

        # --- Create Standard Array Decoder Table
        createStArrDecT(self)

        # --- Create Syndrome Decoder Table
        createSyDecT(self)


    # ==================================== Encoder ==================================== #
    def encode(self, m):
        return np.dot(m, self.G) % 2

    # ==================================== Decoders ==================================== #
    # --- Standard Array Decoder
    def standardArrayDec(self,r):
        decimalValue = bin2dec(r,r.size)
        r, c = np.where(self.cosets == decimalValue)
        return self.codewordsList[c,:], self.possibleMsgs[c,:]

    # --- Syndrome Decoder
    def syndromeDec(self,r):
        # --- Compute syndrome
        syndromeBin = np.squeeze(np.dot(self.H,r.T) % 2)

        # --- Compute the decimal s
        syndromeDec = int(bin2dec(syndromeBin,syndromeBin.size))

        # --- Correct Codeword
        cHat = (r + self.syndromes[syndromeDec,:]) % 2

        # --- Decimal cHat
        cHatDec = bin2dec(cHat,cHat.size)

        msgHat = self.possibleMsgs[np.squeeze(np.where(self.cosets[0,:] == cHatDec)),:]


        return cHat, msgHat

# ==================================== Functions to Create the Tables ==================================== #
def createStArrDecT(self):

    # --- Initialize cosets
    self.cosets = np.zeros((2 ** (self.n - self.k) , self.M))

    # --- Subgroup
    for i in range(self.M):
        self.cosets[0,i] = bin2dec(self.codewordsList[i,:],self.n)

    isIncluded = np.zeros(2 ** self.n)


    isIncluded[bin2dec(self.codewordsList,self.n)] = 1
    isIncluded[0] = 0

    # --- Weight one error patterns
    possibleErrors = np.zeros((2 ** self.n,self.n))


    possibleErrorsTemp = np.array(list(itertools.product([0, 1], repeat=self.n)))
    count = 0
    for i in range(self.n + 1):
        idx = np.squeeze(np.where(np.sum(possibleErrorsTemp, 1) == i))
        possibleErrors[count:count+idx.size,:] = possibleErrorsTemp[idx,:]
        count += idx.size



    i, new = 0, 0
    while True:
        if isIncluded[int(bin2dec(possibleErrors[i,:], possibleErrors[i,:].size))] == 1:
            i += 1
            continue

        # -- New coset Leader
        cosetLeader = possibleErrors[i,:]


        # --- Create Cosets
        for j in range(self.M):

            # --- Find the new element of the coset
            temp = (cosetLeader + self.codewordsList[j,:]) % 2

            # --- Compute its decimal representation
            decimalVal = bin2dec(temp,temp.size)

            # --- Save this value as decimal
            self.cosets[new,j] = decimalVal

            # --- Mark that it has been processed
            isIncluded[int(decimalVal)] = 1

        if new == (2 ** (self.n - self.k)) -1:
            break
        new += 1
        i += 1



def createSyDecT(self):
    self.syndromes = np.zeros((self.M,self.n))
    isIncluded = np.zeros(self.M)
    possibleErrors = findPossibleErrors(self)
    s = 0
    for i in range(2 ** self.n):
        syndromeBin = np.dot(self.H,possibleErrors[i,:]) % 2

        syndromeDec = int(bin2dec(syndromeBin,syndromeBin.size))

        if isIncluded[syndromeDec] == 0:
            self.syndromes[syndromeDec,:] = possibleErrors[i,:]
            isIncluded[syndromeDec] = 1
            s += 1
        if s == self.M:
            break




def findCodewords(self):
    self.possibleMsgs = np.array(list(itertools.product([0, 1], repeat=self.k)))
    return np.array(np.dot(self.possibleMsgs, self.G)) % 2

def bin2dec(binaryVector, n):
    return np.dot(binaryVector, 2 ** np.arange(n-1,-1,-1))

def findPossibleErrors(self):
    possibleErrors = np.zeros((2 ** self.n, self.n))
    possibleErrorsTemp = np.array(list(itertools.product([0, 1], repeat=self.n)))
    count = 0
    for i in range(self.n + 1):
        idx = np.squeeze(np.where(np.sum(possibleErrorsTemp, 1) == i))
        possibleErrors[count:count + idx.size, :] = possibleErrorsTemp[idx, :]
        count += idx.size

    return possibleErrors






