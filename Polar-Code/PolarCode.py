# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib




class PolarCode():
    def __init__(self, n, k, designSNRdB):
        self.n = n
        self.k = k
        self.R = k / n
        self.height = int(np.log2(n))
        self.LLRs = np.zeros(self.n)
        # --- Import Reliability  Sequence
        self.Q = np.genfromtxt("Q_N" + str(n) + "_K" + str(k) + "_SNR" + str(designSNRdB) + "dB.csv", delimiter=',').astype(int)

        self.frozenPos = self.Q[0:n - self.k]
        self.msgPos = self.Q[n - self.k:n]

    def encoder(self, msg, frozenValues):

        # --- Initialization
        codeword = np.zeros(self.n)  # Create an array to work during the encoding
        m = 2  # Bits to combine

        # Define the first bits of the codeword
        codeword[self.msgPos] = msg
        codeword[self.frozenPos] = frozenValues

        # --- Encode
        # Tree Structure
        # For each level
        for d in range(int(np.log2(self.n))):

            # Combine bits at current depth
            for i in range(0, self.n, m):
                # First part of u
                a = codeword[i:i + int(m / 2)]

                # Second part of u
                b = codeword[i + int(m / 2):i + m]

                # Do the addition modulo 2
                c = np.mod(a + b, 2)
                codeword[i:i + m] = np.append([c], [b], axis=1)

            m *= 2

        return codeword

    # =====================================  Decoder =========================== #
    def listDecoder(self, y, nL, prob1):

        # --- Initiazations
        L = np.zeros((self.n, self.height + 1, nL),dtype=np.float64)  # beliefs for all decoders
        uHat = 2 * np.ones((self.n, self.height + 1, nL),dtype=np.float64)  # Decisions in nL decoders

        PML = np.Inf * np.ones(nL)  # Path metric

        # ----------- To Save the results
        nSysDecoded = 0


        PML[0] = 0
        stateVec = np.zeros((2 * self.n - 1))  # State vector, common for all  decoders
        L[:, 0, :] = np.matlib.repmat(y, nL, 1).T  # belief for root
        node, depth, done = 0, 0, 0



        while (done == 0):  # Traversal loop

            # ==================== Leaf node  ==================== #
            # --- Check if you are on the leaf
            if (depth == self.height):

                DM = L[node, self.height, :]  # Decision metric for all decoders

                # --- Check if it is frozen
                if (node in self.frozenPos):
                    uHat[node,  self.height, :] = 0

                    # Decide about the frozen bits
                    dec = 1 * (DM < 0)

                    # Compute the Log-Prob
                    negLogProb, _ = LLR2nLogProb(DM, np.zeros(len(dec)))

                    # Add this to the Path Metric
                    PML = PML + negLogProb

                else:
                    # Make a decision
                    dec = 1 * (DM < 0)

                    # First PML assumes true bit Second assume erroneous
                    negLogProbRightDecision, negLogProbWrongDecision = LLR2nLogProb(DM, dec)

                    # if the bits are not equiprobable and it is time to prune, ask LLM to give the path probability
                    if prob1 != 0.5 and sum(PML) < np.inf:

                        # --- This is the function (compPathProb) that Kaushik has to change
                        # Compute the probability of the path based on the distribution of the bits
                        pathProbRightDecision, pathProbWrongDecision = compPathProb(
                            uHat[self.msgPos[0:nSysDecoded], self.height, :], dec, prob1, nL)

                        # Compute the total path metric = Conditional LLR + Prior LLR ( Negative Log Probabilities)
                        temp = np.concatenate((pathProbRightDecision, pathProbWrongDecision))  # Prior
                        PM2 = np.concatenate((PML + negLogProbRightDecision, PML + negLogProbWrongDecision)) + temp
                        pos, PML = minK(PM2, nL)  # find the path with the smallest path metric
                        PML = PML - temp[pos]  # remove the contribution from LLM
                    else:

                        # Just add the Decision Metric
                        PM2 = np.concatenate((PML + negLogProbRightDecision, PML + negLogProbWrongDecision))
                        pos, PML = minK(PM2, nL)

                    # Find if there is a new list from the wrong decision
                    pos1 = 1 * (pos > (nL - 1))

                    # Find the index of that list
                    idxPos = np.nonzero(pos1)
                    pos[idxPos] = pos[idxPos] - nL  # Normalize the index
                    dec = dec[pos]  # finalize your decision
                    dec[idxPos] = 1 - dec[idxPos]  # for the bit that belongs to the wrong path, change the value
                    L = L[:, :, pos]  # Copy the LLR
                    uHat = uHat[:, :, pos] # select the list that are active
                    uHat[node, self.height, :] = dec

                    # Count the number of decoded symbols
                    nSysDecoded += 1

                if (node == self.n - 1):
                    done = 1
                else:
                    node = int(np.floor(node / 2))
                    depth = int(depth - 1)

            # ==================== Other node  ==================== #
            else:
                # Find the Node position
                npos = int(2 ** depth + node)

                # ==================== L Step  ==================== #
                # ---  If it is the first time that you hit a node
                if (stateVec[npos] == 0):

                    temp = int(2 ** (self.height - depth))
                    # --- Incoming Beliefs
                    Ln = L[temp * node:temp * (node + 1),  depth, :]

                    # --- Break beliefs into two
                    a = Ln[0:int(temp / 2), :]
                    b = Ln[int(temp / 2)::, :]

                    node = int(2 * node)
                    depth = int(depth + 1)

                    # --- Incoming belief length for left child
                    temp = int(temp / 2)

                    # --- Sum - Product # HERE
                    L[temp * node:temp * (node + 1), depth, :] = sumProdLog(a, b)
                    stateVec[npos] = 1


                else:
                    # ==================== R Step  ==================== #
                    if (stateVec[npos] == 1):

                        temp = int(2 ** (self.height - depth))
                        # --- Incoming Beliefs
                        Ln = L[temp * node:temp * (node + 1), depth, :]

                        # --- Incoming msg
                        nodeL = int(2 * node)
                        dL = int(depth + 1)
                        tempL = int(temp / 2)
                        uHatn = uHat[tempL * nodeL:tempL * (nodeL + 1), dL, :]

                        # --- Repetition decoding
                        a = Ln[0:int(temp / 2), :]
                        b = Ln[int(temp / 2)::, :]

                        # --- Move to the next node
                        # Next child left
                        node = int(2 * node + 1)
                        # Next depth
                        depth = int(depth + 1)
                        # Incoming beliefs length
                        temp = int(temp / 2)
                        # Change the state of the node
                        stateVec[npos] = 2

                        # --- Save Beliefs
                        L[temp * node:temp * (node + 1),  depth, :] = g(a, b, uHatn)

                    # ==================== U Step  ==================== #
                    else:
                        temp = int(2 ** (self.height - depth))
                        nodeL = int(2 * node)
                        nodeR = int(2 * node + 1)
                        dC = int(depth + 1)
                        tempC = int(temp / 2)

                        # --- Incoming decisions from the left child
                        uHatL = uHat[tempC * nodeL:tempC * (nodeL + 1), dC, :]

                        # --- Incoming decisions from the right child
                        uHatR = uHat[tempC * nodeR:tempC * (nodeR + 1), dC, :]

                        uHat[temp * node:temp * (node + 1), depth, :] = np.vstack((((uHatL + uHatR) % 2), uHatR))

                        # --- Go back to parent
                        node = int(np.floor(node / 2))
                        depth = int(depth - 1)

        return uHat[self.msgPos, self.height, :], PML

    # ===================================== Functions for Decoders =========================== #



def sumProdLog(a, b):

    # Two Steps
    z1 = np.zeros(a.shape, dtype=np.float64)

    # Step 1:
    temp = a + b
    rows, columns = np.where(temp > 0)
    z1[rows, columns] = temp[rows, columns] + np.log1p(np.exp(-temp[rows, columns]))

    rows, columns = np.where(temp < 0)
    z1[rows, columns] = np.log1p(np.exp(temp[rows, columns]))

    # Step 2:
    return z1 - logdomain_sum(a, b)



def logdomain_sum(x, y):
    """
    Find the addition of x and y in log-domain. It uses log1p to improve numerical stability.

    Parameters
    ----------
    x: float
        any number in the log-domain
    y: float
        any number in the log-domain

    Returns
    ----------
    float
        the result of x + y

    """
    z = np.zeros(x.shape,dtype=np.float64)

    rows, columns= np.where(x > y)
    z[rows,columns] = x[rows,columns] + np.log1p(np.exp(y[rows,columns] - x[rows,columns]))

    rows, columns = np.where(x < y)
    z[rows, columns] = y[rows, columns] + np.log1p(np.exp(x[rows, columns] - y[rows, columns]))

    return z



def g(a, b, c):
    return (b + np.multiply((1 - 2 * c), a))


def minK(a, k):
    idx = (a).argsort()[:k]
    return idx, a[idx]


def minSum(a, b):
    # --- Sign
    signA, signB = 1 - 2 * (1 * (a < 0)), 1 - 2 * (1 * (b < 0))
    sign = np.multiply(signA, signB)
    # --- Magnitude
    magn = np.minimum(abs(a), abs(b))

    return np.multiply(magn, sign)


def LLR2nLogProb(LLR, dec):

    firstPath = np.zeros(LLR.shape[0])
    secondPath = np.zeros(LLR.shape[0])

    # Moon's Book (1st edition) Page 866 eq. 17.94
    # First Path # Correct Path
    xf = -(1 - 2 * dec) * LLR

    loc = np.where(xf > 40)[0]
    firstPath[loc] = xf[loc]
    loc = np.where(xf <= 40)[0]
    firstPath[loc] = np.log(1 + np.exp(xf[loc]))

    # Moon's Book (1st edition) Page 866 eq. 17.94
    # Second Path  # Wrong Path
    dec = (dec + 1) % 2
    xs = (2 * dec - 1) * LLR

    loc = np.where(xs > 40)[0]
    secondPath[loc] = xs[loc]
    loc = np.where(xs <= 40)[0]
    secondPath[loc] = np.log(1 + np.exp(xs[loc]))


    return firstPath, secondPath


def compPathProb(currMsgsHat, newMsgsHat, prob1, nL):
    # Extent the Current path to include the next nL decoders
    extentCurr = np.hstack((currMsgsHat, currMsgsHat))

    # Compute the probability for each bit
    currProb = 1 * (extentCurr == 0) * (1 - prob1) + 1 * (extentCurr == 1) * (prob1)

    # Extent the new decision metric to include the next nL decoders
    extentNew = np.hstack((newMsgsHat, (newMsgsHat + 1) % 2))

    # Compute the probability for each bit
    newProb = 1 * (extentNew == 0) * (1 - prob1) + 1 * (extentNew == 1) * (prob1)

    # Compute the log probability of the 2*nL decoders
    totalProb = sum(np.log(currProb)) + sum(np.log(newProb))

    # Return results
    return -totalProb[0:nL], -totalProb[nL::]
