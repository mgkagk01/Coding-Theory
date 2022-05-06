import numpy as np


class RLL17code():
    def __init__(self):
        self.n = 3
        self.m = 2
        self.R = self.m/self.n
        state1 = np.array([[0, 1, 0, 1], [0, 1, 0, 2], [0, 1, 0, 3], [1, 0, 0, 3]])
        state2 = np.array([[1, 0, 0, 1], [1, 0, 0, 2], [1, 0, 1, 3], [1, 0, 1, 4]])
        state3 = np.array([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3], [0, 0, 1, 4]])
        state4 = np.array([[0, 1, 0, 1], [0, 1, 0, 2], [0, 1, 0, 3], [0, 0, 0, 3]])
        self.transitions = np.stack((state1, state2, state3, state4)) # First dimension is the state, Last column is the next state, rest is the output
        self.I2list = np.array([42, 44, 45, 101, 100, 242, 244, 245, 442, 444, 445, 500, 501])




    def encode(self, x):
        currState, j = 0, 0
        self.x = x # Save the input for performance
        x = np.hstack((x, np.zeros(2*self.m)))
        # To save the codeword, output of the encoder
        output = np.zeros(int((len(x) / self.R)), dtype=int)

        # --- Step 0: reshape x
        xRes = np.reshape(x, (int(len(x)/self.m), self.m))
        # print(currState + 1)
        for i in range(xRes.shape[0]):
            # print('New Symbol')
            # --- Step 1: Find the decimal representation
            idx = bin2dec(xRes[i,:])

            # --- Step 2: Save the codeword
            output[j:j+self.n] = self.transitions[currState, idx, 0:self.n]
            # print( output[j:j+self.n])
            # --- Step 3: Update parameters
            currState = self.transitions[currState, idx, self.n] - 1 # Next State
            # print(currState+1)
            j += self.n


        return output


    def decode(self, y):

        xHat = np.zeros(int((len(y) - 2*self.m - 1) * self.R), dtype=int)
        # print(self.x)
        j, i = 0, 0
        while True:
            # --- Step 1: Find the decimal representation
            hundreds, tens, units, =  bin2dec(y[i:i + self.n]),  bin2dec(y[i + self.n:i + 2 * self.n]), bin2dec(y[i + 2 * self.n:i + 3 * self.n])
            idx = 100 * hundreds + 10 * tens + units
            
            I1 = ((hundreds == 0 and tens == 0) or (hundreds == 0 and tens == 1) or (hundreds == 1) or (hundreds == 2 and tens == 0) or (hundreds == 2 and tens == 1)
                    or (hundreds == 4 and tens == 0) or (hundreds == 4 and tens == 1) or (hundreds == 5)) * 1

            I2 = ((hundreds == 0 and tens == 5) or (hundreds == 2 and tens == 5) or (hundreds == 4 and tens == 0) or (hundreds == 4 and tens == 1) or (hundreds == 4 and tens == 5)
                  or (hundreds == 5 and tens == 2) or (hundreds == 1 and tens == 2) or (hundreds == 0 and tens == 0) or (hundreds == 0 and tens == 1)) * 1
                  

            # --- MSB
            if I1:
                xHat[j] = 1
            
            j += 1

            # --- LSB
            if I2:
                xHat[j] = 1
            elif idx in self.I2list:
                I2 = 1
                xHat[j] = 1

            j += 1
            i += self.n
            # print(I1, I2)
            if i >= y.shape[0] - 2 * self.n:
                break
            
        #
        # if sum((xHat + self.x) % 2) > 0:
        #     print()
        return xHat







def bin2dec(x):
    return int(np.dot(x,  2 ** np.arange(len(x)-1, -1, -1)))