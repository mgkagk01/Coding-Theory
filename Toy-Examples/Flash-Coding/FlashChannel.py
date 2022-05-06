import numpy as np


class FlashChannel():
    def __init__(self, nBitLines, nWordLines, couplingRations, sigma2):
        self.nBitLines = nBitLines
        self.nWordLines = nWordLines
        self.pageSize = nBitLines
        self.thresVoltE2PH = 2
        self.gammaX = couplingRations[0]
        self.gammaY = couplingRations[1]
        self.gammaXY = couplingRations[2]
        self.sigma2 = sigma2
        
        
    def flashChannel(self,x):


        # --- Step 1: Add noise
        noise = np.random.normal(loc=0, scale=np.sqrt(self.sigma2), size=(self.nWordLines, self.nBitLines))
        y = x + noise

        # --- Step 3: Compute the interference
        for w in range(self.nWordLines):
            for b in range(self.nBitLines):
                nGammaX, nGammaY, nGammaXY = 0, 0, 0

                # print(w, b)
                # --- Check if you are in the boundary
                if b == 0: # No cell on the left
                    if w == 0:
                        if x[w,b + 1] == 1 : # PH cell
                            nGammaX += 1

                        if x[w+1, b] == 1: # PH cell
                            nGammaY += 1

                        if x[w+1, b+1] == 1: # PH cell
                            nGammaXY += 1
                    elif w == self.nWordLines -1:
                        if x[w,b + 1] == 1 : # right
                            nGammaX += 1

                        if x[w-1, b] == 1: # up
                            nGammaY += 1

                        if x[w-1, b+1] == 1: # up-right
                            nGammaXY += 1

                    else:
                        activeBlock = np.hstack((np.zeros((3,1)), x[w-1:w + 2, b:b + 2]))
                        nGammaX, nGammaY, nGammaXY = self.findPH(activeBlock)

                elif b == self.nBitLines - 1: # No cell on the right
                    if w == 0:
                        if x[w,b - 1] == 1 : # left
                            nGammaX += 1
                        if x[w+1, b] == 1: # down
                            nGammaY += 1
                        if x[w+1, b-1] == 1: # down-left
                            nGammaXY += 1

                    elif w == self.nWordLines - 1:
                        if x[w - 1,b] == 1 : # up
                            nGammaX += 1
                        if x[w, b -1] == 1: # left
                            nGammaY += 1
                        if x[w-1, b-1] == 1: # up-left
                            nGammaXY += 1
                    else:
                        activeBlock = np.hstack((x[w-1:w + 2, b-1:b + 2], np.zeros((3,1))))
                        nGammaX, nGammaY, nGammaXY = self.findPH(activeBlock)

                    # if w != 0: # if not first row
                    #     if x[w+1, b] == 1: # up
                    #         nGammaY += 1
                    #     if x[w+1, b-1] == 1: # up-left
                    #         nGammaXY += 1

                elif w == 0: # No cell above
                    activeBlock = np.vstack((np.zeros(3), x[w:w + 2, b - 1:b + 2]))
                    nGammaX, nGammaY, nGammaXY = self.findPH(activeBlock)

                elif w == self.nWordLines - 1: # No cell Below
                    activeBlock = np.vstack((x[w-1:w+1, b - 1:b + 2], np.zeros(3)))
                    nGammaX, nGammaY, nGammaXY = self.findPH(activeBlock)
                else:
                    # cells in all directions
                    activeBlock = x[w - 1:w + 2, b-1:b + 2]
                    nGammaX, nGammaY, nGammaXY = self.findPH(activeBlock)








                # --- Find the PH positions

                # print(nGammaX, nGammaY, nGammaXY)
                # --- Add the interference
                y[w,b] = y[w,b] + self.thresVoltE2PH * (nGammaX * self.gammaX + nGammaY * self.gammaY + nGammaXY * self.gammaXY)



        return y


    def findPH(self,activeBlock):
        nGammaX, nGammaY, nGammaXY = 0, 0, 0
        for i in range(activeBlock.shape[0]):
            for j in range(activeBlock.shape[1]):

                if activeBlock[i, j] == 0:
                    continue

                if i == 0:  # First row
                    if j == 1:
                        if activeBlock[i, j] == 1:
                            nGammaY += 1
                        else:
                            nGammaXY += 1
                elif i == 1: # Second row
                    if i == j:
                        continue
                    else:
                        if activeBlock[i, j] == 1:
                            nGammaX += 1

                elif i == 2: # Third row
                    if j == 1:
                        if activeBlock[i, j] == 1:
                            nGammaY += 1

                    else:
                        if activeBlock[i, j] == 1:
                            nGammaXY += 1

        return nGammaX, nGammaY, nGammaXY


