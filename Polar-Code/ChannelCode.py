import numpy as np
from PolarCode  import PolarCode
from utilities import crcEncoder, crcDecoder

class ChannelCode:
    def __init__(self, lenMsg, lenCode, listSize, divisor, designSNRdB, prob1=0.5):
        ''' Parameters '''
        self.lenCode = lenCode  # Length of the code
        self.listSize = listSize  # List size
        self.lenMsg = lenMsg # Length of the message
        self.prob1 = prob1 # Pr(x = 1) = 1 - Pr(x=0)

        self.msgCRC = np.array([])

        ''' For polar code '''
        # Create a polar Code object
        # Polynomial for CRC coding
        self.divisor = divisor
        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.crcCowrdLen = self.lCRC + self.lenMsg
        self.polar = PolarCode(self.lenCode, self.crcCowrdLen, designSNRdB)



    def encode(self, msg, frozenValues):

        # --- Step 1: Append CRC
        self.msgCRC = crcEncoder(msg, self.divisor)


        # --- Step 2: Polar Encoding
        codeword = self.polar.encoder(self.msgCRC, frozenValues)

        return codeword


    def decode(self,y, sigma2):


        cwordHatSoft = (2*y/sigma2)

        # ============ Polar decoder ============ #
        msgCRCHat, PML = self.polar.listDecoder(cwordHatSoft, self.listSize, self.prob1)


        # ============ Check CRC ============ #
        # --- Initialization
        thres, flag = np.Inf, -1


        # --- Check the CRC constraint for all message in the list
        for l in range(self.listSize):
            check = crcDecoder(msgCRCHat[:,l], self.divisor)
            if check:
                # --- Check if its PML is larger than the current PML
                if PML[l] < thres:
                    flag = l
                    thres = PML[l]

        # --- Encode the estimated message
        if thres != np.Inf:
            return 1, msgCRCHat[0:self.lenMsg, flag]
        else:
            return 0, msgCRCHat[0:self.lenMsg, np.argmin(PML)] # Change this from max to min
