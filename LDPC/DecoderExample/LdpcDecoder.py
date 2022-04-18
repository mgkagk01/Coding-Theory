import numpy as np

class Node:
    def __init__(self, id, idxStore, idxRead):
        self.id = id
        self.idxStore = idxStore
        self.idxRead = idxRead
        self.next = None


class LdpcDecoder():
    def __init__(self, fileName, a):
        self.a = a
        self.weight = (2*a) # for the channel LLR
        self.f = open(fileName, "r")
        temp = self.f.readline().split()
        self.V = int(temp[1]) # number of Variable nodes
        self.C = int(temp[0]) # number of Check nodes
        self.nNodes = self.V + self.C # total number of nodes
        self.nNeighbors = np.zeros(self.V + self.C, dtype=int) # number of neighbors

        # --- Count the number of Edges
        self.f.readline() # Skip the next line
        leftDegress = string2numpy(self.f.readline())
        self.E = sum(leftDegress)

        # --- Store the degrees of each node
        self.degrees = np.hstack((string2numpy(self.f.readline()),leftDegress))

        # --- Mapping from a node to the starting location at the message array
        self.node2msgLoc = self.compNode2msgLoc()

        # --- Array to store the messages
        self.messages = np.zeros(2 * self.E)

        # --- Generate Graph
        self.adjList = [None] * (self.nNodes)
        self.generateGraph()
        print()

        # --- Array to store the estimated codeword
        self.softCodeword = np.zeros(self.V)


# ==================================== Functions for Belief Propagation ==================================== #
    def decode(self, y, sigma2, nIter):


        # === Step 1: Initialize the variable nodes with the channel LLR
        self.initVarNodes(y, sigma2)

        for i in range(nIter):

            # === Step 2: Compute the messages from Check nodes to Variable nodes
            self.compCheck2Var()

            # === Step 3: Compute the messages from Variable nodes to Check nodes
            self.compVar2Check()

            # === Step 4: Check if the values of check node
            if self.isCorrect():
                return 1, self.softCodeword

        return 0, self.softCodeword

    # ====== Step 1 ====== #
    def initVarNodes(self,y, sigma2):
        # --- Compute the channel LLR
        self.chLLR = self.weight * y / sigma2
        for i in range(self.V):
            node = self.adjList[i]
            for d in range(self.degrees[i]):

                self.messages[self.node2msgLoc[node.id] + node.idxStore] = self.chLLR[i]
                node = node.next

    # ====== Step 2 ====== #
    def compCheck2Var(self):
        for i in range(self.V, self.nNodes, 1):
            node = self.adjList[i]
            P = np.prod(np.tanh(self.messages[self.node2msgLoc[i]:self.node2msgLoc[i] + self.degrees[i]]/2.0))
            for d in range(self.degrees[i]):

                # --- Compute the message
                L = self.messages[self.node2msgLoc[i] + node.idxRead]/2.0
                msg = 2*np.arctanh(P/np.tanh(L))

                if abs(msg) > 20:
                    msg = np.sign(msg) * 20

                # --- Store the message
                self.messages[self.node2msgLoc[node.id] + node.idxStore] = msg
                node = node.next


    # ====== Step 3 ====== #
    def compVar2Check(self):
        for i in range(self.V):
            node = self.adjList[i]
            S = np.sum(self.messages[self.node2msgLoc[i]:self.node2msgLoc[i] + self.degrees[i]])
            for d in range(self.degrees[i]):
                # --- Compute the message
                msg = S - self.messages[self.node2msgLoc[i] + node.idxRead] + self.chLLR[i]

                if abs(msg) > 20:
                    msg = np.sign(msg) * 20
                # --- Store the message
                self.messages[self.node2msgLoc[node.id] + node.idxStore] = msg

                node = node.next
                if d == 0:
                    # --- Compute the output LLR
                    self.softCodeword[i] = S + self.chLLR[i]


    # ====== Step 4 ====== #
    def isCorrect(self):

        # === Step 1: Compute the hard decision
        hardCodeword = (self.softCodeword < 0) * 1

        # === Step 2: check the value of the Check node
        for i in range(self.V, self.nNodes, 1):
            node = self.adjList[i]
            sum = 0
            for d in range(self.degrees[i]):
                sum += hardCodeword[node.id]
                node = node.next

            if sum % 2 != 0:
                return 0
        return 1


# ==================================== Functions to Construct the Graph ==================================== #
    def generateGraph(self):

        # --- Skip the information for check nodes
        for _ in range(self.C):
            self.f.readline()

        # --- For all variable nodes
        for v in range(self.V):
            # --- Find the neighbors of v
            neighbors = string2numpy(self.f.readline()) - 1 + self.V

            for c in neighbors:
                if c == self.V-1:
                    break

                # --- Add c node to the neighborhood of v, and
                self.addNode(v, c)

                # --- Add this node to the neighborhood of c
                self.addNode(c, v)

                self.nNeighbors[v] += 1
                self.nNeighbors[c] += 1


        self.f.close()

    def addNode(self, sourceID, destinationID):
            idxRead = self.nNeighbors[sourceID]
            idxStore = self.nNeighbors[destinationID]

            # --- Create a new node
            newNode = Node(destinationID, idxStore, idxRead)

            # --- Add this neighbor
            newNode.next = self.adjList[sourceID]
            self.adjList[sourceID] = newNode


    def compNode2msgLoc(self):

        loc = 0
        node2msgLoc = np.zeros(self.nNodes, dtype=int)
        for i in range(self.nNodes):
            node2msgLoc[i] = loc
            loc += self.degrees[i]
        return node2msgLoc

# ==================================== Utility Functions ==================================== #

def printGraph(graph, nNodes):

    for i in range(nNodes):
        temp = graph[i]
        if temp is not None:
            print("=============================================")
            print('Node: ' + str(i))
            print('Neighbors')
            print(temp.id)
            while temp.next is not None:
                temp = temp.next
                print(temp.id)

def string2numpy(string):
    return np.asarray(string.split(), dtype=int)