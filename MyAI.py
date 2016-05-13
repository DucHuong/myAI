import numpy as np
import random as rd
import glob
import os



board = [-1,-1,-1,-1,-1,-1,-1,-1,-1,
     -1, 0, 0, 0, 0, 0, 0, 0, 0,
     -1, 0, 0, 0, 0, 0, 0, 0, 0,
     -1, 0, 0, 0, 0, 0, 0, 0, 0,
     -1, 0, 0, 1, 2, 1, 0, 0, 0,
     -1, 0, 0, 0, 1, 2, 0, 0, 0,
     -1, 0, 0, 0, 0, 0, 0, 0, 0,
     -1, 0, 0, 0, 0, 0, 0, 0, 0,
     -1, 0, 0, 0, 0, 0, 0, 0, 0,
     -1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
test = 0
thongso = []
def sigmoid(x):
        return 1.0 / (1.0 + np.exp(0.0-x) )

def ACi(n, i):
    mu = [4, 34, 64]
    return np.exp(-(n - mu[i])*(n - mu[i]) * 1.0 / (cf.sigmaAC * cf.sigmaAC))

def sigmoidDerivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)




class cf:
    sigma = 0.2
    sigmaAC = 20.0
    selfAdaptive = 0.05

    idxInput = 0
    idxH1 = 1
    idxH2 = 2
    idxH3 = 3

    mutaionRate = 0.01

    INF = 200 #Greater than can overflow memory
    numLayerHidden = 3
    numNodeInput = 64 #Them 1 node bias
    numNodeHid1 = 42 #Them 1 node bias
    numNodeHid2 = 37 #Them 1 bias
    numNodeHid3 = 25 #Them 1 bias
    numNodeOut = 1
    numIdividuNet = 40

    posOutput = 2
    posHid1 = 0
    posHid2 = 1

    matchNum = 2

    pointWin = 5
    pointDraw = 1
    pointLose = 0
    err = 0.0001
    probability_add = 0.25
    probablity_multation = 0.01

    predictNamePop = 'Population'
    predictNameNet = 'NewNet'
    predictNameFol = 'NewG'
    predictPopNet = 'InforNetwork'
    predictH1 = 'H1'
    predictH2 = 'H2'
    predictOut = 'Out'
    extention = '.txt'
    
    IndexFFitness = -3
    IndexFAdaptive = -2
    IndexFNumConn = -1

    popNeuSize = 50

    numTrain = 20
    @staticmethod
    def readConfig():
        name = 'config.txt'
        if not os.path.exists(name):
            return 0
   
        with open(name, 'r') as f:
            l = f.readline()
            f.close()
            return int(l)
    @staticmethod
    def writeConfig(gen):
        name = 'config.txt'
        f = open(name, 'w')
        f.writelines(str(gen))
        f.close()
    @staticmethod
    def writeWinDraw(i, w, d):
        nm = 'WinDraw.txt'
        f = open(nm, 'a')
        ww = "Con thu: " + str(i) + "\tThang: " + str(w) + "\tDraw: " + str(d) + "\n"
        f.writelines(ww)
        f.close()

class Neural:
    def __init__(self, numConnection):
        self.numConn = numConnection
        self.vWeight = []
        self.bias = 0
        #self.fitness = 0
        self.adaptive = [cf.selfAdaptive] * self.numConn

    def create(self):
        self.bias = rd.gauss(0, 1) * cf.sigma
        for i in range(self.numConn):
            w = rd.gauss(0, 1) * cf.sigma
            self.vWeight.append(w)

    def mutiWeight(self, vInput):
        return np.inner(vInput, self.vWeight) + self.bias

class Network:
    #Mac dinh output la 1 node
    def __init__(self):
        self.numberNodeH1 = cf.numNodeHid1
        self.numberNodeH2 = cf.numNodeHid2
        self.numberNodeH3 = cf.numNodeHid3
        self.numberNodeOutput = 1

        self.nodesH1 = []
        self.nodesH2 = []
        self.nodesH3 = []
        self.nodesOut = []
        self.lstLayers = [self.nodesH1, self.nodesH2, self.nodesH3, self.nodesOut]


    def createRand(self):
        for i in range(self.numberNodeH1):
            nn = Neural(cf.numNodeInput)
            nn.create()
            self.nodesH1.append(nn)

        for i in range(self.numberNodeH2):
            nn = Neural(self.numberNodeH1)
            nn.create()
            self.nodesH2.append(nn)

        for i in range(self.numberNodeH3):
            nn = Neural(self.numberNodeH2)
            nn.create()
            self.nodesH3.append(nn)

        for i in range(self.numberNodeOutput):
            nn = Neural(self.numberNodeH3)
            nn.create()
            self.nodesOut.append(nn)

    def net(self, vInput):
        rH1 = []
        for i in range(self.numberNodeH1):
            r = self.nodesH1[i].mutiWeight(vInput)
            rH1.append(r)
        oH1 = [sigmoid(x) for x in rH1]
      
        rH2 = []
        for i in range(self.numberNodeH2):
            r = self.nodesH2[i].mutiWeight(oH1)
            rH2.append(r)
        oH2 = [sigmoid(x) for x in rH2]
        
        rH3 = []
        for i in range(self.numberNodeH3):
            r = self.nodesH3[i].mutiWeight(oH2)
            rH3.append(r)
        oH3 = [sigmoid(x) for x in rH3]
        
        rOut = []
        for i in range(self.numberNodeOutput):
            r = self.nodesOut[i].mutiWeight(oH3)
            rOut.append(r)
        oH4 = [sigmoid(x) for x in rOut]

        return rH1, rH2, rH3, rOut, oH1, oH2, oH3, oH4
    
    def updateWeightNeural(self, vInput, expectValue):
        iH1, iH2, iH3, iH4, oH1, oH2, oH3, oH4 = self.net(vInput)

        err = expectValue - oH4[0]
        
        while abs(err) > cf.err:
            print (err)
            
            delta = err * sigmoidDerivative(iH4[0])
            self.nodesOut[0].vWeight = [self.nodesOut[0].vWeight[x] + self.nodesOut[0].adaptive[x] * oH4[0] * delta for x in range(self.nodesOut[0].numConn)]

            deltaH3_ = [self.nodesOut[0].vWeight[x] * delta * sigmoidDerivative(iH3[x]) for x in range(self.nodesOut[0].numConn)]
            #for i in range(self.numberNodeH3):
            #    node = self.nodesH3[i]
            #    node.vWeight = [node.vWeight[x] + node.adaptive[x] * oH3[i] * deltaH3_[i] for x in range(node.numConn)]
            self.updateWeight(deltaH3_, self.numberNodeH3, self.nodesH3, oH3)

            deltaH2_ = []
            for ii in range(self.numberNodeH2):
                sumD = 0
                for i in range(self.numberNodeH3):
                    sumD = sumD + deltaH3_[i] * self.nodesH3[i].vWeight[ii]
                deltaH2_.append(sumD)
        
            #for i in range(self.numberNodeH2):
            #    node = self.nodesH2[i]
            #    node.vWeight = [node.vWeight[x] + node.adaptive[x] * oH2[i] * deltaH2_[i] for x in range(node.numConn)]
            self.updateWeight(deltaH2_, self.numberNodeH2, self.nodesH2, oH2)
        
            deltaH1_ = []
            for ii in range(self.numberNodeH1):
                sumD = 0
                for i in range(self.numberNodeH2):
                    sumD = sumD + deltaH2_[i] * self.nodesH2[i].vWeight[ii]
                deltaH1_.append(sumD)
            self.updateWeight(deltaH1_, self.numberNodeH1, self.nodesH1, oH1)

            iH1, iH2, iH3, iH4, oH1, oH2, oH3, oH4 = self.net(vInput)
            err = expectValue - oH4[0]        
    def nomalize(vector):
        pass
    def updateWeight(self, delta, numNode, lstNode, oH):
        for i in range(numNode):
            node = lstNode[i]
            node.vWeight = [node.vWeight[x] + node.adaptive[x] * oH[i] * delta[i] for x in range(node.numConn)]

class Network3:
    def __init__(self):
        
        self.alpha = 0.05
    def createRand(self, numNodeInput, numNodeH1, numNodeH2, numNodeH3):
        self.net1 = Network()#numNodeInput, numNodeH1, numNodeH2, numNodeH3)
        self.net1.createRand()

        self.net2 = Network()#numNodeInput, numNodeH1, numNodeH2, numNodeH3)
        self.net2.createRand()

        self.net3 = Network()#numNodeInput, numNodeH1, numNodeH2, numNodeH3)
        self.net3.createRand()

        self.networks = [self.net1, self.net2, self.net3]

    def eval3(self, vInput, numDisc, eV = 0):
        if numDisc <= 20:
            pos_num = 0
            net = self.net1
        elif numDisc <= 40:
            pos_num = 1
            net = self.net2
        else:
            pos_num = 2
            net = self.net3

        net.updateWeightNeural(vInput, eV)

    def eval3_1(self, vInput, numDisc, eV = 0):
        if numDisc <= 20:
            pos_num = 0
            net = self.net1
        elif numDisc <= 40:
            pos_num = 1
            net = self.net2
        else:
            pos_num = 2
            net = self.net3
        h1, h2, h3, out = net.eval(vInput)
                


    def play(self, vInput, numDisc):
        if numDisc <= 20:
            net = self.net1
        elif numDisc <= 40:
            net = self.net2
        else:
            net = self.net3
        return net.eval(vInput)[-1][0]


    def eval(self, vInput, numDisc):
        res = 0
        for i in range(3):
            _res = ACi(numDisc, i) * self.networks[i].eval(vInput)[3][0]


    def deltaWeight(self, vInput, expectValue):
        pass

def readFile(filename):
    f = open(filename, 'r')
    data = f.readlines()
    borad_ = data[0].split(', ')
    board_ = [int(x) for x in borad_]
    #board_1 = []
    #for x in board_:
    #    if x == 1:
    #        board_1.append(3)
    #    else: board.append(x)
    #board_2 = []
    #for x in board_1:
    #    if x == 2:
    #        board_2.append(1)
    #    else: board.append(x)

    #board_ = []
    #for x in board_2:
    #    if x == 3:
    #        board_.append(2)
    #    else: board_.append(x)

    eV = (float(data[1])+ 100.0)/ 200.0
    f.close()
    return board_, eV


def saveNeural(neu, f):
    pass
def saveNet(net, i, directory):
    filename = cf.predictNameNet + str(i) + cf.extention 
    path = os.path.join(directory,  filename)

    f = open(path, 'w')
    number = str(net.numberNodeH1) + ', ' + str(net.numberNodeH2) + ', ' + str(net.numberNodeH3) + ', ' + str(net.numberNodeOutput) + '\n'
    numConn = str(net.nodesH1[0].numConn) + ', ' + str(net.nodesH2[0].numConn) + ', ' + str(net.nodesH3[0].numConn) + ', ' + str(net.nodesOut[0].numConn) + '\n'
    f.writelines(number)
    f.writelines(numConn)

    for node in net.nodesH1:
        bias = str(node.bias) + '\n'
        adap = str(node.adaptive)[1:-1] + '\n'
        w = str(node.vWeight)[1:-1] + '\n'
        f.writelines(bias)
        f.writelines(adap)
        f.writelines(w)

    for node in net.nodesH2:
        bias = str(node.bias) + '\n'
        adap = str(node.adaptive)[1:-1] + '\n'
        w = str(node.vWeight)[1:-1] + '\n'
        f.writelines(bias)
        f.writelines(adap)
        f.writelines(w)

    for node in net.nodesH3:
        bias = str(node.bias) + '\n'
        adap = str(node.adaptive)[1:-1] + '\n'
        w = str(node.vWeight)[1:-1] + '\n'
        f.writelines(bias)
        f.writelines(adap)
        f.writelines(w)

    for node in net.nodesOut:
        bias = str(node.bias) + '\n'
        adap = str(node.adaptive)[1:-1] + '\n'
        w = str(node.vWeight)[1:-1] + '\n'
        f.writelines(bias)
        f.writelines(adap)
        f.writelines(w)

    f.close() 
def saveNet3(net, i):

    directory = cf.predictNameFol + str(i)
    if not os.path.exists(directory):
        os.makedirs(directory)
    saveNet(net.net1, 1, directory)
    saveNet(net.net2, 2, directory)
    saveNet(net.net3, 3, directory)

def train(myNet):
    #co = 0   
    for filename in glob.glob('_second\\*.txt'):
        print("+++++++++++++++++++++++++++++++++++++")
        board, eV = readFile(filename)
        bbb = [x for x in board if x != -1]
        numDisc = 0
        for i in bbb:
            if 0 < i < 3:
                numDisc = numDisc + 1
        myNet.eval3(bbb, numDisc, eV)
        #break
#bbb = [-1, -1, -1, -1, -1, -1, -1, -1, -1,
#       -1, 0, 1, 1, 1, 0, 0, 0, 1, 
#       -1, 0, 0, 1, 1, 1, 2, 1, 0, 
#       -1, 0, 1, 2, 2, 1, 1, 0, 0, 
#       -1, 0, 1, 2, 2, 2, 0, 0, 0, 
#       -1, 0, 2, 2, 1, 1, 2, 1, 0, 
#       -1, 0, 2, 0, 1, 2, 2, 2, 0, 
#       -1, 2, 0, 1, 1, 2, 2, 2, 0, 
#       -1, 0, 2, 2, 2, 2, 2, 2, 2, 
#       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

#eV = (-2.06504545455 + 100.0) / (200.0)
#numDisc = 0
#bbb = [x for x in bbb if x != -1]
#for i in bbb:
#    if 0 < i < 3:
#        numDisc = numDisc + 1
popu = []

#for dem in range(2): #Test tim kiem node toi uu
#    for i in range(42, 60):
#        for j in range(35, i):
#            for k in range(24, j):
#                cf.numNodeHid1 = i
#                cf.numNodeHid2 = j
#                cf.numNodeHid3 = k
#                myNet = Network3()
#                myNet.createRand(cf.numNodeInput, cf.numNodeHid1, cf.numNodeHid2, cf.numNodeHid3)    
#                train(myNet)
#                popu.append(myNet)
#                #p = input("Da ket thuc")
#                print ("Save net " + str(dem)) 
#    p = input("Chuan bi coi i,j,k")
#    mv = 100
#    rr = []
#    #for ii in thongso:
#    #    if ii[0] < mv:
#    #        mv = ii[0]
#    #        rr = ii
#    #print rr
#    p = input("Xong 1")
#    #saveNet3(myNet, dem)


#myNet.eval3(bbb, numDisc, eV)

for dem in range(10): #Test tim kiem node toi uu        
    myNet = Network3()
    myNet.createRand(cf.numNodeInput, cf.numNodeHid1, cf.numNodeHid2, cf.numNodeHid3)    
    train(myNet)
    popu.append(myNet)
    print ("Save net " + str(dem)) 
    #p = input("Ngung")
    saveNet3(myNet, dem)

print ("Da train xong. net toi uu nam trong folder")   
