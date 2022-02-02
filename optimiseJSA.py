# This script will find the optimal poling configuration using simulated
# annealing where the cost function is the purity measure from JSA
# We have the option of randomly Initialising poling config, or adding more domains
# to already optimised ones

import numpy as np
from engineerPMF import *
from simulatedAnnealing import *
# np.random.seed(145)
newFilename = "./data/polingConfig_test.txt"
filename = "./data/polingConfig_032.txt"
readFromFile = False # Set true to read from file
nc = 10000 # number of cycles
addDomains = 20
if (readFromFile):
    # This will pad existing config with up and down poling
    mid = readPolingConfigFile(filename)
    up = np.ones((addDomains,))
    down = -np.ones((addDomains,))
    x_start = np.concatenate((up,mid,down))
    NUMDOMAIN = len(x_start)
else:
    # Initialize random poling configuration
    NUMDOMAIN = 1000 # ~20mm
    x_start = [(-1)**k for k in range(NUMDOMAIN+addDomains)]
    # x_start = 2*np.random.choice(2, NUMDOMAIN)-1 # Randomise initial poling

# Crystal properties
crystalTemp = 32 # deg Celsius
lc = 23.10E-6 # meters
LENGTH = NUMDOMAIN*lc #MICRON
DOMAINWIDTH = lc*np.ones((NUMDOMAIN,))

######################### Pump laser setting ###################################
FWHM = 0.63E-9 # 0.79E-9 # 0.8E-9 FWHM for Guassian pump
delta = 10E-9 #METER
centralWavelength = 1550E-9 #METER
pumpWavelength = 775E-9 #METER
numGrid = 50+1 # resolution of graph

signalWavelength = idlerWavelength = \
np.linspace(centralWavelength-delta, centralWavelength+delta, numGrid)
signal, idler = np.meshgrid(signalWavelength, idlerWavelength)

PEF = pumpEnvFunc(signal, idler, centralWavelength, FWHM, shape = 'sech')
Phi = lambda x: customPhaseMatchedFunc(lc, x, crystalTemp,\
                signal, idler, centralWavelength, pumpWavelength, \
                pumpWaist=300E-6, signalWaist=115E-6 , idlerWaist=115E-6)

######################### Cost function definition #############################
costFunc = lambda x: costFunction(x, PEF, Phi)
xc, optcost, x = simulatedAnnealing(x_start, costFunc, PROBACCEPTIFWORSE = 0.001, \
                                    numCycles = nc, tol = 1E-4)

np.savetxt(newFilename, xc, delimiter='')

# JSA = Phi(xc)*PEF
# JSI = np.abs(JSA)**2
# Phi = np.abs(Phi)**2
# PEF = np.abs(PEF)**2
# JSI /= np.max(JSI) # Normalise JSI

######################### Plotting it altogether ##############################
# plotFigure(Phi, PEF, JSI, signal, idler, colormap = 'cividis')
