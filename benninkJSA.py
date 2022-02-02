# Construct JSI using derived from Bennick's paper.

from engineerPMF import *

# Crystal properties
# Get poling configuration from file
filename = "./data/polingConfig_032.txt"
p = readPolingConfigFile(filename)
# Concat more domains to match Raicol requirement of 25mm crystal
polingConfig = np.concatenate((p,np.ones((7,))))
NUMDOMAIN = len(polingConfig)
# NUMDOMAIN = 650
# polingConfig = [(-1)**k for k in range(NUMDOMAIN)] # Uncomment this for periodic poling
crystalTemp = 20 # degree Celsius
lc = 23.10E-6 #METERS
LENGTH = NUMDOMAIN*lc #MICRON
DOMAINWIDTH = lc*np.ones((NUMDOMAIN,))

# Pump laser setting
FWHM = 0.63E-9 # FWHM for Guassian pump (0.63nm for 1.0ps, 0.79nm for 0.8ps)
delta = 8E-9 #METER
pumpWaist = 300E-6 #METER
signalWaist = idlerWaist = 115E-6 #METER

# Plot properties
centralWavelength = 1550E-9 #METER
pumpWavelength = 775E-9 #775E-9 #METER
numGrid = 100+1 # resolution of graph

signalWavelength = idlerWavelength = \
np.linspace(centralWavelength-delta, centralWavelength+delta, numGrid)
idlerWavelength = idlerWavelength[::-1]
# There is a need to reverse the idlerWavelength array, or else the origin will
# start on the top left corner of grid instead of bottom left.
signal, idler = np.meshgrid(signalWavelength, idlerWavelength)

# Checks if padding the crystal will degrade the purity, it's for
# Raicol order purposes
# polingConfig, lc = repeatPolingConfig(filename,lc,2)

PEF = pumpEnvFunc(signal, idler, centralWavelength, FWHM, shape = 'sech')
customPhi = customPhaseMatchedFunc(lc, polingConfig, crystalTemp,\
                                   idler, signal, centralWavelength, pumpWavelength,\
                                   pumpWaist, signalWaist, idlerWaist)

JSA = customPhi*PEF
JSI = np.abs(JSA)**2
JSI /= np.max(JSI)
purity, entropy = getPurity(JSA) # JSI or JSA

textline1 = 'Domain width: {:0.02f} micron \nNo. of domains: {} \n'.format(lc*1E6, NUMDOMAIN)
textline2 = 'Crystal length: {:0.04f} mm \n'.format(LENGTH*1E3)
textline3 = 'Purity: {:0.02f}%, Entropy: {:0.04f} \n'.format(purity*100, entropy)
textline4 = 'Crystal temperature: {}$^o$'.format(crystalTemp)
text = textline1 + textline2 + textline3 + textline4
print(text)

######################### Plotting it altogether ##############################

# Ploting the dependent HOM of the crystal
# tau = np.linspace(-5E-12, 5E-12, 200)
# Prob = getHOM(idlerWavelength[::-1], signalWavelength, idler,signal, JSI, tau)

# fig2 = plt.figure()
# plt.title('Two-fold coincidence')
# plt.xlabel("Relative delay (s)")
# plt.plot(tau, 0.5-0.5*np.real(Prob))

plotFigure(np.abs(customPhi)**2, np.abs(PEF)**2, JSI, signal, idler,
            colormap = "cividis", nlevels=100, textColour="black",
            filename = "./figures/ppKTP.png", figFormat = "png", toPrint=False,
            summary = text, printDark = False, printText = False)

# domainPos = polingConfig2Pos(polingConfig,lc)
# np.savetxt('./data/domainPos_run32.txt', domainPos, fmt = '%0.08f')
