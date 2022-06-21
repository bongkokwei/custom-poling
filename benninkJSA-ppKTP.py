# Construct JSI using derived from Bennick's paper.

from engineerPMF import *

# Crystal properties
NUMDOMAIN = 650
polingConfig = [(-1)**k for k in range(NUMDOMAIN)]


# Crystal's parameter
crystalTemp = 32 # degree Celsius
lo = 23.10E-6 #METERS
expand = True
if expand == True:
    lc = crystalExpansion(lo, crystalTemp)
else:
    lc = lo

LENGTH = NUMDOMAIN*lc #MICRON
DOMAINWIDTH = lc*np.ones((NUMDOMAIN,))

# Pump laser setting
FWHM = 0.63E-9 # FWHM for Guassian pump (0.63nm for 1.0ps, 0.79nm for 0.8ps)
delta = 10E-9 #METER
pumpWaist = 300E-6 #METER
signalWaist = idlerWaist = 115E-6 #METER

# Plot properties
pumpWavelength = 775E-9 #775E-9 #METER
centralSignalWavelength = 2*pumpWavelength #METER
centralIdlerWavelength = centralSignalWavelength
centralWavelength = np.array([pumpWavelength, centralSignalWavelength, centralIdlerWavelength])

numGrid = 100+1 # resolution of graph
signalWavelength = np.linspace(centralSignalWavelength-delta,
                                centralSignalWavelength+delta, numGrid)
idlerWavelength = np.linspace(centralIdlerWavelength-delta,
                                centralIdlerWavelength+delta, numGrid)

# There is a need to reverse the idlerWavelength array, or else the origin will
# start on the top left corner of grid instead of bottom left.
idlerWavelength = idlerWavelength[::-1]
signal, idler = np.meshgrid(signalWavelength, idlerWavelength)

# Checks if padding the crystal will degrade the purity, it's for
# Raicol order purposes
# polingConfig, lc = repeatPolingConfig(filename,lc,2)

PEF = pumpEnvFunc(signal, idler, centralWavelength, FWHM, shape = 'sech')
customPhi, axis = customPhaseMatchedFunc(lc, polingConfig, crystalTemp,\
                                        signal, idler, centralWavelength,\
                                        pumpWaist, signalWaist, idlerWaist,\
                                        segments=8)

JSA = customPhi*PEF
JSI = np.abs(JSA)**2
# JSI /= np.max(JSI)
purity, entropy = getPurity(JSA) # JSI or JSA

textline0 = 'Periodically-poled\n'
textline1 = 'Domain width: {:0.04f} micron \nNo. of domains: {} \n'.format(lc*1E6, NUMDOMAIN)
textline2 = 'Crystal length: {:0.04f} mm \n'.format(LENGTH*1E3)
textline3 = 'Purity: {:0.02f}%, Entropy: {:0.04f} \n'.format(purity*100, entropy)
textline4 = 'Crystal temperature: {}$^o$C \n'.format(crystalTemp)
textline5 = 'Angle of PMF contour: {:0.02f}$^o$'.format(axis)
text = textline0 + textline1 + textline2 + textline3 + textline4 + textline5
print(text)

######################### Plotting it altogether ##############################

plotFigure(np.abs(customPhi)**2, np.abs(PEF)**2, JSI, signal, idler,
            colormap = "cividis", nlevels=100, textColour="black",
            filename = "./figures/ppKTP.png", figFormat = "png", toPrint=False,
            summary = text, printDark = False, printText = True)

############################## HOM dip ########################################

# Ploting the dependent HOM of the crystal
# customPhiPrime, axis = customPhaseMatchedFunc(lc, polingConfig, crystalTemp,\
#                                             idler, signal, centralWavelength,\
#                                             pumpWaist, idlerWaist, signalWaist)
# JSAPrime = customPhiPrime*PEF
# JSIPrime = np.abs(JSAPrime)**1
# JSIPrime /= np.max(JSIPrime)
# JSAPrime
# tau = np.linspace(-5E-12, 5E-12, 200)
# Prob = getHOM(idlerWavelength[::-1], signalWavelength, idler,signal, JSA, JSA, tau)
#
# fig2 = plt.figure()
# plt.title('Two-fold coincidence')
# plt.xlabel("Relative delay (s)")
# plt.plot(tau, 0.5-0.5*np.real(Prob))
# plt.show()
