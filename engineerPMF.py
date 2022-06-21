# TODO: integrate this script into Crystal's class
# OPTIMIZE: Figure out how to extract width info and for 1D fitting

import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import measure
from numpy.linalg import eig, inv
from scipy.optimize import curve_fit
from scipy.integrate import simps


###############################################################################
# GLOBAL VARIABLE
PI = np.pi
C = 299792458
PERMITTIVITY = 8.85418782E-12
EFFICICIENCYFACTOR = 1.0
CHI2EFF = 1E-8
HBAR = 1.05457173E-34
############################## Helper functions ##############################
def Gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def wavelength2AngFreq(wavelength):
    return (2*PI*C)/wavelength

def angFreq2Wavelength(angFreq):
    return (2*PI*C/angFreq)

def fwhm2Sigma(FWHM, wavelength):
    # approximation to convert FWHM in wavelength to FWHM in angular frequency
    # Using FWHM in ang freq to calc standard deviation for a Gaussian

    angFWHM = (2*PI)*(C/wavelength**2)*FWHM # angular freq FWHM
    sigma = angFWHM/(2*np.sqrt(np.log(2))) # FWHM to gauss sigma
    return angFWHM, sigma

def gen_domain(numDomain, numData):
  # Col: Number of dataset
  # Row: Number of domains
  dataset = np.random.choice([-1,1], (numDomain, numData))
  return dataset

def readPolingConfigFile(filename):
    # Open and read file in data directory, convert string to array
    file = open(filename, 'r')
    polingConfig = np.array(file.read().splitlines()).astype(np.float)
    file.close()
    return polingConfig

def getFWHM(x,y):
    max_y = max(y)  # Find the maximum y value
    xs = [k for k in range(len(x)) if y[k] > max_y/2.0]
    # print([max(xs), min(xs)])
    FWHM = np.abs(x[max(xs)]-x[min(xs)])
    return FWHM

def getSigma(dk,JSA):
    # obtain Gaussian width for fitting
    numGrid = len(JSA)
    #1D slice of JSA
    JSA1Dx = [JSA[k,int(numGrid/2)] for k in range(numGrid)]
    dk1Dx = [dk[k,int(numGrid/2)] for k in range(numGrid)]

    # anti diagonal
    JSA1Dy = [JSA[k,numGrid-k-1] for k in range(numGrid)]
    dk1Dy = [dk[k,numGrid-k-1] for k in range(numGrid)]

    sigmaX = getFWHM(dk1Dy,JSA1Dx)/(2*np.sqrt(np.log(2)))
    sigmaY = getFWHM(dk1Dy,JSA1Dy)/(2*np.sqrt(np.log(2)))
    return sigmaX, sigmaY

def polingConfig2Pos(polingConfig, domainWidth):
    numDomain = len(polingConfig)
    domainPos = np.ones((numDomain,))
    m = 0
    for k in range(numDomain-1):
        if polingConfig[k] != polingConfig[k+1]:
            m += 1
        else:
            domainPos[m] += 1

    domainPos = np.cumsum(domainWidth*domainPos[0:m+1])
    return domainPos


def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def spectrafit(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def plotFigure(Phi, PEF, JSI, signal, idler,\
               nlevels=50, colormap = 'viridis', textColour = 'black',\
               filename = "figures/test.tif", figFormat = "tif", toPrint=False,\
               summary = "HELLO WORLD", printDark = False, printText = True,\
               printTransparent = False):
    numGrid = len(signal)
    mid = np.int(numGrid/2)
    startWavelength = signal[0,0]*1E9
    endWavelength = signal[0,-1]*1E9

    if printDark:
        plt.style.use('dark_background')
    ## FIGURE 1 ##
    # figure1 = plt.figure(figsize=(15,4))
    # figSpc1 = figure1.add_gridspec(nrows = 1, ncols = 3, \
    #                                width_ratios=[1., 1., 1.], height_ratios=[1.])
    # figure1.add_subplot(figSpc1[0,0])
    # plt.xlim([startWavelength, endWavelength])
    # plt.ylim([startWavelength, endWavelength])
    # plt.yticks(color=textColour)
    # plt.xticks(color=textColour)
    # plt.xlabel('Idler wavelength (nm)', color=textColour)
    # plt.ylabel('Signal wavelength (nm)', color=textColour)
    # plt.title('Phase-match Function', color=textColour)
    # plt.locator_params(axis='y', nbins=5) # Number of ticks
    # plt.locator_params(axis='x', nbins=5)
    # phase = plt.contourf(signal*1E9, idler*1E9, Phi, \
    #                      levels = nlevels, cmap = colormap)
    # figure1.add_subplot(figSpc1[0,1])
    # plt.xlim([startWavelength, endWavelength])
    # plt.ylim([startWavelength, endWavelength])
    # plt.xticks(color=textColour)
    # plt.xlabel('Idler wavelength (nm)', color=textColour)
    # plt.yticks([])
    # plt.title('Pump Envelope Function', color=textColour)
    # plt.locator_params(axis='x', nbins=5)
    # pump = plt.contourf(signal*1E9, idler*1E9, PEF,\
    #                     levels = nlevels, cmap = colormap)
    # figure1.add_subplot(figSpc1[0,2])
    # plt.xlim([startWavelength, endWavelength])
    # plt.ylim([startWavelength, endWavelength])
    # plt.xticks(color=textColour)
    # plt.xlabel('Idler wavelength (nm)', color=textColour)
    # plt.yticks([])
    # plt.title('Joint Spectral Intensity', color=textColour)
    # plt.locator_params(axis='x', nbins=5)
    # joint = plt.contourf(signal*1E9, idler*1E9, JSI, \
    #                      levels = nlevels, cmap = colormap)
    #
    # plt.savefig(filename, format = figFormat, transparent = printTransparent)

    ## FIGURE 2 ##
    figure2 = plt.figure(figsize=(15,6))
    figSpc2 = figure2.add_gridspec(nrows = 2, ncols = 4, \
                                   width_ratios=[1.25, 2.5, 2.5, 2.5], \
                                   height_ratios=[2.5, 1.25])

    figure2.suptitle('')
    figure2.add_subplot(figSpc2[0,1])
    plt.xticks([])
    plt.yticks([])
    plt.title('Joint Spectral Intensity')
    plt.imshow(JSI, interpolation='spline16', cmap=colormap)

    contours = measure.find_contours(JSI, np.max(JSI)/2)
    for n, cnt in enumerate(contours):
        center, phi, axes, eccentricity = ellipseSpec(cnt[:, 1], cnt[:, 0])
        plt.plot(cnt[:, 1], cnt[:, 0], '--', linewidth=1,color='lawngreen')

    # Corner text
    delta = 1E-3*(endWavelength-startWavelength)/2 #MICRON
    centerInMicron_s = (center[0]/numGrid*(2*delta)+startWavelength*1E-3)*1E3
    centerInMicron_i = (endWavelength*1E-3 - center[0]/numGrid*(2*delta))*1E3
    centerInMicron = np.array([centerInMicron_s, centerInMicron_i])
    axesInNano = (axes/numGrid*2*delta)*1E3 #NANOMETER
    line1 = 'center = [{0[0]:0.02f}, {0[1]:0.02f}] nm\n'.format(centerInMicron)
    line2 = 'angle of rotation =  {0:0.02f}$^\circ$ \n'.format(phi)
    line3 = 'axes =  [{0[0]:0.04f}, {0[1]:0.04f}] nm\n'.format(axesInNano)
    line4 = 'eccentricity = {0:0.04f}'.format(eccentricity)
    text = line1+line2+line3+line4

    if printText:
        plt.text(0,numGrid-1, text, color='white', multialignment="left")

    wavelength = signal[0]*1E6
    figure2.add_subplot(figSpc2[1,1]) # signal
    plt.yticks([])
    plt.xlabel('Signal wavelength ($\mu$m)')
    signalJSI = np.sum(JSI, axis=0)
    plt.plot(wavelength, signalJSI)
    plt.axvline(x=signal[0,mid]*1E6, linestyle='--', color='m')
    plt.locator_params(axis='x', nbins=5)
    figure2.add_subplot(figSpc2[0,0]) #idler
    plt.xticks([])
    plt.ylabel('Idler wavelength ($\mu$m)')
    plt.locator_params(axis='y', nbins=7)
    idlerJSI = np.sum(JSI, axis=1)
    plt.axhline(y=idler[mid,0]*1E6, linestyle='--', color='m')
    plt.plot(idlerJSI[::-1], wavelength)
    figure2.add_subplot(figSpc2[1,0])
    plt.yticks([])
    plt.locator_params(axis='x', nbins=3)
    plt.plot(wavelength, signalJSI)
    plt.plot(wavelength, idlerJSI[::-1])
    plt.axvline(x=signal[0,mid]*1E6, linestyle='--', color='m')
    plt.xlabel('Wavelength ($\mu$m)')
    figure2.add_subplot(figSpc2[0,2])
    plt.yticks([])
    plt.xticks([])
    plt.title('Pump Envelope Function')
    plt.imshow(PEF, interpolation='spline16', cmap=colormap)
    figure2.add_subplot(figSpc2[0,3])
    plt.xticks([])
    plt.yticks([])
    plt.title('Phase matching function')
    plt.imshow(Phi, interpolation='spline16', cmap=colormap)

    figure2.add_subplot(figSpc2[1,2])
    plt.plot([0],[0])
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xticks([])
    plt.yticks([])
    plt.text(0.1,0.1,summary)


    # solution to aliasing when saving plot to as vecto grpahics
    # https://stackoverflow.com/questions/8263769/
    # hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills/32911283#32911283
    if toPrint:
        # for c in pump.collections:
        #     c.set_edgecolor("face")
        # for c in phase.collections:
        #     c.set_edgecolor("face")
        # for c in joint.collections:
        #     c.set_edgecolor("face")
        # Use tif for ppt pictures, svgz for paper publication
        plt.savefig(filename, format = figFormat, transparent = printTransparent)
    plt.show()

############################## Main equations ##############################
def crystalExpansion(lo, temp, alpha=6.7E-6, beta=11E-9, enable = True):
    if enable:
        return lo*(1 + alpha*(temp-25) + beta*(temp-25)**2)
    else:
        return lo

def sellmeierEqn(wavelength, temperature = 25):
    # Sellmeier equation for ppKtp
    # Sellmeier Equation, ny obtained from Konig and Wong 2004
    # Sellmeier Equation, nz obtained from Fradkin et al. 1999
    # wavelength in micron

    A = [2.09930, 0.922683, 0.0467695, 0.0138408]
    B = [2.12725, 1.18431, 5.14852E-2, 0.6603, 100.00507, 9.68956E-3]

    ny = np.sqrt(A[0] + (A[1]*np.power(wavelength,2)) /
                        (np.power(wavelength,2) - A[2]) -
                        A[3]*np.power(wavelength,2))

    nz = np.sqrt(B[0] + (B[1]*np.power(wavelength,2)) /
                        (np.power(wavelength,2) - B[2]) +
                        (B[3]*np.power(wavelength,2)) /
                        (np.power(wavelength,2) - B[4]) -
                        B[5]*np.power(wavelength,2))

    deltaNy, deltaNz = tempDependence(wavelength, temperature)

    return ny + deltaNy, nz + deltaNz

def diffSellmeierEqn(wavelength, temperature, h = 1E-4):
    # differentiating Sellmeier equation for ppKtp
    # Sellmeier Equation, ny obtained from Konig and Wong 2004
    # Sellmeier Equation, nz obtained from Fradkin et al. 1999
    # wavelength in micron

    nyPos, nzPos = sellmeierEqn(wavelength+h, temperature)
    nyNeg, nzNeg = sellmeierEqn(wavelength-h, temperature)

    dny = (nyPos-nyNeg)/(2*h)
    dnz = (nzPos-nzNeg)/(2*h)

    return dny, dnz

def tempDependence(wavelength = 0.775, temperature = 25):
    # Emanueli and Arie derived the temp dependence of refractive index of KTP

    n1z = lambda w: 9.9587E-6/w**0 + 9.9228E-6/w**1 - 8.9603E-6/w**2 + 4.1010E-6/w**3
    n2z = lambda w: -1.1882E-8/w**0 + 10.459E-8/w**1 - 9.8136E-8/w**2 + 3.1481E-8/w**3
    n1y = lambda w: 6.2897E-6/w**0 + 6.3061E-6/w**1 - 6.0629E-6/w**2 + 2.6486E-6/w**3
    n2y = lambda w: -0.14445E-8/w**0 + 2.2244E-8/w**1 - 3.5770E-8/w**2 + 1.3470E-8/w**3

    refractiveIndexY1 = n1y(wavelength)
    refractiveIndexY2 = n2y(wavelength)
    refractiveIndexZ1 = n1z(wavelength)
    refractiveIndexZ2 = n2z(wavelength)

    deltaNy = refractiveIndexY1*(temperature-25) + refractiveIndexY2*(temperature-25)**2
    deltaNz = refractiveIndexZ1*(temperature-25) + refractiveIndexZ2*(temperature-25)**2
    return deltaNy, deltaNz

def groupIndex(wavelength, temperature=25):
    # ng = n - wavelength*(dn/dwavelength) [inverseGrpVel]

    ny, nz = sellmeierEqn(wavelength, temperature)
    dny, dnz = diffSellmeierEqn(wavelength, temperature)

    grpIdxY = ny - wavelength*dny
    grpIdxZ = nz - wavelength*dnz

    # return grpIdxY, grpIdxZ, ny+delNy, nz+delNz
    return {'grpIdxY': grpIdxY, 'grpIdxZ': grpIdxZ, 'rftIdxY': ny, 'rftIdxZ': nz}

def deltaK(signalWavelength, idlerWavelength, centralWavelength, temp = 25):
    # Constructing the wave vector mismatch grid with first order approximation
    # PRL 105, 253601 (2010)

    omegaIdler = wavelength2AngFreq(idlerWavelength) # meshgrid
    omegaSignal = wavelength2AngFreq(signalWavelength) # meshgrid

    omegaPump = wavelength2AngFreq(centralWavelength[0]) # float
    omegaSignalCentral = wavelength2AngFreq(centralWavelength[1]) # float
    omegaIdlerCentral = wavelength2AngFreq(centralWavelength[2]) # float

    nPump = groupIndex(centralWavelength[0]*1E6, temp) # float
    nCentralSignal = groupIndex(centralWavelength[1]*1E6, temp) # float
    nCentralIdler = groupIndex(centralWavelength[2]*1E6, temp) # float

    inverseGrpVelPump = nPump['grpIdxY']/C
    inverseGrpVelSignal = nCentralSignal['grpIdxY']/C  # float
    inverseGrpVelIdler = nCentralIdler['grpIdxZ']/C  # float

    # Defining Y-axis as signal and Z-axis as idler
    nSignal = groupIndex(signalWavelength*1E6, temp)['rftIdxY']
    nIdler = groupIndex(idlerWavelength*1E6, temp)['rftIdxZ']

    kPump = omegaPump*nPump['rftIdxY']/C
    kCentralY = omegaSignalCentral*nCentralSignal['rftIdxY']/C # signal
    kCentralZ = omegaIdlerCentral*nCentralIdler['rftIdxZ']/C # idler

    kSignal = nSignal*omegaSignal/C # meshgrid
    kIdler  = nIdler*omegaIdler/C # meshgrid

    # First order approximation to wavevector mismatch centered
    # around centralWavelength
    dk0 = kPump - kCentralY - kCentralZ
    dK = dk0 + \
    (inverseGrpVelPump - inverseGrpVelIdler)*(omegaIdler-omegaIdlerCentral) + \
    (inverseGrpVelPump - inverseGrpVelSignal)*(omegaSignal-omegaSignalCentral)

    axis = np.degrees(np.arctan(-(inverseGrpVelPump-inverseGrpVelIdler)/ \
                                 (inverseGrpVelPump-inverseGrpVelSignal)))

    meanGrpIdx = (nCentralSignal['grpIdxY'] + nCentralIdler['grpIdxZ'])/2

    print("Mean group index: {:0.04f}".format(meanGrpIdx))
    # print('dk0 = {}'.format(dk0))
    print("Ideal domain width: {:0.04f} micron".format(np.pi/np.abs(dk0)*1E6))

    return nPump['rftIdxY'], nSignal, nIdler, kPump, kSignal, kIdler, dK, axis

def pumpEnvFunc(idlerWavelength, signalWavelength, centralWavelength, FWHM, \
                shape = 'sech'):
    # centralWavelength = [pump, signal, idler]
    omegaIdler   = wavelength2AngFreq(idlerWavelength)
    omegaSignal  = wavelength2AngFreq(signalWavelength)

    omegaSignalCentral = wavelength2AngFreq(centralWavelength[1])
    omegaIdlerCentral = wavelength2AngFreq(centralWavelength[2])

    angFWHM, sigmaPEF = fwhm2Sigma(FWHM, centralWavelength[0])

    signal = omegaSignal-omegaSignalCentral
    idler = omegaIdler-omegaIdlerCentral

    # The sech and Gaussian PEFs have equal width when tau ~ 0.712sigmaPEF
    if shape == 'gauss':
        PEF = np.exp(-(signal+idler)**2/(2*sigmaPEF**2))
    elif shape == 'sech':
        PEF = np.power(np.cosh(0.5*PI*(1.122/angFWHM)*(signal+idler)), -1)
    return PEF

def gaussPMF(idlerWavelength,signalWavelength, centralWavelength, FWHM, \
             theta=np.pi/4):
    omegaIdler = wavelength2AngFreq(idlerWavelength)
    omegaSignal = wavelength2AngFreq(signalWavelength)
    omegaCentral = wavelength2AngFreq(centralWavelength)
    pumpWavelength = centralWavelength/2 # for degenerate down-conversion

    angFWHM, sigmaPMF = fwhm2Sigma(FWHM, pumpWavelength)
    signal = omegaSignal-omegaCentral
    idler = omegaIdler-omegaCentral

    Phi = np.exp(-(np.sin(theta)*signal-np.cos(theta)*idler)**2/(sigmaPMF**2))
    return np.abs(Phi)

def customPMF(dk, domainWidth, polingConfig):
    # 2D custom phase match function
    # deltaK must be 2 dimensional square matrix
    # domainWidth must be an array
    coord = domainWidth.cumsum() # coord of boundaries along crystal length
    length = coord[-1]
    # coord of midpoints along poling config
    zn = domainWidth/2 + np.concatenate(([0.], coord[:-1]))
    deltaPhi = lambda x: np.sum(polingConfig*\
                                np.sin((x*domainWidth)/2)*\
                                np.exp(1j*x*zn))*(2/(x*length))

    # deltaPhi = lambda x: np.sinc((x-PI/domainWidth[0])*length/2)
    gridLen = len(dk)
    Phi = [[deltaPhi(dk[k,m]) for k in range(gridLen)] for m in range(gridLen)]
    Phi /= np.trace(np.matmul(Phi,Phi))
    return np.abs(Phi)

def customPhaseMatchedFunc(lc, polingConfig, crystalTemp,\
                           signal, idler, centralWavelength,\
                           pumpWaist, signalWaist, idlerWaist,
                           segments=5):

    domainDiv = np.int(segments) # subdivide each domain into smaller segments
    crystalLen = lc*len(polingConfig)
    repPolingConfig = np.repeat(polingConfig, domainDiv) # subdivide for integration
    numSegments = len(repPolingConfig)

    # Add uncertainty to segments
    z = np.linspace(-1,1,numSegments)

    # Eqn 16 of Bennink 2010
    NP = 5.57E18 # mean number of pump photons (the pump energy divided by hbar*omega_p)

    nPump, nSignal, nIdler, kPump, kIdler, kSignal, dk, axis = \
    deltaK(signal, idler, centralWavelength, temp = crystalTemp)

    numGrid = len(signal)
    integral = np.zeros((numGrid, numGrid, numSegments), dtype = 'cdouble')

    # Focal parameter eqn 11
    pumpFocalParam = crystalLen/(kPump*pumpWaist**2)
    signalFocalParam = crystalLen/(kSignal*signalWaist**2)
    idlerFocalParam = crystalLen/(kIdler*idlerWaist**2)

    # auxiliary quantities
    APlus = 1 + (kSignal/kPump)*(signalFocalParam/pumpFocalParam) + \
                (kIdler/kPump)*(idlerFocalParam/pumpFocalParam)

    BPlus = (1 - dk/kPump)*\
            (1 + ((kSignal+dk)/(kPump-dk))*(pumpFocalParam/signalFocalParam) + \
                 ((kIdler+dk)/(kPump-dk))*(pumpFocalParam/idlerFocalParam))

    CPlus = (dk/kPump)*(pumpFocalParam**2/(signalFocalParam*idlerFocalParam))* \
            (APlus/BPlus**2)

    # aggregate focal parameter
    aggreFocalParam = (BPlus/APlus)*((signalFocalParam*idlerFocalParam)/pumpFocalParam)

    constantOne = np.sqrt((8.0*PI**2*EFFICICIENCYFACTOR*HBAR*nSignal*nIdler*NP*crystalLen)/(PERMITTIVITY*nPump))
    constantTwo = CHI2EFF/(signal*idler*np.sqrt(APlus*BPlus))

    # integrand = lambda dl: (np.sqrt(aggreFocalParam)*np.exp(1j*(dk*crystalLen)*dl/2))/ \
    #                        (1 - 1j*aggreFocalParam*dl - CPlus*(aggreFocalParam**2)*dl**2)

    # for k in range(numSegments):
    #     integral[:,:,k] = repPolingConfig[k]*integrand(z[k])

    exponent = np.exp(np.einsum("ij,k -> ijk", 1j*(dk*crystalLen), z/2))
    numerator =  np.einsum("ij,ijk -> ijk", np.sqrt(aggreFocalParam), exponent)
    denom = 1 - 1j*np.einsum("ij,k->ijk", aggreFocalParam, z) - \
                np.einsum("ij,k->ijk", CPlus*(aggreFocalParam**2), z**2)

    integrand = numerator/denom
    integral = np.einsum('k, ijk -> ijk', repPolingConfig, integrand)

    Phi = constantOne*constantTwo*np.trapz(integral, z, axis=-1)
    return Phi, axis

def getPurity(JSA):
    # From the Schmidt decomposition, we can see that w is entangled
    # if and only if w has Schmidt rank strictly greater than 1
    # https://en.wikipedia.org/wiki/Schmidt_decomposition

    u, s, vh = np.linalg.svd(JSA, full_matrices=True)
    s /= np.sqrt(np.sum(s**2)) # Normalise Schmidt coefficients

    # From Dosseva et al. 2016, pg 3
    entropy = -np.sum(np.abs(s)**2*np.log(np.abs(s)**2))
    purity = np.sum(s**4)
    return purity, entropy

def costFunction(x, PEF, Phi):
    # cost function is a function of poling configuration, x
    customPhi = Phi(x)
    JSA = customPhi*PEF
    JSA /= np.trace(np.matmul(JSA,JSA))
    purity, entropy = getPurity(JSA)
    loss = 1-purity
    return loss

def getHOM(idlerWavelength, signalWavelength, idler, signal, JSAPrime, JSA, tau):
    # Calculate dependent HOM dip with JSI
    # arXiv:1711.00080v1, EQN 69 (NAISE)
    omegaSignal = wavelength2AngFreq(signalWavelength)
    omegaIdler = wavelength2AngFreq(idlerWavelength)
    signal = wavelength2AngFreq(signal)
    idler = wavelength2AngFreq(idler)

    prob_conic = lambda tau: (JSAPrime.conj().T@JSA)*np.exp(1j*(idler-signal)*tau)
    Prob = np.array([simps(simps(prob_conic(t), omegaIdler), omegaSignal) for t in tau])
    Prob /= np.max(Prob)

    return Prob

############################## Fitting Ellispe ##############################
# http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipseCenter(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipseEccentricity(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return c/a

def ellipseAngleOfRotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipseAxisLength(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipseAngleOfRotation2(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def plotEllipse(center, phi, axes):
    R = np.arange(0, 2*PI, 0.01)
    a, b = axes
    xx = center[0] + a*np.cos(R)*np.cos(phi+PI/2) - b*np.sin(R)*np.sin(phi+PI/2)
    yy = center[1] + a*np.cos(R)*np.sin(phi+PI/2) + b*np.sin(R)*np.cos(phi+PI/2)
    plt.plot(xx,yy)


def ellipseSpec(x,y, printScrn = False):
    a = fitEllipse(x,y)
    center = ellipseCenter(a)
    #phi = ellipse_angle_of_rotation(a)
    phi = ellipseAngleOfRotation2(a)
    axes = ellipseAxisLength(a)
    eccentricity = ellipseEccentricity(a)

    if printScrn:
        print("center = ",  center)
        print("angle of rotation = ",  phi)
        print("axes = ", axes)
        print("eccentricity = ", eccentricity)

    return center, phi, axes, eccentricity

###############################################################################

def repeatPolingConfig(filename, lc, rep = 4, toSave = False):
    # rep = number of times each element is repeated
    # save to new poling config file

    polingConfig = readPolingConfigFile(filename)
    repPolingConfig = np.repeat(polingConfig, rep)

    if(toSave):
        splitFilename = filename.split('.')
        newFilename = '.' + splitFilename[1] + '_' + str(rep) + 'repeats.' + splitFilename[-1]
        np.savetxt(newFilename, repPolingConfig)
    return repPolingConfig, lc/rep
