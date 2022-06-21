##################################################
# Simulated Annealing
##################################################
import numpy as np
import time

def simulatedAnnealing(x_start, costFunc, numCycles = 200000,
                        PROBACCEPTIFBETTER = 0.999,
                        PROBACCEPTIFWORSE = 0.001, tol = 25.0):

    NUMDOMAIN = len(x_start)
    x = np.zeros((numCycles+1,NUMDOMAIN)) # (NUMCYCLE, NUMVARIABLES)
    x[0] = x_start
    xi = x_start

    # xc = Current best results
    xc = x_start
    fc = costFunc(xi) # current cost function
    fs = np.zeros(numCycles+1)
    fs[0] = fc
    numFlips = 1 # num of domains flipped at each cycle

    start = time.time() # start clock
    for k in range(numCycles):
      # iterates through randomly chosen neighbour config
      if(np.mod(k,200)==0):
        print('Cycle: {:04.0f} with Cost function: {:04.3f}'.format(k,fc))
      randNum = np.random.random()

      # Flipping random domains
      flipDomains = np.random.choice(NUMDOMAIN, numFlips)
      xi[np.int(flipDomains)] *= -1.
      # Calculate cost function
      fnew = costFunc(xi)

      if(fnew >= fc):
        heatFunc = 2*np.power(2,-k/np.double(numCycles))-1
        p = (heatFunc)*PROBACCEPTIFWORSE*(fc/fnew)
        accept = randNum<p
      else:
        accept = (1-PROBACCEPTIFBETTER)*(fnew/fc) <= randNum

      # Replace current config as best config
      if(accept):
        xc = xi
        fc = fnew
      else:
        xi[np.int(flipDomains)] *= -1 #flip it back if not accepted
      if fc <= tol:
        break

      fs[k+1] = fc
      x[k+1] = xc

      # Plot fs on the fly
    # ENDFOR
    print('Cycle: {:05.0f} with Cost function: {:04.3f}'.format(numCycles,fc))

    # Start from left to right, flip every domain flipping each domain
    # and only keeping the new configuration if newcost < optcost
    print('Flipping every domain from left to right')
    optcost = costFunc(xc)
    for m in range(NUMDOMAIN):
      xc[m] *= -1 #flip single domain
      currcost = costFunc(xc)
      print('Flipping domain {}...'.format(m))
      if (currcost > optcost):
        xc[m] *= -1. #flip it back
      else:
        optcost = currcost # update optcost if better config
    # ENDFOR
    print('Cost function after flipping every domain: {:0.3f}'.format(optcost))

    end = time.time()
    print('Time elapsed: {:0.2f} min'.format((end-start)/60))

    return xc, optcost, x
