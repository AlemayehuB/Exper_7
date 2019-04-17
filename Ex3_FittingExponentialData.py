##############################################################################
# PHYS 211 Python Example Scripts
# Example 3 - Fitting Exponential Data
#
# PROGRAM:  This script reads in activity data (t [days], A [microC], dA
#           [microC]) from file and fits the data to an exponential
# INPUT:    Example3_Data.txt
# CREATED:  06/17/2015
# AUTHOR:   David McCowan
# MODIFIED: 07/27/2017 [Minor tweaks and updated to python3]
#           10/12/2017 [Fixed fit initialization parameters]
##############################################################################
import numpy as np
from scipy import loadtxt, optimize
import matplotlib.pyplot as plt

# Here we define our fit function and residual functions
def fitfunc(p, x):
    return p[0]*np.exp(-x/p[1])
def residual(p, x, y, dy):
    return (fitfunc(p, x)-y)/dy

# Read in the data from file
t, A, dA = loadtxt('Example3_Data.txt', unpack=True, skiprows=1)

##############################################################################
# Fit
##############################################################################
p01 = [100.,100.]
pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p01,
                                     args = (t, A, dA), full_output=1)

if cov1 is None:
    print('Fit did not converge')
    print('Success code:', success1)
    print(mesg1)
else:
    print('Fit Converged')
    chisq1 = sum(info1['fvec']*info1['fvec'])
    dof1 = len(t)-len(pf1)
    pferr1 = [np.sqrt(cov1[i,i]) for i in range(len(pf1))]
    print('Converged with chi-squared', chisq1)
    print('Number of degrees of freedom, dof =',dof1)
    print('Reduced chi-squared:', chisq1/dof1)
    print('Inital guess values:')
    print('  p0 =', p01)
    print('Best fit values:')
    print('  pf =', pf1)
    print('Uncertainties in the best fit values:')
    print('  pferr =', pferr1)
    print()
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.errorbar(t, A, dA, fmt='k.', label = 'Data')
    T = np.linspace(t.min(), t.max(), 500)
    ax1.plot(T, fitfunc(pf1, T), 'r-', label = 'Fit')
    
    ax1.set_title('Radioactivity of Co-57 Source')
    ax1.set_xlabel('Time, $t$ (days)')
    ax1.set_ylabel('Activity, $A$ ($\\mu$C)')
    ax1.legend()
    
    textfit = '$A(t) = A_0e^{-t/\\tau}$ \n' \
              '$A_0 = %.1f \pm %.1f$ $\\mu$C \n' \
              '$\\tau = %.0f \pm %.0f$ days\n' \
              '$\chi^2= %.2f$ \n' \
              '$N = %i$ (dof) \n' \
              '$\chi^2/N = % .2f$' \
               % (pf1[0], pferr1[0], pf1[1], pferr1[1], chisq1, dof1,
                  chisq1/dof1)
    ax1.text(0.6, .75, textfit, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top')
    ax1.set_xlim([-25, 575])

    plt.savefig('Example3_Figure1.pdf')
    plt.show()