import numpy as np
from scipy import loadtxt, optimize
import matplotlib.pyplot as plt

# Here we define our fit function and residual functions
def fitfunc(p, x):
    return p[0]*np.exp(-x/p[1]) + p[2]
def residual(p, x, y, dy):
    return (fitfunc(p, x)-y)/dy

# Read in the data from file
ch, c= loadtxt('s&a.tsv', unpack=True, skiprows=51)
dc = 4

##############################################################################
# Fit
##############################################################################
p01 = [112.,60,10]
pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p01,
                                     args = (ch, c, dc), full_output=1)

if cov1 is None:
    print('Fit did not converge')
    print('Success code:', success1)
    print(mesg1)
else:
    print('Fit Converged')
    chisq1 = sum(info1['fvec']*info1['fvec'])
    dof1 = len(c)-len(pf1)
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
    ax1.errorbar(ch, c, dc, fmt='k.', label = 'Data')
    T = np.linspace(ch.min(), ch.max(), 500)
    ax1.plot(T, fitfunc(pf1, T), 'r-', label = 'Fit')

    ax1.set_title('Decay of the Muon')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Total Counts')
    ax1.legend()

    textfit = '$N(ch) = N_0e^{-ch/\\tau}$ + B \n' \
              '$N_0 = %.3f \pm %.3f$ counts \n' \
              '$\\tau = %.1f \pm %.1f$  ch \n' \
              '$B = %.1f \pm %.1f$ counts \n' \
              '$\chi^2= %.2f$ \n' \
              '$N = %i$ (dof) \n' \
              '$\chi^2/N = % .2f$' \
               % (pf1[0], pferr1[0], pf1[1], pferr1[1], pf1[2], pferr1[2],chisq1, dof1,
                  chisq1/dof1)
    ax1.text(0.6, .75, textfit, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top')
    ax1.set_xlim([-25, 575])

    plt.savefig('exp.png')
    plt.show()
