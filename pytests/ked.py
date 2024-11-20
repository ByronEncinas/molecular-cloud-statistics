import numpy as np
import math
import matplotlib.pyplot as plt
AU = 1.496*10.0**11

tvec = [-1000, -10000, -100000]
bvec = np.logspace(9, 16, 8)
vvec = np.linspace(100, 1000, 10)
tList = []
bList = []
vList = []

for i in range(0, 3):
    tList = tList + [str(-tvec[i])]
 
for i in range(0, 8):
    bList = bList +  [str(bvec[i])]

for i in range(0, 10):
    vList = vList + [str(vvec[i])]

def makeSinglePlot(name):
    xvec = np.loadtxt('./data/' + name + 'xvec.txt')
    yvec = np.loadtxt('./data/' + name + 'yvec.txt')
    Evec = np.loadtxt('./data/' + name + 'Evec.txt')
    fig = plt.figure(figsize = (6, 6))
    ax1 = fig.add_subplot(111)
    sc1 = ax1.scatter(xvec, yvec, c=Evec, cmap='viridis', s = 1, rasterized = True)
    ax1.set_xlabel('$x$', fontsize = 18)
    ax1.set_ylabel('$y$', fontsize = 18)
    cbaxes = fig.add_axes([0.85, 0.56, 0.05, 0.32])
    cbar = plt.colorbar(sc1, cax = cbaxes)
    fig.subplots_adjust(left = .15)
    fig.subplots_adjust(right = .8)
    plt.show()

def get_r_Phi(x, y):
    r = np.zeros(len(x))
    phi = np.zeros(len(x))
    for i in range(0, len(x)):
        if x[i] == 0:
            r[i] = np.abs(y[i])
            phi[i] = math.pi/2*np.sign(y[i])
        if x[i] > 0:
            r[i] = math.sqrt(x[i]**2 + y[i]**2)
            phi[i] = np.arctan(y[i]/x[i])
        if x[i] < 0:
            r[i] = math.sqrt(x[i]**2 + y[i]**2)
            phi[i] = math.pi + np.arctan(y[i]/x[i])
    return r, phi

def makeMultiplePlot(heads):
    xvecs = [[] for i in range(0, len(heads))]
    yvecs = [[] for i in range(0, len(heads))]
    Evecs = [[] for i in range(0, len(heads))]
    rvecs = [[] for i in range(0, len(heads))]
    phivecs = [[] for i in range(0, len(heads))]
    phiOffset = [-.139, -.363, -.809]
    for i in range(0, len(heads)):
        xvecs[i] = np.divide(np.loadtxt('./data/' + heads[i] + 'xvec.txt'), AU)
        yvecs[i] = np.divide(np.loadtxt('./data/' + heads[i] + 'yvec.txt'), AU)
        Evecs[i] = np.loadtxt('./data/' + heads[i] + 'Evec.txt')
        rvecs[i], phivecs[i] = get_r_Phi(xvecs[i], yvecs[i])
        for j in range(0, len(phivecs[i])):
            phivecs[i][j] = phivecs[i][j] + phiOffset[i]
    minimumE = 10.0**8
    maximumE = -10.0**8
    for i in range(0, len(Evecs)):
        for j in range(0, len(Evecs[i])):
            if Evecs[i][j] < minimumE:
                minimumE = Evecs[i][j]
            if Evecs[i][j] > maximumE:
                maximumE = Evecs[i][j]
    fig = plt.figure(figsize = (6, 6))
    ax1 = fig.add_subplot(projection = 'polar')
    ax1.grid(False)
    for i in range(0, len(xvecs)):
        sc1 = ax1.scatter(phivecs[i], rvecs[i], c=Evecs[i], cmap='viridis', vmin = -10**6, vmax = maximumE, s = .1, rasterized = True)
    plt.ylim([0, 1000])
    #plt.yticks([250, 500])
    #ax1.set_yticklabels(['250 AU', '500 AU'])
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    #plt.yphilim([-250, 200])
    plt.scatter(x = 0, y = 0, marker = '*', color = 'r', s = 5)
    #ax1.set_xlabel('$\phi$', fontsize = 18)
    #ax1.set_ylabel('$r$, AU', fontsize = 18)
    cbaxes = fig.add_axes([0.85, 0.16, 0.05, 0.62])
    cbar = plt.colorbar(sc1, cax = cbaxes)
    cbaxes.set_yticks([-10**6, -5*10.0**5, 0, 5*10**5])
    cbaxes.set_yticklabels(['$-10^6$', '$-5 \\times 10^5$', '0', '$5 \\times 10^5$'])
    fig.text(.8, .82, '$E_{\\rm tot}$, Joules/kg', fontsize = 13)
    cbaxes.tick_params(axis='y', labelsize=11)

    #fig.text(.6, .65, '$v_0 = 1000$ m/s', fontsize = 9)
    #fig.text(.65, .35, '$v_0 = 300$ m/s', fontsize = 9)
    #fig.text(.4, .38, '$v_0 = 100$ m/s', fontsize = 9)
    ax1.annotate("$v_0 = 1000$ m/s", xy=(1.89, 680), xytext=(2.1, 950), arrowprops=dict(arrowstyle="->"), fontsize = 11)
    ax1.annotate("$v_0 = 300$ m/s", xy=(2.67, 660), xytext=(2.63, 950), arrowprops=dict(arrowstyle="->"), fontsize = 11)
    ax1.annotate("$v_0 = 100$ m/s", xy=(3.0, 670), xytext=(3.25, 950),arrowprops=dict(arrowstyle="->"), fontsize = 11)
    ax1.annotate("1000 AU", xy = (1, 1000), xytext = (1, 1200), arrowprops=dict(arrowstyle="->"), fontsize = 11)
    fig.subplots_adjust(left = .15)
    fig.subplots_adjust(right = .76)
    plt.savefig('introFigure.pdf' , bbox_inches='tight', dpi = 300)
    
makeMultiplePlot(['a', 'b', 'c'])