import numpy as np

import matplotlib.pyplot as plt

def FishGeneratedInterp(u):
    """ Returns fish generated in a year given factor of fish """
    # Taken from Figure 1, roughly
    yData = np.array([0.0, 50, 100, 200, 320, 500, 550, 480, 280, 140, 0])
    xData = np.linspace(0,1,11)
    return np.interp(u, xData, yData)

def FishGeneratedSugested(u):
    return u**2*(1-u)*3712.5  # u^2(1-u) times normalization constant

u = np.linspace(0, 1, 1001)
plt.plot(u*100, FishGeneratedInterp(u), label="Interpolated")
plt.plot(u*100,FishGeneratedSugested(u), label="$3712.5*x^2(1-x)$")
plt.xlabel("Percent of Maximum Fish Population")
plt.ylabel("New Fish per Year")
plt.legend()
plt.grid()
plt.show(block=False)

def ShipEffInterp(u):
    """ Returns ship effectiveness (fish fished per ship per year) given factor of fish """
    # Taken from Figure 1, roughly
    yData = np.array([0.0, 10, 16, 20, 22, 23, 25])
    xData = np.array([0.0, 11, 22, 33, 44, 55,100])/100
    return np.interp(u, xData, yData)


def ShipEffSugested(u):
    return 30.8*u/(u+0.2)  # parameters fitted to be roughly similar to interp

u = np.linspace(0, 1, 1001)
plt.figure()
plt.plot(u*100, ShipEffInterp(u), label="Interpolated")
plt.plot(u*100,ShipEffSugested(u), label="$30.8x/(0.2+x)$")
plt.xlabel("Percent of Maximum Fish Population")
plt.ylabel("Ship Effectiveness")
plt.legend()
plt.grid()
plt.show(block=False)



input("Press ENTER to end")