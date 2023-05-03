import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt


FyData = np.array([0.0, 50, 100, 200, 320, 500, 550, 480, 280, 140, 0])
FxData = np.linspace(0,1,11)
def FishGeneratedInterp(u):
    """ Returns fish generated in a year given factor of fish """
    # Taken from Figure 1, roughly
    return np.interp(u, FxData, FyData)

def FishGeneratedSugested(u):
    return u**2*(1-u)*3712.5  # u^2(1-u) times normalization constant

def FishGeneratedExp(u):
    return 2296*u*(1-u)*np.exp(-7.8*(u-0.64)**2)  # u(1-u)exp(-((u - μ)/σ)^2) times normalization constant

FishGeneratedSpline = spi.PchipInterpolator(FxData,FyData)

u = np.linspace(0, 1, 1001)
# plt.plot(u*100, FishGeneratedInterp(u), label="Linear interpolation")
plt.plot(u*100, FishGeneratedSpline(u), label="Spline interpolation", c="C0")
plt.plot(u*100,FishGeneratedExp(u), label="$2296*x(1-x)e^{(-7.8*(x-0.64)^2)}$", c="C1")
plt.plot(u*100,FishGeneratedSugested(u), label="$3712.5*x^2(1-x)$",c="C2")
plt.xlabel("Percent of Maximum Fish Population")
plt.ylabel("New Fish per Year")
plt.legend()
plt.grid()
plt.show(block=False)


ShipyData = np.array([0.0, 10, 16, 20, 22, 23, 25])
ShipxData = np.array([0.0, 11, 22, 33, 44, 55,100])/100

def ShipEffInterp(u):
    """ Returns ship effectiveness (fish fished per ship per year) given factor of fish """
    # Taken from Figure 1, roughly
    return np.interp(u, ShipxData, ShipyData)

ShipEffSpline = spi.PchipInterpolator(ShipxData,ShipyData)


def ShipEffSugested(u):
    return 30.8*u/(u+0.2)  # parameters fitted to be roughly similar to interp

u = np.linspace(0, 1, 1001)
plt.figure()
# plt.plot(u*100, ShipEffInterp(u),label="Linear interpolation")
plt.plot(u*100, ShipEffSpline(u),label="Spline interpolation", c="C0")
plt.plot(u*100,ShipEffSugested(u), label="$30.8x/(0.2+x)$", c="C1")
plt.xlabel("Percent of Maximum Fish Population")
plt.ylabel("Ship Effectiveness")
plt.legend()
plt.grid()
plt.show(block=False)



input("Press ENTER to end")