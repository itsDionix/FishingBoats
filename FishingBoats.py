import numpy as np
import scipy.interpolate as spi
from scipy.integrate import solve_ivp
import scipy.optimize as sio
import matplotlib.pyplot as plt



# u will generally be normalized by uMax, so u in[0, 1]
uMax = 2000

## The following models are just numerical guesses at the fish generation rate
FyData = np.array([0.0, 50, 100, 200, 320, 500, 550, 480, 280, 140, 0])/uMax
FxData = np.linspace(0,1,11)
def BirthRateInterp(u):
    """ Returns fish generation rate in normalized fish per year """
    # Taken from Figure 1, roughly
    return np.interp(u, FxData, FyData)

def BirthRateSugested(u):
    return 1.6*u**2*(1-u)  # a * u^2(1-u) with fitted a

def BirthRateSimple(u):
    return 7.71*u**3*(1-u)**2 # a*u^3(1-u)^2 with fitted a

# def FishGeneratedExp(u):
#     return 2296*u*(1-u)*np.exp(-7.8*(u-0.64)**2)  # u(1-u)exp(-((u - μ)/σ)^2) times normalization constant

BirthRateSpline = spi.PchipInterpolator(FxData,FyData)


## Plots
u = np.linspace(0, 1, 1001)
uper = u
# plt.plot(u*100, FishGeneratedInterp(u), label="Linear interpolation")
plt.plot(uper, BirthRateSpline(u), label="Spline interpolation", c="C0")
plt.plot(uper,BirthRateSimple(u), label="$7.71*x^3(1-x)^2$", c="C1")
plt.plot(uper,BirthRateSugested(u), label="$1.6*x^2(1-x)$",c="C2")
plt.xlabel("Fish Population [Normalized Fish]")
plt.ylabel("Fish birthrate [Normalized Fish per Year]")
plt.legend()
plt.grid()
plt.show(block=False)

## 
ShipyData = np.array([0.0, 10, 16, 20, 22, 23, 25])/uMax
ShipxData = np.array([0.0, 11, 22, 33, 44, 55,100])/100

def ShipEffInterp(u):
    """ Returns ship effectiveness (fish fished per ship per year) given factor of fish """
    # Taken from Figure 1, roughly
    return np.interp(u, ShipxData, ShipyData)

ShipEffSpline = spi.PchipInterpolator(ShipxData,ShipyData)


def ShipEffSugested(u):
    return 15.6e-3*u/(u+0.203)  # parameters fitted

u = np.linspace(0, 1, 1001)
uper = u
plt.figure()
# plt.plot(u*100, ShipEffInterp(u),label="Linear interpolation")
plt.plot(uper, ShipEffSpline(u),label="Spline interpolation", c="C0")
plt.plot(uper,ShipEffSugested(u), label="$0.0156*x/(0.203+x)$", c="C1")
plt.xlabel("Fish Population [Normalized Fish]")
plt.ylabel("Ship Effectiveness [Normalized Fish per Year per Ship]")
plt.legend()
plt.grid()
plt.show(block=False)

## Choose a model for the rest       *******************************************
chosenBR = BirthRateSimple
chosenSE = ShipEffSugested



## See some over time evolutions
def constYsim(u0, tf, y):
    # Set up dynamic model for constant number of ships y
    # du =  fgen(u) - ShipEffectiveness(u)*y
    def du(_,u):  # Must receive t, u for solve_ivp
        return chosenBR(u) - chosenSE(u)*y
    tf=120
    time_span = [0, tf]
    t_eval=np.linspace(0, tf, 1001)
    sol = solve_ivp(du, time_span, [u0],t_eval=t_eval)
    t = sol.t
    u = sol.y[0,:]
    return t, u
plt.figure()
ys = [1, 5, 10, 15, 20, 21,22,23, 24, 25, 30]
for y in ys:
    y = int(np.round(y))
    t, u = constYsim(0.9, 30, y)
    plt.plot(t, u, label=f"{y:d} ships")
plt.xlabel("t [Years]")
plt.ylabel("Fish population [Normalized Fish]")
plt.ylim([0, 1])
plt.grid()
plt.legend()
plt.show(block=False)

## Calculate equilibrium points
def du(u, y):  # Must receive t, u for solve_ivp
        return chosenBR(u) - chosenSE(u)*y
        
u = np.linspace(-1e-2, 1.01, 1001)
y = np.linspace(1,30, 101)
Y, U = np.meshgrid(y,u)
Z = [du(u,y) for u,y in zip(U.ravel(), Y.ravel())]
Z = np.array(Z).reshape(Y.shape)
plt.figure()
cs = plt.contourf(Y, U, Z, levels=0,
    colors=['#ff8080', '#08ff08'], extend='both')   
cs = plt.contour(Y, U, Z, levels=0, colors='black', linewidths=5) 
plt.scatter([],[], label="Growing population", c='#08ff08')       
plt.scatter([],[], label="Shrinking population", c='#ff8080') 
plt.plot([],[], c='black', label="Equilibrium")
plt.legend()      
plt.xlabel("Ships")
plt.ylabel("Fish population [Normalized Fish]")
plt.title("Equilibrium points")
plt.ylim([0, 1])
plt.grid()
plt.legend()
plt.show(block=False)



input("Press ENTER to end")