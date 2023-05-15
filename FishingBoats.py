import numpy as np
import scipy.interpolate as spi
from scipy.integrate import solve_ivp
import scipy.optimize as sio
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


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
uper = u*100
# plt.plot(u*100, FishGeneratedInterp(u), label="Linear interpolation")
plt.plot(uper, BirthRateSpline(u)*uMax, label="PCHIP interpolation", c="C0")
plt.plot(uper,BirthRateSimple(u)*uMax, label="$15420*x^3(1-x)^2$", c="C1")
plt.plot(uper,BirthRateSugested(u)*uMax, label="$3200*x^2(1-x)$",c="C2")
plt.xlabel("Percent of Maximum Fish Population")
plt.ylabel("Population Change Rate [Fish per Year]")
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
uper = u*100
plt.figure()
# plt.plot(u*100, ShipEffInterp(u),label="Linear interpolation")
plt.plot(uper, ShipEffSpline(u)*uMax,label="PCHIP interpolation", c="C0")
plt.plot(uper,ShipEffSugested(u)*uMax, label="$31.2*x/(0.203+x)$", c="C1")
plt.xlabel("Percent of Maximum Fish Population")
plt.ylabel("Ship Effectiveness [Fish per Year per Ship]")
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
ys = [1, 5, 10, 22,23, 30]
for y in ys:
    y = int(np.round(y))
    t, u = constYsim(0.9, 30, y)
    plt.plot(t, 100*u, label=f"{y:d} ships")
plt.xlabel("t [Years]")
plt.ylabel("Percent of Maximum Fish Population")
plt.ylim([0, 100])
plt.grid()
plt.legend()
plt.show(block=False)

## Calculate equilibrium points
def du(u, y):  # Must receive t, u for solve_ivp
        return chosenBR(u) - chosenSE(u)*y
        

ymax=30
step=1e-3
u = np.arange(-1e-2, 1.01+step, step)
step = 0.5
y = np.arange(0, ymax+step, step)
U, Y = np.meshgrid(u, y)
Z = [du(u,y) for u,y in zip(U.ravel(), Y.ravel())]
Z = np.array(Z).reshape(Y.shape)
plt.figure()
cs = plt.contourf(U*100, Y, Z, levels=0,
    colors=['#ff8080', '#08ff08'], extend='both')   
cs = plt.contour(U*100, Y, Z, levels=0, colors='black', linewidths=5) 
plt.scatter([],[], label="Growing population", c='#08ff08')       
plt.scatter([],[], label="Shrinking population", c='#ff8080') 
# Manually add equilibrium at 0
plt.plot([0, 0],[0, ymax], c='black', label="Equilibrium", linewidth=5)
plt.legend()      
plt.ylabel("Ships")
plt.xlabel("Percent of Maximum Fish Population")
plt.title("Equilibrium points")
plt.xlim([0, 100])
plt.ylim([0, ymax])
plt.grid()
plt.show(block=False)


## Calculate equilibrium points
def dx(u, y, k , c): 
        du = chosenBR(u) - chosenSE(u)*y
        dy = uMax*k*y*(chosenSE(u) - c/uMax)
        if y < -1e-9:
             raise ValueError('y is negative...')
        return np.array([du, dy])
        # else:
        #      return np.array([du, max(0, dy)])

k = 0.1
c = 23
Nu = 102*10+1
ymax=80
u = np.linspace(-1e-2, 1.01, Nu)
step = 0.5
y = np.arange(0, ymax+step, step)
U,Y = np.meshgrid(u,y)
Z = [tuple(dx(u,y, k, c)) for u,y in zip(U.ravel(), Y.ravel())]
du = [z[0] for z in Z]
dy = [z[1] for z in Z]
du = np.array(du).reshape(Y.shape)
dy = np.array(dy ).reshape(Y.shape)

plt.figure()
# cs = plt.contourf(Y, U, Z, levels=0,
#     colors=['#ff8080', '#08ff08'], extend='both')   
cs = plt.contour(U*100,Y, du*100, levels=0, colors='black', linewidths=5) 
sol = sio.root_scalar(lambda u: chosenSE(u)*uMax-c, bracket=[0, 1])
uc = sol.root
plt.plot([0, 100],[0,0], c='black', linestyle="--", linewidth=5)
plt.plot([uc*100, uc*100],[0,ymax], c='black', linestyle="--", linewidth=5)
# plt.scatter([],[], label="Growing population", c='#08ff08')       
# plt.scatter([],[], label="Shrinking population", c='#ff8080') 
## See some over time evolutions
def getTraj(u0, y0, tf):
    # Set up dynamic model for constant number of ships y
    # du =  fgen(u) - ShipEffectiveness(u)*y
    def dxForIVP(_,x):  # Must receive t, u for solve_ivp
        return dx(x[0], x[1], k, c)
    time_span = [0, tf]
    t_eval=np.linspace(0, tf, 3)
    sol = solve_ivp(dxForIVP, time_span, [u0, y0], atol=1e-12,rtol=1e-6, dense_output=True)
    if not sol.success:
        raise ValueError(sol.message)
    t = sol.t
    # u = sol.y[0,:]
    return t, sol.y


# tests = [(0.7, 15), (0.7, 37), (0.9, 30), (0.9, 20)]
tests = [(0.1, 10), (0.1, 60), (0.3, 25), (0.3, 50), (0.4, 15), (0.7, 15), (0.7, 55), (0.9, 30), (0.9, 60)]
for (u0, y0) in tests:
    t, x = getTraj(u0,y0,100)
    plt.plot(x[0,:]*100, x[1,:], c="C0")
    plt.scatter(x[0,0]*100, x[1,0], c="C0")
    print(t[-1])



plt.plot([],[], c='black', label="$\dot{x}=0$")
plt.plot([], [], c='black', linestyle="--",label="$\dot{y}=0$")
# plt.streamplot(U*100, Y, du*100, dy )
plt.legend()      
plt.ylabel("Ships")
plt.xlabel("Percent of Maximum Fish Population")
plt.title(f"Trajectory Analysis\n$k_y={k:.1f},\ c={c:d}$")
plt.xlim([0, 100])
plt.ylim([0, ymax])
plt.grid()
plt.legend()
plt.show(block=False)






input("Press ENTER to end")