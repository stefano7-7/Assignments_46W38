import scipy
from scipy.optimize import Bounds, minimize, LinearConstraint, NonlinearConstraint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import weibull_min

"""
AEP from weibull fit of annual w/s of Task3.py
"""

# read & plot the w/s 
fileName = "Module 7 - Exercises data.xlsx"
df = pd.read_excel(fileName, sheet_name = "Exercise 3")
print(df.head()) 
datetime = df["Date/Time"]
ws_annual = df["Wind Speed (m/s)"].dropna()  # remove NaN

# freq distribution
fig, ax = plt.subplots()
ax.hist(ws_annual, bins=30,density=True, color="blue", edgecolor="black")

# # weibull fit
shape, loc, scale = weibull_min.fit(ws_annual, floc =0)
print("shape k =", shape)
print("scale A =", scale)
ws_weibull = np.linspace(0, ws_annual.max(), 50)
pdf = weibull_min.pdf(ws_weibull, shape, loc,scale)

ax.plot(ws_weibull, pdf, color="red", lw=2, label=f"Weibull fit\nk={shape:.2f}, A={scale:.2f}")
ax.legend()
ax.set_xlabel("w/s (m/s)")
ax.set_title("w/s freq distribution_weibull_fit")
ax.grid(True, alpha = 0.6)
plt.tight_layout()
fig.savefig("ws_freq_distr_weibull.png", dpi = 200, bbox_inches = "tight")

rho = 1.225
hours_year = 8760
Prated_target = 10e6
Vin = 3
Vout = 25
Cp = 0.45
HH = 100

def Prated(rho, D, Cp, Vrated):
    return 0.5*rho*np.pi*D**2/4*Vrated**3

def power_curve(ws_weibull, Vrated, Vin, Vout, Prated):
    """Curva di potenza vettoriale P(V)."""
    ws_weibull = np.asarray(ws_weibull)
    P = np.zeros_like(ws_weibull)
    # sub-rated
    mask_sub = (ws_weibull >= Vin) & (ws_weibull < Vrated)
    P[mask_sub] = Prated * ((ws_weibull[mask_sub] - Vin) / (Vrated - Vin))**3
    # rated
    mask_rated = (ws_weibull >= Vrated) & (ws_weibull <= Vout)
    P[mask_rated] = Prated
    # fuori range = 0
    return P

def AEP(D, Vrated):
    Pr = min(Prated(rho, D, Cp, Vrated), Prated_target)  # cap a 10 MW
    P = power_curve(ws_weibull, Vrated, Vin, Vout, Pr)
    return hours_year * np.trapz(P * pdf, ws_weibull)

# goal: to minimize -AEP, i.e. maximize AEP
def cost(x):
    D, Vrated = x
    return -AEP(D, Vrated)

# 0.5*d + 0 * Vrated <= 0.8*HH
lin_constr = LinearConstraint([[0.5, 0.0]], -np.inf, 0.8 * HH)

# constraint Prated - Prated_target <= 0
def nonlin_func(x):
    D, Vrated = x
    return (Prated(rho, D, Cp, Vrated) - Prated_target)
nonlin_constr = NonlinearConstraint(nonlin_func, -np.inf, 0.0)
bounds = Bounds([70, Vin+0.1], [400, Vout-0.1])

init_guess = np.array([100, 9]) # D, Vrated
res_global = minimize(cost, init_guess, method = "trust-constr",
                      bounds = bounds, constraints = [lin_constr, nonlin_constr], options={"verbose": 1})
D_opt, Vr_opt = res_global.x
Pr_opt = Prated(rho, D_opt, Cp, Vr_opt)
AEP_opt_Wh = AEP(D_opt, Vr_opt)
AEP_opt_GWh = AEP_opt_Wh / 1e9

# print results
print(f"Optimal D = {D_opt:.2f} m")
print(f"Optimal Vrated = {Vr_opt:.2f} m/s")
print(f"Prated(D,Vr) = {Pr_opt/1e6:.3f} MW (<= 10 MW)")
print(f"AEP ≈ {AEP_opt_GWh:.3f} GWh/year")

# power curve optimize
import matplotlib.pyplot as plt

# calcolo della curva ottimizzata
Pr_opt = min(Prated(rho, D_opt, Cp, Vr_opt), Prated_target)
P_opt_curve = power_curve(ws_weibull, Vr_opt, Vin, Vout, Pr_opt)

fig, ax1 = plt.subplots(figsize=(7,4))
# left y-axis: power
ax1.plot(ws_weibull, P_opt_curve/1e6, 'r-', lw=2, label="Power curve (MW)")
ax1.set_xlabel("Wind speed (m/s)")
ax1.set_ylabel("Power (MW)", color="r")
ax1.tick_params(axis="y", labelcolor="r")
ax1.grid(True, alpha=0.5)
# right y-axis: w/s distrib
ax2 = ax1.twinx()
ax2.plot(ws_weibull, pdf, 'b--', lw=2, label="Weibull PDF")
ax2.set_ylabel("w/s weibull", color="b")
ax2.tick_params(axis="y", labelcolor="b")

fig.suptitle(f"Opt power curve D_opt={D_opt:.1f} m, Vrated_opt={Vr_opt:.1f} m/s")
fig.tight_layout()
fig.savefig("power_curve_optimized.png", dpi=200, bbox_inches="tight")
plt.show()
