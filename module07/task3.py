import scipy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import filtfilt, butter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import weibull_min

"""
weibull fit of annual w/s
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
plt.show()
fig.savefig("ws_freq_distr_weibull.png", dpi = 200, bbox_inches = "tight")

