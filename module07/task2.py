import scipy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import filtfilt, butter
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

"""
pitch...
"""

# read & plot the w/s 
fileName = "Module 7 - Exercises data.xlsx"
df = pd.read_excel(fileName, sheet_name = "Exercise 2")
print(df.head()) 
ws_range = df["Wind speed (m/s)"]
pitch = df["Blade pitch (degrees)"]

# interpolate & plot
ws_interp_base = np.linspace(min(ws_range),max(ws_range),100)
f_linear = interp1d(ws_range, pitch, kind="linear")
f_cubic = interp1d(ws_range, pitch, kind="cubic")

fig, ax = plt.subplots()
ax.plot(ws_range,pitch,label="original")
ax.plot(ws_interp_base,f_linear(ws_interp_base),label="linear interp")
ax.plot(ws_interp_base,f_cubic(ws_interp_base),label="cubic interp")

ax.set_xlabel("w/s (m/s)")
ax.set_ylabel("pitch (deg)")
ax.set_title("pitch angle by w/s")
ax.legend()
ax.grid(True, alpha = 0.6)
fig.tight_layout()
#plt.show()

fig.savefig("pitch_vs_ws.png", dpi = 200, bbox_inches = "tight")

# function to get filtered w/s time history
def butter_lowpass_filter(data, f_cutoff,fs, order=4, btype = "low"):
    """
    Zero-phase Butterworth low-pass via filtfilt.
    """
    nyq = 0.5 * fs
    normal_cutoff = np.array(f_cutoff)/ nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return filtfilt(b, a, data)

def filt_ws_history(fileName, sheetName="Exercise 1"):
    """
    filter actual data
    """
    df_ws_history = pd.read_excel(fileName, sheet_name = sheetName)
    time = df_ws_history["Time (s)"]
    ws = df_ws_history["Wind speed (m/s)"]
    
    dt = np.mean(np.diff(time))  # seconds between samples
    fs = 1 / dt       
    ws_signal = np.array(ws)

    f_cut = 0.05
    ws_filtered = butter_lowpass_filter(ws_signal, f_cut, fs, btype="low")
    return time, ws_filtered

# curve fitting
time, ws_filtered = filt_ws_history(fileName, "Exercise 1")
pitch_exp_linear = f_linear(ws_filtered)

b_poly, a_poly = np.polyfit(ws_range, pitch, deg=1)
pitch_exp_polyfit_linear = b_poly * ws_filtered+ a_poly

def linear_model(x, m, c):
    return m*x+c
popt, pcov = curve_fit(linear_model, ws_range, pitch)
y_curve_fit = linear_model(ws_filtered, *popt)

fig_cf, ax_cf = plt.subplots()
ax_cf.plot(ws_filtered, f_linear(ws_filtered), lw=0.5, label="linear interp", alpha=0.9)
ax_cf.plot(ws_filtered, pitch_exp_polyfit_linear , lw=3, label="polyfit linear")
ax_cf.plot(ws_filtered, y_curve_fit, lw=1.5, label="curve_fit linear")
ax_cf.set_xlabel("w/s (m/s)")
ax_cf.set_ylabel("pitch (deg)")
ax_cf.set_title("Pitch vs Wind speed — curve_fit vs interp")
ax_cf.grid(True, alpha=0.6)
ax_cf.legend()
fig_cf.tight_layout()
fig_cf.savefig("pitch_with_interp1d_vs_polyfit_curve_fit.png", dpi=200, bbox_inches="tight")
plt.show()