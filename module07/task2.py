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





# # --- Modello logistico monotono ---
# def logistic_pitch(ws, a, b, x0, k):
#     # min ~ a ; max ~ a+b ; x0 = punto di flesso; k = pendenza
#     return a + b / (1.0 + np.exp(-k * (ws - x0)))

# # Dati puliti e ordinati (se non già fatto)
# ws_arr   = np.asarray(ws_range, dtype=float)
# pitch_arr= np.asarray(pitch, dtype=float)
# mask = np.isfinite(ws_arr) & np.isfinite(pitch_arr)
# ws_arr, pitch_arr = ws_arr[mask], pitch_arr[mask]
# ord_idx = np.argsort(ws_arr)
# ws_arr, pitch_arr = ws_arr[ord_idx], pitch_arr[ord_idx]

# ws_min, ws_max = float(ws_arr.min()), float(ws_arr.max())
# p_min,  p_max  = float(pitch_arr.min()), float(pitch_arr.max())

# # Guess iniziali robusti
# a0   = p_min
# b0   = max(1e-3, p_max - p_min)
# # x0 ≈ ws a cui il pitch è a metà (a + b/2). Stima veloce:
# mid_pitch = a0 + 0.5 * b0
# x0_approx = np.interp(mid_pitch, pitch_arr, ws_arr) if b0 > 0 else ws_arr.mean()
# k0   = 0.5

# p0 = [a0, b0, x0_approx, k0]
# bounds = (
#     [p_min - 20,      1e-5,   ws_min, 1e-4],               # lower
#     [p_min + 20 + b0, 10*b0,  ws_max, 10.0]                # upper
# )

# # Fit
# popt, pcov = curve_fit(logistic_pitch, ws_arr, pitch_arr, p0=p0, bounds=bounds, maxfev=20000)
# a_hat, b_hat, x0_hat, k_hat = popt
# print("curve_fit params:", dict(a=a_hat, b=b_hat, x0=x0_hat, k=k_hat))

# # Curva fitted sul mapping e confronto con lineare
# ws_fit = np.linspace(ws_min, ws_max, 400)
# pitch_fit_curve = logistic_pitch(ws_fit, *popt)

# fig_cf, ax_cf = plt.subplots()
# ax_cf.plot(ws_arr, pitch_arr, "o", label="dati originali", alpha=0.9)
# ax_cf.plot(ws_fit, pitch_fit_curve, lw=2, label="curve_fit (logistica)")
# ax_cf.plot(ws_fit, f_linear(ws_fit), lw=1.5, label="interp lineare")
# ax_cf.set_xlabel("w/s (m/s)")
# ax_cf.set_ylabel("pitch (deg)")
# ax_cf.set_title("Pitch vs Wind speed — curve_fit vs interp")
# ax_cf.grid(True, alpha=0.6)
# ax_cf.legend()
# fig_cf.tight_layout()
# fig_cf.savefig("pitch_vs_ws_curvefit_vs_linear.png", dpi=200, bbox_inches="tight")
# plt.show()

# # === Serie temporale attesa del pitch dai dati filtrati ===
# # NB: clippiamo ws_filtered al range di fit per evitare extrapolazioni instabili
# ws_f_clip = np.clip(ws_filtered, ws_min, ws_max)
# pitch_expected_cf = logistic_pitch(ws_f_clip, *popt)   # da curve_fit
# pitch_expected_li = f_linear(ws_f_clip)                # confronto: interp lineare

# fig_ts, ax_ts = plt.subplots()
# ax_ts.plot(time, pitch_expected_cf, label="pitch (curve_fit logistica)", lw=1.8)
# ax_ts.plot(time, pitch_expected_li, label="pitch (interp lineare)", lw=1.0, alpha=0.8)
# ax_ts.set_xlabel("time (s)")
# ax_ts.set_ylabel("pitch (deg)")
# ax_ts.set_title("Expected blade pitch time series — curve_fit vs interp")
# ax_ts.grid(True, alpha=0.6)
# ax_ts.legend()
# fig_ts.tight_layout()
# fig_ts.savefig("pitch_expected_time_series_curvefit_vs_linear.png", dpi=200, bbox_inches="tight")
# plt.show()

# # (opzionale) metriche rapide di aderenza sul mapping
# res_cf = pitch_arr - logistic_pitch(ws_arr, *popt)
# res_li = pitch_arr - f_linear(ws_arr)
# rmse_cf = np.sqrt(np.mean(res_cf**2))
# rmse_li = np.sqrt(np.mean(res_li**2))
# print(f"RMSE curve_fit: {rmse_cf:.4f}  |  RMSE lineare: {rmse_li:.4f}")
