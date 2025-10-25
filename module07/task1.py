import scipy
from scipy.interpolate import interp1d
from scipy import fftpack
from scipy.signal import welch, iirnotch, filtfilt, butter
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

"""
FFT e PSD of w/s signal
applying notch filter around the freq of max amplitude
"""

# read & plot the w/s 
fileName = "Module 7 - Exercises data.xlsx"
df = pd.read_excel(fileName, sheet_name = "Exercise 1")
print(df.head()) 
time = df["Time (s)"]
ws = df["Wind speed (m/s)"]

fig, ax = plt.subplots()
ax.plot(time,ws, label ="w/s")
ax.set_xlabel("time (s)")
ax.set_ylabel("w/s (m/s)")
ax.set_title("w/s time history")
ax.legend()
ax.grid(True, alpha = 0.6)
fig.tight_layout()
plt.show()

fig.savefig("ws_time_series.png", dpi = 200, bbox_inches = "tight")

# FFT
dt = np.mean(np.diff(time))  # seconds between samples
fs = 1 / dt       
ws_signal = np.array(ws)

spectrum = fftpack.fft(ws_signal)
freqs = fftpack.fftfreq(len(ws_signal), d = dt)
freqs_positive = freqs[freqs > 0]
freqs_positive_log = np.log(freqs_positive)
spectrum_valid = np.abs(spectrum[freqs > 0])

# notch filter
idx_peak = np.argmax(spectrum_valid)
f_peak = freqs_positive[idx_peak] # Frequency to be removed from signal (Hz)
print(f"removed peak freq: {f_peak:.2f} Hz")
Q = 5.0  # Quality factor
b, a = iirnotch(f_peak, Q, fs=fs)
ws_notch = filtfilt(b, a , ws_signal)

spectrum_notch = fftpack.fft(ws_notch)
freqs_notch = fftpack.fftfreq(len(ws_notch), d = dt)
freqs_positive_notch = freqs_notch[freqs_notch > 0]
freqs_positive_log_notch = np.log(freqs_positive_notch)
spectrum_valid_notch = np.abs(spectrum_notch[freqs_notch > 0])

# FFT plot
fig, ax = plt.subplots()
ax.plot(freqs_positive,spectrum_valid, label="original")
ax.plot(freqs_positive_notch,spectrum_valid_notch, label="notched")
ax.set_xlabel("freq (Hz)")
ax.set_ylabel("amplitude")
ax.set_title("w/s FFT spectrum")
ax.grid(True, alpha = 0.6)
ax.legend()
fig.tight_layout()
plt.show()
fig.savefig("ws_notched_FFT.png", dpi = 200, bbox_inches = "tight")

#FFT plot logarithmic freq x-axis
fig, ax = plt.subplots()
ax.plot(freqs_positive,spectrum_valid, label="original")
ax.plot(freqs_positive_notch,spectrum_valid_notch, label="notched")
ax.set_xscale("log")
ax.set_xlabel("freq (Hz)")
ax.set_ylabel("amplitude")
ax.set_title("w/s FFT spectrum")
ax.grid(True, alpha = 0.6)
fig.tight_layout()
plt.show()
fig.savefig("ws_notched_FFT_log_scale.png", dpi = 200, bbox_inches = "tight")

# PFD
s, Sx = welch(ws_notch, fs=fs, nperseg=fs)

# PFD plot logarithmic scale for freqs
fig, ax = plt.subplots()
ax.plot(s,Sx)
# ax.set_xscale("log")
ax.set_xlabel("freq (Hz)")
ax.set_ylabel("v2/Hz")
ax.set_title("w/s PSD spectrum")
ax.grid(True, alpha = 0.6)
fig.tight_layout()
plt.show()
fig.savefig("ws_PSD.png", dpi = 200, bbox_inches = "tight")

# PFD plot logarithmic scale for freqs
fig, ax = plt.subplots()
ax.plot(s,Sx)
ax.set_xscale("log")
ax.set_xlabel("freq (Hz)")
ax.set_ylabel("v2/Hz")
ax.set_title("w/s PSD spectrum")
ax.grid(True, alpha = 0.6)
fig.tight_layout()
plt.show()
fig.savefig("ws_PSD_log_scale.png", dpi = 200, bbox_inches = "tight")