import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

"""
read 3 timeseries from 3 WT with 3 different controllers
plot rpm time series
calculate std deviations and plot them on grouped bar chart to compare performances
"""

# Read data from 3 sheets from same Excel file
fileName = "Module 6 - Exercises Data.xlsx"
df_baseline = pd.read_excel(fileName, sheet_name="Exercise 4 - Baseline")
df_A = pd.read_excel(fileName, sheet_name="Exercise 4 - A")
df_B = pd.read_excel(fileName, sheet_name="Exercise 4 - B")

time = df_baseline["Time (s)"]
rpm_baseline = df_baseline["Rotor speed (rpm)"]
rpm_A = df_A["Rotor speed (rpm)"]
rpm_B = df_B["Rotor speed (rpm)"]

# %%
# plot time histories of 3 WT superposed
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(time, rpm_baseline, label = "baseline", color= "blue")
ax.plot(time, rpm_A, label = "A", color= "green")
ax.plot(time, rpm_B, label = "B", color= "black")
ax.set_title("Baseline vs A vs B controllers impact on rpm")
ax.legend()
ax.grid(True, alpha = 0.6)
fig.tight_layout()
plt.show()

fig.savefig("exercise4_rpm_comparison.png", dpi = 200, bbox_inches = "tight")

# %%
# standard deviation
#normalized
rpm_normalized_A = rpm_A.std() / rpm_baseline.std()
rpm_normalized_B = rpm_B.std() / rpm_baseline.std()

tw_base_moment_baseline = df_baseline["Tower base moment (kNm)"]
tw_base_moment_A = df_A["Tower base moment (kNm)"]
tw_base_moment_B = df_B["Tower base moment (kNm)"]
tw_base_moment_normalized_A = tw_base_moment_A.std() / tw_base_moment_baseline.std()
tw_base_moment_normalized_B = tw_base_moment_B.std() / tw_base_moment_baseline.std()

thrust_baseline = df_baseline["Thrust (kN)"]
thrust_A = df_A["Thrust (kN)"]
thrust_B = df_B["Thrust (kN)"]
thrust_normalized_A = thrust_A.std() / thrust_baseline.std()
thrust_normalized_B = thrust_B.std() / thrust_baseline.std()


categories = ["baseline", "controller A", "controller B"]
# rpm_std = [1.0, rpm_normalized_A, rpm_normalized_B]
# thrust_std = [1.0, thrust_normalized_A, thrust_normalized_B]
# tw_base_moment_std = [1.0, tw_base_moment_normalized_A, tw_base_moment_normalized_B]

base_normalized = [1.0, 1.0, 1.0]
A_normalized = [rpm_normalized_A, thrust_normalized_A, tw_base_moment_normalized_A]
B_normalized = [rpm_normalized_B, thrust_normalized_B, tw_base_moment_normalized_B]

fig, ax = plt.subplots()

x = np.arange(len(categories))   
bar_width = 0.25 

bars_base = ax.bar(x - bar_width, base_normalized, width=bar_width, color="blue", label="Baseline (1.0)")
bars_A = ax.bar(x, A_normalized, width=bar_width, color="green", label="Controller A")
bars_B = ax.bar(x + bar_width, B_normalized, width=bar_width, color="orange", label="Controller B")

ax.set_xticks(x)
ax.set_xticklabels(categories)

ax.set_ylabel("Normalized standard deviation (Baseline = 1.0)")
ax.set_title("Normalized STD comparison – Baseline, A, B")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.6)

# values over bars
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f"{height:.2f}",
            ha="center", va="bottom", fontsize=9
        )

annotate_bars(bars_A)
annotate_bars(bars_B)

# --- Limiti Y e layout ---
ax.set_ylim(0, max(max(A_normalized), max(B_normalized), 1.0) * 1.25)
fig.tight_layout()

# --- Salva e mostra ---
fig.savefig("exercise4_normalized_STD_grouped.png", dpi=200, bbox_inches="tight")
plt.show()
# %%
