import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
fileName = "Module 6 - Exercises Data.xlsx"
df_baseline = pd.read_excel(fileName, sheet_name="Exercise 4 - Baseline")
df_A = pd.read_excel(fileName, sheet_name="Exercise 4 - A")
df_B = pd.read_excel(fileName, sheet_name="Exercise 4 - B")

time = df_baseline["Time (s)"]
rpm_baseline = df_baseline["Rotor speed (rpm)"]
rpm_A = df_A["Rotor speed (rpm)"]
rpm_B = df_B["Rotor speed (rpm)"]

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
