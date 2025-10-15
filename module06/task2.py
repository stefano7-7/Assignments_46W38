import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_excel("Module 6 - Exercises Data.xlsx", sheet_name="Exercise 2")

ws      = df["Wind speed (m/s)"]
power   = df["Power (kW)"]
thrust  = df["Thrust (kN)"]
rpm     = df["Rotor speed (rpm)"]
pitch   = df["Blade pitch (degrees)"]

# --- Colors  ---
c_power  = "#1f77b4"  # blue
c_thrust = "red" # red
c_rpm    = "green"  # green
c_pitch  = "purple"

# ---------- FIGURE 2: one figure, two subplots, each with twin axes ----------
fig, (axA, axB) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Subplot A: Power (left) + Thrust (right)
axA2 = axA.twinx()
lA1, = axA.plot(ws, power,  marker='o', linewidth=2, color=c_power,  label="Power (kW)")
lA2, = axA2.plot(ws, thrust, marker='s', linewidth=2, color=c_thrust, label="Thrust (kN)")

axA.set_ylabel("Power (kW)",  color=c_power)
axA2.set_ylabel("Thrust (kN)", color=c_thrust)
axA.tick_params(axis='y', labelcolor=c_power)
axA2.tick_params(axis='y', labelcolor=c_thrust)
axA.set_title("A) Power vs Thrust")
axA.grid(True, linestyle="--", alpha=0.6)

# Combined legend for subplot A
linesA = [lA1, lA2]
labelsA = [ln.get_label() for ln in linesA]
axA.legend(linesA, labelsA, loc="best", frameon=False)

# Subplot B: RPM (left) + Pitch (right)
axB2 = axB.twinx()
lB1, = axB.plot(ws, rpm,   marker='^', linewidth=2, color=c_rpm,   label="Rotor speed (rpm)")
lB2, = axB2.plot(ws, pitch, marker='d', linewidth=2, color=c_pitch, label="Blade pitch (degrees)")

axB.set_xlabel("Wind speed [m/s]")
axB.set_ylabel("Rotor speed (rpm)",    color=c_rpm)
axB2.set_ylabel("Blade pitch (degrees)", color=c_pitch)
axB.tick_params(axis='y', labelcolor=c_rpm)
axB2.tick_params(axis='y', labelcolor=c_pitch)
axB.set_title("B) RPM vs Pitch")
axB.grid(True, linestyle="--", alpha=0.6)

# Combined legend for subplot B
linesB = [lB1, lB2]
labelsB = [ln.get_label() for ln in linesB]
axB.legend(linesB, labelsB, loc="best", frameon=False)

fig.tight_layout()
fig.savefig("exercise2_subplots.png", dpi=200, bbox_inches="tight")

plt.show()
