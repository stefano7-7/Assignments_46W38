import pandas as pd
import matplotlib.pyplot as plt

# read file
df = pd.read_excel("Module 6 - Exercises Data.xlsx", sheet_name="Exercise 2")

# print headers
print(df.head())

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Wind speed (m/s)"], df["Power (kW)"],  marker='o', linewidth=2, label="Power (kW)")
ax.plot(df["Wind speed (m/s)"], df["Thrust (kN)"], marker='s', linewidth=2, label="Thrust (kN)")
ax.plot(df["Wind speed (m/s)"], df["Rotor speed (rpm)"],  marker='^', linewidth=2, label="Pitch angle (°)")
ax.plot(df["Wind speed (m/s)"], df["Blade pitch (degrees)"], marker='d', linewidth=2, label="RPM")

ax.set_xlabel("Wind speed [m/s]")
ax.set_title("WT metrics vs. Wind Speed")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()
plt.savefig('exercise2.png',format = 'png', dpi = 200, bbox_inches = 'tight')