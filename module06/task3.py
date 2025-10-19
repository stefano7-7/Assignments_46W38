import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('Module 6 - Exercises Data.xlsx', 
                   sheet_name="Exercise 3", index_col = 0)

TSR = df.index.values.astype(float) 
Pitch = df.columns.values.astype(float)
Cp = df.values.astype(float)

Pitch_2D, TSR_2D = np.meshgrid(Pitch, TSR)

fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')

Cp = ax.plot_surface(TSR_2D, Pitch_2D, Cp, cmap='viridis', edgecolor='none')

ax.set_xlabel("Blade Pitch Angle θ (degrees)")
ax.set_ylabel("Tip Speed Ratio λ")
ax.set_zlabel("$C_p$")
ax.set_title("3D Surface of $C_p$ vs Pitch and TSR")

plt.tight_layout()
plt.show()

fig.savefig("exercise_3_3Dplot.png",dpi=200, bbox_inches="tight")

