"""test program"""

# ========= pylint my_script.py===============
# # my_script.py:1:0: C0114: Missing module
# docstring (missing-module-docstring)
# my_script.py:5:0: C0116: Missing function or
# method docstring (missing-function-docstring)
# my_script.py:5:0: C0103: Function name "Calculate_Power"
# doesn't conform to snake_case naming style (invalid-name)
# my_script.py:13:0: C0116: Missing function or
# method docstring (missing-function-docstring)
# my_script.py:14:4: R1705: Unnecessary "elif" after "return",
#  remove the leading "el" from "elif" (no-else-return)
# my_script.py:23:0: C0103: Constant name "status"
# doesn't conform to UPPER_CASE naming style (invalid-name)
# my_script.py:2:0: W0611: Unused import sys (unused-import)
# -----------------------------------
# Your code has been rated at 5.88/10

# ==========flake8 my_script.py==============
# my_script.py:2:1: F401 'sys' imported but unused


import math


def calculate_power(windspeed, rotor_diameter, air_density=1.225):
    """calculate power"""
    if windspeed < 0:
        return 0
    area = math.pi * (rotor_diameter / 2) ** 2
    power = 0.5 * air_density * area * (windspeed**3)
    return power


def check_turbine_status(wind_speed, cut_in=3, cut_out=25):
    """determine status"""
    if wind_speed < cut_in:
        return "stopped"
    if wind_speed > cut_out:
        return "braked"
    return "operating"


# result is in the whitelist for names on whitelist
result = calculate_power(15.5, 120)
# pylint: disable=invalid-name
status = check_turbine_status(15.5)
# pylint: enable=invalid-name
print(f"Power output: {result} W, Status: {status}")
