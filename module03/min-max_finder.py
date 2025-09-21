"""
This script ask User to input a series of numbers or input "end" to stop
2 blocks:
    1) first input is asked and used as initial value for number_min and number_max 
    2) loop checking whether the last input is higher or lower than the previous ones
Note: inputs are string and are changed into numbers
"""
number_string = input("write the first number (or type END to finish): ")
if number_string.lower() == "end":
    print("No numbers entered")
    exit()  # termina subito il programma
number_min = int(number_string)
number_max = number_min
list_numbers = [number_min]

while number_string != "end":
    number_string = input("write another number (or type END to finish): ")
    if number_string.lower() == "end":
        break
    number_int = int(number_string)
    list_numbers.append(number_int)
    if number_int < number_min: 
        number_min = number_int
    elif number_int > number_max:
        number_max = number_int

print(f"minimum is: {number_min}")
print(f"maximum is: {number_max}")
print(f"list of the numbers is: {list_numbers}")

