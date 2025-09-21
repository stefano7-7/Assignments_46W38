"""
This script ask User to input a series of numbers or input "end" to stop
2 blocks:
    1) first input is asked and used as initial value for number_min and number_max 
    2) loop checking whether the last input is higher or lower than the previous ones
    3) printing results
Note: inputs are string and are changed into numbers
"""
number_string = input("write the first number (or type END to finish): ")
if number_string.lower() == "end": #handle both END and end
    print(f"minimum is: {None}")
    print(f"maximum is: {None}")
    exit()  # handle the case of END as first input

# inizialization of min and max and the list of inputs
number_min = int(number_string)
number_max = number_min
list_numbers = [number_min]

# loop asking for inputs
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

# printing results
print(f"minimum is: {number_min}")
print(f"maximum is: {number_max}")
print(f"list of the numbers is: {list_numbers}")

# input sorting order
order = input('do you want to sort in ascending order? write YES or NO: ')
ascending = order.lower() == "yes"

# sorting the list of numbers
if ascending:
    sorted_list = [number_min]
    temporary_min = number_min
    while True:
        temporary_list = [n for n in list_numbers if n > temporary_min]
        if temporary_list:
            temporary_min = min(temporary_list)
            sorted_list.append(temporary_min)
        else:
            break   # exit from loop if there are no more elements
else:
    sorted_list = [number_max]
    temporary_max = number_max
    while True:
        temporary_list = [n for n in list_numbers if n < temporary_max]
        if temporary_list:
            temporary_max = max(temporary_list)
            sorted_list.append(temporary_max)
        else:
            break   # exit from loop if there are no more elements
print(sorted_list)
