number_min = int(input("write the first number (or type END to finish): "))
number_max = number_min
list_numbers = [number_min]
number_string = None 
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
print(f"list of the numbers is {list_numbers}")
