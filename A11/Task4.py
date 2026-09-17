def get_input():
    hourly_rate = int(input(str("Employees hourly rate is: ")))
    hours_worked = int(input(str("Employees hours worked are: ")))
    bonus = int(input(str("Employees Bonus is: ")))
    return hourly_rate,hours_worked,bonus


def calculate_salary(hourly_rate,hours_worked,bonus):
    salary = (hourly_rate * hours_worked) + bonus
    return salary

hourly_rate, hours_worked, bonus = get_input()
wage = calculate_salary(hourly_rate,hours_worked,bonus)
print ("Total monthly income is: " , wage) 



