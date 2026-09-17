marks = []

for i in range (2000 , 3200):    

    if i % 7 == 0 and i % 5 != 0:
        marks.append(i)
    else:
        print("Numbers don't fall under this criteria")

print(marks)

