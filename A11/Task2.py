n = int(input("Number to test is: "))
divisibility = []
for i in range (1,n):
    if n % i == 0:
        divisibility.append(i)
if sum(divisibility) == n:
    print(f"{n} is a perfect number")
else:
    print(f"{n} is not a perfect number")