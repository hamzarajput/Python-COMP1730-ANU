new_file = open ('teninput.txt', 'a')

for i in range(1 , 11):
    s = int(input(str("Input value is: ")))
    print (s)
    new_file.write(str(s))

new_file.write("\n")

new_file.close()
