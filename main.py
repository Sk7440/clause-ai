n = int(input("Students: "))
highest = 0
lowest = 100
for i in range(n):
    m = float(input("Marks: "))
    if m > highest:
        highest = m
    if m < lowest:
        lowest = m
print("Highest:", highest, "Lowest:", lowest)
