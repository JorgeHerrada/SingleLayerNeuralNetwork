# use it to generate you own test data

# Define range
min = -15
max = 15

# output file
fileName = "testData.csv"

# Open file in write mode
with open(fileName, "w") as file:
    # generate all combinations
    for x in range(min, max):
        for y in range(min, max):
            # write to file
            file.write(f"{x},{y}\n")

# close file
file.close()
