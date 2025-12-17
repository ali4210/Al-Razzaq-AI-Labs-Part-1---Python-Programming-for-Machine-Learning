# Enhanced for loop with iteration tracking
numbers = [1, 2, 3, 4, 5]
for i, number in enumerate(numbers):
    print(f'Iteration {i + 1} - Number is: {number}')

# Enhanced while loop with iteration tracking
count = 0
while count < 5:
    print(f'Start of iteration {count + 1}')
    print('Count is:', count)
    print(f'End of iteration {count + 1}\n')
    count += 1