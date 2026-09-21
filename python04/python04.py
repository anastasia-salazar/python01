import numpy as np 
import numpy.ma as ma

# Assignment 1: Array Creation and Manipulation
# 1. Create a NumPy array of shape (5, 5) filled with random integers between 1 and 20. Replace all the elements in the third column with 1.
# 2. Create a NumPy array of shape (4, 4) with values from 1 to 16. Replace the diagonal elements with 0
array = np.random.randint(1, 21, size=(5, 5))

print("Array: ", array)
print("Column 2: ", array[:, 2])
array[:, 2] = 1
print("Column 2: ", array[:, 2])

array3 = np.arange(1, 17).reshape(4, 4)
print("Array 3: \n", array3)
np.fill_diagonal(array3, 0)
print("Diagonals Filled: \n", array3)


# Assignment 2: Array Indexing and Slicing
# 1. Create a NumPy array of shape (6, 6) with values from 1 to 36. Extract the sub-array consisting of the 3rd to 5th rows and 2nd to 4th columns.
# 2. Create a NumPy array of shape (5, 5) with random integers. Extract the elements on the border.
print("\nAssignment 2: Array Indexing and Slicing")
array = np.arange(1, 37).reshape(6, 6)
print("Array: \n", array)
print("Sub-array: \n", array[2:5, 1:4])

array2 = np.random.randint(1, 21, size=(5,5))
print("Array 2: \n", array2)
bottom = array2[-1, :]
top = array2[0, :]
left = array2[1: -1, 0]
right = array2[1: -1, -1]
border = np.concatenate((top, right, bottom, left))
print("Border: ", border)

# Assignment 3: Array Operations
# 1. Create two NumPy arrays of shape (3, 4) filled with random integers. Perform element-wise addition, subtraction, multiplication, and division.
# 2. Create a NumPy array of shape (4, 4) with values from 1 to 16. Compute the row-wise and column-wise sum.
print("\nAssignment 3: Array Operations")
array1 = np.random.randint(1, 21, size=(3,4))
array3 = np.random.randint(1, 21, size=(3,4))

total = array1 + array3 
difference = array1 - array3
product = array1 * array3
quotient = array1 / array3\

print(f"Sum: \n{total}\nDifference: \n{difference}\nProduct: \n{product}\nQuotient: \n{quotient}")

array4 = np.arange(1, 17).reshape(4,4)
print("Array 4: \n", array4)
print("Row-wise sum: ", np.sum(array4, axis=1))
print("Column-wise sum: ", np.sum(array4, axis=0))


# Assignment 4: Statistical Operations
print("\nAssignment 4: Statistical Operations")
array = np.random.randint(1, 21, size=(5, 5))
print("Array: \n", array)

mean = np.mean(array)
median = np.median(array)
std_dev = np.std(array)
variance = np.var(array)

print(f"Mean: {mean}\nMedian: {median}\nStandard Deviation: {std_dev}\nVariance: {variance}")

array4 = np.arange(1, 10).reshape(3,3)
print("\nArray: \n", array4)
normalized = (array4 - array4.mean()) / array4.std()
print("Normalized Array: \n", normalized)
print("Normalized Mean: ", normalized.mean())
print("Normalized Std Dev: ", normalized.std())



# Assignment 5: Broadcasting
print("\nAssignment 5: Broadcasting")
array1 = np.random.randint(1, 21, size=(3, 3))
array2 = np.array([7, 15, 9])
print("Array 1: \n", array1)
print("Array 2: \n", array2)

result = array1 + array2
print("Result: \n", result)

array3 = np.random.randint(1, 21, size=(4, 4))
print("Array 3: \n", array3)
array = np.random.randint(1, 21, size=(4, 1))
print("Array: \n", array)

result = array3 - array
print("Result: \n", result)


# Assignment 6: Linear Algebra
print("\nAssignment 6: Linear Algebra")
matrix = np.random.randint(1, 21, size=(3, 3))
print("Matrix: \n", matrix)

det = np.linalg.det(matrix)
inv = np.linalg.inv(matrix)
eig = np.linalg.eig(matrix)

print(f"Determinant: {det}\nInverse: \n{inv}\nEigenvalues: {eig[0]}\nEigenvectors: \n{eig[1]}")
print("Check (should be ~identity matrix):\n", matrix @ inv)

arrayA = np.random.randint(1, 21, size=(2, 3))
arrayB = np.random.randint(1, 21, size=(3, 2))
print("Array A: \n", arrayA)
print("Array B: \n", arrayB)

result = arrayA @ arrayB
print("Result of Matrix Multiplication: \n", result)


# Assignment 7: Advanced Array Manipulation
print("\nAssignment 7: Advanced Array Manipulation")
arr = np.arange(1, 10).reshape(3, 3)
print("Original Array: \n", arr)
arr1 = arr.reshape(1, 9)
print(f"Reshaped Array: \n", arr1, "Shape: \n", arr1.shape)
arr2 = arr.reshape(9, 1)
print(f"Reshaped Array: \n", arr2, "Shape: \n", arr2.shape)

array = np.random.randint(1, 21, size=(5,5))
print("\nArray: \n", array)
flat = array.flatten()
print("Flattened Array: ", flat)
notflat = flat.reshape(5,5)
print("Reshaped Array: \n", notflat)
print("Matches original?", np.array_equal(array, notflat))



# Assignment 8: Fancy Indexing and Boolean Indexing
print("\nAssignment 8: Fancy Indexing and Boolean Indexing")
matrix = np.random.randint(1, 21, size=(5, 5))
print("Matrix: \n", matrix)
# Fancy Indexing
rows = [0, 0, -1, -1]
cols = [0, -1, 0, -1]
arr = matrix[rows, cols]
print("Fancy Indexed Elements: ", arr)

matrix2 = np.random.randint(1, 21, size=(4,4))
print("Matrix 2: \n", matrix2)
# Boolean Indexing
mask = matrix2 > 10
print(mask) 
matrix2[matrix2 > 10] = 10
print("Modified Matrix 2: \n", matrix2)


# Assignment 9: Structured Arrays 
print("\nAssignment 9: Structured Arrays")
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('height', 'f4')])
data = np.array([
    ('Alice', 25, 5.5), 
    ('Bob', 30, 6.0), 
    ('Charlie', 22, 5.8),
    ('Diana', 28, 5.6)
], dtype=dt)
print("Structured Array: \n", data)

sorted_data = np.sort(data, order='age')
print("Data sorted by age: \n", sorted_data)


dt2 = np.dtype([('x', 'i4'), ('y', 'i4')])
data2 = np.array([
    (5, 6),
    (7, 12),
    (14, 6)
], dtype=dt2)
print("\nStructured Array 2: \n", data2)

for i in range(len(data2)):
    for j in range(i+1, len(data2)):
        dist = np.sqrt((data2['x'][j] - data2['x'][i])**2 + (data2['y'][j] - data2['y'][i])**2)
        print(f"Distance between points {i} and {j}: {dist}")


# Assignment 10: Masked Arrays
print("\nAssignment 10: Masked Arrays")
arr = np.random.randint(1, 21, size=(4, 4))
print("Array: \n", arr)

masked = ma.masked_greater(arr, 10)
print("Masked Array: \n", masked)
print(f"Sum of the unmasked values: {masked.sum()}")

array = np.random.randint(1, 21, size=(3, 3))
print("\nArray: \n", array)
diag_mask = np.eye(3, dtype=bool)
masked = ma.masked_array(array, mask=diag_mask)
print("Masked Array: \n", masked)

print(f"Filled masked array: \n{masked.filled(masked.mean())}")

