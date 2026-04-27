print("hello world!!!")
print("In this file we going to check conceptual and numerical difference between python list"
      "python list boosted by numba and numpy np.array")
import numpy as np
import time
from numba import njit
import matplotlib.pyplot as plt
list_a = [1, True, "hui"]     # different data type
array_a = np.array([1, 2, 3, 4]) # same data type only

array_a = np.array([1, "fvdf"])
print(array_a) # no error int -> str (1 -> "1")

print(list_a)
list_a.append(13)       # unfixed size
print(list_a)
try:
      array_b = np.array([1,2,3])
      print(array_b)
      array_b[4] = 3                      # fixed size
      print(array_b)
except IndexError as e:
      print("исключение indexError: ", e)

# matrix job
matrix_list_a = [[1, 2], [2, 3]]
matrix_list_b = [[2, 1], [2, 0]]
list_sum = matrix_list_a + matrix_list_b # no sum like operation
print(list_sum) # [[1, 2], [2, 3], [2, 1], [2, 0]]

a = np.array([[1, 1], [1, 1]])
b = a
sum_matrix = a + b # sum as matrix
print(sum_matrix) #[[2 2][2 2]]

# some basic method in ndarrays
print(np.shape(a)) # 2, 2 form of matrix
print(np.zeros((3, 3))) # create zero matrix
print(np.dot(sum_matrix, a))
print(a.dtype)

# time difference

# square matrix
n = 1000 #size
A = np.random.random((n, n))
B = np.random.random((n, n))

# rectangle
m = 100
A_r = np.random.random((n, m))
B_r = np.random.random((m, n))

# wrong rectangle
A_ir = np.random.random((n, m))
B_ir = np.random.random((n, n))
#     'proverka na duraka'
def is_mult_available(A, B):
      return A.shape[1] == B.shape[0]

# Pure python multiply
def matrix_multiply(A, B):
      if not is_mult_available(A, B):
            raise ValueError("Matrix shapes are not compatible for multiplication")

      rows_a, cols_a = A.shape
      _, cols_b = B.shape
      C = np.zeros((rows_a, cols_b))

      for i in range(rows_a):
            for j in range(cols_b):
                  for k in range(cols_a):
                        C[i, j] += A[i, k] * B[k, j]
      return C

# boosted by numba
#print("to use numba we need annotate as decorator")
@njit
def matrix_multiply_boost(A, B):
      rows_a, cols_a = A.shape
      rows_b, cols_b = B.shape

      if cols_a != rows_b:
            raise ValueError("Matrix shapes are not compatible for multiplication")

      C = np.zeros((rows_a, cols_b))

      for i in range(rows_a):
            for j in range(cols_b):
                  for k in range(cols_a):
                        C[i, j] += A[i, k] * B[k, j]
      return C

# vectorized
def matrix_mult(A, B):
      if not is_mult_available(A, B):
            raise ValueError("Matrix shapes are not compatible for multiplication")
      return np.dot(A, B)

#comparing
def comparing_performance(max_size=500, step=100):
      sizes = list(range(step, max_size + 1, step))
      pure_python_times = []
      numba_times = []
      numpy_times = []

      # warm up numba compilation once before benchmarking
      warmup_a = np.random.random((2, 2))
      warmup_b = np.random.random((2, 2))
      matrix_multiply_boost(warmup_a, warmup_b)

      for size in sizes:
            A_test = np.random.random((size, size))
            B_test = np.random.random((size, size))

            start = time.time()
            matrix_multiply(A_test, B_test)
            pure_time = time.time() - start
            pure_python_times.append(pure_time)

            start = time.time()
            matrix_multiply_boost(A_test, B_test)
            numba_time = time.time() - start
            numba_times.append(numba_time)

            start = time.time()
            matrix_mult(A_test, B_test)
            numpy_time = time.time() - start
            numpy_times.append(numpy_time)

            print(
                  f"size={size}x{size} | "
                  f"Pure Python: {pure_time:.6f} sec | "
                  f"Numba: {numba_time:.6f} sec | "
                  f"NumPy: {numpy_time:.6f} sec"
            )

      plt.figure(figsize=(10, 6))
      plt.plot(sizes, pure_python_times, marker='o', label='Pure Python')
      plt.plot(sizes, numba_times, marker='o', label='Numba')
      plt.plot(sizes, numpy_times, marker='o', label='NumPy')
      plt.xlabel('Matrix size (n x n)')
      plt.ylabel('Execution time (seconds)')
      plt.title('Matrix multiplication performance comparison')
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.show()

      plt.figure(figsize=(10, 6))
      speedup_numba = [pure / numba for pure, numba in zip(pure_python_times, numba_times)]
      speedup_numpy = [pure / numpy_t for pure, numpy_t in zip(pure_python_times, numpy_times)]
      plt.plot(sizes, speedup_numba, marker='o', label='Numba speedup vs Pure Python')
      plt.plot(sizes, speedup_numpy, marker='o', label='NumPy speedup vs Pure Python')
      plt.xlabel('Matrix size (n x n)')
      plt.ylabel('Speedup (times faster)')
      plt.title('Acceleration relative to Pure Python')
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.show()

      return {
            'sizes': sizes,
            'pure_python_times': pure_python_times,
            'numba_times': numba_times,
            'numpy_times': numpy_times,
      }

results = comparing_performance(max_size=500, step=100)
