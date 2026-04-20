import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    from numba import njit

    return mo, njit, np, plt, time


@app.cell
def _(mo):
    mo.md(r"""
    # NumPy vs Numba

    Cравниваем обычные Python-списки, массивы NumPy,
    ручное умножение матриц, JIT-ускорение через Numba и векторизованную
    операцию `np.dot`.

    Основная идея: NumPy дает удобные массивы фиксированного размера и
    быстрые векторизованные операции, а Numba может ускорять циклы,
    написанные в стиле обычного Python.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Списки Python и массивы NumPy

    Список Python может хранить значения разных типов и менять размер.
    Массив NumPy обычно приводит элементы к одному типу и имеет фиксированную
    форму после создания.
    """)
    return


@app.cell
def _(np):
    list_a = [1, True, "hui"]
    homogeneous_array = np.array([1, 2, 3, 4])
    mixed_array = np.array([1, "fvdf"])

    print("Python list:", list_a)
    print("Homogeneous NumPy array:", homogeneous_array)
    print("Mixed NumPy array:", mixed_array)
    print("Mixed array dtype:", mixed_array.dtype)

    list_a.append(13)
    print("List after append:", list_a)

    try:
        array_b = np.array([1, 2, 3])
        print("Fixed-size array:", array_b)
        array_b[4] = 3
    except IndexError as e:
        print("Исключение IndexError:", e)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Операции над матрицами

    Для вложенных списков оператор `+` склеивает списки. Для массивов NumPy
    тот же оператор выполняет поэлементное сложение матриц одинаковой формы.
    """)
    return


@app.cell
def _(np):
    matrix_list_a = [[1, 2], [2, 3]]
    matrix_list_b = [[2, 1], [2, 0]]
    list_sum = matrix_list_a + matrix_list_b

    a = np.array([[1, 1], [1, 1]])
    b = a
    sum_matrix = a + b

    print("List + list:", list_sum)
    print("Array + array:")
    print(sum_matrix)
    return a, sum_matrix


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Базовые возможности `ndarray`

    У массивов NumPy есть форма (`shape`), тип данных (`dtype`) и готовые
    операции линейной алгебры. Например, можно быстро создать нулевую
    матрицу или перемножить матрицы через `np.dot`.
    """)
    return


@app.cell
def _(a, np, sum_matrix):
    print("Shape:", np.shape(a))
    print("Zeros:")
    print(np.zeros((3, 3)))
    print("Dot product:")
    print(np.dot(sum_matrix, a))
    print("dtype:", a.dtype)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Тестовые матрицы

    Подготовим квадратные, прямоугольные и несовместимые по размеру матрицы.
    Последний вариант нужен для проверки, что функция корректно отлавливает
    невозможное умножение.
    """)
    return


@app.cell
def _(np):
    n = 1000
    m = 100

    A = np.random.random((n, n))
    B = np.random.random((n, n))

    A_r = np.random.random((n, m))
    B_r = np.random.random((m, n))

    A_ir = np.random.random((n, m))
    B_ir = np.random.random((n, n))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Проверка совместимости

    Матрицы можно перемножать только тогда, когда число столбцов первой
    матрицы равно числу строк второй.
    """)
    return


@app.function
def is_mult_available(A, B):
    return A.shape[1] == B.shape[0]


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Умножение матриц на чистом Python

    Это прямой алгоритм из трех вложенных циклов. Он хорошо показывает идею
    матричного умножения, но на больших размерах работает медленно.
    """)
    return


@app.cell
def _(np):
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

    return (matrix_multiply,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Умножение матриц с Numba

    Декоратор `@njit` компилирует функцию перед выполнением. Первый запуск
    включает время компиляции, поэтому перед бенчмарком функцию нужно
    один раз прогреть на маленьких матрицах.
    """)
    return


@app.cell
def _(njit, np):
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

    return (matrix_multiply_boost,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Векторизованное умножение NumPy

    `np.dot` использует оптимизированные низкоуровневые реализации линейной
    алгебры. Для таких задач это обычно самый простой и быстрый вариант.
    """)
    return


@app.cell
def _(np):
    def matrix_mult(A, B):
        if not is_mult_available(A, B):
            raise ValueError("Matrix shapes are not compatible for multiplication")

        return np.dot(A, B)

    return (matrix_mult,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Сравнение производительности

    Функция ниже запускает три реализации на квадратных матрицах разных
    размеров, печатает время выполнения и строит два графика:

    - абсолютное время выполнения;
    - ускорение Numba и NumPy относительно чистого Python.
    """)
    return


@app.cell
def _(matrix_mult, matrix_multiply, matrix_multiply_boost, np, plt, time):
    def comparing_performance(max_size=500, step=100):
        sizes = list(range(step, max_size + 1, step))
        pure_python_times = []
        numba_times = []
        numpy_times = []

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
        plt.plot(sizes, pure_python_times, marker="o", label="Pure Python")
        plt.plot(sizes, numba_times, marker="o", label="Numba")
        plt.plot(sizes, numpy_times, marker="o", label="NumPy")
        plt.xlabel("Matrix size (n x n)")
        plt.ylabel("Execution time (seconds)")
        plt.title("Matrix multiplication performance comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        speedup_numba = [
            pure / numba for pure, numba in zip(pure_python_times, numba_times)
        ]
        speedup_numpy = [
            pure / numpy_t for pure, numpy_t in zip(pure_python_times, numpy_times)
        ]
        plt.plot(sizes, speedup_numba, marker="o", label="Numba speedup vs Pure Python")
        plt.plot(sizes, speedup_numpy, marker="o", label="NumPy speedup vs Pure Python")
        plt.xlabel("Matrix size (n x n)")
        plt.ylabel("Speedup (times faster)")
        plt.title("Acceleration relative to Pure Python")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return {
            "sizes": sizes,
            "pure_python_times": pure_python_times,
            "numba_times": numba_times,
            "numpy_times": numpy_times,
        }

    return (comparing_performance,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Запуск эксперимента

    По умолчанию сравнение идет на размерах `100, 200, 300, 400, 500`.
    Если выполнение занимает слишком много времени, можно уменьшить
    `max_size` или увеличить `step`.
    """)
    return


@app.cell
def _(comparing_performance):
    results = comparing_performance(max_size=500, step=100)
    return


if __name__ == "__main__":
    app.run()
