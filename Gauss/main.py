filename = 'matrix'
output_file = open('output', 'w')   # файл в який записуються всі вихідні дані
infile = open(filename, 'r')
len_matrix = len(infile.readlines()) - 1


# ФУНКЦІЯ ДЛЯ ЗЧИТУВАННЯ МАТРИЦІ З ФАЙЛУ
def read_matrix(matrix):
    with open(filename, 'r') as f:
        row_in_file = 0
        for line in f.readlines():
            if row_in_file < len_matrix:
                items = line.strip().split(' ')
                matrix.append([float(item) for item in items])
                row_in_file += 1
    return matrix


# ФУНКЦІЯ ДЛЯ ЗЧИТУВАННЯ ВЕКТОРА З ФАЙЛУ
def read_vector(vector):
    with open(filename, 'r') as f:
        row_in_file = 0
        for line in f.readlines():
            if row_in_file >= len_matrix:
                items = line.strip().split(' ')
                for item in items:
                    vector.append(float(item))
            row_in_file += 1
    return vector


# ФУНКЦІЯ ДЛЯ СТВОРЕННЯ МАТРИЦЬ Аі (створює матрицю А1, А2 і так далі)
def make_matrix_Ai(matrix, matrix_Ai, k):
    for i in range(k):
        matrix_Ai.append(matrix[i])

    for i in range(k, len(matrix)):
        row = [0]*(k)
        for j in range(k, len(matrix[i])):
            k_ = k - 1
            matrix_Ai_element = matrix[i][j] - (matrix[i][k_]/matrix[k_][k_])*matrix[k_][j]
            matrix_Ai_element = round(matrix_Ai_element, 8)
            row.append(matrix_Ai_element)
        matrix_Ai.append(row)
    return matrix_Ai


# ФУНКЦІЯ ДЛЯ СТВОРЕННЯ ВЕКТОРІВ bі
def make_vector_bi(vector, matrix, k):
    vector_bi = []
    for i in range(k):
        vector_bi.append(vector[i])

    for i in range(k, len(vector)):
        k_ = k - 1
        bi_element = vector[i] - (matrix[i][k_]/matrix[k_][k_])*vector[k_]
        bi_element = round(bi_element, 8)
        vector_bi.append(bi_element)
    return vector_bi

# ФУНКЦІЯ ЩО ШУКАЄ ГОЛОВНИЙ ЕЛЕМЕНТ СТОВПЦІВ(повертає його індекс по рядках)
def find_column_maximum_index(matrix, column_number):
    column_array = []
    for i in range(column_number - 1, column_number):
        for j in range(column_number - 1, len(matrix[i])):
            column_array.append(abs(matrix[j][i]))
    # print("Головний ел-т " + str(max(column_array)))

    return column_array.index(max(column_array)) + column_number - 1


# ФУНКЦІЯ ЩО МІНЯЄ МІСЦЯМИ ЕЛЕМЕНТИ (використовую для зміни рядків матриці місцями)
def swap(value_1, value_2):
    temp = value_1
    value_1 = value_2
    value_2 = temp
    return value_1, value_2


# ФУНКЦІЯ ШУКАЄ ГОЛОВНІ ЕЛ-ТИ СТОВПЦІВ, МІНЯЄ МІСЦЯМИ РЯДКИ М-ЦІ ТА ЕЛ-ТИ ВЕКТОРА
def search_max_and_swap(matrix_Ai, vector_bi, k):
    index_maximum_A = find_column_maximum_index(matrix_Ai, k)
    # if k - 1 != index_maximum_A:
    #     print("Міняємо місцями рядки {0}, {1}".format(k-1, index_maximum_A))
    matrix_Ai[k-1], matrix_Ai[index_maximum_A] = swap(matrix_Ai[k-1], matrix_Ai[index_maximum_A])
    vector_bi[k-1], vector_bi[index_maximum_A] = swap(vector_bi[k-1], vector_bi[index_maximum_A])

    print_matrix_and_vector(matrix_Ai, vector_bi, "Swaped", "Swaped")


# ФУНКЦІЯ ДЛЯ ПОШУКУ МІНОРІВ МАТРИЦІ
def find_matrix_minor(matrix,i,j):
    return [row[:j] + row[j+1:] for row in (matrix[:i]+matrix[i+1:])]


# ФУНКЦІЯ ДЛЯ ПОШУКУ ВИЗНАЧНИКА МАТРИЦІ
def find_matrix_determinant(matrix):
    # для матриці 2 на 2
    if len(matrix) == 2:
        return (matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0])

    determinant = 0
    for c in range(len(matrix)):
        determinant += ((-1)**c)*matrix[0][c]*find_matrix_determinant(find_matrix_minor(matrix,0,c))
    return determinant


# ФУНКЦІЯ ДЛЯ ПОШУКУ ТА ВИВОДУ ІКСІВ
def find_x(matrix_A, vector_b):

    # якщо існує нульовий рядок - розвязків безліч
    zero_rows_existance: bool = False
    for i in range(len_matrix):
        if matrix_A[len_matrix - 1] == [0] * len_matrix:
            zero_rows_existance = True

    #  якщо нульових рядків немає, тоді знаходимо ікси
    if zero_rows_existance == False :
        x3 = round(vector_b[len_matrix - 1]/matrix_A[len_matrix - 1][len_matrix - 1], 8)
        x2 = round(((vector_b[len_matrix - 2] - x3*matrix_A[len_matrix - 2][len_matrix - 1])/matrix_A[len_matrix-2][len_matrix-2]), 8)
        x1 = round((vector_b[len_matrix - 3] - x3*matrix_A[len_matrix - 3][len_matrix - 1] - x2*matrix_A[len_matrix - 3][len_matrix - 2])/matrix_A[len_matrix - 3][len_matrix - 3], 8)
        x_array = [x1, x2, x3]

        if len_matrix > 3:
            x0 = round((vector_b[len_matrix - 4] - x3*matrix_A[len_matrix - 4][len_matrix - 1] - x2*matrix_A[len_matrix - 4][len_matrix - 2] -x1*matrix_A[len_matrix - 4][len_matrix - 3])/matrix_A[len_matrix-4][len_matrix-4], 4)
            x_array.insert(0, x0)
        # виводимо ікси у файл та на консоль
        i = 1
        print("")
        for x in x_array:
            output_file.write("\nx{0} = {1}".format(i, x))
            print("x{0} = {1}".format(i, x))
            i+=1
    else:
        output_file.write("\nСистема не має розвязків")
        print("\nСистема не має розвязків")


# ФУНКЦІЯ ДЛЯ ВИВОДУ МАТРИЦІ ТА ВЕКТОРА
def print_matrix_and_vector(matrix, vector, matrix_name, vector_name):
    output_file.write("Матриця " + matrix_name + ":\n")
    print("\nМатриця " + matrix_name + ":")
    for row in matrix:
        output_file.write(str(row) + "\n")
        print(row)
    output_file.write("\nВектор " + vector_name + ":\n" + str(vector) + "\n\n")
    print("Вектор " + vector_name + ":\n")
    print(vector)

matrix = []
vector = []

# створення матриці A та вектора b (зчитування з файлу)
matrix_A = read_matrix(matrix)
vector_b = read_vector(vector)
print("\nВизначник матриці А: det =", str(find_matrix_determinant(matrix_A)))
output_file.write("\nВизначник матриці А: det = " + str(find_matrix_determinant(matrix_A)) + "\n________________________________\n\n")
print_matrix_and_vector(matrix_A, vector_b, "A", "b")

search_max_and_swap(matrix_A, vector_b, 1)

d = find_matrix_determinant(matrix_A)

# створення матриці A1 та вектора b1
matrix_A1 = []
make_matrix_Ai(matrix_A, matrix_A1, 1)
vector_b1 = make_vector_bi(vector_b, matrix_A, 1)
print_matrix_and_vector(matrix_A1, vector_b1, "A1", "b1")

search_max_and_swap(matrix_A1, vector_b1, 2)


# створення матриці A2 та вектора b2
if len_matrix > 2:
    matrix_A2 = []
    make_matrix_Ai(matrix_A1, matrix_A2, 2)
    vector_b2 = make_vector_bi(vector_b1, matrix_A1, 2)
    print_matrix_and_vector(matrix_A2, vector_b2, "A2", "b2")
    # створення матриці A3 та вектора b3 (якщо розмірність матриці це потребує/дозволяє)
    if len_matrix > 3:
        matrix_A3 = []

        make_matrix_Ai(matrix_A2, matrix_A3, 3)
        vector_b3 = make_vector_bi(vector_b2, matrix_A2, 3)
        print_matrix_and_vector(matrix_A3, vector_b3, "A3", "b3")

        find_x(matrix_A3, vector_b3)
    else:
        find_x(matrix_A2, vector_b2)

    search_max_and_swap(matrix_A2, vector_b2, 3)
elif len_matrix == 2:
    find_x(matrix_A1, vector_b1)