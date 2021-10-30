# QR decomposition (Householder method)

import numpy as np

filename = 'matrix'
infile = open(filename, 'r')
len_matrix = len(infile.readlines()) - 1


def read_matrix():
    matrix = []
    with open(filename, 'r') as f:
        row_in_file = 0
        for line in f.readlines():
            if row_in_file < len_matrix:
                items = line.strip().split(' ')
                matrix.append([float(item) for item in items])
                row_in_file += 1
    return matrix


def read_vector():
    vector = []
    with open(filename, 'r') as f:
        row_in_file = 0
        for line in f.readlines():
            if row_in_file >= len_matrix:
                items = line.strip().split(' ')
                for item in items:
                    vector.append(float(item))
            row_in_file += 1
    return vector


def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(round(matrix[i][j], 4), "\t", end='')
        print()
    print()


def print_vector(vector):
    for el in vector:
        print(round(el, 4), "\t", end='')


def find_vector_norm(vector):
    norm = np.linalg.norm(vector)
    return norm


def crop_matrix(matrix, index_to_reduce):
    croped_matrix = matrix[:]
    for k in range(index_to_reduce):
        croped_matrix = np.delete(croped_matrix, 0, 0)       # delete zero row in matrix
        croped_matrix = np.delete(croped_matrix, 0, 1)       # delete zero column in matrix

    return croped_matrix


def pull_vector_from_matrix(matrix):
    vector = []
    for row in matrix:
        vector.append(row[0])
    return vector


def find_alpha(vector_a_norm, vector_e_norm):
    alpha = vector_a_norm/vector_e_norm
    return alpha


def subtract_vectors(vec1, vec2):
    return [vec1_i - vec2_i for vec1_i, vec2_i in zip(vec1, vec2)]


def multiply_number_and_vector(number, vector):
    return [number*vector_i for vector_i in vector]


def scalar_multiply_vectors(vec1, vec2):
    return sum(vec1_i*vec2_i for vec1_i, vec2_i in zip(vec1, vec2))


def make_omega_matrix(vector_w, size):
    omega_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            omega_matrix[i][j] = vector_w[i]*vector_w[j]
    return omega_matrix


def make_U_matrix(e_matrix, omega_matrix):
    size = len(e_matrix)
    V_matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            V_matrix[i][j] = e_matrix[i][j] - 2*omega_matrix[i][j]
    return V_matrix


def multiply_matrixes(matrix_U, matrix_A):
    size = len(matrix_U)
    m = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            for k in range(size):
                m[i][j] += matrix_U[i][k] * matrix_A[k][j]

    for i in range(size):
        for j in range(size):
            m[i][j] = m[i][j]
    return m


def add_rows_to_U(U_to_add):
    zero_row = [0]*len(U_to_add)
    one_and_zeros_column = [0]*len(U_to_add)
    one_and_zeros_column.insert(0, 1)
    U2 = np.row_stack((zero_row, U_to_add))
    U2 = np.column_stack((one_and_zeros_column, U2))
    return U2


def print_x(matrix_A, vector_b):
    matrix_A_inv = np.linalg.inv(matrix_A)
    x_vec = matrix_A_inv @ np.array(vector_b)
    print(x_vec)


if len_matrix >= 2:
    matrix_A = read_matrix()
    vector_b = read_vector()

    print("Matrix A:")
    print_matrix(matrix_A)
    print("Vector b", vector_b)

    e1_matrix = np.eye(len_matrix)

    a1 = pull_vector_from_matrix(matrix_A)
    e1 = pull_vector_from_matrix(e1_matrix)
    print("Vector a ", a1, " \nVector e ", e1)

    a1_norm = find_vector_norm(a1)
    e1_norm = find_vector_norm(e1)

    alpha1 = find_alpha(a1_norm, e1_norm)
    print("alpha = ", alpha1)

    vector_a1_substract_alpha_e1 = subtract_vectors(a1, multiply_number_and_vector(alpha1, e1))
    scalar_multiply_a1_and_vector_a1_substract_alpha_e1 = scalar_multiply_vectors(a1, vector_a1_substract_alpha_e1)
    print("\n(a - alpha*e) = ", vector_a1_substract_alpha_e1, "\na(a - alpha*e) = ", scalar_multiply_a1_and_vector_a1_substract_alpha_e1)

    w1 = multiply_number_and_vector(1/np.sqrt(2*scalar_multiply_a1_and_vector_a1_substract_alpha_e1), vector_a1_substract_alpha_e1)
    omega1_matrix = make_omega_matrix(w1, len_matrix)
    print("Vector w ", w1)
    print("\nMatrix Omega")
    print_matrix(omega1_matrix)

    print("\nMatrix (E - 2ww^T)")
    U1 = make_U_matrix(e1_matrix, omega1_matrix)
    print("Matrix U1")
    print_matrix(U1)

    U1_A = multiply_matrixes(U1, matrix_A)
    U1_B = np.dot(U1, vector_b)

    print("Matrix U1*A")
    print_matrix(U1_A)
    print("Vector U1*b", U1_B)
    print("\n____________________________________________________")

    if len_matrix >= 3:
        U1A_reduced = crop_matrix(U1_A, 1)
        e2_matrix= crop_matrix(e1_matrix, 1)
        a2 = pull_vector_from_matrix(U1A_reduced)
        e2 = pull_vector_from_matrix(e2_matrix)
        print("Vector a ", a2, " \nVector e ", e2)

        a2_norm = find_vector_norm(a2)
        e2_norm = find_vector_norm(e2)
        alpha2 = find_alpha(a2_norm, e2_norm)
        print("alpha = ", alpha2)

        vector_a2_substract_alpha2_e2 = subtract_vectors(a2, multiply_number_and_vector(alpha2, e2))
        scalar_multiply_a2_and_vector_a2_substract_alpha2_e2 = scalar_multiply_vectors(a2, vector_a2_substract_alpha2_e2)
        print("\n(a - alpha*e) = ", vector_a2_substract_alpha2_e2, "\na(a - alpha*e) = ",
              scalar_multiply_a2_and_vector_a2_substract_alpha2_e2)
        w2 = multiply_number_and_vector(1/np.sqrt(2*scalar_multiply_a2_and_vector_a2_substract_alpha2_e2), vector_a2_substract_alpha2_e2)
        omega2_matrix = make_omega_matrix(w2, len_matrix - 1)
        print("Vector w ", w2)
        print("\nMatrix (E - 2ww^T)")
        print_matrix(omega2_matrix)

        U2_part = make_U_matrix(e2_matrix, omega2_matrix)
        print_matrix(U2_part)
        U2 = add_rows_to_U(U2_part)
        print("Matrix U2")
        print_matrix(U2)

        U2_U1_A = multiply_matrixes(U2, U1_A)
        print("Matrix U2_U1_A")
        print_matrix(U2_U1_A)

        U2_U1_B = np.dot(U2, U1_B)
        print("Vector U2_U1_B")
        print_vector(U2_U1_B)
        print("\n____________________________________________________")
        if len_matrix >= 4:
            U2_U1_A_reduced = crop_matrix(U2_U1_A, 2)
            e3_matrix = crop_matrix(e1_matrix, 2)
            a3 = pull_vector_from_matrix(U2_U1_A_reduced)
            e3 = pull_vector_from_matrix(e3_matrix)
            print("Vector a ", a3, " \nVector e ", e3)

            a3_norm = find_vector_norm(a3)
            e3_norm = find_vector_norm(e3)
            alpha3 = find_alpha(a3_norm, e3_norm)
            print("alpha3 = ", alpha3)

            vector_a3_substract_alpha3_e3 = subtract_vectors(a3, multiply_number_and_vector(alpha3, e3))
            scalar_multiply_a3_and_vector_a3_substract_alpha3_e3 = scalar_multiply_vectors(a3,
                                                                                           vector_a3_substract_alpha3_e3)
            w3 = multiply_number_and_vector(1 / math.sqrt(2 * scalar_multiply_a3_and_vector_a3_substract_alpha3_e3),
                                            vector_a3_substract_alpha3_e3)
            omega3_matrix = make_omega_matrix(w3, len_matrix - 2)
            print("Vector w ", w3)
            print("\nMatrix Omega")
            print_matrix(omega3_matrix)

            print("\nMatrix (E - 2ww^T)")
            U3_part = make_U_matrix(e3_matrix, omega3_matrix)
            print_matrix(U3_part)
            U3 = add_rows_to_U(U3_part)
            U3 = add_rows_to_U(U3)
            print("Matrix U3 ")
            print_matrix(U3)

            U3_U2_U1_A = multiply_matrixes(U3, U2_U1_A)
            print("Matrix U3_U2_U1_A")
            print_matrix(U3_U2_U1_A)

            U3_U2_U1_B = np.dot(U3, U2_U1_B)
            print("Vector U3_U2_U1_B")
            print_vector(U3_U2_U1_B)

            print_x(U3_U2_U1_A, U3_U2_U1_B)
        else:
            print_x(U2_U1_A, U2_U1_B)
    else:
        print_x(U1_A, U1_B)
