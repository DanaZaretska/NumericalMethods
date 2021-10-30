# Finding the largest eigen value and eigen vector using PM(Power Method) algorithm
import numpy

matrix_file = 'matrix'
len_matrix = len(open(matrix_file, 'r').readlines()) - 1


def print_matrix(matrix):
    for i in range(len_matrix):
        for j in range(len_matrix):
            print(matrix[i][j], "\t", end='')
        print()
    print()


def print_vector(vector, k: str):
    if k:
        print("Vector y_" + k + ": ", end=' ')

    for el in vector:
        print(round(el, 4), "\t", end='')
    print()


def read_matrix():
    matrix = []
    for i in range(len_matrix):
        matrix.append([float(item) for item in list(open(matrix_file))[i].strip().split()])

    return matrix


def read_vector():
    return [float(item) for item in list(open(matrix_file))[len_matrix].strip().split()]


def multiply_number_and_vector(number, vector):
    return [number*vector_i for vector_i in vector]


def multiply_matrix_and_vector(matrix, vector):
    return numpy.array(matrix).dot(numpy.array(vector))


def find_y_wave_k(vector_y_k_minus_1):
    return multiply_matrix_and_vector(matrix_A, vector_y_k_minus_1)


def find_y_k(vector_y_wave_k):
    return multiply_number_and_vector(1 / max(vector_y_wave_k), vector_y_wave_k)


def find_mu_k(vector_y_wave_k, vector_y_k_minus_1):
    mu_k = max(vector_y_wave_k) if 0 in vector_y_k_minus_1 else sum(y_wave_i/y_i for y_wave_i, y_i in zip(vector_y_wave_k, vector_y_k_minus_1)) / len_matrix
    return round(mu_k, 6)


def check_approximation(mu_k, mu_k_minus_1):
    return abs(mu_k - mu_k_minus_1) <= eps * abs(mu_k)


eps = float(input("Enter eps: "))

matrix_A = read_matrix()
vector_y_0 = read_vector()

vector_y_wave_1 = find_y_wave_k(vector_y_0)
vector_y_1 = find_y_k(vector_y_wave_1)
mu_1 = find_mu_k(vector_y_wave_1, vector_y_0)

vector_y_wave_2 = find_y_wave_k(vector_y_1)
vector_y_2 = find_y_k(vector_y_wave_2)
mu_2 = find_mu_k(vector_y_wave_2, vector_y_1)

vector_y_k_minus_1, vector_y_k = vector_y_2[:], []
mu_k_minus_1, mu_k = mu_1, mu_2

print_vector(vector_y_wave_1, str("wave_1"))
print_vector(vector_y_1, str(1))
print("Mu_1 = ", mu_1, "\n")

print_vector(vector_y_wave_2, str("wave_2"))
print_vector(vector_y_2, str(2))
print("Mu_2 = ", mu_2, "\n")

iter_number = 2
while not check_approximation(mu_k, mu_k_minus_1):
    vector_y_wave_k = find_y_wave_k(vector_y_k_minus_1)
    vector_y_k = find_y_k(vector_y_wave_k)

    mu_k_minus_1 = mu_k
    mu_k = find_mu_k(vector_y_wave_k, vector_y_k_minus_1)
    vector_y_k_minus_1 = vector_y_k[:]

    iter_number += 1
    print_vector(vector_y_wave_k, str("wave_" + str(iter_number)))
    print_vector(vector_y_k, str(iter_number))
    print("Mu_{0} = {1}".format(iter_number, mu_k) + '\n')

print(iter_number, " iterations")
print("Eigen value: ", mu_k)
print("Eigen vector: ", end='')
print_vector(vector_y_k, "")
