import numpy as np
import re


def multiply(A, B):
    return A.dot(B)


def inverse(A):
    return np.linalg.inv(A)


def transpose(A):
    return A.transpose()


def get_eigen_value(A):
    return np.linalg.eigvals(A)


def get_eigen_vector(A):
    return np.linalg.eig(A)


def set_pivot_to_one(A, cur_row, cur_col, no_of_col):
    val = A[cur_row, cur_col]
    for j in range(no_of_col):
        A[cur_row, j] = A[cur_row, j] / val


def set_lower_triangular_elements_zero(A, cur_row, cur_col, no_of_col):
    val = A[cur_row, cur_col]
    for j in range(no_of_col):
        A[cur_row, j] = A[cur_row, j] - val * A[cur_col, j]


def get_upper_triangular_matrix(A, no_of_row, no_of_col):
    for cur_row in range(no_of_row):
        for cur_col in range(no_of_col):
            if cur_row == cur_col:
                set_pivot_to_one(A, cur_row, cur_col, no_of_col)
            if cur_row > cur_col:
                set_lower_triangular_elements_zero(A, cur_row, cur_col, no_of_col)


def get_last_row(rhs_lst, size, lhs):
    arr = np.empty(size)
    coeff_at_lhs = 1.0
    if lhs.split('F')[0] == '':
        coeff_at_lhs = 1.0
    elif lhs.split('F')[0] == '-':
        coeff_at_lhs = -1.0
    else:
        coeff_at_lhs = float(lhs.split('F')[0])
    for i in range(size):
        coeff_at_rhs = 0.0
        if rhs_lst[i].split('F')[0] == '':
            coeff_at_rhs = 1.0
        elif rhs_lst[i].split('F')[0] == '-':
            coeff_at_rhs = -1.0
        else:
            coeff_at_rhs = float(rhs_lst[i].split('F')[0])
        arr[i] = coeff_at_rhs / coeff_at_lhs
    return arr


def construct_coeff_matrix(lhs, rhs):
    rhs_lst = re.findall(r'-?[0-9]{0,}?F\(k\+?[0-9]?\)+?', rhs.replace(' ',''))
    rhs_lst.sort(key=lambda x: x[x.index('F'):])
    if rhs_lst.__contains__(''):
        rhs_lst.remove('')
    size = len(rhs_lst)
    lst_row = get_last_row(rhs_lst, size, lhs)
    coeff_matrix = np.zeros(size * size).reshape(size, size)
    for i in range(size - 1):
        coeff_matrix[i][i + 1] = 1.0
    for i in range(size):
        coeff_matrix[size - 1][i] = lst_row[i]
    print(coeff_matrix)
    return coeff_matrix


def solve_diff_eqn(input):
    diff_eqn_parts = input.split('=')
    coeff_matrix = construct_coeff_matrix(diff_eqn_parts[0], diff_eqn_parts[1])
    return np.linalg.eigvals(coeff_matrix)


# checking stability of a linear system
def check_stability_of_linear_system(input):
    eigen_values = solve_diff_eqn(input)
    print(eigen_values)
    is_system_unstable = False
    for i in range(len(eigen_values)):
        if np.iscomplex(eigen_values[i]):
            val = np.absolute(eigen_values[i])
            if -1 <= val <= 1:
                is_system_unstable = False
            else:
                is_system_unstable = True
                break
        else:
            if -1 <= eigen_values[i] <= 1:
                is_system_unstable = False
            else:
                is_system_unstable = True
                break
    if is_system_unstable:
        print('System is unstable')
    else:
        print('System is stable')


if __name__ == "__main__":
    print('Hello World!!!')

    # A = np.array([1,2,3,4,5,6]).reshape(3,2)
    # B = np.array([6,5,4,3,2,1]).reshape(2,3)
    # print(A)
    # print(B)
    # print(multiply(A,B))
    # print('------------------------------------------------------------')
    # A = np.array([1,2,6,4,5,3,7,8,9]).reshape(3,3)
    # print(A)
    # print(inverse(A))
    # print('------------------------------------------------------------')
    # A = np.array([1, 2, 6, 4, 5, 3, 7, 8, 9]).reshape(3, 3)
    # print(A)
    # print(transpose(A))
    # print('------------------------------------------------------------')
    # A = np.array([1, 2, 6, 4, 7, 3, 7, 8, 9]).reshape(3, 3)
    # print(A)
    # print(get_eigen_value(A))
    # print(get_eigen_vector(A))

    A = np.array([1.0, 2.0, 6.0, 7.0, 4.0, 5.0, 9.0, 5.0, 7.0, 8.0, 9.0, 1.0]).reshape(4, 3)
    print(A)
    get_upper_triangular_matrix(A, 4, 3)
    print(A)

    print('--------------------------------------------------------------')

    # input = '2F(k+3) = 3F(k+2) + 4F(k+1) + 2F(k)'
    input = '2F(k+3) = -3F(k+2)+2F(k)'
    check_stability_of_linear_system(input)
