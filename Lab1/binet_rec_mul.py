import numpy as np

def is_power_of_two(n: int) -> bool:
    while n != 1:
        if n % 2 != 0: return False
        n = n // 2
    return True

def split_matrix_in_quarters(A: np.array):
    half_i = A.shape[0] // 2
    return  A[:half_i, :half_i], \
            A[:half_i, half_i:], \
            A[half_i:, :half_i],\
            A[half_i:, half_i:]
            
def join_matrices(A11, A12, A21, A22):
    A1 = np.hstack((A11, A12))
    A2 = np.hstack((A21, A22))
    return np.vstack((A1, A2))

def binet_rec_mat_mul(A: np.array, B: np.array, k: int) -> np.array:
    # assert len(A.shape) == 2 and A.shape[0] == A.shape[1] and A.shape == B.shape, "The input matrices have wrong dimensions"
    # n = A.shape[0]
    # assert is_power_of_two(n), "Input matrices are not powers of two"
    
    if k == 0:
        return A*B
    
    rec_step = lambda A1, B1, A2, B2: binet_rec_mat_mul(A1, B1, k-1) + binet_rec_mat_mul(A2, B2, k-1)

    A11, A12, A21, A22 = split_matrix_in_quarters(A)
    B11, B12, B21, B22 = split_matrix_in_quarters(B)
    
    C11 = rec_step(A11, B11, A12, B21)
    C12 = rec_step(A11, B12, A12, B22)
    C21 = rec_step(A21, B11, A22, B21)
    C22 = rec_step(A21, B12, A22, B22)
    
    return join_matrices(C11, C12, C21, C22)
    

if __name__=="__main__":
    A = np.arange(64).reshape(8,8)
    B = np.arange(64).reshape(8,8)
    
    expected_result =  A @ B
    
    actual_result = binet_rec_mat_mul(A, B, 3)
    assert np.array_equal(actual_result, expected_result), "Results differ!"
    