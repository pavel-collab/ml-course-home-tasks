import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    #TODO: посмотреть в заданиях по вычматам
    #* В начале генерируем начальный вектор нужной размерности
    r0 = np.ones(np.shape(data)[0])

    for _ in range(num_steps):
        r_tmp = np.dot(data, r0)
        r_k = r_tmp / np.sqrt(np.dot(r_tmp, r_tmp))

        #! DEBUG, need to remove from final implementation
        mu_k = np.dot(r0, r_tmp) / np.dot(r0, r0)

        r0 = r_k

    #* last mu_k is a value, that we have return
    return mu_k, r0