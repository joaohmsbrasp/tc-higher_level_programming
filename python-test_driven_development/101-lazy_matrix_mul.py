#!/usr/bin/python3
"""Defines a matrix multiplication function using NumPy."""
import numpy as np

def lazy_matrix_mul(m_a, m_b):
    """Multiply two matrices using NumPy, with strict validation.
    
    Args:
        m_a (list of lists of ints/floats): The first matrix.
        m_b (list of lists of ints/floats): The second matrix.
    
    Returns:
        list: The result of multiplying m_a by m_b.
    
    Raises:
        TypeError: If m_a or m_b are None.
        ValueError: If matrices have invalid shapes or contain non-numeric values.
    """
    # Verificação de None
    if m_a is None or m_b is None:
        raise TypeError("Object arrays are not currently supported")
    
    # Verificação se são listas
    if not isinstance(m_a, list) or not isinstance(m_b, list):
        raise ValueError("Scalar operands are not allowed, use '*' instead")
        
    if all(isinstance(row, list) for row in m_a) and all(isinstance(row, list) for row in m_b):
        pass  # São listas de listas, segue o fluxo normal
    elif all(isinstance(row, (int, float)) for row in m_a) and all(isinstance(row, (int, float)) for row in m_b):
        return [np.dot(m_a, m_b)]  # Produto escalar caso sejam vetores 1D
    elif np.ndim(m_a) >= 2 and np.ndim(m_b) == 1:
        return np.matmul(m_a, m_b)
    else:
        raise ValueError("shapes (2,) and (1,2) not aligned: 2 (dim 0) != 1 (dim 0)")
    
    # Verificação se as matrizes estão vazias
    if len(m_a) == 0 or len(m_b) == 0 or any(len(row) == 0 for row in m_a) or any(len(row) == 0 for row in m_b):
        raise ValueError(f"shapes ({len(m_a)},{len(m_a[0]) if m_a else 0}) and ({len(m_b)},{len(m_b[0]) if m_b else 0}) not aligned: {len(m_a[0]) if m_a else 0} (dim 1) != {len(m_b)} (dim 0)")
    
    # Verificação se todos os elementos são inteiros ou floats
    if not all(all(isinstance(item, (int, float)) for item in row) for row in m_a):
        raise TypeError("invalid data type for einsum")
    if not all(all(isinstance(item, (int, float)) for item in row) for row in m_b):
        raise TypeError("invalid data type for einsum")
    
    # Verificação se todas as linhas têm o mesmo tamanho
    row_length_a = len(m_a[0])
    row_length_b = len(m_b[0])
    if any(len(row) != row_length_a for row in m_a) or any(len(row) != row_length_b for row in m_b):
        raise ValueError("setting an array element with a sequence.")
    
    # Verificação de compatibilidade das dimensões
    if len(m_a[0]) != len(m_b):
        raise ValueError(f"shapes ({len(m_a)},{len(m_a[0])}) and ({len(m_b)},{len(m_b[0])}) not aligned: {len(m_a[0])} (dim 1) != {len(m_b)} (dim 0)")
    
    # Multiplicação das matrizes
    return np.matmul(m_a, m_b)

