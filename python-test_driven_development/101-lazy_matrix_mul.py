#!/usr/bin/python3
"""Defines a matrix multiplication function using NumPy."""
import numpy as np


def lazy_matrix_mul(m_a, m_b):
    """Return the multiplication of two matrices.
    Args:
        m_a (list of lists of ints/floats): The first matrix.
        m_b (list of lists of ints/floats): The second matrix.
    """

    if not isinstance(m_a, list) or not isinstance(m_b, list):
        raise ValueError("Scalar operands are not allowed, use '*' instead")
        
    # Verifica se as matrizes são listas de listas e se contêm apenas números
    if not all(isinstance(row, list) for row in m_a) or not all(isinstance(row, list) for row in m_b):
        raise TypeError("invalid data type for einsum")
        
    # Verifica se todas as linhas têm o mesmo número de colunas
    if any(len(row) != len(m_a[0]) for row in m_a):
        raise ValueError("setting an array element with a sequence.")

    if any(len(row) != len(m_b[0]) for row in m_b):
        raise ValueError("setting an array element with a sequence.")

    if not all(isinstance(num, (int, float)) for row in m_a for num in row) or not all(isinstance(num, (int, float)) for row in m_b for num in row):
        raise TypeError("invalid data type for einsum")
        
    # Obtém as dimensões das matrizes
    rows_a = len(m_a)
    cols_a = len(m_a[0]) if rows_a > 0 else 0
    rows_b = len(m_b)
    cols_b = len(m_b[0]) if rows_b > 0 else 0

    # Verifica se todas as linhas das matrizes possuem o mesmo número de colunas
    if any(len(row) != cols_a for row in m_a) or any(len(row) != cols_b for row in m_b):
        raise TypeError("Each row of the matrices must have the same size")

    # Verifica se as dimensões permitem a multiplicação
    if cols_a != rows_b:
        raise ValueError(f"shapes ({rows_a},{cols_a}) and ({rows_b},{cols_b}) not aligned: {cols_a} (dim 1) != {rows_b} (dim 0)")
        
    if m_a is None or m_b is None:
        raise TypeError("Object arrays are not currently supported")

    return (np.matmul(m_a, m_b))
