#!/usr/bin/python3

from functools import reduce
def calc_average(a_dictionary):
    """
    Calcula e imprime a média dos valores das chaves 'age' e 'salary' em um dicionário.
    É obrigatório usar a função reduce.

    Args:
        a_dictionary (dict): Um dicionário contendo as chaves 'age' e 'salary' (opcionalmente outras chaves).
    """

    keys_to_average = ['age', 'salary']
    valid_values = [a_dictionary[key] for key in keys_to_average if key in a_dictionary and isinstance(a_dictionary[key], (int, float))]

    if not valid_values:
        print("Não há valores 'age' e/ou 'salary' válidos para calcular a média.")
        return

    total = reduce(lambda x, y: x + y, valid_values)
    average = total / len(valid_values)

    return total, average
